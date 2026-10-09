"""Standalone Transformers + PEFT implementation of the evaluation pipeline."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .parsing import parse_forecast
from .prompt import build_forecast_prompt
from .adapters import load_adapter


def normalize_history(history):
    """Float32, population std, history-only statistics: matches train/dataset.py."""
    x = np.asarray(history, dtype=np.float32)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("history must contain at least two finite numbers")
    mean, std = float(x.mean()), float(x.std())
    scaled = (x - mean) / (std + 1e-8)
    # The evaluator reconstructs raw values from its float32 normalized input.
    llm_values = scaled * std + mean
    return scaled, mean, std, llm_values


def evaluation_prompt(values, text, horizon, timestamps=None, forecast_timestamps=None):
    keep = min(len(values), 64)
    return build_forecast_prompt(
        text_list=list(text[-keep:]), values=list(values[-keep:]), pred_len=horizon,
        timestamps=list(timestamps[-keep:]) if timestamps is not None else None,
        forecast_timestamps=list(forecast_timestamps) if forecast_timestamps is not None else None,
        text_window=16,
    )


def covariate_frames(history, median, variance, stds, *, seed=42, noise_std=0.05, use_confidence=True):
    """Shifted forecast/confidence covariates with evaluation's legacy RNG order.

    history/median are normalized; variance is in original units squared.
    One call corresponds to one evaluation batch. No global RNG is changed.
    """
    history, median = map(np.asarray, (history, median))
    variance = np.asarray(variance) if use_confidence else None
    if history.ndim != 2 or median.ndim != 2 or (use_confidence and variance.shape != median.shape):
        raise ValueError("Expected [batch, context] and matching [batch, horizon] arrays")
    if len(history) != len(median) or len(stds) != len(history) or median.shape[1] < 1:
        raise ValueError("Batch sizes must match and horizon must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    rng = np.random.RandomState(seed)
    context_len, horizon = history.shape[1], median.shape[1]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=context_len, freq="D")
    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    context, future = [], []
    for i in range(len(history)):
        noisy = history[i, 1:] + rng.randn(context_len - 1) * noise_std
        hist_cov = np.concatenate([noisy, [median[i, 0]]])
        fut_cov = np.concatenate([median[i, 1:], [median[i, -1] + rng.randn() * noise_std]])
        context.append(pd.DataFrame(dict(
            id=f"series_{i}", timestamp=dates, target=history[i].astype(np.float64),
            llm_cov=hist_cov.astype(np.float64),
        )))
        future.append(pd.DataFrame(dict(
            id=f"series_{i}", timestamp=future_dates,
            llm_cov=fut_cov.astype(np.float64),
        )))
        if use_confidence:
            confidence = variance[i] / (float(stds[i]) ** 2 + 1e-8)
            hist_confidence = rng.randn(context_len) * noise_std
            context[-1]['confidence'] = hist_confidence.astype(np.float64)
            future[-1]['confidence'] = confidence.astype(np.float64)
    return pd.concat(context, ignore_index=True), pd.concat(future, ignore_index=True)


@dataclass
class Forecast:
    values: np.ndarray
    normalized_values: np.ndarray
    llm_median: np.ndarray
    llm_variance: np.ndarray | None
    responses: list[str]
    parse_info: list[dict]
    history_mean: float
    history_std: float
    prompt: str
    rollouts: np.ndarray
    mode: str


class CoCo:
    """Two LoRA-adapted models; no inference server or research-repo imports."""

    def __init__(self, llm, tokenizer, chronos):
        self.llm, self.tokenizer, self.chronos = llm, tokenizer, chronos
        self.backend = 'transformers'
        self._vllm = None

    @classmethod
    def from_pretrained(
        cls, *, device="cuda", dtype=torch.bfloat16,
        llm_base="Qwen/Qwen3-4B-Instruct-2507", chronos_base="amazon/chronos-2",
        llm_adapter="SWARM-Lab/CoCo-Qwen3-4B-LoRA",
        chronos_adapter="SWARM-Lab/CoCo-Chronos2-LoRA",
        llm_revision=None, chronos_revision=None,
        llm_adapter_revision=None, chronos_adapter_revision=None,
        local_files_only=False, backend='transformers', vllm_options=None,
    ):
        from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if backend not in {'transformers','vllm'}:
            raise ValueError('backend must be "transformers" or "vllm"')
        if backend == 'transformers' and vllm_options:
            raise ValueError('vllm_options requires backend="vllm"')
        if backend == 'vllm' and str(device) not in {'cuda','cuda:0'}:
            raise ValueError('For vLLM select physical GPUs with CUDA_VISIBLE_DEVICES and use device="cuda:0"')

        tokenizer = AutoTokenizer.from_pretrained(
            llm_base, revision=llm_revision, local_files_only=local_files_only,
        )
        generator = None
        llm = None
        if backend == 'transformers':
            llm = AutoModelForCausalLM.from_pretrained(
                llm_base, revision=llm_revision, torch_dtype=dtype,
                device_map=device, local_files_only=local_files_only,
            )
            llm = load_adapter(
                llm, llm_adapter, revision=llm_adapter_revision, local_files_only=local_files_only,
            ).eval()
        else:
            from .vllm_backend import VLLMGenerator
            generator = VLLMGenerator(llm_base,llm_adapter,tokenizer,dtype=dtype,
                base_revision=llm_revision,adapter_revision=llm_adapter_revision,
                local_files_only=local_files_only,options=vllm_options)
        try:
            base = Chronos2Model.from_pretrained(
                chronos_base, revision=chronos_revision, torch_dtype=dtype,
                device_map=device, local_files_only=local_files_only,
            )
            chronos = load_adapter(
                base, chronos_adapter, revision=chronos_adapter_revision, local_files_only=local_files_only,
            ).merge_and_unload().eval()
        except Exception:
            if generator is not None:
                generator.close()
            raise
        model = cls(llm, tokenizer, Chronos2Pipeline(chronos))
        model.backend, model._vllm = backend, generator
        return model

    def close(self):
        """Release model resources and the optional vLLM worker."""
        if self._vllm is not None:
            self._vllm.close()
        self.llm = None
        self.chronos = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @torch.inference_mode()
    def generate(self, prompt, *, temperature=0.0, seed=42, max_new_tokens=4096):
        """One completion. Greedy by default; seeded sampling is opt-in."""
        if temperature < 0 or max_new_tokens < 1:
            raise ValueError("temperature must be nonnegative and max_new_tokens positive")
        if self._vllm is not None:
            return self._vllm.generate_many(prompt,count=1,temperature=temperature,
                seed=seed,max_new_tokens=max_new_tokens)[0]
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.llm.device)
        # Do not inherit model-specific sampling defaults in greedy mode.
        generation = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          temperature=temperature if temperature > 0 else 1.0,
                          top_k=0 if temperature > 0 else 50, top_p=1.0,
                          pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                        else self.tokenizer.eos_token_id))
        devices = [self.llm.device.index] if self.llm.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            # Explicit kwargs override Qwen's generation defaults, including do_sample=True.
            output = self.llm.generate(**inputs, **generation)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.inference_mode()
    def predict_from_covariates(self, history, median, variance, stds, *, seed=42, noise_std=0.05, use_confidence=True):
        context, future = covariate_frames(history, median, variance, stds, seed=seed, noise_std=noise_std, use_confidence=use_confidence)
        pred = self.chronos.predict_df(
            context, future_df=future, prediction_length=np.asarray(median).shape[1],
            quantile_levels=[0.5], id_column="id", timestamp_column="timestamp", target="target",
        )
        grouped = pred.groupby("id")
        return np.stack([
            grouped.get_group(f"series_{i}").sort_values("timestamp")["predictions"].to_numpy()
            for i in range(len(history))
        ]).astype(np.float32)

    def predict(
        self, history, text=None, *, prediction_length=16, timestamps=None,
        forecast_timestamps=None, mode="rollouts", n_rollouts=5, temperature=1.0, seed=42,
        noise_std=0.05, max_new_tokens=4096,
    ):
        """Forecast one series in original units; text must align with history only.

        Default: five sampled rollouts, median and variance covariates.
        mode="greedy" overrides n_rollouts/temperature to 1/0 and omits the
        confidence channel entirely. Chronos2 receives the complete history.
        """
        if mode not in {"rollouts", "greedy"}:
            raise ValueError('mode must be "rollouts" or "greedy"')
        if mode == "greedy":
            n_rollouts, temperature = 1, 0.0
        elif n_rollouts < 2 or temperature <= 0:
            raise ValueError('Rollout mode requires n_rollouts >= 2 and temperature > 0; use mode="greedy" otherwise')
        if prediction_length < 1 or n_rollouts < 1:
            raise ValueError("prediction_length and n_rollouts must be positive")
        scaled, mean, std, llm_values = normalize_history(history)
        text = list(text) if text is not None else [None] * len(scaled)
        if len(text) != len(scaled):
            raise ValueError("text must align one-to-one with history (no future annotations)")
        if timestamps is not None and len(timestamps) != len(scaled):
            raise ValueError("timestamps must align with history")
        if forecast_timestamps is not None and len(forecast_timestamps) != prediction_length:
            raise ValueError("forecast_timestamps must match prediction_length")
        prompt = evaluation_prompt(llm_values.tolist(), text, prediction_length, timestamps, forecast_timestamps)
        if self._vllm is not None:
            responses = self._vllm.generate_many(prompt,count=n_rollouts,temperature=temperature,
                seed=seed,max_new_tokens=max_new_tokens)
        else:
            responses = [self.generate(prompt,temperature=temperature,seed=seed+k,max_new_tokens=max_new_tokens)
                         for k in range(n_rollouts)]
        info, rollouts = [], []
        for k, response in enumerate(responses):
            vals, parse_info = parse_forecast(response, prediction_length, k, llm_values)
            if not np.isfinite(vals).all():
                raise ValueError("LLM generated non-finite forecast values")
            info.append(parse_info)
            rollouts.append(vals)
        rollouts = np.asarray(rollouts, dtype=np.float64)
        median = np.median(rollouts, axis=0)
        variance = np.var(rollouts, axis=0).astype(np.float32) if mode == "rollouts" else None
        # The evaluator passes these tensors to Chronos2 in float32.
        median_scaled = ((median - mean) / (std + 1e-8)).astype(np.float32)
        normalized = self.predict_from_covariates(
            scaled[None], median_scaled[None], variance[None] if variance is not None else None,
            [std], seed=seed, noise_std=noise_std, use_confidence=mode == "rollouts",
        )[0]
        return Forecast(
            normalized * std + mean, normalized, median, variance,
            responses, info, mean, std, prompt, rollouts, mode,
        )
