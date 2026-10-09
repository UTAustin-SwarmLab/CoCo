"""Optional local vLLM engine. Imported only when backend='vllm'."""
from pathlib import Path


class VLLMGenerator:
    def __init__(self, base, adapter, tokenizer, *, dtype, base_revision=None,
                 adapter_revision=None, local_files_only=False, options=None):
        from huggingface_hub import snapshot_download
        from peft import PeftConfig
        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError('Install the optional backend with: uv sync --extra vllm') from exc

        if not Path(adapter).is_dir():
            adapter = snapshot_download(adapter, revision=adapter_revision,
                allow_patterns=['adapter_config.json','adapter_model.safetensors'],
                local_files_only=local_files_only)
        if local_files_only and not Path(base).is_dir():
            base = snapshot_download(base, revision=base_revision, local_files_only=True)
        config = PeftConfig.from_pretrained(adapter, local_files_only=True)
        if config.peft_type != 'LORA':
            raise ValueError('The vLLM backend requires a LoRA adapter')
        settings = dict(max_model_len=8192, max_num_seqs=8,
                        gpu_memory_utilization=0.7, enforce_eager=True,
                        enable_prefix_caching=False, enable_chunked_prefill=False)
        options = dict(options or {})
        protected = {'model','revision','dtype','enable_lora','max_lora_rank','generation_config','seed'}
        if protected.intersection(options):
            raise ValueError(f'Use top-level arguments, not vllm_options, for: {sorted(protected.intersection(options))}')
        settings.update(options)
        self.engine = LLM(model=base, revision=base_revision, dtype=str(dtype).removeprefix('torch.'),
            enable_lora=True, max_lora_rank=max(8, config.r), seed=42,
            generation_config='vllm', **settings)
        self.request = LoRARequest('coco', 1, str(adapter))
        self.tokenizer = tokenizer

    def generate_many(self, prompt, *, count, temperature, seed, max_new_tokens):
        from vllm import SamplingParams
        ids = self.tokenizer.apply_chat_template(
            [{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True)
        parameters = [SamplingParams(n=1, temperature=temperature, top_p=1.0, top_k=-1,
                                     max_tokens=max_new_tokens, seed=seed+k) for k in range(count)]
        outputs = self.engine.generate([{'prompt_token_ids':ids} for _ in range(count)],
            parameters, lora_request=self.request, use_tqdm=False)
        if len(outputs) != count:
            raise RuntimeError('vLLM returned an unexpected number of rollouts')
        return [out.outputs[0].text for out in outputs]

    def close(self):
        """Release the local worker process; safe to call more than once."""
        engine = getattr(self, 'engine', None)
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            self.engine = None
