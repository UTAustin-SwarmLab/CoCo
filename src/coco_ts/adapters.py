"""Strict LoRA loading, including the submission's vLLM-style Qwen export."""
import torch
from peft import PeftConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights


def canonical_keys(weights):
    """vLLM exports model.layers.*; PEFT expects base_model.model.model.layers.*."""
    if weights and all(key.startswith('model.layers.') for key in weights):
        return {'base_model.model.'+key: value for key, value in weights.items()}
    return weights


def load_adapter(base, adapter, *, revision=None, local_files_only=False):
    kwargs = dict(revision=revision, local_files_only=local_files_only)
    config = PeftConfig.from_pretrained(adapter, **kwargs)
    config.inference_mode = True
    weights = canonical_keys(load_peft_weights(adapter, device='cpu', **kwargs))
    model = get_peft_model(base, config)
    expected = get_peft_model_state_dict(model)
    if set(expected) != set(weights):
        missing, extra = set(expected)-set(weights), set(weights)-set(expected)
        raise ValueError(f'Adapter key mismatch: {len(missing)} missing, {len(extra)} unexpected; '
                         f'examples: {sorted(missing)[:3]}, {sorted(extra)[:3]}')
    for key in expected:
        if expected[key].shape != weights[key].shape:
            raise ValueError(f'Adapter shape mismatch: {key}')
    result = set_peft_model_state_dict(model, weights)
    missing_lora = [key for key in result.missing_keys if 'lora_' in key]
    if missing_lora or result.unexpected_keys:
        raise ValueError(f'Incomplete adapter load: {missing_lora[:3]}, {result.unexpected_keys[:3]}')
    # Check every tensor against the checkpoint, not merely the absence of warnings.
    loaded = get_peft_model_state_dict(model)
    for key, value in loaded.items():
        if not torch.equal(value.detach().cpu(), weights[key].to(dtype=value.dtype)):
            raise ValueError(f'Loaded adapter tensor differs from checkpoint: {key}')
    model.requires_grad_(False)
    return model.eval()
