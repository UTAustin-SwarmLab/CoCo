"""Run with: uv run python examples/forecast.py --device cuda:0"""
import argparse
import numpy as np
from coco_ts import CoCo

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--llm-base', default='Qwen/Qwen3-4B-Instruct-2507', help='Hub ID or local cached base-model directory')
    p.add_argument('--mode', choices=['rollouts','greedy'], default='rollouts')
    p.add_argument('--backend', choices=['transformers','vllm'], default='transformers')
    p.add_argument('--gpu-memory-utilization', type=float, default=.7)
    args = p.parse_args()
    options = {'gpu_memory_utilization':args.gpu_memory_utilization} if args.backend=='vllm' else None
    with CoCo.from_pretrained(device=args.device,llm_base=args.llm_base,backend=args.backend,vllm_options=options) as model:
        history = 100 + .1*np.arange(64) + np.sin(np.arange(64)/5)
        result = model.predict(history, ['Demand is gradually increasing.']*64, prediction_length=16,mode=args.mode)
        print('CoCo forecast:', result.values.tolist())
        print('LLM parsing:', result.parse_info)


if __name__ == '__main__':
    main()
