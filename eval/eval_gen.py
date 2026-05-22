import argparse
import random
import time

import torch
from transformers import AutoTokenizer

from lib.utils import clean
from lib.utils import shard_model as sm
from lib.utils.unsafe_import import model_from_hf_path
from model.cache_utils import StaticCache

from eval.interactive_gen import (
    decode_one_tokens,
    generate,
    llama_arg_fn,
)
import eval.interactive_gen as interactive_gen

torch.set_grad_enabled(False)


EVAL_PROMPTS = [
    "What is macOS?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
]


def build_model(hf_path, max_new_tokens, max_mem_ratio):
    model, model_str = model_from_hf_path(hf_path, max_mem_ratio=max_mem_ratio)

    sharded = False
    layer_device_map = None
    if 'llama' in model_str.lower():
        n_shards = model.lm_head.weight.device.index + 1
        if n_shards > 1:
            sharded = True
            del model.model.layers
            clean()
            cpumodel, _ = model_from_hf_path(hf_path, device_map='cpu')
            nlayers = len(cpumodel.model.layers)
            shards = [torch.nn.ModuleList([]) for _ in range(n_shards)]
            layer_device_map = []
            for i in range(n_shards):
                for j in range(int(nlayers * i / n_shards),
                               int(nlayers * (i + 1) / n_shards)):
                    shards[i].append(cpumodel.model.layers[j])
                    layer_device_map.append(i)
                shards[i] = {'device': i, 'arg_fn': llama_arg_fn, 'shard': shards[i]}
            model.model.layers = [
                sm.ShardDecoderLayers(shards, model.lm_head.weight.dtype)
            ]

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    if not sharded:
        past_kv = StaticCache(model.config, 1, 2 * max_new_tokens,
                              device=0, dtype=model.dtype)
    else:
        past_kv = StaticCache(model.config, 1, 2 * max_new_tokens,
                              layer_device_map=layer_device_map,
                              dtype=model.dtype)

    return model, tokenizer, past_kv, sharded


def maybe_compile(model, sharded):
    print('Capturing CUDA graphs, may take some time. If you are running a '
          'model over multiple GPUs, the first generation will be very slow '
          'due to compiling the model.')
    if not sharded:
        interactive_gen.decode_one_tokens = torch.compile(
            decode_one_tokens, mode="max-autotune", fullgraph=True)
    else:
        for shard in model.model.layers[0].shards:
            shard.forward = torch.compile(shard.forward,
                                          mode='max-autotune',
                                          fullgraph=True)


def format_prompt(tokenizer, prompt):
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
    return prompt


def main(args):
    model, tokenizer, past_kv, sharded = build_model(
        args.hf_path, args.max_new_tokens, args.max_mem_ratio)

    warmup_text = "This is a test of this large language model"
    callback = lambda x: x
    generate(model, tokenizer, warmup_text, 8, args.top_k, callback, past_kv)

    if not args.no_compile:
        maybe_compile(model, sharded)

    generate(model, tokenizer, warmup_text, 16, args.top_k, callback, past_kv)

    per_run_tps = []
    print("\n" + "=" * 70)
    print(f"eval_gen: {len(EVAL_PROMPTS)} prompts, max_new_tokens={args.max_new_tokens}, top_k={args.top_k}")
    print("=" * 70)

    for i, prompt in enumerate(EVAL_PROMPTS):
        text = format_prompt(tokenizer, prompt)
        run_t0 = time.time()
        ids, decoded, decode_tps = generate(model, tokenizer, text,
                                            args.max_new_tokens, args.top_k,
                                            callback, past_kv)
        run_wall = time.time() - run_t0

        print(f"\n--- Prompt {i + 1}/{len(EVAL_PROMPTS)} ---")
        print(f"USER:   {prompt}")
        print(f"OUTPUT: {decoded[0]}")
        print(f"Decode throughput: {decode_tps:.02f} tok/sec  "
              f"(wall: {run_wall:.02f}s, max_new_tokens={args.max_new_tokens})")
        per_run_tps.append(decode_tps)

    print("\n" + "=" * 70)
    print("Summary (decode throughput, tokens/sec)")
    print("=" * 70)
    for i, tps in enumerate(per_run_tps):
        print(f"  Prompt {i + 1}: {tps:.02f} tok/sec")
    avg_tps = sum(per_run_tps) / len(per_run_tps)
    print(f"  Average : {avg_tps:.02f} tok/sec")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Non-interactive generation eval: runs a fixed prompt set '
                    'and reports per-prompt + average decode throughput.')
    parser.add_argument('--hf_path', type=str, required=True,
                        help='Path to checkpoint (hfized).')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of new tokens per prompt.')
    parser.add_argument('--top_k', type=int, default=32,
                        help='Top-k for sampling.')
    parser.add_argument('--no_compile', action='store_true',
                        help='Skip torch.compile / CUDA-graph capture.')
    parser.add_argument('--disable_tf32', action='store_true',
                        help='Disable TF32 for FP32 matmuls.')
    parser.add_argument('--max_mem_ratio', type=float, default=0.7,
                        help='Per-GPU memory budget passed to model_from_hf_path.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for python/torch RNGs.')

    args = parser.parse_args()

    if not args.disable_tf32:
        torch.set_float32_matmul_precision('high')

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    main(args)
