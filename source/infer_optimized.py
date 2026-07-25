"""
CogNet Optimized Inference Engine
==================================
Inference for the optimized CogNet model.
Supports: generate, analyze, benchmark

Usage:
    python infer_optimized.py generate --prompt "The future of AI is" --max-tokens 100
    python infer_optimized.py analyze --prompt "CogNet is"
    python infer_optimized.py benchmark
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognet_1b_optimized import CogNet1BOptimized, create_cognet_1b_optimized


# ─── Model & Tokenizer Loading ───────────────────────────────────────────────

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
TOKENIZER_PATH = os.path.join(CKPT_DIR, 'bpe_tokenizer_32000.json')


_model_cache = {'model': None, 'tokenizer': None, 'device': None, 'loaded': False}


def load_model_and_tokenizer(checkpoint_path: Optional[str] = None):
    """Load model and tokenizer with caching."""
    if _model_cache['loaded']:
        return _model_cache['model'], _model_cache['tokenizer'], _model_cache['device']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = None
    if os.path.exists(TOKENIZER_PATH):
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
            print(f"Loaded BPE tokenizer (vocab={tokenizer.get_vocab_size()})")
        except ImportError:
            pass

    if tokenizer is None:
        # Fallback: simple char tokenizer
        print("Using fallback character tokenizer")
        tokenizer = _SimpleCharTokenizer()

    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else tokenizer.vocab_size

    # Create model
    model = create_cognet_1b_optimized(
        vocab_size=vocab_size,
        max_seq_len=4096,
        use_gradient_checkpointing=False,
    )

    # Load weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CKPT_DIR, 'best.pt')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(CKPT_DIR, 'latest.pt')

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        step = ckpt.get('step', ckpt.get('metrics', {}).get('step', '?'))
        print(f"Loaded model from {checkpoint_path} (step={step})")
    else:
        print("WARNING: No trained weights found. Using random initialization.")

    model = model.to(device)
    model.eval()

    # Try to compile for inference
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled for inference")
    except Exception:
        pass

    _model_cache['model'] = model
    _model_cache['tokenizer'] = tokenizer
    _model_cache['device'] = device
    _model_cache['loaded'] = True

    return model, tokenizer, device


class _SimpleCharTokenizer:
    """Fallback character tokenizer."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self._id_to_char = {i: chr(i) for i in range(min(vocab_size, 256))}
        self._char_to_id = {v: k for k, v in self._id_to_char.items()}

    def encode(self, text):
        return [self._char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join(self._id_to_char.get(i, ' ') for i in ids)

    def get_vocab_size(self):
        return self.vocab_size


# ─── Actions ──────────────────────────────────────────────────────────────────

def handle_generate(prompt: str, max_tokens: int = 100,
                    temperature: float = 0.8, top_k: int = 40) -> Dict:
    """Generate text from a prompt."""
    model, tokenizer, device = load_model_and_tokenizer()

    ids = tokenizer.encode(prompt)
    if not isinstance(ids, list):
        ids = ids.ids if hasattr(ids, 'ids') else list(ids)
    if len(ids) == 0:
        ids = [0]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_tokens,
            temperature=temperature, top_k=top_k,
        )
    elapsed = time.time() - t0

    gen_ids = output_ids[0].tolist()
    gen_text = tokenizer.decode(gen_ids) if hasattr(tokenizer, 'decode') else str(gen_ids[:100])

    return {
        'action': 'generate',
        'prompt': prompt,
        'generated_text': gen_text,
        'num_tokens': len(gen_ids),
        'time_seconds': elapsed,
        'tokens_per_second': len(gen_ids) / max(elapsed, 0.001),
    }


def handle_analyze(prompt: str) -> Dict:
    """Analyze logits, entropy, and top predictions."""
    model, tokenizer, device = load_model_and_tokenizer()

    ids = tokenizer.encode(prompt)
    if not isinstance(ids, list):
        ids = ids.ids if hasattr(ids, 'ids') else list(ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        result = model(input_ids, return_stats=True)
        logits = result['logits']

    # Last token predictions
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum().item()

    # Top 10 predictions
    topk_vals, topk_ids = torch.topk(probs, min(10, probs.size(0)))
    top_predictions = []
    for prob, tid in zip(topk_vals.tolist(), topk_ids.tolist()):
        char = tokenizer.decode([tid]) if hasattr(tokenizer, 'decode') else f'token_{tid}'
        top_predictions.append({
            'token_id': tid,
            'char': char,
            'probability': prob,
        })

    return {
        'action': 'analyze',
        'prompt': prompt,
        'entropy': entropy,
        'top_predictions': top_predictions,
    }


def handle_benchmark() -> Dict:
    """Benchmark model throughput."""
    model, tokenizer, device = load_model_and_tokenizer()
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else tokenizer.vocab_size

    # Param count
    params = sum(p.numel() for p in model.parameters())

    # Warmup
    warmup_input = torch.randint(0, vocab_size, (1, 128), device=device)
    with torch.no_grad():
        for _ in range(5):
            model(warmup_input)

    # Benchmark different sequence lengths
    results = {}
    for seq_len in [128, 256, 512, 1024, 2048]:
        try:
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

            # Timed runs
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()

            n_runs = 20
            with torch.no_grad():
                for _ in range(n_runs):
                    model(input_ids)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - t0

            tokens_per_sec = (seq_len * n_runs) / elapsed
            latency_ms = (elapsed / n_runs) * 1000

            results[f'seq_{seq_len}'] = {
                'tokens_per_second': tokens_per_sec,
                'latency_ms': latency_ms,
            }
            print(f"  seq_len={seq_len:>5d}: {tokens_per_sec:>10,.0f} tokens/s, {latency_ms:.1f}ms latency")
        except torch.cuda.OutOfMemoryError:
            results[f'seq_{seq_len}'] = {'error': 'OOM'}
            print(f"  seq_len={seq_len:>5d}: OOM")

    return {
        'action': 'benchmark',
        'parameters': params,
        'device': str(device),
        'results': results,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CogNet Optimized Inference')
    parser.add_argument('action', choices=['generate', 'analyze', 'benchmark'])
    parser.add_argument('--prompt', type=str, default='The ')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint:
        # Override checkpoint path
        global TOKENIZER_PATH
        # Load from specified checkpoint
        _model_cache['loaded'] = False

    if args.action == 'generate':
        result = handle_generate(args.prompt, args.max_tokens, args.temperature, args.top_k)
    elif args.action == 'analyze':
        result = handle_analyze(args.prompt)
    elif args.action == 'benchmark':
        result = handle_benchmark()

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
