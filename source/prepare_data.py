"""
CogNet Data Preparation Script
===============================
Prepares and tokenizes multiple datasets for training:
- Wikipedia (multilingual)
- Code datasets (The Stack, CodeParrot)
- Books (BookCorpus)
- Common Crawl subsets
- Custom local files

Outputs pre-tokenized .pt files for maximum training throughput.

Usage:
    python prepare_data.py --output-dir ./data_cache --vocab-size 32000
    python prepare_data.py --output-dir ./data_cache --datasets wiki code books
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ─── Dataset Configs ─────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    'wiki': {
        'path': 'wikimedia/wikipedia',
        'subset': '20231101.en',
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 5_000_000_000,  # 5B chars
        'description': 'Wikipedia English',
    },
    'wiki_fr': {
        'path': 'wikimedia/wikipedia',
        'subset': '20231101.fr',
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 2_000_000_000,
        'description': 'Wikipedia French',
    },
    'code': {
        'path': 'bigcode/the-stack',
        'subset': 'data',
        'split': 'train',
        'text_field': 'content',
        'max_docs': None,
        'max_chars': 5_000_000_000,
        'description': 'The Stack (multi-language code)',
        'languages': ['python', 'javascript', 'java', 'cpp', 'c', 'rust', 'go', 'typescript'],
    },
    'code_python': {
        'path': 'bigcode/the-stack',
        'subset': 'data',
        'split': 'train',
        'text_field': 'content',
        'max_docs': None,
        'max_chars': 3_000_000_000,
        'description': 'Python code from The Stack',
        'languages': ['python'],
    },
    'books': {
        'path': 'bookcorpus/bookcorpus',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 3_000_000_000,
        'description': 'BookCorpus',
    },
    'c4': {
        'path': 'allenai/c4',
        'subset': 'en',
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 10_000_000_000,
        'description': 'C4 (Colossal Clean Crawled Corpus)',
    },
    'openwebtext': {
        'path': 'openwebtext',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 5_000_000_000,
        'description': 'OpenWebText',
    },
    'alpaca': {
        'path': 'tatsu-lab/alpaca',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 500_000_000,
        'description': 'Alpaca instruction data',
        'format_fn': 'alpaca_format',
    },
    'redpajama': {
        'path': 'togethercomputer/RedPajama-Data-1T',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'max_docs': None,
        'max_chars': 10_000_000_000,
        'description': 'RedPajama 1T',
    },
}


def alpaca_format(example: Dict) -> str:
    """Format Alpaca data into text."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


# ─── Tokenizer Training ──────────────────────────────────────────────────────

def train_bpe_tokenizer(output_dir: str, vocab_size: int = 32000,
                         sample_files: Optional[List[str]] = None) -> str:
    """
    Train a BPE tokenizer on sample text data.
    Returns the path to the saved tokenizer.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Metaspace, ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        print("ERROR: 'tokenizers' library not installed.")
        print("Install with: pip install tokenizers")
        sys.exit(1)

    tokenizer_path = os.path.join(output_dir, f"bpe_tokenizer_{vocab_size}.json")
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer already exists at {tokenizer_path}")
        return tokenizer_path

    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "[PAD]",    # 0
            "[UNK]",    # 1
            "[BOS]",    # 2
            "[EOS]",    # 3
        ],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    if sample_files and len(sample_files) > 0:
        print(f"Training on {len(sample_files)} files...")
        tokenizer.train(sample_files, trainer)
    else:
        print("No sample files provided. Training on built-in data...")
        # Generate diverse sample text for tokenizer training
        sample_texts = []
        # English
        sample_texts.extend([
            "The quick brown fox jumps over the lazy dog. " * 500,
            "Science and technology have transformed our understanding of the universe. " * 500,
            "In the field of artificial intelligence, neural networks learn from data. " * 500,
        ])
        # French
        sample_texts.extend([
            "Le renard brun rapide saute par-dessus le chien paresseux. " * 500,
            "La science et la technologie ont transforme notre comprehension de l'univers. " * 500,
        ])
        # Code
        sample_texts.extend([
            "def hello_world():\n    print('Hello, World!')\n    return True\n" * 500,
            "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n" * 500,
            "import torch\nimport torch.nn as nn\nmodel = nn.Sequential(nn.Linear(768, 768))\n" * 500,
            "function fibonacci(n) {\n  if (n <= 1) return n;\n  return fibonacci(n-1) + fibonacci(n-2);\n}\n" * 500,
        ])
        tokenizer.train_from_iterator(sample_texts, trainer)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer_path


# ─── Data Processing ─────────────────────────────────────────────────────────

def process_dataset(name: str, config: Dict, tokenizer, output_dir: str,
                    seq_len: int = 4096) -> Optional[str]:
    """
    Process a single dataset and save as pre-tokenized .pt file.
    Returns the output path or None if failed.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {name} — {config.get('description', '')}")
    print(f"{'='*60}")

    output_path = os.path.join(output_dir, f"{name}_packed_seq{seq_len}.pt")
    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return output_path

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return None

    # Load dataset
    print(f"Loading {config['path']}...")
    try:
        if config.get('subset'):
            ds = load_dataset(
                config['path'],
                config['subset'],
                split=config['split'],
                streaming=True,
                trust_remote_code=True,
            )
        else:
            ds = load_dataset(
                config['path'],
                split=config['split'],
                streaming=True,
                trust_remote_code=True,
            )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    # Filter by language if specified (for code datasets)
    if config.get('languages'):
        languages = set(config['languages'])
        def lang_filter(example):
            return example.get('language', '') in languages
        ds = ds.filter(lang_filter)

    # Tokenize
    all_ids = []
    doc_count = 0
    total_chars = 0
    max_chars = config.get('max_chars', 5_000_000_000)
    text_field = config.get('text_field', 'text')
    format_fn_name = config.get('format_fn')

    t0 = time.time()

    for example in ds:
        # Get text
        if format_fn_name == 'alpaca_format':
            text = alpaca_format(example)
        else:
            text = example.get(text_field, '')

        if not text or len(text.strip()) < 20:
            continue

        # Tokenize
        ids = tokenizer.encode(text)
        if isinstance(ids, list):
            all_ids.extend(ids)
        elif hasattr(ids, 'ids'):
            all_ids.extend(ids.ids)
        else:
            all_ids.extend(list(ids))

        # Add EOS between documents
        all_ids.append(3)  # [EOS] token id

        doc_count += 1
        total_chars += len(text)

        if doc_count % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {doc_count:,} docs | {len(all_ids):,} tokens | "
                  f"{total_chars/1e9:.2f}B chars | {elapsed:.0f}s")

        if total_chars >= max_chars:
            print(f"  Reached char limit ({max_chars/1e9:.1f}B)")
            break

        if config.get('max_docs') and doc_count >= config['max_docs']:
            print(f"  Reached doc limit ({config['max_docs']:,})")
            break

    if len(all_ids) == 0:
        print("  No tokens collected!")
        return None

    # Save
    elapsed = time.time() - t0
    print(f"\n  Final: {doc_count:,} docs, {len(all_ids):,} tokens, {total_chars/1e9:.2f}B chars")
    print(f"  Time: {elapsed:.0f}s ({doc_count/max(elapsed,1):,.0f} docs/s)")

    # Pack into sequences and save
    import torch
    tensor_data = torch.tensor(all_ids, dtype=torch.long)
    torch.save(tensor_data, output_path)
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"  Saved to {output_path} ({size_gb:.2f} GB)")

    return output_path


# ─── Merge Datasets ──────────────────────────────────────────────────────────

def merge_datasets(paths: List[str], output_path: str):
    """Merge multiple pre-tokenized datasets into one."""
    print(f"\nMerging {len(paths)} datasets...")
    all_data = []

    for path in paths:
        if not os.path.exists(path):
            print(f"  Skipping (not found): {path}")
            continue
        data = torch.load(path, map_location='cpu', weights_only=True)
        all_data.append(data)
        print(f"  {path}: {len(data):,} tokens")

    if not all_data:
        print("  No data to merge!")
        return

    merged = torch.cat(all_data, dim=0)
    print(f"  Total: {len(merged):,} tokens")

    torch.save(merged, output_path)
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"  Saved to {output_path} ({size_gb:.2f} GB)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CogNet Data Preparation')
    parser.add_argument('--output-dir', type=str, default='./data_cache',
                        help='Output directory for processed data')
    parser.add_argument('--vocab-size', type=int, default=32000,
                        help='BPE vocabulary size')
    parser.add_argument('--seq-len', type=int, default=4096,
                        help='Sequence length for packing')
    parser.add_argument('--datasets', nargs='+',
                        default=['wiki', 'code'],
                        choices=list(DATASET_CONFIGS.keys()) + ['all'],
                        help='Datasets to process')
    parser.add_argument('--merge', action='store_true',
                        help='Merge all datasets into one file')
    parser.add_argument('--local-data', type=str, default=None,
                        help='Path to local data directory with .txt/.py files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Train tokenizer
    tokenizer_path = train_bpe_tokenizer(args.output_dir, args.vocab_size)

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"\nTokenizer loaded: {tokenizer.get_vocab_size()} vocab")

    # Process datasets
    if 'all' in args.datasets:
        datasets_to_process = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_process = args.datasets

    output_paths = []
    for name in datasets_to_process:
        config = DATASET_CONFIGS[name]
        path = process_dataset(name, config, tokenizer, args.output_dir, args.seq_len)
        if path:
            output_paths.append(path)

    # Process local data
    if args.local_data and os.path.exists(args.local_data):
        print(f"\nProcessing local data from {args.local_data}...")
        local_ids = []
        for ext in ['*.txt', '*.md', '*.py', '*.js', '*.java', '*.c', '*.cpp', '*.rs', '*.go']:
            for fpath in Path(args.local_data).rglob(ext):
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    ids = tokenizer.encode(text)
                    if isinstance(ids, list):
                        local_ids.extend(ids)
                    elif hasattr(ids, 'ids'):
                        local_ids.extend(ids.ids)
                    local_ids.append(3)  # EOS
                except Exception as e:
                    print(f"  Skipping {fpath}: {e}")

        if local_ids:
            local_path = os.path.join(args.output_dir, "local_packed_seq{args.seq_len}.pt")
            torch.save(torch.tensor(local_ids, dtype=torch.long), local_path)
            output_paths.append(local_path)
            print(f"  Local data: {len(local_ids):,} tokens")

    # Merge
    if args.merge and len(output_paths) > 1:
        merge_path = os.path.join(args.output_dir, f"train_packed_seq{args.seq_len}.pt")
        merge_datasets(output_paths, merge_path)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
