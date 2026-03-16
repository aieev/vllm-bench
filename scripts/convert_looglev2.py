"""
LooGLE-v2 → vLLM custom benchmark JSONL 변환기

Usage:
    python convert_loogle.py --max-context-chars 50000 --num-samples 100 --output loogle_bench.jsonl
    python convert_loogle.py --max-context-tokens 8192 --output loogle_bench_8k.jsonl
"""

import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert LooGLE-v2 to vLLM custom bench JSONL")
    parser.add_argument("--output", type=str, default="loogle_bench.jsonl")
    parser.add_argument("--num-samples", type=int, default=None, help="Max samples to convert (None = all)")
    parser.add_argument("--max-context-chars", type=int, default=50000, help="Max context length in characters (rough limit)")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="If set, uses tokenizer to truncate by token count (overrides --max-context-chars)")
    parser.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer name (required if --max-context-tokens is set)")
    parser.add_argument("--output-tokens", type=int, default=256, help="Output tokens per request")
    parser.add_argument("--task-filter", type=str, default=None, help="Filter by task name (e.g. 'Legal Case Retrieval', 'Timeline Reorder')")
    parser.add_argument("--source-filter", type=str, default=None, help="Filter by source (e.g. 'Law', 'Science')")
    parser.add_argument("--group-by-context", action="store_true", help="Sort by context to maximize prefix cache hits")
    args = parser.parse_args()

    # Load tokenizer if token-based truncation
    tokenizer = None
    if args.max_context_tokens:
        if not args.tokenizer:
            parser.error("--tokenizer is required when using --max-context-tokens")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load dataset
    print("Loading LooGLE-v2 dataset...")
    ds = load_dataset("MuLabPKU/LooGLE-v2", split="test")
    print(f"  Total rows: {len(ds)}")

    # Filter
    if args.task_filter:
        ds = ds.filter(lambda x: x["task"] == args.task_filter)
        print(f"  After task filter '{args.task_filter}': {len(ds)}")
    if args.source_filter:
        ds = ds.filter(lambda x: x["source"] == args.source_filter)
        print(f"  After source filter '{args.source_filter}': {len(ds)}")

    # Convert
    entries = []
    for row in ds:
        context = row["context"]

        # Truncate context
        if tokenizer and args.max_context_tokens:
            tokens = tokenizer.encode(context, add_special_tokens=False)
            if len(tokens) > args.max_context_tokens:
                tokens = tokens[: args.max_context_tokens]
                context = tokenizer.decode(tokens, skip_special_tokens=True)
        elif args.max_context_chars:
            context = context[: args.max_context_chars]

        instruction = row.get("instruction", "")
        question = row.get("question", "")
        options = row.get("options", "")

        # Build prompt
        if options and options != "[]":
            user_msg = f"{context}\n\nQuestion: {question}\nOptions: {options}"
        else:
            user_msg = f"{context}\n\nQuestion: {question}"

        # Format as chat (will be processed by chat template)
        prompt = f"{instruction}\n\n{user_msg}" if instruction else user_msg

        entry = {"prompt": prompt, "output_tokens": args.output_tokens}

        # Keep context ID for grouping (same context = same prefix)
        # LooGLE-v2 has multiple questions per document
        entry["_context_hash"] = hash(row["context"][:200])
        entries.append(entry)

    # Sort by context for prefix caching benefit
    if args.group_by_context:
        entries.sort(key=lambda x: x["_context_hash"])
        print("  Sorted by context (prefix cache friendly)")

    # Limit samples
    if args.num_samples:
        entries = entries[: args.num_samples]

    # Write JSONL (remove internal fields)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in entries:
            out = {"prompt": entry["prompt"], "output_tokens": entry["output_tokens"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Stats
    prompt_lens = [len(e["prompt"]) for e in entries]
    print(f"\nDone! Written {len(entries)} samples to {args.output}")
    print(f"  Prompt length (chars): min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)//len(prompt_lens)}")


if __name__ == "__main__":
    main()
