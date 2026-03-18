#!/usr/bin/env python3
"""
벤치마크 결과 JSON에서 OpenRouter식 per-request TPS를 계산.

Per-request TPS = output_tokens / e2e_latency (각 요청별)
e2e_latency = ttft + sum(itls)

Usage:
  python scripts/analyze_results.py bench-dataset/results/some-result.json
  python scripts/analyze_results.py bench-dataset/results/*.json
  python scripts/analyze_results.py --workload sharegpt bench-dataset/results/*.json
  python scripts/analyze_results.py --summary-only bench-dataset/results/some-result.json
  python scripts/analyze_results.py --generate-rows bench-dataset/results/*.json
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path


def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def analyze_file(path):
    with open(path) as f:
        data = json.load(f)

    ttfts = data.get("ttfts", [])
    itls = data.get("itls", [])
    output_lens = data.get("output_lens", [])

    if not ttfts or not itls or not output_lens:
        return None

    n = min(len(ttfts), len(itls), len(output_lens))
    per_request_tps = []
    e2e_latencies = []

    for i in range(n):
        e2e = ttfts[i] + sum(itls[i])
        if e2e > 0 and output_lens[i] > 0:
            per_request_tps.append(output_lens[i] / e2e)
            e2e_latencies.append(e2e)

    if not per_request_tps:
        return None

    metadata = data.get("metadata", {}) or {}

    def meta(key):
        return data.get(key, "") or metadata.get(key, "")

    return {
        "file": os.path.basename(path),
        "model_id": data.get("model_id", "unknown"),
        "backend": data.get("backend", "unknown"),
        "num_prompts": data.get("num_prompts", 0),
        "completed": data.get("completed", 0),
        "request_rate": data.get("request_rate"),
        "max_concurrency": data.get("max_concurrency"),
        "duration": data.get("duration", 0),
        "output_throughput": data.get("output_throughput", 0),
        "per_req_tps_mean": statistics.mean(per_request_tps),
        "per_req_tps_median": statistics.median(per_request_tps),
        "per_req_tps_p1": percentile(per_request_tps, 1),
        "per_req_tps_std": statistics.stdev(per_request_tps) if len(per_request_tps) > 1 else 0,
        "e2e_mean": statistics.mean(e2e_latencies),
        "e2e_median": statistics.median(e2e_latencies),
        "e2e_p99": percentile(e2e_latencies, 99),
        "date": data.get("date", ""),
        "mean_ttft_ms": data.get("mean_ttft_ms", 0),
        "median_ttft_ms": data.get("median_ttft_ms", 0),
        "p99_ttft_ms": data.get("p99_ttft_ms", 0),
        "gpu": meta("gpu"),
        "gpu_mem_util": meta("gpu_mem_util"),
        "max_model_len": meta("max_model_len"),
        "max_batched_tokens": meta("max_batched_tokens"),
        "max_num_seqs": meta("max_num_seqs"),
        "replica": meta("replica"),
        "quant": meta("quant"),
        "workload": meta("workload"),
    }


CONFIG_FIELDS = ["gpu", "gpu_mem_util", "max_model_len", "max_batched_tokens", "max_num_seqs", "replica", "quant"]


def config_key(r):
    return tuple(r.get(f, "") for f in CONFIG_FIELDS)


def config_label(r):
    gpu = r.get("gpu", "")
    if not gpu:
        return "unknown config"
    parts = [gpu.replace("-", " ")]
    replica = r.get("replica", "")
    if replica:
        parts.append(f"x{replica}")
    quant = r.get("quant", "")
    if quant:
        parts.append(quant)
    mem = r.get("gpu_mem_util", "")
    if mem:
        parts.append(f"mem={mem}")
    model_len = r.get("max_model_len", "")
    if model_len:
        parts.append(f"model_len={model_len}")
    batched = r.get("max_batched_tokens", "")
    if batched:
        parts.append(f"batched={batched}")
    seqs = r.get("max_num_seqs", "")
    if seqs:
        parts.append(f"seqs={seqs}")
    return " | ".join(parts)


def _sort_key(r):
    mc = r["max_concurrency"]
    rr = r["request_rate"]
    mc_val = mc if isinstance(mc, (int, float)) else 0
    if rr is None or (isinstance(rr, float) and rr == float("inf")):
        rr_val = float("inf")
    elif isinstance(rr, (int, float)):
        rr_val = float(rr)
    else:
        rr_val = 0
    return (mc_val, rr_val, r.get("date", ""))


def concurrency_label(result):
    rr = result["request_rate"]
    mc = result["max_concurrency"]
    parts = []
    if rr is not None and isinstance(rr, (int, float)) and rr != float("inf"):
        parts.append(f"RR={rr:.0f}")
    if mc:
        parts.append(f"C={mc}")
    return ",".join(parts) if parts else "unlimited"


def print_single(r):
    print(f"  Model:          {r['model_id']}")
    print(f"  Completed:      {r['completed']}/{r['num_prompts']} requests")
    print(f"  Load:           {concurrency_label(r)}")
    print(f"  Duration:       {r['duration']:.1f}s")
    if r["workload"]:
        print(f"  Workload:       {r['workload']}")
    if r["gpu"]:
        print(f"  GPU:            {r['gpu']} (mem={r['gpu_mem_util']}, model_len={r['max_model_len']}, batched={r['max_batched_tokens']}, seqs={r['max_num_seqs']})")
    print()
    print(f"  Aggregate TPS:  {r['output_throughput']:.2f} tok/s")
    print()
    print(f"  Per-request TPS (output_tokens / e2e_latency):")
    print(f"    Mean:         {r['per_req_tps_mean']:.2f} tok/s")
    print(f"    Median:       {r['per_req_tps_median']:.2f} tok/s")
    print(f"    P1 (worst):   {r['per_req_tps_p1']:.2f} tok/s")
    print(f"    Std:          {r['per_req_tps_std']:.2f}")
    print()
    print(f"  E2E Latency:")
    print(f"    Mean:         {r['e2e_mean']:.3f}s")
    print(f"    Median:       {r['e2e_median']:.3f}s")
    print(f"    P99:          {r['e2e_p99']:.3f}s")


def print_summary(r):
    print(f"  📊 Per-request TPS: {r['per_req_tps_mean']:.1f} tok/s (mean), {r['per_req_tps_median']:.1f} tok/s (median)")


def print_comparison_table(results):
    results.sort(key=_sort_key)

    has_workload = any(r["workload"] for r in results)
    if has_workload:
        header = f"{'Workload':<20} {'Load':<18} {'Agg TPS':>10} {'Per-req Mean':>13} {'Per-req Med':>12} {'Per-req P1':>11} {'E2E Mean':>10} {'TTFT Mean':>10}"
    else:
        header = f"{'Load':<18} {'Agg TPS':>10} {'Per-req Mean':>13} {'Per-req Med':>12} {'Per-req P1':>11} {'E2E Mean':>10} {'TTFT Mean':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        cols = []
        if has_workload:
            cols.append(f"{r['workload'] or '?':<20}")
        cols.append(f"{concurrency_label(r):<18}")
        cols.append(f"{r['output_throughput']:>10.1f}")
        cols.append(f"{r['per_req_tps_mean']:>13.1f}")
        cols.append(f"{r['per_req_tps_median']:>12.1f}")
        cols.append(f"{r['per_req_tps_p1']:>11.1f}")
        cols.append(f"{r['e2e_mean']:>9.2f}s")
        cols.append(f"{r['mean_ttft_ms']:>9.1f}ms")
        print(" ".join(cols))


def generate_matrix_rows(results):
    results.sort(key=lambda r: (r["workload"] or "", _sort_key(r)))

    model_map = {"qwen3.5-9b": "Qwen3.5-9B"}

    for r in results:
        model = model_map.get(r["model_id"], r["model_id"])
        gpu = r["gpu"].replace("-", " ") if r["gpu"] else ""
        replica = r["replica"] or ""
        quant = r["quant"] or ""
        mem = r["gpu_mem_util"] or ""
        batched = r["max_batched_tokens"] or ""
        seqs = r["max_num_seqs"] or ""
        workload = r["workload"] or "?"
        conc = concurrency_label(r)
        ot = f"{r['output_throughput']:.2f}"
        pr = f"{r['per_req_tps_mean']:.2f}"
        ttft_mean = f"{r['mean_ttft_ms']:.2f}"
        ttft_med = f"{r['median_ttft_ms']:.2f}"
        ttft_p99 = f"{r['p99_ttft_ms']:.2f}"

        print(f"| {model} | {gpu} | {replica} | {quant} | {mem} | {batched} | {seqs} | {workload} | {conc} | {ot} | {pr} | {ttft_mean} | {ttft_med} | {ttft_p99} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze vLLM benchmark results for per-request TPS")
    parser.add_argument("files", nargs="+", help="Result JSON files")
    parser.add_argument("--workload", help="Filter by workload keyword in filename or metadata")
    parser.add_argument("--summary-only", action="store_true", help="Print one-line summary only")
    parser.add_argument("--generate-rows", action="store_true", help="Generate markdown table rows from results with metadata")
    args = parser.parse_args()

    results = []
    for path in args.files:
        if not Path(path).exists():
            print(f"⚠️  File not found: {path}", file=sys.stderr)
            continue
        try:
            r = analyze_file(path)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"⚠️  Failed to parse {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        if not r:
            print(f"⚠️  No detailed data in: {os.path.basename(path)}", file=sys.stderr)
            continue
        if args.workload:
            wl = r.get("workload", "") or ""
            if args.workload not in wl and args.workload not in os.path.basename(path):
                continue
        results.append(r)

    if not results:
        print("No valid results to analyze.", file=sys.stderr)
        sys.exit(1)

    if args.generate_rows:
        generate_matrix_rows(results)
    elif args.summary_only:
        print_summary(results[-1])
    elif len(results) == 1:
        print(f"\n📊 {results[0]['file']}")
        print("=" * 50)
        print_single(results[0])
    else:
        groups = {}
        for r in results:
            key = config_key(r)
            groups.setdefault(key, []).append(r)

        for key, group in groups.items():
            label = config_label(group[0])
            print(f"\n📊 [{label}] ({len(group)} results)")
            print(f"   Model: {group[0]['model_id']}")
            print("=" * 115)
            print_comparison_table(group)
            print()

        print("  Agg TPS = aggregate output throughput (total tokens / total time)")
        print("  Per-req = per-request TPS (output_tokens / e2e_latency per request)")
        print("  P1 = 1st percentile (worst per-request TPS)")


if __name__ == "__main__":
    main()
