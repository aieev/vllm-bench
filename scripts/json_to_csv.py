#!/usr/bin/env python3
"""
vLLM 벤치마크 결과 JSON을 benchmark-data.csv에 추가.

Usage:
  python scripts/json_to_csv.py bench-dataset/results/some-result.json
  python scripts/json_to_csv.py bench-dataset/results/*.json
  python scripts/json_to_csv.py --latest          # 가장 최근 결과 1개
  python scripts/json_to_csv.py --latest 5        # 가장 최근 결과 5개
"""
import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

CSV_PATH = 'docs/benchmark-data.csv'
CSV_HEADERS = [
    'model', 'workload', 'date',
    'gpu_name', 'gpu_count', 'replica', 'quant', 'gpu_mem_util',
    'max_model_len', 'max_batched_tokens', 'max_num_seqs', 'vllm_version',
    'num_prompts', 'completed', 'max_concurrency', 'request_rate',
    'duration_s', 'total_input_tokens', 'total_output_tokens',
    'output_throughput', 'per_req_tps_mean', 'per_req_tps_median',
    'per_req_tps_p1', 'e2e_mean_s',
    'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
    'mean_tpot_ms', 'median_tpot_ms',
    'mean_itl_ms', 'median_itl_ms',
]


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def extract_row(path):
    with open(path) as f:
        d = json.load(f)

    # Per-request TPS (calculated from detailed data)
    ttfts = d.get('ttfts', [])
    itls = d.get('itls', [])
    output_lens = d.get('output_lens', [])

    per_req_tps = []
    e2e_latencies = []
    if ttfts and itls and output_lens:
        n = min(len(ttfts), len(itls), len(output_lens))
        for i in range(n):
            e2e = ttfts[i] + sum(itls[i])
            if e2e > 0 and output_lens[i] > 0:
                per_req_tps.append(output_lens[i] / e2e)
                e2e_latencies.append(e2e)

    def meta(key):
        v = d.get(key, '')
        if not v:
            m = d.get('metadata', {}) or {}
            v = m.get(key, '')
        return v

    def fmt(v, f='.2f'):
        if v is None or v == '' or v == 0:
            return ''
        if isinstance(v, float):
            return f'{v:{f}}'
        return str(v)

    rr = d.get('request_rate')
    if rr is None or (isinstance(rr, float) and rr == float('inf')):
        rr_s = 'inf'
    else:
        rr_s = fmt(rr)

    row = {
        'model': d.get('model_id', ''),
        'workload': meta('workload'),
        'date': d.get('date', '')[:8] if d.get('date') else '',
        'gpu_name': meta('gpu'),
        'gpu_count': '1',
        'replica': meta('replica'),
        'quant': meta('quant'),
        'gpu_mem_util': meta('gpu_mem_util'),
        'max_model_len': meta('max_model_len'),
        'max_batched_tokens': meta('max_batched_tokens'),
        'max_num_seqs': meta('max_num_seqs'),
        'vllm_version': '',
        'num_prompts': str(d.get('num_prompts', '')),
        'completed': str(d.get('completed', '')),
        'max_concurrency': str(d.get('max_concurrency', '')),
        'request_rate': rr_s,
        'duration_s': fmt(d.get('duration')),
        'total_input_tokens': str(d.get('total_input_tokens', '')),
        'total_output_tokens': str(d.get('total_generated_tokens', '')),
        'output_throughput': fmt(d.get('output_throughput')),
        'per_req_tps_mean': fmt(statistics.mean(per_req_tps), '.1f') if per_req_tps else '',
        'per_req_tps_median': fmt(statistics.median(per_req_tps), '.1f') if per_req_tps else '',
        'per_req_tps_p1': fmt(percentile(per_req_tps, 1), '.1f') if per_req_tps else '',
        'e2e_mean_s': fmt(statistics.mean(e2e_latencies), '.3f') if e2e_latencies else '',
        'mean_ttft_ms': fmt(d.get('mean_ttft_ms')),
        'median_ttft_ms': fmt(d.get('median_ttft_ms')),
        'p99_ttft_ms': fmt(d.get('p99_ttft_ms')),
        'mean_tpot_ms': fmt(d.get('mean_tpot_ms')),
        'median_tpot_ms': fmt(d.get('median_tpot_ms')),
        'mean_itl_ms': fmt(d.get('mean_itl_ms')),
        'median_itl_ms': fmt(d.get('median_itl_ms')),
    }
    return row


def _row_key(row):
    return (row.get('model'), row.get('workload'), row.get('date'),
            row.get('max_concurrency'), row.get('request_rate'))


def _load_existing_keys(csv_path):
    keys = set()
    if not Path(csv_path).exists() or os.path.getsize(csv_path) == 0:
        return keys
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            keys.add(_row_key(row))
    return keys


def append_to_csv(rows, csv_path=CSV_PATH):
    existing = _load_existing_keys(csv_path)
    new_rows = [r for r in rows if _row_key(r) not in existing]
    skipped = len(rows) - len(new_rows)

    if skipped:
        print(f"  ⏭️  {skipped} duplicate(s) skipped")

    if not new_rows:
        print("No new rows to add.")
        return 0

    exists = Path(csv_path).exists() and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not exists:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description='Append vLLM benchmark JSON results to CSV')
    parser.add_argument('files', nargs='*', help='Result JSON files')
    parser.add_argument('--latest', nargs='?', const=1, type=int, help='Use N most recent results (default: 1)')
    parser.add_argument('--csv', default=CSV_PATH, help=f'CSV output path (default: {CSV_PATH})')
    parser.add_argument('--dry-run', action='store_true', help='Print rows without writing')
    args = parser.parse_args()

    files = args.files
    if args.latest:
        all_files = sorted(Path('bench-dataset/results').glob('*.json'))
        files = [str(f) for f in all_files[-args.latest:]]

    if not files:
        parser.print_help()
        sys.exit(1)

    rows = []
    for f in files:
        try:
            row = extract_row(f)
            if not row['model'] or not row['workload']:
                print(f"⚠️  Skipped {os.path.basename(f)}: missing model or workload", file=sys.stderr)
                continue
            rows.append(row)
            model = row['model']
            wl = row['workload']
            tps = row['per_req_tps_mean'] or '—'
            ttft = row['median_ttft_ms'] or '—'
            print(f"  ✅ {model} / {wl}: {tps} tok/s, TTFT {ttft}ms")
        except Exception as e:
            print(f"⚠️  Failed {os.path.basename(f)}: {e}", file=sys.stderr)

    if not rows:
        print("No valid results to add.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run — {len(rows)} rows would be appended to {args.csv}")
        print(','.join(CSV_HEADERS))
        for row in rows:
            print(','.join(row.get(h, '') for h in CSV_HEADERS))
    else:
        added = append_to_csv(rows, args.csv)
        if added:
            print(f"\n✅ {added} rows appended to {args.csv}")


if __name__ == '__main__':
    main()
