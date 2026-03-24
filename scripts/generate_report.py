#!/usr/bin/env python3
"""
CSV 데이터에서 OpenRouter 벤치마크 리포트 HTML을 생성.

Usage:
  python scripts/generate_report.py docs/benchmark-data.csv
  python scripts/generate_report.py docs/benchmark-data.csv -o docs/benchmark-report.html
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path


def load_data(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in r:
                v = r[k].strip()
                if v == '' or v == 'inf':
                    r[k] = v
                else:
                    try:
                        r[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        r[k] = v
            rows.append(r)
    return rows


def models(rows):
    seen = []
    for r in rows:
        if r['model'] not in seen:
            seen.append(r['model'])
    return seen


def workloads(rows):
    seen = []
    for r in rows:
        if r['workload'] not in seen:
            seen.append(r['workload'])
    return seen


MODEL_COLORS = ['#6366f1', '#a855f7', '#ea580c', '#0891b2', '#dc2626', '#16a34a']
MODEL_BADGES = [
    ('badge-b', '#eef2ff', '#6366f1'),
    ('badge-p', '#faf5ff', '#9333ea'),
    ('', '#fff7ed', '#ea580c'),
    ('', '#ecfeff', '#0891b2'),
    ('', '#fef2f2', '#dc2626'),
    ('', '#f0fdf4', '#16a34a'),
]
WORKLOAD_COLORS = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']
TTFT_COLORS = ['#22c55e', '#4ade80', '#86efac', '#bbf7d0']

WORKLOAD_COLORS = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff', '#eef2ff', '#f5f3ff']
CONTEXT_COLORS = ['#22c55e', '#f59e0b', '#64748b']

WORKLOAD_CATEGORIES = {
    'apps_coding': 'Code',
    'vision_single': 'Vision',
    'random_1k': '1K context',
    'random_10k': '10K context',
    'random_100k': '100K context',
}

# Display names shown in charts and tables
WORKLOAD_DISPLAY = {
    'apps_coding': 'Coding',
    'vision_single': 'Single Image',
    'random_1k': 'Short context (1K)',
    'random_10k': 'Medium context (10K)',
    'random_100k': 'Long context (100K)',
}

# Actual dataset/workload names shown in footnotes
WORKLOAD_ACTUAL = {
    'apps_coding': 'zed-industries/zeta (apps_coding)',
    'vision_single': '1MP image + 1K text tokens (vision_single)',
    'random_1k': 'Random 1K tokens (random_1k)',
    'random_10k': 'Random 10K tokens (random_10k)',
    'random_100k': 'Random 100K tokens (random_100k)',
}

ALL_WORKLOADS = ['apps_coding', 'vision_single', 'random_1k', 'random_10k', 'random_100k']


def val(r, key, fmt='.1f'):
    v = r.get(key, '')
    if v == '' or v is None or v == 0 or v == 0.0:
        return '—'
    if isinstance(v, float):
        return f'{v:{fmt}}'
    return str(v)


def chart_val(v):
    """Return None for zero/empty values so Chart.js skips the bar."""
    if isinstance(v, (int, float)) and v > 0:
        return v
    return None


def generate_html(rows):
    ms = models(rows)
    ws = workloads(rows)
    today = date.today().isoformat()

    # Compute ranges
    all_tps = [r['per_req_tps_mean'] for r in rows if isinstance(r.get('per_req_tps_mean'), (int, float))]
    all_ttft = [r['median_ttft_ms'] for r in rows if isinstance(r.get('median_ttft_ms'), (int, float))]
    all_agg = [r['output_throughput'] for r in rows if isinstance(r.get('output_throughput'), (int, float))]

    tps_range = f"{min(all_tps):.0f}–{max(all_tps):.0f}" if all_tps else "—"
    ttft_range = f"{min(all_ttft)/1000:.2f}–{max(all_ttft)/1000:.2f}" if all_ttft else "—"
    agg_peak = f"{max(all_agg):,.0f}" if all_agg else "—"

    # Model legend + badges
    badge_html = ''
    legend_html = ''
    for i, m in enumerate(ms):
        bg, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        badge_html += f'    <span class="badge" style="background:{tbg};color:{tc};border:1px solid {tc}30">{m}</span>\n'
        legend_html += f'  <span><span class="swatch" style="background:{MODEL_COLORS[i % len(MODEL_COLORS)]}"></span><b>{m}</b></span>\n'

    # Helper: display name for workload
    def wl_display(wl):
        return WORKLOAD_DISPLAY.get(wl, wl)

    # Per-model mini cards
    model_cards_html = '<div style="display:grid;grid-template-columns:repeat(' + str(len(ms)) + ',1fr);gap:1px;background:#e5e7eb;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;margin:24px 0">\n'
    for i, m in enumerate(ms):
        bg, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        m_rows = [r for r in rows if r['model'] == m]
        # Best throughput (per-req)
        tps_vals = [r['per_req_tps_mean'] for r in m_rows if isinstance(r.get('per_req_tps_mean'), (int, float)) and r['per_req_tps_mean'] > 0]
        best_tps = f"{max(tps_vals):.1f}" if tps_vals else "—"
        # Best TTFT (lowest, excluding zeros)
        ttft_vals = [r['median_ttft_ms'] for r in m_rows if isinstance(r.get('median_ttft_ms'), (int, float)) and r['median_ttft_ms'] > 0]
        best_ttft = f"{min(ttft_vals):.0f}" if ttft_vals else "—"
        # Best workload name for throughput
        best_wl = "—"
        if tps_vals:
            best_r = max((r for r in m_rows if isinstance(r.get('per_req_tps_mean'), (int, float)) and r['per_req_tps_mean'] > 0), key=lambda r: r['per_req_tps_mean'])
            best_wl = wl_display(best_r['workload'])

        model_cards_html += f'''  <div style="background:#fff;padding:18px;text-align:center">
    <div style="font-size:13px;font-weight:600;color:{color};margin-bottom:2px">{m}</div>
    <div style="font-size:9px;color:#9ca3af;margin-bottom:6px">{'MoE · ' + m.split('-')[-1] + ' active' if 'A' in m.split('-')[-1] and m.split('-')[-1][0] == 'A' else 'Dense'}</div>
    <div style="font-size:28px;font-weight:700">{best_tps} <span style="font-size:13px;font-weight:400;color:#9ca3af">tok/s</span></div>
    <div style="font-size:11px;color:#6b7280;margin-top:2px">Best output speed · {best_wl}</div>
    <div style="margin-top:10px;font-size:14px;font-weight:600">{best_ttft}<span style="font-size:11px;font-weight:400;color:#9ca3af">ms</span></div>
    <div style="font-size:10px;color:#9ca3af">Best TTFT</div>
  </div>\n'''
    model_cards_html += '</div>'

    # Performance matrix rows
    matrix_rows = ''
    for i, m in enumerate(ms):
        bg, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        short = m
        for j, r in enumerate(rows):
            if r['model'] != m:
                continue
            sep = ' style="border-top:2px solid #e5e7eb"' if j == 0 and i > 0 else ''
            wl = r['workload']
            tps = val(r, 'per_req_tps_mean')
            ttft = val(r, 'median_ttft_ms', '.0f')
            tps_s = f'{tps} tok/s' if tps != '—' else '—'
            ttft_s = f'{ttft}ms' if ttft != '—' else '—'
            matrix_rows += f'      <tr{sep}><td><span class="tag" style="background:{tbg};color:{tc}">{short}</span></td><td>{wl_display(wl)}</td><td class="r b">{tps_s}</td><td class="r">{ttft_s}</td></tr>\n'

    # Latency table rows
    latency_rows = ''
    for i, m in enumerate(ms):
        bg, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        short = m
        first = True
        for r in rows:
            if r['model'] != m:
                continue
            sep = ' style="border-top:2px solid #e5e7eb"' if first and i > 0 else ''
            first = False
            wl = r['workload']
            latency_rows += f'      <tr{sep}><td><span class="tag" style="background:{tbg};color:{tc}">{short}</span></td><td>{wl_display(wl)}</td>'
            latency_rows += f'<td class="r">{val(r,"mean_ttft_ms",".0f")}ms</td>'
            latency_rows += f'<td class="r b">{val(r,"median_ttft_ms",".0f")}ms</td>'
            latency_rows += f'<td class="r">{val(r,"p99_ttft_ms",".0f")}ms</td></tr>\n'

    # Throughput table rows
    tp_rows = ''
    for i, m in enumerate(ms):
        bg, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        short = m
        first = True
        for r in rows:
            if r['model'] != m:
                continue
            sep = ' style="border-top:2px solid #e5e7eb"' if first and i > 0 else ''
            first = False
            wl = r['workload']
            agg = val(r, 'output_throughput')
            tps = val(r, 'per_req_tps_mean')
            agg_s = f'{agg} tok/s' if agg != '—' else '—'
            tps_s = f'{tps} tok/s' if tps != '—' else '—'
            tp_rows += f'      <tr{sep}><td><span class="tag" style="background:{tbg};color:{tc}">{short}</span></td><td>{wl_display(wl)}</td><td class="r">{agg_s}</td><td class="r b">{tps_s}</td></tr>\n'

    # Chart data
    model_labels = json.dumps(ms)

    # All workloads colors: Medium_coding (blue), Vision (purple), 1K (green), 10K (orange), 100K (gray)
    ALL_COLORS = ['#6366f1', '#a855f7', '#22c55e', '#f59e0b', '#64748b']

    # Output Speed by Input Token Count (all workloads)
    speed_datasets = ''
    for j, wl in enumerate(ws):
        data = []
        for m in ms:
            v = next((r['per_req_tps_mean'] for r in rows if r['model'] == m and r['workload'] == wl and isinstance(r.get('per_req_tps_mean'), (int, float))), None)
            data.append(chart_val(v))
        speed_datasets += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(data)}, backgroundColor: '{ALL_COLORS[j % len(ALL_COLORS)]}', borderRadius: 4 }},\n"

    # TTFT by Input Token Count (all workloads)
    ttft_datasets = ''
    for j, wl in enumerate(ws):
        data = []
        for m in ms:
            v = next((r['median_ttft_ms'] for r in rows if r['model'] == m and r['workload'] == wl and isinstance(r.get('median_ttft_ms'), (int, float))), None)
            data.append(chart_val(v) / 1000 if chart_val(v) else None)
        ttft_datasets += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(data)}, backgroundColor: '{ALL_COLORS[j % len(ALL_COLORS)]}', borderRadius: 4 }},\n"

    # Aggregate throughput (all workloads)
    agg_datasets = ''
    for j, wl in enumerate(ws):
        data = []
        for m in ms:
            v = next((r['output_throughput'] for r in rows if r['model'] == m and r['workload'] == wl and isinstance(r.get('output_throughput'), (int, float))), None)
            data.append(chart_val(v))
        agg_datasets += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(data)}, backgroundColor: '{ALL_COLORS[j % len(ALL_COLORS)]}', borderRadius: 4 }},\n"

    # Workload mapping footnote for methodology
    workload_mapping = ' · '.join(f'{wl_display(w)} = {WORKLOAD_ACTUAL.get(w, w)}' for w in ws)
    dataset_names = ', '.join(f'{WORKLOAD_ACTUAL.get(w, w)}' for w in ws)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIEEV Provider Performance Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fff; color: #111827; line-height: 1.5; }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 0 32px; }}
  @media print {{ body {{ background: #fff; }} .page-break {{ page-break-before: always; }} .container {{ padding: 0 16px; }} }}
  .header {{ border-bottom: 2px solid #111827; padding: 40px 0 20px; }}
  .header h1 {{ font-size: 26px; font-weight: 700; }}
  .header .badges {{ display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }}
  .badge {{ display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 12px; font-weight: 500; }}
  .header .meta {{ color: #9ca3af; font-size: 13px; margin-top: 8px; }}
  .model-legend {{ display: flex; gap: 20px; margin: 20px 0 8px; font-size: 13px; align-items: center; flex-wrap: wrap; }}
  .model-legend .swatch {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; margin-right: 6px; vertical-align: middle; }}
  .metrics-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #e5e7eb; border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; margin: 12px 0 24px; }}
  .mc {{ background: #fff; padding: 18px; text-align: center; }}
  .mc .label {{ font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }}
  .mc .val {{ font-size: 26px; font-weight: 700; margin: 2px 0; }}
  .mc .val .u {{ font-size: 13px; font-weight: 400; color: #9ca3af; }}
  .mc .sub {{ font-size: 10px; color: #6b7280; }}
  .section {{ padding: 28px 0 12px; }}
  .section h2 {{ font-size: 19px; font-weight: 700; margin-bottom: 2px; }}
  .section .sd {{ font-size: 13px; color: #9ca3af; margin-bottom: 18px; }}
  .divider {{ border: none; border-top: 1px solid #e5e7eb; margin: 6px 0 0; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; margin-bottom: 14px; }}
  .card h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 2px; }}
  .card .cd {{ font-size: 11px; color: #9ca3af; margin-bottom: 12px; }}
  .cb {{ position: relative; height: 280px; }}
  table.dt {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  .dt th {{ text-align: left; padding: 7px 6px; border-bottom: 2px solid #e5e7eb; color: #9ca3af; font-weight: 500; font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .dt td {{ padding: 6px 6px; border-bottom: 1px solid #f3f4f6; }}
  .dt .r {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .dt .b {{ font-weight: 600; }}
  .dt .m {{ color: #d1d5db; }}
  .tag {{ display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 9px; font-weight: 600; }}
  .fn {{ font-size: 10px; color: #d1d5db; margin-top: 8px; }}
  footer {{ border-top: 1px solid #e5e7eb; padding: 16px 0; margin-top: 20px; text-align: center; font-size: 11px; color: #d1d5db; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>AIEEV — Provider Performance Report</h1>
  <div class="badges">
{badge_html}    <span class="badge" style="background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0">FP8</span>
    <span class="badge" style="background:#f9fafb;color:#6b7280;border:1px solid #e5e7eb">vLLM 0.18.1</span>
  </div>
  <div class="meta">Provider Benchmark Report · {today}</div>
</div>

{model_cards_html}

<!-- PAGE 2: Throughput -->
<div class="page-break"></div>
<div class="section">
  <h2>1. Output Speed</h2>
  <div class="sd">How fast is each request? · Tokens per second by context length · Higher is better</div>
  <hr class="divider">
</div>

<div class="card">
  <h3>Output Speed by Input Token Count (Context Length)</h3>
  <div class="cd">Tokens per second · Higher is better</div>
  <div class="cb">
    <canvas id="speedChart"></canvas>
  </div>
</div>

<div class="card">
  <table class="dt">
    <thead><tr><th>Model</th><th>Workload</th><th class="r b">Output Speed (tok/s)</th><th class="r">TTFT Median (Time to First Token)</th></tr></thead>
    <tbody>
{matrix_rows}    </tbody>
  </table>
</div>

<!-- PAGE 3: Latency -->
<div class="page-break"></div>
<div class="section">
  <h2>2. Latency</h2>
  <div class="sd">How quickly does the response start? · Time to First Token by context length · Lower is better</div>
  <hr class="divider">
</div>

<div class="card">
  <h3>Latency by Input Token Count (Context Length)</h3>
  <div class="cd">Time to First Token · Seconds · Lower is better</div>
  <div class="cb">
    <canvas id="ttftChart"></canvas>
  </div>
</div>

<div class="card">
  <table class="dt">
    <thead><tr><th>Model</th><th>Workload</th><th class="r">TTFT Mean</th><th class="r b">TTFT Median</th><th class="r">TTFT P99</th></tr></thead>
    <tbody>
{latency_rows}    </tbody>
  </table>
</div>

<!-- PAGE 4: Capacity -->
<div class="page-break"></div>
<div class="section">
  <h2>3. Capacity</h2>
  <div class="sd">How much load can the server handle? · Total tokens processed across all concurrent requests · Higher is better</div>
  <hr class="divider">
</div>

<div class="card">
  <h3>Total Throughput Under Concurrent Load</h3>
  <div class="cd">Total tokens/sec across all concurrent requests</div>
  <div class="cb">
    <canvas id="aggChart"></canvas>
  </div>
</div>

<div class="card">
  <table class="dt">
    <thead><tr><th>Model</th><th>Workload</th><th class="r b">Server Throughput (total tok/s)</th><th class="r">Per-Request (tok/s)</th></tr></thead>
    <tbody>
{tp_rows}    </tbody>
  </table>
  <div class="fn" style="margin-top:12px; font-size:11px; color:#9ca3af; line-height:1.6">
    <b>Test methodology:</b> All benchmarks run with <b>vllm bench serve</b> (vLLM 0.18.1) against a live endpoint.<br>
    500 requests per workload · Max concurrency: 16 · Request rate: unlimited (all sent immediately, capped by concurrency).<br>
    Throughput = output_tokens / generation_time per request. Total Throughput = total output tokens / total benchmark duration.<br>
    <b>Workload mapping:</b> {workload_mapping}
  </div>
</div>

<footer>AIEEV Provider Benchmark · {' + '.join(ms)} · {today}</footer>

</div>

<script>
const base = {{ responsive: true, maintainAspectRatio: false, animation: false }};
const noGrid = {{ display: false }};
const noBorder = {{ display: false }};
const lightGrid = {{ color: '#f3f4f6' }};

// Highlight plugin: click legend to highlight, click again to reset
const highlightPlugin = {{
  id: 'legendHighlight',
  beforeInit(chart) {{ chart._highlightIndex = -1; }},
  afterInit(chart) {{
    const origClick = chart.options.plugins.legend.onClick;
    chart.options.plugins.legend.onClick = function(e, item, legend) {{
      const ci = item.datasetIndex;
      const chart = legend.chart;
      if (chart._highlightIndex === ci) {{
        chart._highlightIndex = -1;
        chart.data.datasets.forEach(ds => {{ ds.backgroundColor = ds._origColor || ds.backgroundColor; }});
      }} else {{
        chart._highlightIndex = ci;
        chart.data.datasets.forEach((ds, i) => {{
          if (!ds._origColor) ds._origColor = ds.backgroundColor;
          ds.backgroundColor = i === ci ? ds._origColor : ds._origColor + '25';
        }});
      }}
      chart.update();
    }};
  }}
}};
Chart.register(highlightPlugin);

new Chart(document.getElementById('speedChart'), {{
  type: 'bar',
  data: {{
    labels: {model_labels},
    datasets: [
{speed_datasets}    ]
  }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'rectRounded' }} }},
    tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw + ' tok/s' }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Output speed (tokens / second)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }}, x: {{ grid: noGrid, border: noBorder }} }} }}
}});

new Chart(document.getElementById('ttftChart'), {{
  type: 'bar',
  data: {{
    labels: {model_labels},
    datasets: [
{ttft_datasets}    ]
  }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'rectRounded' }} }},
    tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw + 's' }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Seconds to First Token', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }}, x: {{ grid: noGrid, border: noBorder }} }} }}
}});

new Chart(document.getElementById('aggChart'), {{
  type: 'bar',
  data: {{
    labels: {model_labels},
    datasets: [
{agg_datasets}    ]
  }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'rectRounded' }} }},
    tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw.toFixed(1) + ' tok/s' }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Total throughput (tokens / second)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }}, x: {{ grid: noGrid, border: noBorder }} }} }}
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark report HTML from CSV')
    parser.add_argument('csv', help='Path to benchmark-data.csv')
    parser.add_argument('-o', '--output', default='docs/benchmark-report.html', help='Output HTML path')
    args = parser.parse_args()

    rows = load_data(args.csv)
    html = generate_html(rows)
    Path(args.output).write_text(html)
    print(f"Report generated: {args.output}")


if __name__ == '__main__':
    main()
