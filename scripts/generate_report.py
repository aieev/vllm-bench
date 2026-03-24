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
from collections import defaultdict
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


def unique_ordered(rows, key):
    seen = []
    for r in rows:
        v = r[key]
        if v not in seen:
            seen.append(v)
    return seen


def models(rows):
    return unique_ordered(rows, 'model')


def workloads(rows):
    return unique_ordered(rows, 'workload')


MODEL_COLORS = ['#6366f1', '#a855f7', '#ea580c', '#0891b2', '#dc2626', '#16a34a', '#d97706', '#0d9488']
MODEL_BADGES = [
    ('badge-b', '#eef2ff', '#6366f1'),
    ('badge-p', '#faf5ff', '#9333ea'),
    ('', '#fff7ed', '#ea580c'),
    ('', '#ecfeff', '#0891b2'),
    ('', '#fef2f2', '#dc2626'),
    ('', '#f0fdf4', '#16a34a'),
    ('', '#fffbeb', '#d97706'),
    ('', '#f0fdfa', '#0d9488'),
]

WORKLOAD_DISPLAY = {
    'apps_coding': 'Coding',
    'vision_single': 'Single Image',
    'random_1k': 'Short context (1K)',
    'random_10k': 'Medium context (10K)',
    'random_100k': 'Long context (100K)',
    'sharegpt': 'ShareGPT',
    'mt_bench': 'MT-Bench',
    'blazedit': 'BlazeEdit',
    'blazedit_10k': 'BlazeEdit 10K',
    'random_4k': 'Context (4K)',
    'random_32k': 'Long context (32K)',
    'random_128k': 'Long context (128K)',
}

WORKLOAD_ACTUAL = {
    'apps_coding': 'zed-industries/zeta (apps_coding)',
    'vision_single': '1MP image + 1K text tokens (vision_single)',
    'random_1k': 'Random 1K tokens (random_1k)',
    'random_10k': 'Random 10K tokens (random_10k)',
    'random_100k': 'Random 100K tokens (random_100k)',
}

LINE_COLORS = ['#6366f1', '#a855f7', '#22c55e', '#f59e0b', '#ea580c', '#0891b2', '#dc2626', '#16a34a']
ALL_COLORS = ['#6366f1', '#a855f7', '#22c55e', '#f59e0b', '#64748b', '#ea580c', '#0891b2', '#dc2626']


def val(r, key, fmt='.1f'):
    v = r.get(key, '')
    if v == '' or v is None or v == 0 or v == 0.0:
        return '—'
    if isinstance(v, float):
        return f'{v:{fmt}}'
    return str(v)


def chart_val(v):
    if isinstance(v, (int, float)) and v > 0:
        return v
    return None


def conc_label(r):
    rr = r.get('request_rate', '')
    mc = r.get('max_concurrency', '')
    parts = []
    if rr and rr != 'inf' and isinstance(rr, (int, float)):
        parts.append(f'RR={rr:.0f}')
    if mc and isinstance(mc, (int, float)):
        parts.append(f'C={mc:.0f}')
    return ', '.join(parts) if parts else '—'


def replica_val(r):
    v = r.get('replica', '')
    if v == '' or v is None:
        return '—'
    return str(int(v)) if isinstance(v, float) else str(v)


def wl_display(wl):
    return WORKLOAD_DISPLAY.get(wl, wl)


def generate_html(rows):
    ms = models(rows)
    ws = workloads(rows)
    today = date.today().isoformat()

    # --- Summary stats ---
    all_tps = [r['per_req_tps_mean'] for r in rows if isinstance(r.get('per_req_tps_mean'), (int, float))]
    all_ttft = [r['median_ttft_ms'] for r in rows if isinstance(r.get('median_ttft_ms'), (int, float))]
    all_agg = [r['output_throughput'] for r in rows if isinstance(r.get('output_throughput'), (int, float))]

    # --- Badges ---
    badge_html = ''
    for i, m in enumerate(ms):
        _, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        badge_html += f'    <span class="badge" style="background:{tbg};color:{tc};border:1px solid {tc}30">{m}</span>\n'

    # --- Model summary cards ---
    model_cards_html = '<div style="display:grid;grid-template-columns:repeat(' + str(min(len(ms), 4)) + ',1fr);gap:1px;background:#e5e7eb;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;margin:24px 0">\n'
    for i, m in enumerate(ms):
        _, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        m_rows = [r for r in rows if r['model'] == m]
        tps_vals = [r['per_req_tps_mean'] for r in m_rows if isinstance(r.get('per_req_tps_mean'), (int, float)) and r['per_req_tps_mean'] > 0]
        best_tps = f"{max(tps_vals):.1f}" if tps_vals else "—"
        ttft_vals = [r['median_ttft_ms'] for r in m_rows if isinstance(r.get('median_ttft_ms'), (int, float)) and r['median_ttft_ms'] > 0]
        best_ttft = f"{min(ttft_vals):.0f}" if ttft_vals else "—"
        best_wl = "—"
        if tps_vals:
            best_r = max((r for r in m_rows if isinstance(r.get('per_req_tps_mean'), (int, float)) and r['per_req_tps_mean'] > 0), key=lambda r: r['per_req_tps_mean'])
            best_wl = wl_display(best_r['workload'])

        model_cards_html += f'''  <div style="background:#fff;padding:18px;text-align:center">
    <div style="font-size:13px;font-weight:600;color:{color};margin-bottom:2px">{m}</div>
    <div style="font-size:28px;font-weight:700">{best_tps} <span style="font-size:13px;font-weight:400;color:#9ca3af">tok/s</span></div>
    <div style="font-size:11px;color:#6b7280;margin-top:2px">Best output speed · {best_wl}</div>
    <div style="margin-top:10px;font-size:14px;font-weight:600">{best_ttft}<span style="font-size:11px;font-weight:400;color:#9ca3af">ms</span></div>
    <div style="font-size:10px;color:#9ca3af">Best TTFT</div>
  </div>\n'''
    model_cards_html += '</div>'

    # =================================================================
    #  Section 1: Output Speed table (with RR, C, Replica columns)
    # =================================================================
    matrix_rows = ''
    for i, m in enumerate(ms):
        _, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        first = True
        for r in rows:
            if r['model'] != m:
                continue
            sep = ' style="border-top:2px solid #e5e7eb"' if first and i > 0 else ''
            first = False
            tps = val(r, 'per_req_tps_mean')
            ttft = val(r, 'median_ttft_ms', '.0f')
            matrix_rows += f'      <tr{sep}><td><span class="tag" style="background:{tbg};color:{tc}">{m}</span></td><td>{wl_display(r["workload"])}</td><td class="r">{replica_val(r)}</td><td class="r">{conc_label(r)}</td><td class="r b">{tps} tok/s</td><td class="r">{ttft}ms</td></tr>\n'

    # =================================================================
    #  Section 2: Latency table
    # =================================================================
    latency_rows = ''
    for i, m in enumerate(ms):
        _, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        first = True
        for r in rows:
            if r['model'] != m:
                continue
            sep = ' style="border-top:2px solid #e5e7eb"' if first and i > 0 else ''
            first = False
            latency_rows += f'      <tr{sep}><td><span class="tag" style="background:{tbg};color:{tc}">{m}</span></td><td>{wl_display(r["workload"])}</td><td class="r">{replica_val(r)}</td><td class="r">{conc_label(r)}</td>'
            latency_rows += f'<td class="r">{val(r,"mean_ttft_ms",".0f")}ms</td>'
            latency_rows += f'<td class="r b">{val(r,"median_ttft_ms",".0f")}ms</td>'
            latency_rows += f'<td class="r">{val(r,"p99_ttft_ms",".0f")}ms</td></tr>\n'

    # =================================================================
    #  Section 3: Capacity table
    # =================================================================
    tp_rows = ''
    for i, m in enumerate(ms):
        _, tbg, tc = MODEL_BADGES[i % len(MODEL_BADGES)]
        first = True
        for r in rows:
            if r['model'] != m:
                continue
            sep = ' style="border-top:2px solid #e5e7eb"' if first and i > 0 else ''
            first = False
            agg = val(r, 'output_throughput')
            tps = val(r, 'per_req_tps_mean')
            tp_rows += f'      <tr{sep}><td><span class="tag" style="background:{tbg};color:{tc}">{m}</span></td><td>{wl_display(r["workload"])}</td><td class="r">{replica_val(r)}</td><td class="r">{conc_label(r)}</td><td class="r">{agg} tok/s</td><td class="r b">{tps} tok/s</td></tr>\n'

    # =================================================================
    #  Existing bar charts (workload comparison)
    # =================================================================
    model_labels = json.dumps(ms)

    speed_datasets = ''
    for j, wl in enumerate(ws):
        data = []
        for m in ms:
            v = next((r['per_req_tps_mean'] for r in rows if r['model'] == m and r['workload'] == wl and isinstance(r.get('per_req_tps_mean'), (int, float))), None)
            data.append(chart_val(v))
        speed_datasets += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(data)}, backgroundColor: '{ALL_COLORS[j % len(ALL_COLORS)]}', borderRadius: 4 }},\n"

    ttft_datasets = ''
    for j, wl in enumerate(ws):
        data = []
        for m in ms:
            v = next((r['median_ttft_ms'] for r in rows if r['model'] == m and r['workload'] == wl and isinstance(r.get('median_ttft_ms'), (int, float))), None)
            data.append(chart_val(v) / 1000 if chart_val(v) else None)
        ttft_datasets += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(data)}, backgroundColor: '{ALL_COLORS[j % len(ALL_COLORS)]}', borderRadius: 4 }},\n"

    agg_datasets = ''
    for j, wl in enumerate(ws):
        data = []
        for m in ms:
            v = next((r['output_throughput'] for r in rows if r['model'] == m and r['workload'] == wl and isinstance(r.get('output_throughput'), (int, float))), None)
            data.append(chart_val(v))
        agg_datasets += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(data)}, backgroundColor: '{ALL_COLORS[j % len(ALL_COLORS)]}', borderRadius: 4 }},\n"

    workload_mapping = ' · '.join(f'{wl_display(w)} = {WORKLOAD_ACTUAL.get(w, w)}' for w in ws)

    # =================================================================
    #  Section 4: Saturation Analysis (TPS & TTFT vs Concurrency)
    #  Group by (model, workload, replica) — X axis = concurrency
    # =================================================================
    saturation_charts_html = ''
    saturation_charts_js = ''
    sat_chart_idx = 0

    # Collect groups that have multiple concurrency values
    sat_groups = defaultdict(list)
    for r in rows:
        mc = r.get('max_concurrency', '')
        if not isinstance(mc, (int, float)):
            continue
        key = (r['model'], r['workload'], replica_val(r))
        sat_groups[key].append(r)

    # Filter to groups with 2+ different concurrency values
    sat_groups = {k: sorted(v, key=lambda x: x['max_concurrency']) for k, v in sat_groups.items() if len(set(x['max_concurrency'] for x in v)) >= 2}

    if sat_groups:
        # Group by (model, replica) for combined charts
        model_replica_groups = defaultdict(lambda: defaultdict(list))
        for (model, wl, rep), rr_rows in sat_groups.items():
            model_replica_groups[(model, rep)][wl] = rr_rows

        for (model, rep), wl_dict in model_replica_groups.items():
            tps_id = f'satTps_{sat_chart_idx}'
            ttft_id = f'satTtft_{sat_chart_idx}'
            sat_chart_idx += 1

            label = f'{model} (replica {rep})'

            saturation_charts_html += f'''
<div class="card">
  <h3>{label} — Per-Request TPS vs Concurrency</h3>
  <div class="cd">Higher is better · Each line = one workload</div>
  <div class="cb"><canvas id="{tps_id}"></canvas></div>
</div>
<div class="card">
  <h3>{label} — TTFT vs Concurrency</h3>
  <div class="cd">Lower is better · Each line = one workload</div>
  <div class="cb"><canvas id="{ttft_id}"></canvas></div>
</div>
'''
            # Collect all concurrency values across workloads for this model
            all_conc = sorted(set(r['max_concurrency'] for rr_rows in wl_dict.values() for r in rr_rows))
            conc_labels = json.dumps([int(c) for c in all_conc])

            tps_ds = ''
            ttft_ds = ''
            for j, (wl, rr_rows) in enumerate(wl_dict.items()):
                conc_to_tps = {r['max_concurrency']: r.get('per_req_tps_mean') for r in rr_rows}
                conc_to_ttft = {r['max_concurrency']: r.get('median_ttft_ms') for r in rr_rows}
                tps_data = [chart_val(conc_to_tps.get(c)) for c in all_conc]
                ttft_data = [chart_val(conc_to_ttft.get(c)) for c in all_conc]
                color = LINE_COLORS[j % len(LINE_COLORS)]
                tps_ds += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(tps_data)}, borderColor: '{color}', backgroundColor: '{color}', tension: 0.3, pointRadius: 4, fill: false }},\n"
                ttft_ds += f"      {{ label: '{wl_display(wl)}', data: {json.dumps(ttft_data)}, borderColor: '{color}', backgroundColor: '{color}', tension: 0.3, pointRadius: 4, fill: false }},\n"

            saturation_charts_js += f"""
new Chart(document.getElementById('{tps_id}'), {{
  type: 'line',
  data: {{ labels: {conc_labels}, datasets: [{tps_ds}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Per-Request TPS (tok/s)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }},
              x: {{ title: {{ display: true, text: 'Max Concurrency', font: {{ size: 11 }} }}, grid: noGrid, border: noBorder }} }} }}
}});
new Chart(document.getElementById('{ttft_id}'), {{
  type: 'line',
  data: {{ labels: {conc_labels}, datasets: [{ttft_ds}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Median TTFT (ms)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }},
              x: {{ title: {{ display: true, text: 'Max Concurrency', font: {{ size: 11 }} }}, grid: noGrid, border: noBorder }} }} }}
}});
"""

    # =================================================================
    #  Section 5: Replica Scaling (TPS & TTFT vs Replica count)
    #  Group by (model, workload, concurrency) — X axis = replica
    # =================================================================
    replica_charts_html = ''
    replica_charts_js = ''
    rep_chart_idx = 0

    rep_groups = defaultdict(list)
    for r in rows:
        rep = r.get('replica', '')
        if not isinstance(rep, (int, float)):
            continue
        key = (r['model'], r['workload'], r.get('max_concurrency', ''))
        rep_groups[key].append(r)

    rep_groups = {k: sorted(v, key=lambda x: x['replica']) for k, v in rep_groups.items() if len(set(x['replica'] for x in v)) >= 2}

    if rep_groups:
        # Group by model for combined charts
        model_groups = defaultdict(lambda: defaultdict(list))
        for (model, wl, mc), rr_rows in rep_groups.items():
            model_groups[model][(wl, mc)] = rr_rows

        for model, wl_mc_dict in model_groups.items():
            tps_id = f'repTps_{rep_chart_idx}'
            ttft_id = f'repTtft_{rep_chart_idx}'
            rep_chart_idx += 1

            replica_charts_html += f'''
<div class="card">
  <h3>{model} — Per-Request TPS vs Replica Count</h3>
  <div class="cd">Higher is better · Each line = one workload + concurrency combo</div>
  <div class="cb"><canvas id="{tps_id}"></canvas></div>
</div>
<div class="card">
  <h3>{model} — TTFT vs Replica Count</h3>
  <div class="cd">Lower is better · Each line = one workload + concurrency combo</div>
  <div class="cb"><canvas id="{ttft_id}"></canvas></div>
</div>
'''
            all_rep = sorted(set(r['replica'] for rr_rows in wl_mc_dict.values() for r in rr_rows))
            rep_labels = json.dumps([int(rp) for rp in all_rep])

            tps_ds = ''
            ttft_ds = ''
            for j, ((wl, mc), rr_rows) in enumerate(wl_mc_dict.items()):
                rep_to_tps = {r['replica']: r.get('per_req_tps_mean') for r in rr_rows}
                rep_to_ttft = {r['replica']: r.get('median_ttft_ms') for r in rr_rows}
                tps_data = [chart_val(rep_to_tps.get(rp)) for rp in all_rep]
                ttft_data = [chart_val(rep_to_ttft.get(rp)) for rp in all_rep]
                color = LINE_COLORS[j % len(LINE_COLORS)]
                mc_label = f'C={int(mc)}' if isinstance(mc, (int, float)) else ''
                series_label = f'{wl_display(wl)} ({mc_label})'
                tps_ds += f"      {{ label: '{series_label}', data: {json.dumps(tps_data)}, borderColor: '{color}', backgroundColor: '{color}', tension: 0.3, pointRadius: 4, fill: false }},\n"
                ttft_ds += f"      {{ label: '{series_label}', data: {json.dumps(ttft_data)}, borderColor: '{color}', backgroundColor: '{color}', tension: 0.3, pointRadius: 4, fill: false }},\n"

            replica_charts_js += f"""
new Chart(document.getElementById('{tps_id}'), {{
  type: 'line',
  data: {{ labels: {rep_labels}, datasets: [{tps_ds}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Per-Request TPS (tok/s)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }},
              x: {{ title: {{ display: true, text: 'Replica Count', font: {{ size: 11 }} }}, grid: noGrid, border: noBorder }} }} }}
}});
new Chart(document.getElementById('{ttft_id}'), {{
  type: 'line',
  data: {{ labels: {rep_labels}, datasets: [{ttft_ds}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Median TTFT (ms)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }},
              x: {{ title: {{ display: true, text: 'Replica Count', font: {{ size: 11 }} }}, grid: noGrid, border: noBorder }} }} }}
}});
"""

    # =================================================================
    #  Saturation section visibility
    # =================================================================
    saturation_section = ''
    if sat_groups:
        saturation_section = f'''
<div class="page-break"></div>
<div class="section">
  <h2>4. Saturation Analysis</h2>
  <div class="sd">How does performance change as concurrency increases? · Find the sweet spot before latency degrades</div>
  <hr class="divider">
</div>
{saturation_charts_html}
'''

    replica_section = ''
    if rep_groups:
        section_num = 5 if sat_groups else 4
        replica_section = f'''
<div class="page-break"></div>
<div class="section">
  <h2>{section_num}. Replica Scaling</h2>
  <div class="sd">How does performance scale with more replicas? · Same model, same concurrency, different replica counts</div>
  <hr class="divider">
</div>
{replica_charts_html}
'''

    # =================================================================
    #  Assemble HTML
    # =================================================================
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
  .container {{ max-width: 960px; margin: 0 auto; padding: 0 32px; }}
  @media print {{ body {{ background: #fff; }} .page-break {{ page-break-before: always; }} .container {{ padding: 0 16px; }} }}
  .header {{ border-bottom: 2px solid #111827; padding: 40px 0 20px; }}
  .header h1 {{ font-size: 26px; font-weight: 700; }}
  .header .badges {{ display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }}
  .badge {{ display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 12px; font-weight: 500; }}
  .header .meta {{ color: #9ca3af; font-size: 13px; margin-top: 8px; }}
  .section {{ padding: 28px 0 12px; }}
  .section h2 {{ font-size: 19px; font-weight: 700; margin-bottom: 2px; }}
  .section .sd {{ font-size: 13px; color: #9ca3af; margin-bottom: 18px; }}
  .divider {{ border: none; border-top: 1px solid #e5e7eb; margin: 6px 0 0; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; margin-bottom: 14px; }}
  .card h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 2px; }}
  .card .cd {{ font-size: 11px; color: #9ca3af; margin-bottom: 12px; }}
  .cb {{ position: relative; height: 300px; }}
  table.dt {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  .dt th {{ text-align: left; padding: 7px 6px; border-bottom: 2px solid #e5e7eb; color: #9ca3af; font-weight: 500; font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .dt td {{ padding: 6px 6px; border-bottom: 1px solid #f3f4f6; }}
  .dt .r {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .dt .b {{ font-weight: 600; }}
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
{badge_html}  </div>
  <div class="meta">Provider Benchmark Report · {today}</div>
</div>

{model_cards_html}

<!-- Section 1: Output Speed -->
<div class="page-break"></div>
<div class="section">
  <h2>1. Output Speed</h2>
  <div class="sd">How fast is each request? · Tokens per second · Higher is better</div>
  <hr class="divider">
</div>

<div class="card">
  <h3>Output Speed by Workload</h3>
  <div class="cd">Tokens per second · Higher is better</div>
  <div class="cb"><canvas id="speedChart"></canvas></div>
</div>

<div class="card">
  <table class="dt">
    <thead><tr><th>Model</th><th>Workload</th><th class="r">Replica</th><th class="r">Load</th><th class="r b">Output Speed</th><th class="r">TTFT Median</th></tr></thead>
    <tbody>
{matrix_rows}    </tbody>
  </table>
</div>

<!-- Section 2: Latency -->
<div class="page-break"></div>
<div class="section">
  <h2>2. Latency</h2>
  <div class="sd">How quickly does the response start? · Time to First Token · Lower is better</div>
  <hr class="divider">
</div>

<div class="card">
  <h3>Latency by Workload</h3>
  <div class="cd">Time to First Token · Seconds · Lower is better</div>
  <div class="cb"><canvas id="ttftChart"></canvas></div>
</div>

<div class="card">
  <table class="dt">
    <thead><tr><th>Model</th><th>Workload</th><th class="r">Replica</th><th class="r">Load</th><th class="r">TTFT Mean</th><th class="r b">TTFT Median</th><th class="r">TTFT P99</th></tr></thead>
    <tbody>
{latency_rows}    </tbody>
  </table>
</div>

<!-- Section 3: Capacity -->
<div class="page-break"></div>
<div class="section">
  <h2>3. Capacity</h2>
  <div class="sd">Total tokens processed across all concurrent requests · Higher is better</div>
  <hr class="divider">
</div>

<div class="card">
  <h3>Total Throughput Under Concurrent Load</h3>
  <div class="cd">Total tokens/sec across all concurrent requests</div>
  <div class="cb"><canvas id="aggChart"></canvas></div>
</div>

<div class="card">
  <table class="dt">
    <thead><tr><th>Model</th><th>Workload</th><th class="r">Replica</th><th class="r">Load</th><th class="r">Server Throughput</th><th class="r b">Per-Request TPS</th></tr></thead>
    <tbody>
{tp_rows}    </tbody>
  </table>
  <div class="fn" style="margin-top:12px; font-size:11px; color:#9ca3af; line-height:1.6">
    <b>Test methodology:</b> All benchmarks run with <b>vllm bench serve</b> against a live endpoint.<br>
    Throughput = output_tokens / generation_time per request. Server Throughput = total output tokens / total benchmark duration.<br>
    <b>Workload mapping:</b> {workload_mapping}
  </div>
</div>

{saturation_section}

{replica_section}

<footer>AIEEV Provider Benchmark · {' · '.join(ms)} · {today}</footer>

</div>

<script>
const base = {{ responsive: true, maintainAspectRatio: false, animation: false }};
const noGrid = {{ display: false }};
const noBorder = {{ display: false }};
const lightGrid = {{ color: '#f3f4f6' }};

const highlightPlugin = {{
  id: 'legendHighlight',
  beforeInit(chart) {{ chart._highlightIndex = -1; }},
  afterInit(chart) {{
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
  data: {{ labels: {model_labels}, datasets: [{speed_datasets}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'rectRounded' }} }},
    tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw + ' tok/s' }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Output speed (tokens / second)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }}, x: {{ grid: noGrid, border: noBorder }} }} }}
}});

new Chart(document.getElementById('ttftChart'), {{
  type: 'bar',
  data: {{ labels: {model_labels}, datasets: [{ttft_datasets}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'rectRounded' }} }},
    tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw + 's' }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Seconds to First Token', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }}, x: {{ grid: noGrid, border: noBorder }} }} }}
}});

new Chart(document.getElementById('aggChart'), {{
  type: 'bar',
  data: {{ labels: {model_labels}, datasets: [{agg_datasets}] }},
  options: {{ ...base, plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'rectRounded' }} }},
    tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.raw.toFixed(1) + ' tok/s' }} }} }},
    scales: {{ y: {{ title: {{ display: true, text: 'Total throughput (tokens / second)', font: {{ size: 11 }} }}, grid: lightGrid, border: noBorder }}, x: {{ grid: noGrid, border: noBorder }} }} }}
}});

{saturation_charts_js}
{replica_charts_js}
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
    if not rows:
        print("No data in CSV.", file=sys.stderr)
        sys.exit(1)
    html = generate_html(rows)
    Path(args.output).write_text(html)
    print(f"Report generated: {args.output}")


if __name__ == '__main__':
    main()
