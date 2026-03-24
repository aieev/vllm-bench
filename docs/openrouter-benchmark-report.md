# OpenRouter Provider Benchmark Report

## Qwen3.5-9B on RTX 5090 (FP8) — Inference Performance

> **TL;DR**
>
> **Per-request speed:** **56–61 tok/s** across conversation, code, and math workloads — consistent regardless of task type.
>
> **Time to first token:** **112–173ms median** — sub-200ms for all standard workloads.
>
> **Aggregate throughput:** Up to **1,822 tok/s** under concurrent load (32 parallel requests).
>
> **Long-context (32K):** Handles 32K-token inputs at **11.9 tok/s** per-request. TTFT scales with prefill length (~49s for 32K tokens).

---

## Infrastructure

| | |
|---|---|
| **Model** | Qwen3.5-9B |
| **GPU** | NVIDIA RTX 5090 (1x) |
| **Quantization** | FP8 |
| **GPU Memory Utilization** | 85% |
| **Max Context Length** | 131,072 tokens |
| **Max Batched Tokens** | 8,192 |
| **Max Concurrent Sequences** | 32 |
| **Serving Engine** | vLLM 0.18.1rc1 |
| **Tokenizer** | lovedheart/Qwen3.5-9B-FP8 |

---

## Benchmark Results

Benchmarks run on **2026-03-23** using `vllm bench serve` with Poisson-distributed request arrivals.

### Per-Request Performance

The metric that matters most for end-user experience: how fast each individual request gets tokens.

| Workload | Category | Per-req TPS (mean) | Per-req TPS (median) | TTFT Median | TPOT Median |
|---|---|---|---|---|---|
| sharegpt | Conversation | **56.0** tok/s | **60.8** tok/s | **112ms** | 15.7ms |
| instructor_coder | Code | **58.6** tok/s | **58.2** tok/s | **173ms** | 16.3ms |
| aimo | Math/Reasoning | **60.5** tok/s | **60.6** tok/s | **117ms** | 15.7ms |
| mt_bench | Multi-turn | **61.3** tok/s | **61.0** tok/s | **122ms** | 16.0ms |
| random_32k | Long-context (32K) | 11.9 tok/s | 10.6 tok/s | 49,367ms | 46.0ms |

### Aggregate Throughput Under Load

Total tokens processed per second across all concurrent requests.

| Workload | Concurrency | Request Rate | Agg TPS | Peak TPS |
|---|---|---|---|---|
| sharegpt | C=16 | 50 req/s | **930.8** tok/s | 1,099 tok/s |
| instructor_coder | C=32 | 200 req/s | **1,822.7** tok/s | — |
| aimo | C=32 | 200 req/s | **1,795.5** tok/s | — |
| mt_bench | C=32 | 5 req/s | **1,033.1** tok/s | 1,911 tok/s |
| random_32k | C=32 | inf | **314.5** tok/s | — |

### Latency Breakdown

| Workload | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | TPOT P99 | ITL Median |
|---|---|---|---|---|---|---|
| sharegpt | 113ms | **112ms** | **188ms** | 15.7ms | 19.4ms | 14.5ms |
| instructor_coder | 180ms | **173ms** | **292ms** | 16.3ms | — | — |
| aimo | 130ms | **117ms** | **264ms** | 15.7ms | — | — |
| mt_bench | 125ms | **122ms** | **180ms** | 16.0ms | 16.9ms | 14.5ms |
| random_32k | 44,551ms | 49,367ms | 75,965ms | 44.4ms | 48.7ms | 20.1ms |

### Detailed Results Matrix

Full benchmark configuration for reproducibility.

| Model | GPU | Replica | Quant | Mem Util | Batched Tokens | Num Seqs | Workload | Concurrency | Agg TPS | Per-req TPS | TTFT Mean | TTFT Med | TTFT P99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | RTX 5090 | 1 | FP8 | 0.85 | 8192 | 32 | sharegpt | RR=50,C=16 | 930.79 | 56.00 | 113.47 | 111.74 | 188.00 |
| Qwen3.5-9B | RTX 5090 | 1 | FP8 | 0.85 | 8192 | 32 | instructor_coder | RR=200,C=32 | 1822.71 | 58.59 | 180.03 | 172.61 | 292.36 |
| Qwen3.5-9B | RTX 5090 | 1 | FP8 | 0.85 | 8192 | 32 | aimo | RR=200,C=32 | 1795.52 | 60.51 | 130.21 | 117.18 | 263.85 |
| Qwen3.5-9B | RTX 5090 | 1 | FP8 | 0.85 | 8192 | 32 | mt_bench | RR=5,C=32 | 1033.05 | 61.34 | 124.84 | 121.71 | 179.58 |
| Qwen3.5-9B | RTX 5090 | 1 | FP8 | 0.85 | 8192 | 32 | random_32k | C=32 | 314.51 | 11.87 | 44551.37 | 49366.89 | 75964.86 |

---

## Workload Coverage

5 workloads tested, covering conversation, code generation, math reasoning, multi-turn dialogue, and long-context processing.

| Workload | Category | Prompts | Status |
|---|---|---|---|
| **sharegpt** | Conversation (ShareGPT_V3) | 500 | Tested |
| **instructor_coder** | Code generation (InstructCoder) | 500 | Tested |
| **aimo** | Math/Reasoning (NuminaMath-CoT) | 500 | Tested |
| **mt_bench** | Multi-turn dialogue | 80 | Tested |
| **random_32k** | Long-context (32K input tokens) | 100 | Tested |

---

## Key Observations

- **Consistent per-request speed:** 56–61 tok/s across all standard workloads (sharegpt, code, math, multi-turn). The model maintains stable per-request performance regardless of task type.
- **Low TTFT:** Median TTFT under 175ms for all standard workloads. P99 under 300ms. Users experience near-instant response start.
- **TPOT stability:** 15.7–16.3ms per output token across workloads, indicating consistent decode throughput.
- **Long-context trade-off:** 32K-token inputs are handled but TTFT is dominated by prefill time (~49s median). Per-request TPS drops to ~12 tok/s due to the long prefill phase.
- **Aggregate throughput scales with concurrency:** instructor_coder and aimo reach ~1,800 tok/s at C=32, showing efficient batching under load.
