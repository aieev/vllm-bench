# OpenRouter Provider Benchmark Report

## Provider Overview

| 항목 | 값 |
|---|---|
| Company Name | |
| Contact Email | |
| API Base URL | |
| Terms of Service URL | |
| Supported Models | |

## Infrastructure Summary

| 항목 | 값 |
|---|---|
| GPU Type | |
| GPU Count | |
| Quantization | |
| GPU Memory Utilization | |
| Max Context Length | |
| Max Batched Tokens | |
| Max Concurrent Sequences | |
| Replica Count | |
| vLLM Version | |

## API Compatibility

### Endpoint

- [x] OpenAI-compatible `/v1/chat/completions` endpoint
- [ ] `/v1/completions` endpoint

### Supported Features

- [ ] tools (function calling)
- [ ] json_mode
- [ ] structured_outputs
- [ ] logprobs
- [ ] reasoning
- [ ] web_search

### Supported Sampling Parameters

temperature, top_p, top_k, min_p, top_a, frequency_penalty, presence_penalty, repetition_penalty, stop, seed, max_tokens, logit_bias, logprobs, top_logprobs

## Pricing

Rates in USD per token (string format). 예: `"0.000002"` = $2/M tokens.

| Model | Prompt ($/token) | Completion ($/token) | Image ($/token) | Request ($/req) |
|---|---|---|---|---|
| | | | | |

## Benchmark Results

### Test Configuration

| 항목 | 값 |
|---|---|
| Benchmark Tool | vllm-bench (vllm bench serve) |
| Date | |
| Max Concurrency | |
| Request Rate | |

### Results Matrix

`python scripts/analyze_results.py --generate-rows bench-dataset/results/*.json` 출력을 아래에 붙여넣기:

| Model | GPU | Replica | Quant | Mem Util | Max Batched Tokens | Max Num Seqs | Workload | Concurrency | Agg TPS | Per-req TPS | TTFT Mean | TTFT Med | TTFT P99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

### Error & Success Rates

| Workload | Completed | Total | Success Rate |
|---|---|---|---|
| | | | |

## Workload Coverage

실행한 워크로드에 체크:

- [ ] **sharegpt** — 실제 대화 데이터 (500 prompts)
- [ ] **instructor_coder** — 코드 생성 (500 prompts)
- [ ] **zeta** — 코드 에디터 워크로드 (500 prompts)
- [ ] **blazedit** — 코드 편집 5K char (50 prompts)
- [ ] **mt_bench** — 멀티턴 대화 (80 prompts)
- [ ] **aimo** — 수학 추론 (500 prompts)
- [ ] **aimo_aime** — 수학 올림피아드 (90 prompts)
- [ ] **numinamath** — 수학 확장 (500 prompts)
- [ ] **burstgpt** — 버스트 트래픽 (100 prompts)
- [ ] **random_32k** — Long-context 32K (100 prompts)
- [ ] **random_64k** — Long-context 64K (100 prompts)
- [ ] **random_128k** — Long-context 128K (100 prompts)

## Reliability

| 항목 | 값 |
|---|---|
| Uptime SLA Target | |
| Monitoring | |
| Error Handling | |
| Rate Limiting | |

## Data Policy

| 항목 | 값 |
|---|---|
| Data Retention | |
| Privacy | |
| Logging | |
