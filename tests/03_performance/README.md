# Performance Tests (k6)

k6 load and soak tests for vLLM endpoints with SSE streaming support.

## Prerequisites

- Docker (for building custom k6 with xk6-sse)
- A running vLLM instance

## Build Custom k6

```sh
docker build -t k6-sse -f tests/03_performance/Dockerfile.k6 .
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_BASE_URL` | vLLM API base URL (e.g. `http://localhost:8000/v1`) | `http://localhost:8000/v1` |
| `VLLM_API_KEY` | API key for authentication | (empty) |
| `MODEL_NAME` | Model name to use | `default` |
| `SOAK_DURATION` | Duration for soak test | `30m` |

## Run Load Test

Ramps from 1 to 10 VUs over 30s, holds for 1m, then ramps down.

```sh
docker run --rm --network host \
  -v $(pwd)/tests/03_performance/scripts:/scripts \
  -e VLLM_BASE_URL=http://localhost:8000/v1 \
  -e VLLM_API_KEY=your-api-key \
  -e MODEL_NAME=your-model \
  k6-sse run /scripts/load_test.js
```

## Run Soak Test

Constant 5 VUs for 30 minutes (configurable via `SOAK_DURATION`).

```sh
docker run --rm --network host \
  -v $(pwd)/tests/03_performance/scripts:/scripts \
  -e VLLM_BASE_URL=http://localhost:8000/v1 \
  -e VLLM_API_KEY=your-api-key \
  -e MODEL_NAME=your-model \
  -e SOAK_DURATION=30m \
  k6-sse run /scripts/soak_test.js
```

## Custom Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm_ttft` | Trend | Time to first token (ms) |
| `llm_tps` | Trend | Tokens per second |
| `llm_tbt` | Trend | Time between tokens (ms) |
| `llm_total_tokens` | Counter | Total tokens generated |
| `llm_errors` | Counter | Failed requests |
| `llm_error_rate` | Rate | Error rate (soak test only) |

## Thresholds

**Load test:**
- TTFT p95 < 5000ms
- Error count < 5

**Soak test:**
- Error rate < 1%
- TTFT p99 < 10000ms
