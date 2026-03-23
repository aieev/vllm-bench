# OpenRouter Metrics Guide

vllm-bench 결과를 OpenRouter 프로바이더 제출용으로 매핑하는 가이드.

Last verified: 2026-03-23

## OpenRouter의 프로바이더 라우팅 시그널

OpenRouter는 세 가지 시그널로 프로바이더에게 트래픽을 분배합니다:

1. **Uptime** (최우선): 최근 30초간 장애가 없는 프로바이더 우선
2. **Pricing** (가격 가중치): Inverse-square 가중치 — $1/M tokens 프로바이더는 $3/M tokens 대비 9배 높은 라우팅 확률
3. **Performance** (성능): 5분 롤링 윈도우 기준 p50/p75/p90/p99 백분위 측정
   - Throughput: tokens/sec
   - Latency: 응답 시간 (초)

벤치마크 정확도가 중앙값에서 1 표준편차 이하인 프로바이더는 deprioritize됩니다.

Sources:
- https://openrouter.ai/docs/guides/routing/provider-selection
- https://openrouter.ai/docs/guides/guides/for-providers

## vllm-bench → OpenRouter 메트릭 매핑

| vllm-bench 메트릭 | OpenRouter 메트릭 | 설명 |
|---|---|---|
| `output_throughput` | Throughput (tokens/sec) | 집계 출력 처리량 (전체 tokens / 전체 시간) |
| `mean_ttft_ms` | TTFT | 첫 토큰까지 지연 시간 (ms) |
| `median_ttft_ms` | TTFT (median) | TTFT 중앙값 |
| `p99_ttft_ms` | TTFT (p99) | TTFT 99번째 백분위 |
| `per_req_tps_mean` | Per-request speed | 요청별 출력 속도 (output_tokens / e2e_latency) |
| `completed` / `num_prompts` | Success rate | 요청 완료율 |

## 워크로드 → OpenRouter 벤치마크 카테고리 매핑

OpenRouter는 Artificial Analysis 기반 벤치마크 (programming, math, science, long-context 등)를 사용합니다.

| 카테고리 | vllm-bench 워크로드 | 설명 | 실행 명령 |
|---|---|---|---|
| **Code** | `instructor_coder` | HuggingFace InstructCoder 코드 생성 | `./run-bench-remote.sh instructor_coder` |
| **Code** | `zeta` | Zed Industries 코드 에디터 워크로드 | `./run-bench-remote.sh zeta` |
| **Code** | `blazedit` | 5K char 코드 편집 | `./run-bench-remote.sh blazedit` |
| **Code** | `blazedit_10k` | 10K char 코드 편집 | `./run-bench-remote.sh blazedit_10k` |
| **Math/Reasoning** | `aimo` | NuminaMath-CoT 수학 추론 | `./run-bench-remote.sh aimo` |
| **Math/Reasoning** | `aimo_aime` | AIMO 올림피아드 고난도 추론 (90 prompts) | `./run-bench-remote.sh aimo_aime` |
| **Math/Reasoning** | `numinamath` | NuminaMath-1.5 확장 수학 | `./run-bench-remote.sh numinamath` |
| **Conversation** | `sharegpt` | ShareGPT_V3 실제 대화 데이터 (500 prompts) | `./run-bench-remote.sh sharegpt` |
| **Conversation** | `mt_bench` | MT-Bench 구조화된 멀티턴 평가 (80 prompts) | `./run-bench-remote.sh mt_bench` |
| **Long-context** | `random_32k` | 32K input 합성 스트레스 테스트 | `./run-bench-remote.sh random_32k` |
| **Long-context** | `random_64k` | 64K input 한계 테스트 | `./run-bench-remote.sh random_64k` |
| **Long-context** | `random_128k` | 128K input 최대 context 테스트 | `./run-bench-remote.sh random_128k` |
| **Burst** | `burstgpt` | 실제 버스트 트래픽 패턴 | `./run-bench-remote.sh burstgpt` |

Note: `all` 명령은 7개 코어 워크로드만 실행 (sharegpt, prefix_repetition, instructor_coder, mt_bench, blazedit, aimo, burstgpt). 나머지는 개별 실행 필요.

## 우선순위 워크로드 (OpenRouter 제출용)

OpenRouter 라우팅에 가장 영향을 미치는 순서:

1. **sharegpt** — 실제 사용 패턴과 가장 유사, 범용 성능 지표
2. **instructor_coder** / **zeta** — 코드 생성은 OpenRouter 트래픽의 상당 부분
3. **aimo** — 수학/추론 벤치마크 (Artificial Analysis 카테고리)
4. **mt_bench** — 멀티턴 대화 품질
5. **random_32k+** — Long-context 지원 증명

## 결과 매트릭스 형식

`analyze_results.py --generate-rows` 출력 형식:

```
| Model | GPU | Replica | Quant | Mem Util | Max Batched Tokens | Max Num Seqs | Workload | Concurrency | Aggregate TPS | Per-request TPS | TTFT Mean | TTFT Median | TTFT P99 |
```

실행:
```sh
python scripts/analyze_results.py --generate-rows bench-dataset/results/*.json
```

각 컬럼 설명:

| 컬럼 | 설명 |
|---|---|
| Model | 서빙 모델 ID |
| GPU | GPU 타입 (예: RTX 5090) |
| Replica | 리플리카 수 |
| Quant | 양자화 옵션 (FP8, INT8 등) |
| Mem Util | GPU 메모리 활용률 (0.0-1.0) |
| Max Batched Tokens | 최대 배치 토큰 수 |
| Max Num Seqs | 최대 동시 시퀀스 수 |
| Workload | 벤치마크 워크로드명 |
| Concurrency | 동시 요청 설정 (RR=request rate, C=max concurrency) |
| Aggregate TPS | 집계 출력 처리량 (tok/s) |
| Per-request TPS | 요청별 평균 출력 속도 (tok/s) |
| TTFT Mean | 첫 토큰 지연 평균 (ms) |
| TTFT Median | 첫 토큰 지연 중앙값 (ms) |
| TTFT P99 | 첫 토큰 지연 99번째 백분위 (ms) |
