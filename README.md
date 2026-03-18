# vLLM 벤치마크 방법

## 로컬 벤치마크 (GPU 서버)

```sh
cat .env # LLM 모델 및 vLLM 버전 확인
uv venv && uv pip install huggingface_hub datasets
docker compose up -d
./run-bench.sh sharegpt
```

## 원격 벤치마크 (macOS 등에서 원격 vLLM 서버 대상)

Docker 없이 `vllm bench serve`를 네이티브로 실행합니다.

### 사전 준비

```sh
make setup
# 또는: uv venv && uv pip install vllm huggingface_hub datasets
```

### .env 설정

`.env.example`을 복사하여 설정합니다.

```sh
cp .env.example .env
```

```sh
MODEL_NAME=qwen3.5-9b                                    # 서버에 배포된 모델명
VLLM_BASE_URL=https://your-server.example.com/endpoint    # 원격 vLLM 서버 URL (필수)
VLLM_API_KEY=sk-xxx                                       # API 키 (선택)
VLLM_TOKENIZER=org/tokenizer-name                         # 토크나이저 (선택, 기본값: MODEL_NAME)

# GPU 설정 (결과 JSON에 메타데이터로 저장됨)
GPU_NAME=RTX-5090
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=131072
MAX_BATCHED_TOKENS=65536
MAX_NUM_SEQS=256
REPLICA=1
QUANT_OPT=FP8
```

### 실행

`MAX_CONCURRENCY`의 기본값은 32입니다.

```sh
./run-bench-remote.sh sharegpt                        # 단일
./run-bench-remote.sh sharegpt mt_bench aimo           # 여러 개
./run-bench-remote.sh all                              # 전체 (기본 7개)
./run-bench-remote.sh random_32k random_128k           # long-context (all 미포함)
MAX_CONCURRENCY=64 ./run-bench-remote.sh sharegpt      # concurrency 오버라이드

# Make
make remote-sharegpt
make remote-all
```

### 결과 분석

```sh
# 전체 결과 비교 (GPU 설정별 그룹핑, 워크로드별 per-request TPS)
make analyze

# 단일 파일 분석
python scripts/analyze_results.py bench-dataset/results/some-result.json
```

### 워크로드 목록

| 워크로드 | 데이터셋 | Prompts | 특성 |
|----------|----------|---------|------|
| `sharegpt` | ShareGPT_V3 | 500 | 실제 대화 데이터 |
| `prefix_repetition` | 내장 생성 | 500 | Prefix caching 효과 측정 |
| `instructor_coder` | InstructCoder (HF) | 500 | 코드 생성 |
| `mt_bench` | MT-Bench (HF) | 80 | 멀티턴 대화 |
| `blazedit` | edit_5k_char (HF) | 50 | 코드 편집 |
| `aimo` | NuminaMath-CoT (HF) | 500 | 수학 추론 |
| `burstgpt` | BurstGPT | 100 | 버스트 트래픽 |
| `random_32k` | Random (32K input) | 100 | Long-context 테스트 |
| `random_128k` | Random (128K input) | 100 | 최대 컨텍스트 한계 테스트 |

### 주요 메트릭

- **Output TPS** (tok/s): 집계 출력 토큰 처리량 (전체 tokens / 전체 시간)
- **Per-request TPS** (tok/s): 요청별 출력 속도 (output_tokens / e2e_latency)
- **TTFT** (ms): 첫 토큰까지 지연 시간
- **TPOT** (ms): 토큰당 출력 시간
- **ITL** (ms): 토큰 간 지연 시간

# Reference
- https://docs.vllm.ai/en/latest/benchmarking/cli/
- https://huggingface.co/datasets/lmms-lab/GQA
- https://huggingface.co/datasets/MuLabPKU/LooGLE-v2
