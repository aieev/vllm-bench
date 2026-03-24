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
GPU_MEM_UTIL=0.92
MAX_MODEL_LEN=131072
MAX_BATCHED_TOKENS=32768
MAX_NUM_SEQS=256
REPLICA=1
QUANT_OPT=FP8
```

### 실행

`MAX_CONCURRENCY`의 기본값은 32입니다.

```sh
./run-bench-remote.sh sharegpt                            # 단일
./run-bench-remote.sh sharegpt mt_bench aimo               # 여러 개
./run-bench-remote.sh all                                  # 전체 (기본 7개)
./run-bench-remote.sh random_32k                           # long-context (all 미포함)

# 환경변수 오버라이드
MAX_CONCURRENCY=16 ./run-bench-remote.sh sharegpt          # 동시 요청 수 변경
REQUEST_RATE=2 ./run-bench-remote.sh sharegpt               # 초당 요청 수 (미설정시 워크로드별 기본값)
NUM_PROMPTS=50 ./run-bench-remote.sh random_32k             # 요청 수 변경


# Make
make remote-sharegpt
make remote-all
```

| 환경변수 | 설명 | 기본값 |
|----------|------|--------|
| `MAX_CONCURRENCY` | 최대 동시 요청 수 | 32 |
| `REQUEST_RATE` | 초당 요청 생성 수 (Poisson) | 워크로드별 기본값 |
| `NUM_PROMPTS` | 총 요청 수 | 워크로드별 기본값 |


### 결과 분석

```sh
# 전체 결과 비교 (GPU 설정별 그룹핑, 워크로드별 per-request TPS)
make analyze

# 단일 파일 분석
python scripts/analyze_results.py bench-dataset/results/some-result.json

# OpenRouter 리포트용 행 생성
python scripts/analyze_results.py --generate-rows bench-dataset/results/*.json
```

### OpenRouter 벤치마크 리포트 생성

벤치마크 결과를 CSV에 기록하고 HTML/PDF 리포트를 생성합니다.

```sh
# 1. 벤치마크 실행 (OpenRouter 제출용 5개 워크로드)
MAX_CONCURRENCY=16 NUM_PROMPTS=500 ./run-bench-remote.sh apps_coding vision_single random_1k random_10k random_100k

# 2. 결과를 CSV에 기록
#    docs/benchmark-data.csv 에 데이터 입력

# 3. HTML 리포트 생성
python scripts/generate_report.py docs/benchmark-data.csv

# 4. PDF 변환
cd /tmp && node html2pdf.mjs /path/to/benchmark-report.html /path/to/benchmark-report.pdf
```

#### OpenRouter 제출용 워크로드 (Artificial Analysis 기준)

| 워크로드 | 리포트 표시명 | 데이터셋 | Input tokens | Output tokens | 설명 |
|----------|---------------|----------|-------------|--------------|------|
| `apps_coding` | Coding | zed-industries/zeta | ~527 | 자유 | 코드 에디터 워크로드 |
| `vision_single` | Single Image | Random-MM (1MP) | 1K + 1 image | 1,024 | 1MP 이미지 + 텍스트 |
| `random_1k` | Short context (1K) | Random | 1,024 | 256 | 짧은 context |
| `random_10k` | Medium context (10K) | Random | 10,240 | 256 | 중간 context |
| `random_100k` | Long context (100K) | Random | 102,400 | 256 | 긴 context |

#### CSV 데이터 형식

`docs/benchmark-data.csv`:
```
model,workload,num_prompts,max_concurrency,request_rate,output_throughput,per_req_tps_mean,per_req_tps_median,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,mean_itl_ms,median_itl_ms
Qwen3.5-9B,apps_coding,50,16,inf,206.51,38.1,51.8,...
```

### 전체 워크로드 목록

| 워크로드 | 데이터셋 | Prompts | 특성 |
|----------|----------|---------|------|
| `apps_coding` | Zeta (Zed Industries) | 500 | 코드 에디터 워크로드 (OpenRouter용) |
| `vision_single` | Random-MM (1MP image) | 500 | 비전 벤치마크 (OpenRouter용) |
| `random_1k` | Random (1K input) | 500 | Short context (OpenRouter용) |
| `random_10k` | Random (10K input) | 500 | Medium context (OpenRouter용) |
| `random_100k` | Random (100K input) | 500 | Long context (OpenRouter용) |
| `sharegpt` | ShareGPT_V3 | 500 | 실제 대화 데이터 |
| `prefix_repetition` | 내장 생성 | 500 | Prefix caching 효과 측정 |
| `instructor_coder` | InstructCoder (HF) | 500 | 코드 생성 |
| `mt_bench` | MT-Bench (HF) | 80 | 멀티턴 대화 |
| `blazedit` | edit_5k_char (HF) | 50 | 코드 편집 |
| `aimo` | NuminaMath-CoT (HF) | 500 | 수학 추론 |
| `aimo_aime` | AIMO-AIME (HF) | 90 | 수학 올림피아드 고난도 추론 |
| `numinamath` | NuminaMath-1.5 (HF) | 500 | 수학 추론 (NuminaMath 확장) |
| `zeta` | Zeta (Zed Industries) | 500 | 코드 에디터 워크로드 |
| `blazedit_10k` | edit_10k_char (HF) | 50 | 긴 코드 편집 (10K chars) |
| `burstgpt` | BurstGPT | 100 | 버스트 트래픽 |
| `random_5k` | Random (5K input) | 100 | 짧은 context 테스트 |
| `random_20k` | Random (20K input) | 100 | 프로그래밍 워크로드 시뮬레이션 |
| `random_32k` | Random (32K input) | 100 | Long-context 테스트 |
| `random_64k` | Random (64K input) | 100 | Long-context 한계 테스트 |
| `random_128k` | Random (128K input) | 100 | 최대 context 한계 테스트 |

### 주요 메트릭

- **Output Speed** (tok/s): 요청별 출력 속도 (output_tokens / generation_time) — OpenRouter의 "Throughput"에 해당
- **Total Throughput** (tok/s): 서버 전체 출력 처리량 (전체 tokens / 전체 시간)
- **TTFT** (ms): Time to First Token — 첫 토큰까지 지연 시간
- **TPOT** (ms): 토큰당 출력 시간
- **ITL** (ms): 토큰 간 지연 시간

# Reference
- https://docs.vllm.ai/en/latest/benchmarking/cli/
- https://artificialanalysis.ai/methodology/performance-benchmarking
- https://openrouter.ai/docs/guides/routing/provider-selection
- https://huggingface.co/datasets/zed-industries/zeta
- https://huggingface.co/datasets/lmms-lab/GQA
- https://huggingface.co/datasets/MuLabPKU/LooGLE-v2
