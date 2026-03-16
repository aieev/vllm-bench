#!/bin/bash
set -euo pipefail
set -a
source "$(dirname "$0")/.env"
set +a

# ============================================================
#  공통 설정
# ============================================================
BENCH_IMAGE="lmcache/vllm-openai:v0.3.14"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(pwd)/bench-dataset"
HF_CACHE="$HOME/.cache/huggingface"

# 호스트 Python: uv .venv 우선, 없으면 conda_env, 없으면 시스템 python3
_python() {
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    "${SCRIPT_DIR}/.venv/bin/python" "$@"
  elif [[ -x "${SCRIPT_DIR}/.conda_env/bin/python" ]]; then
    "${SCRIPT_DIR}/.conda_env/bin/python" "$@"
  else
    python3 "$@"
  fi
}

# docker run 공통 래퍼
_docker_bench() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${BENCH_DIR}:/bench-dataset" \
    "${BENCH_IMAGE}" \
    "$@"
}

# uv pip install + vllm bench serve 래퍼 (추가 패키지 필요할 때)
_docker_bench_with_deps() {
  local deps="$1"; shift
  _docker_bench /bin/bash -c \
    "uv pip install ${deps} --python /opt/venv/bin/python --quiet && vllm bench serve $*"
}

# vllm bench serve 공통 인자
_common_args() {
  echo "--save-result --save-detailed --result-dir /bench-dataset/results"
}

# ============================================================
#  데이터셋 확인 및 다운로드/변환
# ============================================================

# HuggingFace Hub에서 단일 파일을 BENCH_DIR로 복사
_hf_download() {
  local repo_id="$1" filename="$2" repo_type="${3:-dataset}"
  _python - "$repo_id" "$filename" "$repo_type" "${BENCH_DIR}/${filename}" <<'PYEOF'
import sys, shutil, os
from huggingface_hub import hf_hub_download
repo_id, filename, repo_type, dest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(os.path.dirname(dest), exist_ok=True)
src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
shutil.copy(src, dest)
print(f"  → {dest}")
PYEOF
}

check_data() {
  local name="$1"
  case "$name" in

    sharegpt)
      local f="${BENCH_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
      if [[ ! -f "$f" ]]; then
        echo "📥 [sharegpt] ShareGPT_V3 다운로드 중..."
        _hf_download "anon8231489123/ShareGPT_Vicuna_unfiltered" \
          "ShareGPT_V3_unfiltered_cleaned_split.json"
      fi
      ;;

    sharegpt_image)
      local f="${BENCH_DIR}/sharegpt4v_coco_only.json"
      if [[ ! -f "$f" ]]; then
        echo "📥 [sharegpt_image] ShareGPT4V (COCO only) 다운로드 중..."
        _hf_download "Lin-Chen/ShareGPT4V" "sharegpt4v_coco_only.json"
      fi
      ;;

    burstgpt)
      local f="${BENCH_DIR}/BurstGPT_without_fails_2.csv"
      if [[ ! -f "$f" ]]; then
        echo "📥 [burstgpt] BurstGPT CSV 다운로드 중..."
        _hf_download "HKUST-LM/BurstGPT" "BurstGPT_without_fails_2.csv"
      fi
      ;;

    looglev2)
      local f="${BENCH_DIR}/looglev2_long.jsonl"
      if [[ ! -f "$f" ]]; then
        echo "🔄 [looglev2] LooGLE-v2 다운로드 + 변환 중..."
        _python "${SCRIPT_DIR}/bench-dataset/convert_looglev2.py" \
          --max-context-chars 80000 \
          --num-samples 100 \
          --output "$f"
      fi
      ;;

    gqa_sorted|gqa_shuffled)
      local sorted="${BENCH_DIR}/gqa_data/gqa_sorted.json"
      local shuffled="${BENCH_DIR}/gqa_data/gqa_shuffled.json"
      if [[ ! -f "$sorted" || ! -f "$shuffled" ]]; then
        echo "🔄 [gqa] GQA 이미지 다운로드 + 변환 중..."
        _python "${SCRIPT_DIR}/bench-dataset/convert_gqa.py" \
          --max-images 398 \
          --output-dir "${BENCH_DIR}/gqa_data"
      fi
      ;;

    vision_arena|instructor_coder|mt_bench|prefix_repetition)
      # vLLM이 --dataset-name hf / 내장 생성으로 자동 처리
      ;;

    *)
      echo "⚠️  [check_data] '$name' 에 대한 데이터 확인 규칙이 없습니다. 스킵."
      ;;
  esac
}

# ============================================================
#  서버 대기
# ============================================================
wait_vllm_ready() {
  local timeout=${1:-300}
  local elapsed=0

  echo "⏳ vLLM 서버 대기 중... (timeout: ${timeout}s)"
  until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed >= timeout )); then
      echo "❌ 타임아웃! vLLM 서버가 ${timeout}초 내에 준비되지 않음"
      return 1
    fi
  done

  echo "✅ vLLM 서버 준비 완료! (${elapsed}s)"
  echo "----------------------------------------"

  # 모델 정보 출력
  local model_id
  model_id=$(curl -s http://localhost:8000/v1/models | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "Unknown")
  local vllm_ver
  vllm_ver=$(curl -s http://localhost:8000/version | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "Unknown")

  echo "📌 모델: ${model_id}"
  echo "📌 vLLM: ${vllm_ver}"
  echo "📌 시각: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------"
}

# ============================================================
#  벤치마크 워크로드 정의
# ============================================================

sharegpt() {
  _docker_bench vllm bench serve $(_common_args) \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /bench-dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 50 \
    --num-prompts 5000
}

sharegpt_image() {
  _docker_bench vllm bench serve $(_common_args) \
    --backend openai-chat \
    --model "${VL_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}" \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /bench-dataset/sharegpt4v_coco_only.json \
    --request-rate 100 \
    --num-prompts 1000
}

burstgpt() {
  _docker_bench_with_deps "pandas" \
    $(_common_args) \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name burstgpt \
    --dataset-path /bench-dataset/BurstGPT_without_fails_2.csv \
    --num-prompts 2000
}

vision_arena() {
  _docker_bench_with_deps "datasets" \
    $(_common_args) \
    --backend openai-chat \
    --model "${VL_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 2000
}

prefix_repetition() {
  _docker_bench vllm bench serve $(_common_args) \
    --backend openai \
    --model "${MODEL_NAME}" \
    --endpoint /v1/completions \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 512 \
    --prefix-repetition-suffix-len 128 \
    --prefix-repetition-num-prefixes 5 \
    --prefix-repetition-output-len 128 \
    --request-rate 200 \
    --num-prompts 2000
}

instructor_coder() {
  _docker_bench_with_deps "datasets" \
    $(_common_args) \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --request-rate 200 \
    --num-prompts 2000
}

mt_bench() {
  _docker_bench_with_deps "datasets" \
    $(_common_args) \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --request-rate 200 \
    --num-prompts 80
}

looglev2() {
  # 사전 준비: python convert_looglev2.py --max-context-chars 80000 --num-samples 100 --output bench-dataset/looglev2_long.jsonl
  _docker_bench vllm bench serve $(_common_args) \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name custom \
    --dataset-path /bench-dataset/looglev2_long.jsonl \
    --request-rate 5 \
    --num-prompts 100
}

gqa_sorted() {
  # 사전 준비: python convert_gqa.py --max-images 398 --output-dir bench-dataset
    _docker_bench_with_deps "pandas" \
    $(_common_args) \
    --backend openai-chat \
    --model "${VL_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}" \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /bench-dataset/gqa_data/gqa_sorted.json \
    --request-rate 1 \
    --max-concurrency 1 \
    --disable-shuffle \
    --num-prompts 300
}

gqa_shuffled() {
    _docker_bench_with_deps "pandas" \
    $(_common_args) \
    --backend openai-chat \
    --model "${VL_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}" \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /bench-dataset/gqa_data/gqa_shuffled.json \
    --request-rate 1 \
    --disable-shuffle \
    --num-prompts 1000
}

# ============================================================
#  사용 가능한 벤치마크 목록 (여기에 추가하면 자동 반영)
# ============================================================
ALL_BENCHMARKS=(
  sharegpt
  sharegpt_image
  burstgpt
  vision_arena
  prefix_repetition
  instructor_coder
  mt_bench
  looglev2
  gqa_sorted
  gqa_shuffled
)

VISION_REQUIRED=(
  vision_arena
  sharegpt_image
  gqa_sorted
  gqa_shuffled
)

# ============================================================
#  메인 실행
# ============================================================
usage() {
  cat <<EOF
Usage: $0 [벤치마크이름 ...]

  벤치마크를 선택적으로 실행합니다.
  인자가 없으면 사용 가능한 목록을 출력합니다.

Examples:
  $0 looglev2                     # 단일 실행
  $0 looglev2 mt_bench sharegpt   # 여러 개 연속 실행
  $0 all                          # 전체 실행

Available benchmarks:
$(printf '  - %s\n' "${ALL_BENCHMARKS[@]}")
  - all (전체 실행)
EOF
}

run_bench() {
  local targets=("$@")

  # 인자 없으면 usage
  if (( ${#targets[@]} == 0 )); then
    usage
    return 0
  fi

  # "all" 처리
  if [[ "${targets[0]}" == "all" ]]; then
    targets=("${ALL_BENCHMARKS[@]}")
  fi

  # 유효성 검사
  for name in "${targets[@]}"; do
    if ! declare -f "$name" > /dev/null 2>&1; then
      echo "❌ 알 수 없는 벤치마크: '$name'"
      echo "   사용 가능: ${ALL_BENCHMARKS[*]}"
      return 1
    fi
  done

  # 서버 대기
  wait_vllm_ready

  # 순차 실행
  local total=${#targets[@]}
  local idx=0
  for name in "${targets[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "=========================================="
    echo "  [${idx}/${total}] 🚀 ${name}"
    echo "=========================================="
    local start_ts=$(date +%s)

    check_data "$name"
    if "$name"; then
      local elapsed=$(( $(date +%s) - start_ts ))
      echo "  ✅ ${name} 완료 (${elapsed}s)"
    else
      local elapsed=$(( $(date +%s) - start_ts ))
      echo "  ❌ ${name} 실패 (${elapsed}s)"
    fi
  done

  echo ""
  echo "=========================================="
  echo "  🏁 전체 벤치마크 완료!"
  echo "  결과: ${BENCH_DIR}/results/"
  echo "=========================================="
}

run_bench "$@"