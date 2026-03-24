#!/bin/bash
set -euo pipefail
ENV_FILE="$(dirname "$0")/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ .env 파일이 없습니다. .env.example을 복사하여 설정하세요:"
  echo "   cp .env.example .env"
  exit 1
fi
set -a
source "$ENV_FILE"
set +a

# ============================================================
#  공통 설정
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(pwd)/bench-dataset"
BASE_URL="${VLLM_BASE_URL:?VLLM_BASE_URL must be set in .env}"
API_KEY="${VLLM_API_KEY:-}"
TOKENIZER="${VLLM_TOKENIZER:-${MODEL_NAME}}"
TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-}"
REQUEST_RATE="${REQUEST_RATE:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-}"

GPU_NAME="${GPU_NAME:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
REPLICA="${REPLICA:-}"
QUANT_OPT="${QUANT_OPT:-}"

_python() {
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    "${SCRIPT_DIR}/.venv/bin/python" "$@"
  else
    python3 "$@"
  fi
}

_vllm_bench() {
  local cmd
  if [[ -x "${SCRIPT_DIR}/.venv/bin/vllm" ]]; then
    cmd="${SCRIPT_DIR}/.venv/bin/vllm"
  else
    cmd="vllm"
  fi
  OPENAI_API_KEY="${API_KEY}" "${cmd}" bench serve "$@"
}

_get_cache_metrics() {
  local auth_header=()
  if [[ -n "${API_KEY}" ]]; then
    auth_header=(-H "Authorization: Bearer ${API_KEY}")
  fi
  local metrics
  metrics=$(curl -sf --connect-timeout 5 -m 10 "${auth_header[@]}" "${BASE_URL}/metrics" 2>/dev/null) || return 1
  local queries hits
  queries=$(echo "$metrics" | grep '^vllm:prefix_cache_queries_total' | awk '{print $2}')
  hits=$(echo "$metrics" | grep '^vllm:prefix_cache_hits_total' | awk '{print $2}')
  echo "${queries:-0} ${hits:-0}"
}

_print_cache_stats() {
  local before_queries=$1 before_hits=$2 after_queries=$3 after_hits=$4
  local delta_queries=$(( ${after_queries%.*} - ${before_queries%.*} ))
  local delta_hits=$(( ${after_hits%.*} - ${before_hits%.*} ))
  if (( delta_queries > 0 )); then
    local hit_rate
    hit_rate=$(awk "BEGIN {printf \"%.1f\", ($delta_hits / $delta_queries) * 100}")
    echo "  📊 Prefix Cache: ${delta_queries} queries, ${delta_hits} hits (${hit_rate}% hit rate)"
  else
    echo "  📊 Prefix Cache: no queries during this benchmark"
  fi
}

_default_prompts() {
  if [[ -n "${NUM_PROMPTS}" ]]; then
    echo "--num-prompts ${NUM_PROMPTS}"
  else
    echo "--num-prompts $1"
  fi
}

_common_args() {
  local workload="${1:-}"
  local args="--save-result --save-detailed --result-dir ${BENCH_DIR}/results"
  args+=" --base-url ${BASE_URL}"
  args+=" --tokenizer ${TOKENIZER}"
  if [[ -n "${TOKENIZER_MODE}" ]]; then
    args+=" --tokenizer-mode ${TOKENIZER_MODE}"
  fi
  args+=" --trust-remote-code"
  args+=" --ready-check-timeout-sec 0"
  if [[ -n "${REQUEST_RATE}" ]]; then
    args+=" --request-rate ${REQUEST_RATE}"
  fi
  if [[ -n "${MAX_CONCURRENCY}" ]]; then
    args+=" --max-concurrency ${MAX_CONCURRENCY}"
  fi
  local metadata=""
  [[ -n "${GPU_NAME}" ]] && metadata+=" gpu=${GPU_NAME}"
  [[ -n "${GPU_MEM_UTIL}" ]] && metadata+=" gpu_mem_util=${GPU_MEM_UTIL}"
  [[ -n "${MAX_MODEL_LEN}" ]] && metadata+=" max_model_len=${MAX_MODEL_LEN}"
  [[ -n "${MAX_BATCHED_TOKENS}" ]] && metadata+=" max_batched_tokens=${MAX_BATCHED_TOKENS}"
  [[ -n "${MAX_NUM_SEQS}" ]] && metadata+=" max_num_seqs=${MAX_NUM_SEQS}"
  [[ -n "${REPLICA}" ]] && metadata+=" replica=${REPLICA}"
  [[ -n "${QUANT_OPT}" ]] && metadata+=" quant=${QUANT_OPT}"
  [[ -n "${workload}" ]] && metadata+=" workload=${workload}"
  if [[ -n "${metadata}" ]]; then
    args+=" --metadata${metadata}"
  fi
  echo "$args"
}

# ============================================================
#  데이터셋 확인 및 다운로드/변환
# ============================================================

check_data() {
  local name="$1"
  case "$name" in

    apps_coding|vision_single|random_1k|random_10k|random_100k|sharegpt|mt_bench|blazedit)
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
  local auth_header=()
  if [[ -n "${API_KEY}" ]]; then
    auth_header=(-H "Authorization: Bearer ${API_KEY}")
  fi

  echo "⏳ vLLM 서버 대기 중... (${BASE_URL}, timeout: ${timeout}s)"

  until curl -sf --connect-timeout 5 -m 10 "${auth_header[@]}" "${BASE_URL}/v1/models" > /dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed >= timeout )); then
      echo "❌ 타임아웃! vLLM 서버가 ${timeout}초 내에 준비되지 않음"
      return 1
    fi
  done

  echo "✅ vLLM 서버 준비 완료! (${elapsed}s)"
  echo "----------------------------------------"

  local model_id
  model_id=$(curl -s "${auth_header[@]}" "${BASE_URL}/v1/models" | \
    _python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "Unknown")
  local vllm_ver
  vllm_ver=$(curl -s "${auth_header[@]}" "${BASE_URL}/version" | \
    _python -c "import sys,json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "Unknown")

  echo "📌 모델: ${model_id}"
  echo "📌 vLLM: ${vllm_ver}"
  echo "📌 시각: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------"
}

# ============================================================
#  벤치마크 워크로드 정의
# ============================================================

apps_coding() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path zed-industries/zeta \
    $(_default_prompts 500)
}

vision_single() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name random-mm \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-mm-base-items-per-request 1 \
    --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --random-mm-bucket-config '{(1024, 1024, 1): 1.0}' \
    $(_default_prompts 500)
}

random_1k() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    $(_default_prompts 500)
}

random_10k() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 10240 \
    --random-output-len 256 \
    $(_default_prompts 500)
}

random_100k() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 102400 \
    --random-output-len 256 \
    $(_default_prompts 500)
}


# ============================================================
#  추가 워크로드 (all에 미포함, 개별 실행 가능)
# ============================================================

sharegpt() {
  local f="${BENCH_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
  if [[ ! -f "$f" ]]; then
    echo "📥 [sharegpt] ShareGPT_V3 다운로드 중..."
    _python - "anon8231489123/ShareGPT_Vicuna_unfiltered" "ShareGPT_V3_unfiltered_cleaned_split.json" "dataset" "${f}" <<'PYEOF'
import sys, shutil, os
from huggingface_hub import hf_hub_download
repo_id, filename, repo_type, dest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(os.path.dirname(dest), exist_ok=True)
src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
shutil.copy(src, dest)
print(f"  → {dest}")
PYEOF
  fi
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path "${f}" \
    $(_default_prompts 500)
}

mt_bench() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    $(_default_prompts 80)
}

blazedit() {
  _vllm_bench $(_common_args "${FUNCNAME[0]}") \
    --backend openai-chat \
    --model "${MODEL_NAME}" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path vdaita/edit_5k_char \
    $(_default_prompts 50)
}

# ============================================================
#  사용 가능한 벤치마크 목록 (여기에 추가하면 자동 반영)
# ============================================================
ALL_BENCHMARKS=(
  apps_coding
  vision_single
  random_1k
  random_10k
  random_100k
)


# ============================================================
#  메인 실행
# ============================================================
usage() {
  cat <<EOF
Usage: $0 [벤치마크이름 ...]

  원격 vLLM 서버에 대해 벤치마크를 실행합니다.
  vllm bench serve를 네이티브로 실행합니다 (Docker 불필요).

  .env 설정:
    VLLM_BASE_URL        원격 서버 URL (필수)
    VLLM_API_KEY         API 키 (선택)
    VLLM_TOKENIZER       토크나이저 (선택, 기본값: MODEL_NAME)
    VLLM_TOKENIZER_MODE  토크나이저 모드 (Mistral 모델: mistral)

  환경 변수:
    MAX_CONCURRENCY  최대 동시 요청 수 (기본값: 32)
    REQUEST_RATE     초당 요청 수 (선택, 미설정시 unlimited)
    NUM_PROMPTS      총 요청 수 (선택, 미설정시 워크로드별 기본값)

  사전 준비:
    uv pip install vllm huggingface_hub datasets

Examples:
  $0 apps_coding                                        # 단일 실행
  $0 apps_coding random_1k random_10k                    # 여러 개 연속 실행
  $0 all                                                # 전체 실행 (OpenRouter 5개)
  MAX_CONCURRENCY=16 NUM_PROMPTS=50 $0 all               # 설정 오버라이드
  VLLM_TOKENIZER_MODE=mistral $0 apps_coding             # Mistral 모델

Available benchmarks:
$(printf '  - %s\n' "${ALL_BENCHMARKS[@]}")
  - all (전체 실행)
EOF
}

run_bench() {
  local targets=("$@")

  if (( ${#targets[@]} == 0 )); then
    usage
    return 0
  fi

  if [[ "${targets[0]}" == "all" ]]; then
    targets=("${ALL_BENCHMARKS[@]}")
  fi

  for name in "${targets[@]}"; do
    if ! declare -f "$name" > /dev/null 2>&1; then
      echo "❌ 알 수 없는 벤치마크: '$name'"
      echo "   사용 가능: ${ALL_BENCHMARKS[*]}"
      return 1
    fi
  done

  mkdir -p "${BENCH_DIR}/results"
  wait_vllm_ready

  local total=${#targets[@]}
  local idx=0
  local run_start_ts=$(date +%s)
  for name in "${targets[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "=========================================="
    echo "  [${idx}/${total}] 🚀 ${name}"
    echo "=========================================="
    local start_ts=$(date +%s)

    check_data "$name"

    local cache_before before_queries=0 before_hits=0
    if cache_before=$(_get_cache_metrics); then
      read -r before_queries before_hits <<< "$cache_before"
    fi

    if "$name"; then
      local elapsed=$(( $(date +%s) - start_ts ))
      echo "  ✅ ${name} 완료 (${elapsed}s)"
      local latest_result
      latest_result=$(ls -t "${BENCH_DIR}/results/"*.json 2>/dev/null | head -1)
      if [[ -n "$latest_result" ]]; then
        _python "${SCRIPT_DIR}/scripts/analyze_results.py" --summary-only "$latest_result" || true
      fi
    else
      local elapsed=$(( $(date +%s) - start_ts ))
      echo "  ❌ ${name} 실패 (${elapsed}s)"
    fi

    local cache_after after_queries=0 after_hits=0
    if cache_after=$(_get_cache_metrics); then
      read -r after_queries after_hits <<< "$cache_after"
      _print_cache_stats "$before_queries" "$before_hits" "$after_queries" "$after_hits"
    fi
  done

  local total_elapsed=$(( $(date +%s) - run_start_ts ))
  local total_min=$(( total_elapsed / 60 ))
  local total_sec=$(( total_elapsed % 60 ))
  echo ""
  echo "=========================================="
  echo "  🏁 전체 벤치마크 완료! (${total_min}m ${total_sec}s)"
  echo "  결과: ${BENCH_DIR}/results/"
  echo "=========================================="
}

run_bench "$@"
