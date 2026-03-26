#!/usr/bin/env bash
# vLLM 서버 시작 스크립트 (CaaS 배포용 템플릿)
# 환경변수로 설정을 주입합니다.

set -euo pipefail

MODEL="${HF_MODEL:-lovedheart/Qwen3.5-9B-FP8}"
SERVED_NAME="${MODEL_NAME:-qwen3.5-9b}"
GPU_MEM="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
API_KEY="${VLLM_API_KEY:-}"

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --gpu-memory-utilization "$GPU_MEM"
  --max-model-len "$MAX_MODEL_LEN"
  --kv-cache-dtype fp8
  --enable-chunked-prefill
  --enable-prefix-caching
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --enable-prompt-tokens-details
)

[[ -n "$API_KEY" ]] && ARGS+=(--api-key "$API_KEY")

exec vllm serve "${ARGS[@]}" "$@"
