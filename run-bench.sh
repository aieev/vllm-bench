#!/bin/bash
set -a
source "$(dirname "$0")/.env"
set +a

BENCH_IMAGE="lmcache/vllm-openai:v0.3.14"
BENCH_COMMON_ARGS=(
  --save-result
  --save-detailed
  --result-dir /bench-dataset/results
  --model ${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}
)

# ShareGPT
function sharegpt() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535
    -v $(pwd)/bench-dataset:/bench-dataset \
    vllm/vllm-openai:latest-cu130 \
    vllm bench serve \
    --model ${MODEL_NAME:-ministral/Ministral-3b-instruct} \
    --endpoint /v1/completions \
    --save-result \
    --save-detailed \
    --dataset-name sharegpt \
    --dataset-path /bench-dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 50 \
    --num-prompts 5000
}


# ShareGPT4V (Image)
function sharegpt_imgage() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    -v $(pwd)/bench-dataset:/bench-dataset \
    lmcache/vllm-openai:v0.3.14 \
    /bin/bash -c "vllm bench serve \
    --save-result \
    --save-detailed \
    --result-dir /bench-dataset/results \
    --backend openai-chat \
    --model ${MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct} \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /bench-dataset/sharegpt4v_coco_only.json \
    --request-rate 100 \
    --num-prompts 1000"
}

function burstgpt() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    -v $(pwd)/bench-dataset:/bench-dataset \
    lmcache/vllm-openai:v0.3.14 \
    /bin/bash -c "uv pip install pandas --python /opt/venv/bin/python && vllm bench serve \
    --save-result \
    --save-detailed \
    --model ${MODEL_NAME:-ministral/Ministral-3b-instruct} \
    --endpoint /v1/completions \
    --dataset-name burstgpt \
    --dataset-path /bench-dataset/BurstGPT_without_fails_2.csv \
    --num-prompts 2000"
}

function vision_arena() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535 \
    -v $(pwd)/bench-dataset:/bench-dataset \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    lmcache/vllm-openai:v0.3.14 \
    /bin/bash -c "uv pip install datasets --python /opt/venv/bin/python && vllm bench serve \
    --save-result \
    --save-detailed \
    --result-dir /bench-dataset/results \
    --backend openai-chat \
    --model ${VL_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct} \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 2000"
}

# Prefix Repetition
function prefix_repetition() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535 \
    -v $(pwd)/bench-dataset:/bench-dataset \
    lmcache/vllm-openai:v0.3.14 \
    vllm bench serve \
    --backend openai \
    --model ${MODEL_NAME:-ministral/Ministral-3b-instruct} \
    --endpoint /v1/completions \
    --save-result \
    --save-detailed \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 512 \
    --prefix-repetition-suffix-len 128 \
    --prefix-repetition-num-prefixes 5 \
    --prefix-repetition-output-len 128 \
    --request-rate 200 \
    --num-prompts 2000
}

function instructor_coder() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535 \
    -v $(pwd)/bench-dataset:/bench-dataset \
    lmcache/vllm-openai:v0.3.14 \
      /bin/bash -c "uv pip install datasets --python /opt/venv/bin/python && vllm bench serve \
    --save-result \
    --save-detailed \
    --result-dir /bench-dataset/results \
    --model ${MODEL_NAME:-Qwen/Qwen3-4B-Instruct} \
    --endpoint /v1/completions \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --request-rate 200 \
    --num-prompts 2000"
}

function mt_bench() {
  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535 \
    -v $(pwd)/bench-dataset:/bench-dataset \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    lmcache/vllm-openai:v0.3.14 \
      /bin/bash -c "uv pip install datasets --python /opt/venv/bin/python && vllm bench serve \
    --save-result \
    --save-detailed \
    --result-dir /bench-dataset/results \
    --model ${MODEL_NAME:-Qwen/Qwen3-4B-Instruct} \
    --endpoint /v1/completions \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --request-rate 200 \
    --num-prompts 2000"
}

function LooGLEv2() {
  # LooGLEv2 벤치마크는 커스텀 데이터셋
  # 긴 컨텍스트를 필요로 하므로, 기존 데이터로부터 일부 샘플링하고, 이를 JSONL 형식으로 변환된 데이터를 사용합니다.
  # python ./convert_looglev2.py --max-context-chars 80000 --num-samples 100 --output ./looglev2_long.jsonl 
      #   --max-model-len 32768
      # --gpu-memory-utilization 0.95
      # --swap-space 4
      # --kv-cache-dtype fp8
      # --block-size 16
      # --max-num-batched-tokens 32768
      # --max-num-seqs 8

  docker run --rm \
    --network host \
    --gpus all \
    --entrypoint "" \
    --ulimit nofile=65535:65535 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/bench-dataset:/bench-dataset \
    lmcache/vllm-openai:v0.3.14 \
      /bin/bash -c "uv pip install pandas --python /opt/venv/bin/python && vllm bench serve \
    --save-result \
    --save-detailed \
    --backend openai-chat \
    --result-dir /bench-dataset/results \
    --model ${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507} \
    --endpoint /v1/chat/completions \
    --dataset-name custom \
    --dataset-path /bench-dataset/looglev2_long.jsonl \
    --request-rate 5 \
    --num-prompts 100"
}

funcion swe_multimodal() {
}
function wait_vllm_ready() {
  echo "⏳ vLLM 서버 초기화 및 모델 로딩 대기 중..."
  
  # 1. 모델이 로드될 때까지 대기
  until curl -s http://localhost:8000/v1/models | grep -q "model"; do
    echo "waiting for model load..."
    sleep 5
  done

  echo -e "\n✅ vLLM 서버 준비 완료!"
  echo "----------------------------------------"
  
  # 2. 로드된 모델 이름 추출
  local MODEL_INFO=$(curl -s http://localhost:8000/v1/models)
  local MODEL_NAME=$(echo "$MODEL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('data', [{}])[0].get('id', 'Unknown'))" 2>/dev/null)
  
  # 3. vLLM 버전 추출 (vLLM 버전에 따라 /version 엔드포인트가 없을 수도 있으므로 예외 처리)
  local VERSION_INFO=$(curl -s http://localhost:8000/version 2>/dev/null)
  local VLLM_VERSION=$(echo "$VERSION_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('version', 'Unknown'))" 2>/dev/null)
  
  # 4. 준비 완료 시점
  local READY_TIME=$(date '+%Y-%m-%d %H:%M:%S')

  # 5. 결과 출력
  echo "📌 [vLLM 구동 상태 정보]"
  echo " - 준비 시점: $READY_TIME"
  echo " - vLLM 버전: $VLLM_VERSION"
  echo " - 서빙 모델: $MODEL_NAME"
  echo " - 엔드포인트: http://localhost:8000/v1"
  echo "----------------------------------------"
}

function run_bench() {
  benchlist=(
    sharegpt
    sharegpt_imgage
    instructor_coder
    vision_arena
    prefix_repetition
    burstgpt
    LooGLEv2
    mt_bench
  )

  benchname = bnechlist["LooGLEv2"]

  wait_vllm_ready
  LooGLEv2
  # mt_bench
  # instructor_coder
  # vision_arena
  # prefix_repetition
  # burstgpt

}

run_bench