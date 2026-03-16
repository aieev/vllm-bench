# vLLM 벤치마크 방법

```sh
cat .env # LLM 모델 및 vLLM 버전 확인.
uv venv && uv pip install huggingface_hub datasets
docker compose up -d # vLLM 띄우기
./run-bench.sh sharegpt # vLLM 벤치마크 실행하기
```