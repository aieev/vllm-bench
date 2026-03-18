PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
RESULTS_DIR := bench-dataset/results

.PHONY: setup remote-% remote-all analyze

setup:
	uv venv && uv pip install vllm huggingface_hub datasets

remote-%:
	./run-bench-remote.sh $(subst -,_,$*)

remote-all:
	./run-bench-remote.sh all

analyze:
	$(PYTHON) scripts/analyze_results.py $(RESULTS_DIR)/*.json
