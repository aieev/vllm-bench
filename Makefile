PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
RESULTS_DIR := bench-dataset/results
CSV := docs/benchmark-data.csv

.PHONY: setup bench analyze csv csv-empty report pdf test-unit test-func test-all test-endpoint test-all-endpoints

setup:
	uv venv && uv pip install "vllm==0.11.0" "transformers<5" huggingface_hub datasets

bench:
	./run-bench-remote.sh all

analyze:
	$(PYTHON) scripts/analyze_results.py $(RESULTS_DIR)/*.json

csv:
	$(PYTHON) scripts/json_to_csv.py $(RESULTS_DIR)/*.json --csv $(CSV)

csv-empty:
	@head -1 $(CSV) > $(CSV).tmp && mv $(CSV).tmp $(CSV)
	@echo "✅ $(CSV) emptied (header only)"

report:
	$(PYTHON) scripts/generate_report.py $(CSV)

pdf:
	cd /tmp && node html2pdf.mjs $(CURDIR)/docs/benchmark-report.html $(CURDIR)/docs/benchmark-report.pdf

test-unit:
	$(PYTHON) -m pytest tests/01_unit_static/ -v

test-func:
	$(PYTHON) -m pytest tests/02_functional/ -v

test-all:
	$(PYTHON) -m pytest tests/ -v

test-endpoint:
	@test -n "$(EP)" || (echo "Usage: make test-endpoint EP=qwen3.5-9b-4090" && exit 1)
	TEST_ENDPOINT=$(EP) $(PYTHON) -m pytest tests/ -v

test-all-endpoints:
	@$(PYTHON) -c "import yaml; eps=yaml.safe_load(open('endpoints.yaml'))['endpoints']; [print(e['name']) for e in eps]" | \
	while read ep; do \
		echo "\n===== $$ep ====="; \
		TEST_ENDPOINT=$$ep $(PYTHON) -m pytest tests/ -v --tb=short || true; \
	done
