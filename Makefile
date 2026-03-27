PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
RESULTS_DIR := bench-dataset/results
CSV := docs/benchmark-data.csv

.PHONY: setup bench analyze csv csv-empty report pdf

setup:
	uv venv && uv pip install vllm huggingface_hub datasets

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
