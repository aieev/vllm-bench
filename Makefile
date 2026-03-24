PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
RESULTS_DIR := bench-dataset/results
CSV := docs/benchmark-data.csv

.PHONY: setup remote-% remote-all analyze csv csv-latest csv-all report pdf

setup:
	uv venv && uv pip install vllm huggingface_hub datasets

remote-%:
	./run-bench-remote.sh $(subst -,_,$*)

remote-all:
	./run-bench-remote.sh all

analyze:
	$(PYTHON) scripts/analyze_results.py $(RESULTS_DIR)/*.json

csv:
	@echo "Usage: make csv-latest [N=5]  or  make csv-all"

csv-latest:
	$(PYTHON) scripts/json_to_csv.py --latest $(or $(N),1) --csv $(CSV)

csv-all:
	$(PYTHON) scripts/json_to_csv.py $(RESULTS_DIR)/*.json --csv $(CSV)

report:
	$(PYTHON) scripts/generate_report.py $(CSV)

pdf:
	cd /tmp && node html2pdf.mjs $(CURDIR)/docs/benchmark-report.html $(CURDIR)/docs/benchmark-report.pdf
