SHELL := /bin/bash
PYTHON := python/.venv/bin/python
UV := uv

.PHONY: all setup-env train-tier1 export-tier2 verify-models build run test clean help

all: train-tier1 export-tier2 verify-models build

help:
	@echo "Sentix (PolySent) Build & Pipeline Automation"
	@echo "============================================="
	@echo "  make setup-env      - Create uv virtualenv and install Python dependencies"
	@echo "  make train-tier1    - Train Tier-1 TF-IDF + Logistic Regression and export ONNX + Go JSON"
	@echo "  make export-tier2   - Export Tier-2 DistilBERT to ONNX with INT8 dynamic quantization"
	@echo "  make verify-models  - Run numerical parity and latency tests across tiers"
	@echo "  make build          - Build Go Sentix Gateway binary"
	@echo "  make test           - Run Go test suite"
	@echo "  make clean          - Clean generated binaries and temporary files"

setup-env:
	$(UV) venv python/.venv --python 3.13
	$(UV) pip install -r python/requirements.txt --python python/.venv

train-tier1:
	@echo "==> Training and exporting Tier-1 TF-IDF..."
	$(PYTHON) python/sentix_ml/train_tfidf.py

export-tier2:
	@echo "==> Exporting Tier-2 DistilBERT to FP32 & INT8 ONNX..."
	$(PYTHON) python/sentix_ml/export_distilbert.py

verify-models:
	@echo "==> Verifying model inference and parity..."
	$(PYTHON) python/sentix_ml/verify_inference.py

build:
	@echo "==> Building Sentix Go Gateway..."
	mkdir -p bin
	go build -o bin/sentix ./cmd/sentix

run: build
	@echo "==> Running Sentix Gateway on :8080..."
	./bin/sentix -port 8080

test: test-go test-python

test-go:
	@echo "==> Running Go unit tests with race detector..."
	go test -v -race ./...

test-python:
	@echo "==> Running Python verification and RL loop..."
	$(PYTHON) python/sentix_ml/verify_inference.py
	$(PYTHON) python/sentix_ml/rl_loop.py

clean:
	rm -rf bin/
