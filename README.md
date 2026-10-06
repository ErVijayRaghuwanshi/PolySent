# Sentix (PolySent) - Task-Driven Sentiment Gateway

[![Go Version](https://img.shields.io/badge/go-1.22%2B-blue.svg)](https://golang.org)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-brightgreen.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18%2B-informational.svg)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Sentix is a high-performance, unified sentiment analysis and semantic classification engine designed to bridge the gap between offline NLP model development (Python) and production-grade, low-latency microservice serving (Go).

---

## 🏛️ Architecture Overview

Sentix operates as a **Task-Driven Sentiment Gateway** that dynamically routes inference requests across three performance tiers:

```
[ Client Requests ] ──► ( REST / gRPC API Gateway )
                               │
                               ▼
            ┌────────────────────────────────────────┐
            │       Sentix Go Inference Gateway      │
            │  - Request Validation & SLA Router     │
            │  - Fast C-Binding Tokenization Engine  │
            │  - Worker Pool & Session Manager       │
            └───────┬──────────────┬───────────────┬─┘
                    │              │               │
       (Latency < 1ms)      (Latency 15-40ms)      (Deep Context)
                    ▼              ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Tier 1:      │ │ Tier 2:      │ │ Tier 3:      │
            │ Vectorized   │ │ ONNX Runtime │ │ Local LLM    │
            │ TF-IDF +     │ │ (DistilBERT/ │ │ (vLLM /      │
            │ LogReg       │ │  XLM-R /     │ │  Ollama API) │
            │ (Pure Go)    │ │  DeBERTa)    │ │              │
            └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🚀 Status & Milestone Progress

### Phase 1: Model Training & ONNX Export (Completed)
- [x] **Tier-1 Sub-Millisecond Engine**:
  - Trained TF-IDF + Logistic Regression with sublinear TF and n-gram (1, 2) features.
  - Exported ONNX graph via `skl2onnx` (`models/tier1_tfidf/model.onnx`).
  - Serialized vocabulary, IDF weights, and coefficients to `weights.json` for **pure Go zero-Cgo execution**.
  - **Benchmark**: **6.1 microseconds per inference** (`~191,000 ops/sec`), easily surpassing the `< 1ms` SLA.
- [x] **Tier-2 Balanced Transformer Engine**:
  - Exported fine-tuned DistilBERT to ONNX (`models/tier2_distilbert/model_fp32.onnx`).
  - Applied Post-Training Dynamic INT8 Quantization (`model_int8.onnx`).
  - Model size compressed from **255.55 MB to 64.47 MB** (**74.8% reduction**).
  - Quantization speedup: **2.98x** (batch latency reduced from 46.56 ms to 15.62 ms).
  - Exported native `tokenizer.json` and `vocab.txt` for direct C-Go Hugging Face tokenizers bindings.
- [x] **Parity & Verification Suite**:
  - `python/sentix_ml/verify_inference.py` ensures 100% numerical parity between Python ONNX and Pure Go execution.
  - Go unit tests and benchmark suite in `internal/engine/tier1/tfidf_test.go`.

---

## 📂 Project Structure

```
PolySent/
├── Makefile                          # Build and automation targets
├── go.mod                            # Go module definition
├── cmd/
│   └── sentix/                       # Go server entrypoint
├── internal/
│   ├── engine/
│   │   ├── types.go                  # Domain models (Request, Response, Analyzer interface)
│   │   └── tier1/
│   │       ├── tfidf.go              # Pure Go sub-millisecond TF-IDF engine
│   │       └── tfidf_test.go         # Tier-1 unit and benchmark tests
│   ├── router/                       # SLA-driven dynamic request router
│   ├── tokenizer/                    # C-Go Hugging Face tokenizers wrapper
│   └── gateway/                      # HTTP/REST API handlers
├── models/
│   ├── tier1_tfidf/
│   │   ├── model.onnx                # skl2onnx exported ONNX model
│   │   ├── weights.json              # Pure Go vocabulary & linear weights
│   │   └── metadata.json             # Model metadata & schema
│   └── tier2_distilbert/
│       ├── model_fp32.onnx           # Baseline PyTorch FP32 ONNX
│       ├── model_int8.onnx           # Quantized INT8 ONNX
│       ├── tokenizer.json            # FastTokenizer definition for Go
│       ├── vocab.txt                 # WordPiece vocabulary
│       └── metadata.json             # Model contract and benchmark stats
└── python/
    ├── pyproject.toml                # Python package metadata
    ├── requirements.txt              # ML dependencies (torch, onnx, optimum, scikit-learn)
    └── sentix_ml/
        ├── train_tfidf.py            # Tier-1 training & dual export script
        ├── export_distilbert.py      # Tier-2 ONNX export & INT8 quantization script
        └── verify_inference.py       # Cross-tier parity and benchmark verification
```

---

## ⚡ Quickstart

### 1. Python ML Pipeline Setup
```bash
# Setup environment and install dependencies using uv
make setup-env

# Train Tier-1 TF-IDF & export models
make train-tier1

# Export Tier-2 DistilBERT & apply dynamic INT8 quantization
make export-tier2

# Verify numerical parity and latency
make verify-models
```

### 2. Run Go Benchmarks & Tests
```bash
go test -v -bench=. ./internal/engine/tier1/...
```

---

## 📊 Benchmark Summary

| Metric | Tier 1 (Pure Go TF-IDF) | Tier 2 (DistilBERT FP32) | Tier 2 (DistilBERT INT8) |
| :--- | :--- | :--- | :--- |
| **Model Size** | 183 KB (`weights.json`) | 255.55 MB | **64.47 MB** (-74.8%) |
| **P50 Latency** | **0.006 ms (6.1 µs)** | ~46.5 ms (batch=4) | **~15.6 ms (batch=4)** |
| **Throughput** | **~191,000 req/sec/core** | - | - |
| **SLA Target** | < 1 ms | 15–40 ms | 15–40 ms |
| **Primary Use** | `ultra_fast` coarse sentiment | `balanced` high-accuracy | `balanced` high-accuracy |

---

## 🗺️ Next Steps
1. **Phase 2: Core Go Engine & Tokenizer Integration**:
   - Integrate `yalue/onnxruntime_go` with session pool manager.
   - Integrate native tokenization C-bindings using `models/tier2_distilbert/tokenizer.json`.
   - Implement HTTP REST server (`POST /v1/sentiment/analyze`) in `cmd/sentix`.
2. **Phase 3: Routing Logic & Tier-3 LLM Gateway**:
   - Rule-based & confidence-based fallback router.
   - Ollama / vLLM HTTP adapter for `deep_context` requests.
3. **Phase 4: Benchmarking & Containerization**:
   - k6 load-testing suite (50 - 1,000 QPS).
   - Multi-stage Dockerfile packaging models and Go binary.
