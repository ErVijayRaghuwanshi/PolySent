# Sentix Architecture & Design Documentation

Welcome to the comprehensive technical documentation for **Sentix (PolySent)**: the high-performance, unified sentiment analysis and semantic classification engine.

Sentix is designed to bridge the gap between offline NLP model experimentation (Python) and production-grade, low-latency microservice serving (Go).

---

## 🗺️ Documentation Map

The documentation is organized into two primary tiers: **High-Level Design (HLD)** and **Low-Level Design (LLD)** across both the **Python ML** and **Go Serving Gateway** engineering tracks.

```
docs/
├── README.md                          # You are here: Architecture index & reading guide
├── architecture/
│   ├── HLD_OVERVIEW.md                # System-Wide High-Level Design (Vision, Multi-Tier SLAs, RL Loop)
│   ├── HLD_PYTHON_ML.md               # Python ML High-Level Design (Offline Pipeline, Serialization)
│   └── HLD_GO_GATEWAY.md              # Go Inference Gateway High-Level Design (Cgo, Session Pool, Router)
└── lld/
    ├── LLD_PYTHON_PIPELINE.md         # Python Low-Level Design (TF-IDF, ONNX Tracing, INT8 Quant, HF Datasets, RL)
    ├── LLD_GO_GATEWAY.md              # Go Low-Level Design (WordPiece Tokenizer, Session Pool, UCB1 Bandit, Swagger UI)
    └── LLD_DATA_CONTRACTS.md          # Tensor Schemas, weights.json, metadata.json & Experience Buffer Specs
```

---

## 🧭 Reading Pathways

Select your pathway based on your role and focus area:

### 1. For Machine Learning Engineers
- Start with **[HLD_PYTHON_ML.md](architecture/HLD_PYTHON_ML.md)** for an overview of the training pipeline, ONNX serialization, and quantization strategies.
- Dive into **[LLD_PYTHON_PIPELINE.md](lld/LLD_PYTHON_PIPELINE.md)** for script-level implementation details, Hugging Face `datasets` ingestion, dynamic INT8 quantization, and offline RL policy optimization.
- Check **[LLD_DATA_CONTRACTS.md](lld/LLD_DATA_CONTRACTS.md)** to inspect the exact tensor signatures, dynamic axes, and metadata schemas expected by the Go runtime.

### 2. For Go & Backend Systems Engineers
- Start with **[HLD_GO_GATEWAY.md](architecture/HLD_GO_GATEWAY.md)** to understand the concurrency model, zero-Cgo Tier-1 execution, and session pooling architecture.
- Dive into **[LLD_GO_GATEWAY.md](lld/LLD_GO_GATEWAY.md)** for details on the pure Go WordPiece tokenizer, thread-safe ONNX Runtime session pool (`chan *DynamicAdvancedSession`), and the Contextual Bandit (UCB1) routing algorithm.
- Explore **[HLD_OVERVIEW.md](architecture/HLD_OVERVIEW.md)** to see the end-to-end request flow and SLA routing guarantees.

### 3. For Platform, DevOps & SRE Engineers
- Read **[HLD_OVERVIEW.md](architecture/HLD_OVERVIEW.md)** for deployment topologies, shared dynamic library (`libonnxruntime`) discovery, and containerization strategies.
- Consult the API and telemetry contracts in **[LLD_DATA_CONTRACTS.md](lld/LLD_DATA_CONTRACTS.md)** and the live OpenAPI 3.0 specification in [api/openapi.yaml](../api/openapi.yaml).

---

## ⚡ Core Architectural Pillars

| Performance Tier | Technology Stack | Latency SLA | Target Use Cases |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Sub-millisecond)** | Vectorized TF-IDF + Logistic Regression (Pure Go / ONNX) | **< 1 ms** (empirically ~6 µs) | Coarse sentiment, high-volume real-time ingestion, `ultra_fast` SLA |
| **Tier 2 (Balanced / High-Accuracy)** | Fine-tuned DistilBERT via ONNX Runtime C-API (INT8 Quantized) | **15–40 ms** (empirically ~8 ms) | Deep semantic accuracy, aspect-based sentiment (ABSA), `balanced` SLA |
| **Tier 3 (Zero-Shot / Complex Context)** | Local LLM fallback (Ollama / vLLM API) | **200–800 ms** | Sarcasm, ambiguous queries, multi-turn reasoning, `deep_context` SLA |
| **Reinforcement Learning Loop** | Online Contextual Bandit (UCB1) + Offline Experience Buffer | **Real-time update** | Adaptive SLA-driven routing, active learning dataset augmentation |
