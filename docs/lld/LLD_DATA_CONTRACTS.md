# Data, Tensor & Interface Contracts Specification (LLD)

---

## 1. ONNX Tensor Interface Contract

The Tier-2 Transformer engine (`models/tier2_distilbert/model_int8.onnx`) strictly enforces the following ONNX tensor signatures:

### Input Tensors:
| Tensor Name | ONNX Type | Go Type | Shape | Dynamic Axes | Semantics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `input_ids` | `tensor(int64)` | `[]int64` | `[batch_size, sequence_length]` | Axis 0: `batch_size`<br>Axis 1: `sequence_length` | Token vocabulary IDs from `tokenizer.json` / `vocab.txt` |
| `attention_mask` | `tensor(int64)` | `[]int64` | `[batch_size, sequence_length]` | Axis 0: `batch_size`<br>Axis 1: `sequence_length` | Binary mask ($1$ for actual tokens, $0$ for padding) |

### Output Tensors:
| Tensor Name | ONNX Type | Go Type | Shape | Dynamic Axes | Semantics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `logits` | `tensor(float)` | `[]float32` | `[batch_size, num_classes]` | Axis 0: `batch_size` | Unnormalized log-odds classification scores |

---

## 2. Zero-Cgo Weights Schema (`weights.json`)

Exported by Python to `models/tier1_tfidf/weights.json` for Pure Go execution:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SentixTier1Weights",
  "type": "object",
  "required": [
    "model_name",
    "version",
    "classes",
    "vocabulary_size",
    "ngram_range",
    "sublinear_tf",
    "vocabulary",
    "idf",
    "coefficients",
    "intercepts"
  ],
  "properties": {
    "model_name": { "type": "string", "example": "sentix-tier1-tfidf-logreg" },
    "version": { "type": "string", "example": "1.0.0" },
    "classes": {
      "type": "array",
      "items": { "type": "string" },
      "example": ["negative", "neutral", "positive"]
    },
    "vocabulary_size": { "type": "integer", "example": 3000 },
    "ngram_range": {
      "type": "array",
      "items": { "type": "integer" },
      "example": [1, 2]
    },
    "sublinear_tf": { "type": "boolean", "example": true },
    "vocabulary": {
      "type": "object",
      "additionalProperties": { "type": "integer" },
      "description": "Mapping from term string to feature index"
    },
    "idf": {
      "type": "array",
      "items": { "type": "number" },
      "description": "Inverse document frequencies indexed by feature ID"
    },
    "coefficients": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" }
      },
      "description": "K x M matrix of linear coefficients"
    },
    "intercepts": {
      "type": "array",
      "items": { "type": "number" },
      "description": "K-dimensional bias vector"
    }
  }
}
```

---

## 3. Model Metadata Schema (`metadata.json`)

Exported alongside every model in `models/*/metadata.json`:

```json
{
  "tier": 2,
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "architecture": "DistilBertForSequenceClassification",
  "num_classes": 2,
  "id2label": {
    "0": "NEGATIVE",
    "1": "POSITIVE"
  },
  "label2id": {
    "NEGATIVE": 0,
    "POSITIVE": 1
  },
  "fp32_model_path": "model_fp32.onnx",
  "int8_model_path": "model_int8.onnx",
  "tokenizer_path": "tokenizer.json",
  "fp32_size_mb": 255.55,
  "int8_size_mb": 64.47,
  "size_reduction_pct": 74.77,
  "benchmark_latency_ms": {
    "fp32": 46.56,
    "int8": 15.62,
    "speedup": 2.98
  }
}
```

---

## 4. Experience Replay Log Schema (`data/rl_feedback.jsonl`)

Appended by `internal/router/bandit.go` when `POST /v1/sentiment/feedback` is called:

```json
{
  "timestamp": "2026-10-07T00:32:57Z",
  "request_id": "req-18dc0606cfc20c10-1",
  "text": "The display is crisp and vivid, but customer service was frustratingly slow.",
  "engine_used": "onnx_distilbert_int8",
  "predicted_label": "NEGATIVE",
  "correct_label": "NEGATIVE",
  "reward": 1.0,
  "latency_ms": 9.377,
  "comment": "Accurate negative polarity on mixed review"
}
```

---

## 5. API Payloads Contract

### 1. `POST /v1/sentiment/analyze`
**Request Payload**:
```json
{
  "text": "The display is crisp and vivid, but customer service was slow.",
  "task": "document_level",
  "strategy": "balanced",
  "language": "auto",
  "aspects": ["display", "customer service"]
}
```

**Response Payload**:
```json
{
  "request_id": "req-18dc0606cfc20c10-1",
  "status": "success",
  "task": "document_level",
  "engine_used": "onnx_distilbert_int8",
  "latency_ms": 9.377,
  "overall": {
    "label": "NEGATIVE",
    "score": 0.9991
  },
  "aspects": [
    {
      "aspect": "display",
      "sentiment": "NEGATIVE",
      "confidence": 0.9991
    }
  ]
}
```

### 2. `POST /v1/sentiment/feedback`
**Request Payload**:
```json
{
  "request_id": "req-18dc0606cfc20c10-1",
  "text": "The display is crisp...",
  "engine_used": "onnx_distilbert_int8",
  "predicted_label": "NEGATIVE",
  "correct_label": "NEGATIVE",
  "reward": 1.0,
  "latency_ms": 9.377
}
```

**Response Payload**:
```json
{
  "status": "success",
  "message": "reinforcement learning feedback recorded successfully",
  "updated_stats": {
    "total_feedback_pulls": 1,
    "exploration_factor": 1.414,
    "arms": {
      "tier2": {
        "engine": "onnx_distilbert_int8",
        "average_reward": 1.0,
        "pull_count": 1
      }
    }
  }
}
```
