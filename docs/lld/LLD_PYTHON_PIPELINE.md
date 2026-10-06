# Python ML Pipeline Low-Level Design (LLD)

---

## 1. Module Overview & Dependency Graph

```
python/sentix_ml/
├── __init__.py               # Package metadata and version definition
├── dataset_loader.py         # Standardized Hugging Face dataset loading
├── train_tfidf.py            # Tier 1 TF-IDF training & dual serialization
├── export_distilbert.py      # Tier 2 DistilBERT ONNX export & INT8 quantization
├── verify_inference.py       # Cross-tier numerical parity & benchmarking
└── rl_loop.py                # Experience replay buffer analysis & active learning
```

---

## 2. Ingestion Module: `dataset_loader.py`

### Class / Function Signatures
```python
def load_sentiment_dataset(
    dataset_name: str = "sst2",
    split: str = "train",
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str]]:
```

### Dataset Catalog Configuration
```python
DATASET_CATALOG = {
    "sst2": {
        "repo": "stanfordnlp/sst2",
        "config": None,
        "text_col": "sentence",
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"}
    },
    "tweet_eval": {
        "repo": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"}
    }
}
```

### Ingestion Logic:
1. If `dataset_name == "synthetic"`: loads the self-contained baseline with deterministic prefix/suffix expansions.
2. If `dataset_name` is in `DATASET_CATALOG`: invokes `datasets.load_dataset(repo, config, split=split)`.
3. Maps raw numeric labels to standardized string polarities (`negative`, `neutral`, `positive`).
4. Selects up to `max_samples` if configured and returns sanitized `(texts, labels)` tuples.

---

## 3. Tier 1 Training & Serialization: `train_tfidf.py`

### Mathematical Formulations

#### 1. Sublinear Term Frequency ($TF$):
To prevent long documents or repetitive keywords from disproportionately dominating linear logits:
$$TF(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

#### 2. Inverse Document Frequency ($IDF$):
$$IDF(t) = \ln\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$

#### 3. Euclidean L2 Normalization:
$$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^M v_i^2}}$$

#### 4. Multinomial Linear Classification & Softmax:
For each class $c \in \{0, \dots, K-1\}$:
$$z_c = b_c + \sum_{j=1}^M w_{c, j} \cdot v_{\text{norm}, j}$$
$$P(y = c \mid x) = \frac{\exp(z_c - \max_k z_k)}{\sum_{k=0}^{K-1} \exp(z_k - \max_k z_k)}$$

### Dual Serialization Logic:

#### 1. Export to ONNX (`model.onnx`):
- Uses `skl2onnx.convert_sklearn`.
- Defines input type: `StringTensorType([None, 1])`.
- Target Opset: `target_opset=15`.
- Contains both the string tokenizer, TF-IDF n-gram scanner, and logistic regression operators within a single ONNX graph.

#### 2. Export to Pure Go Weights (`weights.json`):
- Extracts vocabulary mapping `vocab: Dict[str, int]`.
- Extracts float arrays `idf: List[float]`.
- Normalizes binary classification:
  - Scikit-Learn stores binary classifiers as a single weight vector `coef_[0]`.
  - Sentix normalizes this into symmetric 2-class multinomial format:
    $$\text{Row } 0 \text{ (negative)}: [0, 0, \dots, 0], \quad b_0 = 0.0$$
    $$\text{Row } 1 \text{ (positive)}: \text{coef\_}[0], \quad b_1 = \text{intercept\_}[0]$$
    This ensures `softmax([0, z])` is mathematically identical to `sigmoid(z)`.

---

## 4. Tier 2 DistilBERT Export & Quantization: `export_distilbert.py`

### Tracing Pipeline:
```python
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "logits": {0: "batch_size"}
}

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    fp32_onnx_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=14,
    do_constant_folding=True,
    dynamo=False  # Legacy TorchScript tracing ensures single-file self-contained ONNX
)
```

### Post-Training Dynamic INT8 Quantization:
```python
quantize_dynamic(
    model_input=fp32_onnx_path,
    model_output=int8_onnx_path,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False
)
```

### Tokenizer Preservation:
- Calls `tokenizer.save_pretrained(output_dir)`.
- Guarantees generation of `tokenizer.json` (Hugging Face FastTokenizer format) and extracts `vocab.txt` for microsecond Go tokenization.

---

## 5. Verification & Parity Suite: `verify_inference.py`

### Parity Verification Function:
`pure_go_tfidf_predict(text, weights_path)`
- Re-implements the exact Go tokenization regex `(?u)\b\w\w+\b`.
- Computes sublinear TF, IDF, Euclidean norm, dot product, and softmax in pure Python.
- Compares predictions against `onnxruntime.InferenceSession("models/tier1_tfidf/model.onnx")`.
- Assert statement enforces:
  $$\text{Label}_{\text{ONNX}} \equiv \text{Label}_{\text{PureGo}}, \quad |\text{Prob}_{\text{ONNX}} - \text{Prob}_{\text{PureGo}}| < 10^{-4}$$

---

## 6. Offline RL Policy Optimizer: `rl_loop.py`

### Replay Record Ingestion:
Reads `data/rl_feedback.jsonl`:
```json
{
  "timestamp": "2026-10-07T00:00:00Z",
  "request_id": "req-xxx",
  "text": "The display is crisp but customer service was slow.",
  "engine_used": "onnx_distilbert_int8",
  "predicted_label": "NEGATIVE",
  "correct_label": "NEGATIVE",
  "reward": 1.0,
  "latency_ms": 9.3
}
```

### Analytical Outputs:
- **Empirical Tier Rewards**: Mean reward per engine $\mu_e = \frac{1}{N_e} \sum r_i$.
- **Latency-to-Reward Trade-off**: $\text{Efficiency}_e = \frac{\mu_e}{\text{AvgLatencyMs}_e}$.
- **Candidate Active Learning Extraction**: Samples where `reward < 0` or `predicted_label != correct_label` are extracted and serialized to `data/rl_summary.json` for active retraining.
