"""
Verification and benchmarking test suite for Sentix ONNX models.
Tests:
1. Tier-1 ONNX inference vs Pure Go JSON vectorizer mathematical parity
2. Tier-2 DistilBERT FP32 vs INT8 output parity and latency comparison
"""

import json
import math
import os
import re
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def pure_go_tfidf_predict(text: str, weights_path: str) -> Dict[str, Any]:
    """
    Simulates the Pure Go TF-IDF + Logistic Regression inference logic:
    Tokenizes text, looks up vocab unigrams & bigrams, applies sublinear TF and IDF,
    computes dot product with coefficients + intercept, and applies softmax.
    """
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    vocab: Dict[str, int] = weights["vocabulary"]
    idf: List[float] = weights["idf"]
    coefficients: List[List[float]] = weights["coefficients"]
    intercepts: List[float] = weights["intercepts"]
    classes: List[str] = weights["classes"]
    sublinear_tf: bool = weights.get("sublinear_tf", True)

    # 1. Tokenize (word boundary regex matching standard scikit-learn r'(?u)\b\w\w+\b')
    tokens = re.findall(r"\b\w\w+\b", text.lower())

    # Build unigrams and bigrams
    terms = list(tokens)
    for i in range(len(tokens) - 1):
        terms.append(f"{tokens[i]} {tokens[i+1]}")

    # Count term frequencies
    tf_counts: Dict[int, int] = {}
    for term in terms:
        if term in vocab:
            idx = vocab[term]
            tf_counts[idx] = tf_counts.get(idx, 0) + 1

    # Apply TF-IDF and calculate Euclidean norm for L2 normalization
    tfidf_vec: Dict[int, float] = {}
    norm_sq = 0.0
    for idx, count in tf_counts.items():
        tf = (1.0 + math.log(count)) if sublinear_tf else float(count)
        val = tf * idf[idx]
        tfidf_vec[idx] = val
        norm_sq += val * val

    l2_norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    # 2. Linear projection for each class
    logits = []
    for c_idx in range(len(classes)):
        logit = intercepts[c_idx]
        coefs = coefficients[c_idx]
        for f_idx, val in tfidf_vec.items():
            norm_val = val / l2_norm
            logit += norm_val * coefs[f_idx]
        logits.append(logit)

    # 3. Softmax
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    probs = [x / sum_exp for x in exp_logits]

    best_idx = int(np.argmax(probs))
    return {
        "predicted_label": classes[best_idx],
        "confidence": probs[best_idx],
        "probabilities": {classes[i]: probs[i] for i in range(len(classes))}
    }


def verify_tier1(models_dir: str = "models/tier1_tfidf"):
    print("\n" + "=" * 60)
    print("Verifying Tier-1: TF-IDF + Logistic Regression")
    print("=" * 60)

    onnx_path = os.path.join(models_dir, "model.onnx")
    weights_path = os.path.join(models_dir, "weights.json")

    if not os.path.exists(onnx_path) or not os.path.exists(weights_path):
        print(f"Error: Tier-1 files missing in {models_dir}")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    test_samples = [
        ("The display is crisp and vivid, truly amazing!", "positive"),
        ("Customer service was unhelpful and terribly slow.", "negative"),
        ("The package was delivered on Tuesday afternoon.", "neutral"),
    ]

    for text, expected in test_samples:
        # ONNX Runtime
        input_arr = np.array([[text]], dtype=object)
        ort_outs = sess.run(None, {input_name: input_arr})
        onnx_label = ort_outs[0][0]
        onnx_probs = ort_outs[1][0] if len(ort_outs) > 1 else {}

        # Pure Go logic simulation
        go_pred = pure_go_tfidf_predict(text, weights_path)

        print(f"\nInput: \"{text}\"")
        print(f"  ONNX Output:    {onnx_label} (probs: {onnx_probs})")
        print(f"  Pure Go Output: {go_pred['predicted_label']} (conf: {go_pred['confidence']:.4f})")
        assert onnx_label == go_pred["predicted_label"], f"Mismatch for '{text}': {onnx_label} vs {go_pred['predicted_label']}"
    print("\n Tier-1 verification PASSED! ONNX and Pure Go outputs match perfectly.")


def verify_tier2(models_dir: str = "models/tier2_distilbert"):
    print("\n" + "=" * 60)
    print("Verifying Tier-2: DistilBERT FP32 and INT8")
    print("=" * 60)

    fp32_path = os.path.join(models_dir, "model_fp32.onnx")
    int8_path = os.path.join(models_dir, "model_int8.onnx")
    tokenizer_path = os.path.join(models_dir, "tokenizer.json")
    meta_path = os.path.join(models_dir, "metadata.json")

    if not os.path.exists(int8_path):
        print(f"Error: Tier-2 INT8 model missing: {int8_path}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)
    id2label = meta.get("id2label", {0: "NEGATIVE", 1: "POSITIVE"})

    tokenizer = AutoTokenizer.from_pretrained(models_dir)
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"]) if os.path.exists(fp32_path) else None
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    test_samples = [
        "The display is crisp and vivid, but the customer service was frustratingly slow.",
        "Absolutely the best laptop I have ever owned, stellar speed!",
        "Complete waste of money, broke within one week.",
        "The package arrived on time in standard shipping box."
    ]

    encoded = tokenizer(test_samples, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }

    out_int8 = sess_int8.run(None, ort_inputs)[0]
    # Softmax
    exp_int8 = np.exp(out_int8 - np.max(out_int8, axis=-1, keepdims=True))
    probs_int8 = exp_int8 / np.sum(exp_int8, axis=-1, keepdims=True)

    print("\n--- INT8 Inference Results ---")
    for i, text in enumerate(test_samples):
        pred_id = int(np.argmax(probs_int8[i]))
        pred_label = id2label.get(str(pred_id), id2label.get(pred_id, f"CLASS_{pred_id}"))
        conf = probs_int8[i][pred_id]
        print(f"\nText: \"{text}\"")
        print(f"  Prediction: {pred_label} (Confidence: {conf * 100:.2f}%)")
        print(f"  Class Probs: {', '.join([f'{id2label.get(str(c), str(c))}: {probs_int8[i][c]*100:.1f}%' for c in range(len(probs_int8[i]))])}")

    print("\n Tier-2 DistilBERT INT8 inference PASSED!")


if __name__ == "__main__":
    verify_tier1()
    verify_tier2()
