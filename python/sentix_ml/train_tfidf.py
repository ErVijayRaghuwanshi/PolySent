"""
Train TF-IDF + Logistic Regression model for Tier-1 sub-millisecond sentiment analysis.
Exports:
1. ONNX model via skl2onnx (for ONNX Runtime)
2. JSON weights & vocabulary (for zero-overhead pure Go inference)
3. Model metadata
"""

import argparse
import json
import os
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

try:
    from sentix_ml.dataset_loader import load_sentiment_dataset, SYNTHETIC_DATASET, expand_dataset
except ImportError:
    from dataset_loader import load_sentiment_dataset, SYNTHETIC_DATASET, expand_dataset


def train_and_export(
    output_dir: str = "models/tier1_tfidf",
    dataset_name: str = "synthetic",
    max_samples: Optional[int] = None,
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, Any]:
    """Trains TF-IDF + Logistic Regression and exports ONNX + Go-compatible JSON weights."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"Sentix Tier-1: Training Vectorized TF-IDF on '{dataset_name}'")
    print("=" * 60)

    # 1. Prepare data
    raw_texts, raw_labels = load_sentiment_dataset(dataset_name, split="train", max_samples=max_samples)
    X_train, X_val, y_train, y_val = train_test_split(
        raw_texts, raw_labels, test_size=0.2, random_state=42, stratify=raw_labels
    )
    print(f"Training set: {len(X_train)} samples | Validation set: {len(X_val)} samples")

    # 2. Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs"
        ))
    ])

    # 3. Train
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.3f}s")

    # 4. Evaluate
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    # 5. Extract weights for pure Go implementation
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    classes = [str(c) for c in clf.classes_]
    vocab = {str(word): int(idx) for word, idx in tfidf.vocabulary_.items()}
    idf = [float(val) for val in tfidf.idf_]
    if len(classes) == 2 and len(clf.coef_) == 1:
        coefficients = [
            [0.0 for _ in range(len(clf.coef_[0]))],
            [float(v) for v in clf.coef_[0]]
        ]
        intercepts = [0.0, float(clf.intercept_[0])]
    else:
        coefficients = [[float(v) for v in row] for row in clf.coef_]
        intercepts = [float(v) for v in clf.intercept_]

    go_weights = {
        "model_name": "sentix-tier1-tfidf-logreg",
        "version": "1.0.0",
        "task": "document_level_sentiment",
        "classes": classes,
        "vocabulary_size": len(vocab),
        "ngram_range": list(ngram_range),
        "sublinear_tf": True,
        "vocabulary": vocab,
        "idf": idf,
        "coefficients": coefficients,
        "intercepts": intercepts
    }

    weights_path = os.path.join(output_dir, "weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(go_weights, f, indent=2)
    print(f"Exported Pure Go weights to: {weights_path} ({os.path.getsize(weights_path) / 1024:.1f} KB)")

    # 6. Export to ONNX via skl2onnx
    onnx_path = os.path.join(output_dir, "model.onnx")
    initial_type = [("text", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=15)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model to: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")

    # 7. Export metadata
    metadata = {
        "tier": 1,
        "name": "sentix-tier1-tfidf-logreg",
        "engine": "tfidf_logistic_regression",
        "classes": classes,
        "vocab_size": len(vocab),
        "latency_target": "<1ms",
        "accuracy": float(acc),
        "onnx_model_file": "model.onnx",
        "weights_file": "weights.json",
        "inputs": [
            {"name": "text", "type": "string", "shape": [None, 1]}
        ],
        "outputs": [
            {"name": "label", "type": "string", "shape": [None]},
            {"name": "probabilities", "type": "map(string, float) or tensor", "shape": [None, len(classes)]}
        ]
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to: {meta_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export Sentix Tier-1 TF-IDF + Logistic Regression")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "sst2", "tweet_eval"], help="Dataset to train on")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of training samples")
    parser.add_argument("--output-dir", type=str, default="models/tier1_tfidf", help="Output directory")
    args = parser.parse_args()

    train_and_export(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        max_samples=args.max_samples
    )
