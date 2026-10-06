"""
Train TF-IDF + Logistic Regression model for Tier-1 sub-millisecond sentiment analysis.
Exports:
1. ONNX model via skl2onnx (for ONNX Runtime)
2. JSON weights & vocabulary (for zero-overhead pure Go inference)
3. Model metadata
"""

import json
import os
import time
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


# Comprehensive sentiment dataset across varied domains (reviews, electronics, food, customer service)
SYNTHETIC_DATASET = [
    # Positive (class: positive, id: 2)
    ("The display is crisp, vibrant, and looks absolutely stunning.", "positive"),
    ("Battery life easily lasts two full days with heavy usage.", "positive"),
    ("Customer support was remarkably fast, polite, and resolved my issue immediately.", "positive"),
    ("Exceptional build quality with premium materials and sleek design.", "positive"),
    ("Fast shipping and arrived in pristine packaging ahead of schedule.", "positive"),
    ("The sound quality is rich with punchy bass and crystal clear vocals.", "positive"),
    ("Intuitive user interface that makes navigation a complete breeze.", "positive"),
    ("Outstanding performance with zero lag even under heavy multitasking.", "positive"),
    ("I love everything about this device, exceeded all my expectations.", "positive"),
    ("Great value for money, definitely worth every penny spent.", "positive"),
    ("Highly recommend this to anyone looking for a reliable product.", "positive"),
    ("Flawless camera quality, takes crisp low-light photos effortlessly.", "positive"),
    ("The software is smooth, responsive, and packed with handy features.", "positive"),
    ("Incredible experience from start to finish, five stars!", "positive"),
    ("Super easy setup, was up and running within two minutes.", "positive"),
    ("The keyboard feels tactile and very satisfying to type on.", "positive"),
    ("Remarkable efficiency and saves tons of time daily.", "positive"),
    ("Best purchase I have made all year, thoroughly impressed.", "positive"),
    ("Very pleased with the build and overall durability.", "positive"),
    ("Fast charging is phenomenal, goes to 80% in 20 minutes.", "positive"),
    ("Brilliant design and works exactly as advertised.", "positive"),
    ("Delightful experience, customer service was top tier.", "positive"),
    ("The picture clarity is breathtaking, true deep blacks.", "positive"),
    ("Very quiet fan noise and thermals remain cool throughout.", "positive"),
    ("Love the compact form factor and premium metallic finish.", "positive"),

    # Neutral (class: neutral, id: 1)
    ("The package was delivered on Tuesday afternoon.", "neutral"),
    ("The phone weighs approximately 180 grams and has a 6.1 inch screen.", "neutral"),
    ("It comes with standard accessories including a charging cable.", "neutral"),
    ("The software update was released yesterday morning.", "neutral"),
    ("Battery capacity is rated at 4500 milliamp hours.", "neutral"),
    ("It operates at standard room temperature under normal load.", "neutral"),
    ("The device supports both Wi-Fi 6 and Bluetooth 5.2.", "neutral"),
    ("The item matches the dimensions specified in the user manual.", "neutral"),
    ("There are three color options available: black, white, and gray.", "neutral"),
    ("Shipped via standard postal carrier with tracking number provided.", "neutral"),
    ("It functions as expected, neither exceptional nor disappointing.", "neutral"),
    ("The sound is average, typical for speakers in this price range.", "neutral"),
    ("Display resolution is 1080p with a 60Hz refresh rate.", "neutral"),
    ("The product was manufactured in November according to the label.", "neutral"),
    ("Contains a quad-core processor and 8GB of system memory.", "neutral"),
    ("The interface is standard and follows conventional layouts.", "neutral"),
    ("Received the confirmation email shortly after placing the order.", "neutral"),
    ("The battery lasted about 6 hours of continuous video playback.", "neutral"),
    ("The warranty covers repairs for a period of twelve months.", "neutral"),
    ("Average performance, meets basic baseline criteria.", "neutral"),
    ("The device has two USB ports and one HDMI connection.", "neutral"),
    ("It does the job it was designed to do, nothing more nothing less.", "neutral"),
    ("The packaging consists of plain brown cardboard box.", "neutral"),
    ("Settings can be adjusted in the preferences menu.", "neutral"),
    ("Standard build quality consistent with entry-level hardware.", "neutral"),

    # Negative (class: negative, id: 0)
    ("The customer service was frustratingly slow and completely unhelpful.", "negative"),
    ("Battery drains rapidly in less than three hours on standby.", "negative"),
    ("Build quality feels cheap, fragile, and made of flimsy plastic.", "negative"),
    ("Constant software crashes and freezes every time I open an app.", "negative"),
    ("Terrible audio quality with heavy distortion and muffled sound.", "negative"),
    ("Extremely disappointing purchase, stopped working after two days.", "negative"),
    ("The screen flickers uncontrollably and shows strange artifacts.", "negative"),
    ("Overheats terribly even when performing basic simple tasks.", "negative"),
    ("Shipping was delayed by three weeks and arrived damaged.", "negative"),
    ("Horrible return process, refused to honor the warranty.", "negative"),
    ("The buttons feel loose, rattle constantly, and stick.", "negative"),
    ("Waste of money, would not recommend this to anyone.", "negative"),
    ("Camera takes blurry, washed-out images with terrible autofocus.", "negative"),
    ("Wi-Fi connection drops constantly every few minutes.", "negative"),
    ("The charging port broke within the first week of gentle use.", "negative"),
    ("Completely broken out of the box, would not even power on.", "negative"),
    ("Loud whining noise from the fan that is impossible to ignore.", "negative"),
    ("Terrible ergonomics, painful and uncomfortable to hold.", "negative"),
    ("Misleading advertisement, features listed were missing.", "negative"),
    ("Worst customer support experience I have ever encountered.", "negative"),
    ("The companion app is full of bugs and crashes repeatedly.", "negative"),
    ("Excessive bloatware preinstalled that cannot be uninstalled.", "negative"),
    ("Very slow performance, lags on even basic menu navigation.", "negative"),
    ("Fell apart within a month of minimal usage.", "negative"),
    ("Defective item, highly frustrating and deeply dissatisfied.", "negative"),
]


def expand_dataset(base_data: List[Tuple[str, str]], repeat: int = 5) -> Tuple[List[str], List[str]]:
    """Expands dataset with variations to ensure robust statistical learning."""
    prefixes = ["Honestly, ", "In my opinion, ", "To be frank, ", "Overall, ", ""]
    suffixes = ["", " in general.", " without doubt.", " for sure.", " personally."]
    
    texts = []
    labels = []
    for text, label in base_data:
        texts.append(text)
        labels.append(label)
        for p in prefixes:
            for s in suffixes:
                if p or s:
                    texts.append(f"{p}{text.lower()}{s}")
                    labels.append(label)
    return texts, labels


def train_and_export(
    output_dir: str = "models/tier1_tfidf",
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, Any]:
    """Trains TF-IDF + Logistic Regression and exports ONNX + Go-compatible JSON weights."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Sentix Tier-1: Training Vectorized TF-IDF + Logistic Regression")
    print("=" * 60)

    # 1. Prepare data
    raw_texts, raw_labels = expand_dataset(SYNTHETIC_DATASET)
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
    train_and_export()
