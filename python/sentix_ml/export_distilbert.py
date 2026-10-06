"""
Export and quantize DistilBERT for Sentix Tier-2 sentiment inference.
Supports:
1. Exporting PyTorch DistilBERT to ONNX with dynamic batch & sequence dimensions
2. Post-training Dynamic INT8 Quantization via ONNX Runtime
3. Exporting tokenizer artifacts (tokenizer.json, vocab.txt, config) for Go C-Go integration
4. Outputting metadata with class mappings and input/output contracts
"""

import argparse
import json
import os
import time
from typing import Dict, Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MULTILINGUAL_3CLASS_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


def export_distilbert(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = "models/tier2_distilbert",
    opset_version: int = 17,
) -> Dict[str, Any]:
    """Downloads DistilBERT, exports to ONNX FP32 and INT8, and saves tokenizer artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print(f"Sentix Tier-2: Exporting DistilBERT ({model_id}) to ONNX + INT8")
    print("=" * 65)

    # 1. Load tokenizer and model
    print(f"Loading model and tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    num_labels = model.config.num_labels
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}
    print(f"Loaded model with {num_labels} classes: {id2label}")

    # 2. Save native tokenizer files for Go C-bindings
    print("Exporting tokenizer artifacts for Go C-Go bindings...")
    tokenizer.save_pretrained(output_dir)
    # Ensure tokenizer.json exists (FastTokenizer format)
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        tokenizer.backend_tokenizer.save(tokenizer_json_path)
    print(f"Saved tokenizer to: {tokenizer_json_path}")

    # 3. Export to FP32 ONNX
    fp32_onnx_path = os.path.join(output_dir, "model_fp32.onnx")
    print(f"Exporting PyTorch model to FP32 ONNX: {fp32_onnx_path}...")

    # Dummy inputs for tracing
    dummy_text = "Sentix provides lightning-fast sentiment analysis."
    encoded = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True
    )
    dummy_input_ids = encoded["input_ids"]
    dummy_attention_mask = encoded["attention_mask"]

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    t0 = time.time()
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        fp32_onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )
    export_time = time.time() - t0
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    print(f"FP32 ONNX exported in {export_time:.2f}s (Size: {fp32_size_mb:.2f} MB)")

    # Validate FP32 ONNX model
    onnx_proto = onnx.load(fp32_onnx_path)
    onnx.checker.check_model(onnx_proto)
    print("FP32 ONNX model structure validated successfully.")

    # 4. Post-training Dynamic INT8 Quantization
    int8_onnx_path = os.path.join(output_dir, "model_int8.onnx")
    print(f"Applying Post-Training Dynamic INT8 Quantization: {int8_onnx_path}...")
    t1 = time.time()
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False
    )
    quant_time = time.time() - t1
    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    reduction = (1.0 - (int8_size_mb / fp32_size_mb)) * 100
    print(f"INT8 ONNX quantized in {quant_time:.2f}s (Size: {int8_size_mb:.2f} MB, {reduction:.1f}% reduction)")

    # 5. Benchmark FP32 vs INT8 in Python
    print("\nBenchmarking FP32 vs INT8 inference latency...")
    test_queries = [
        "The display is crisp and vivid, but the customer service was frustratingly slow.",
        "Absolutely amazing performance and stellar battery life!",
        "Poor quality, broke immediately after arrival.",
        "Shipped on Monday with tracking number provided."
    ]
    inputs = tokenizer(test_queries, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    session_fp32 = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])

    # Warmup
    _ = session_fp32.run(None, ort_inputs)
    _ = session_int8.run(None, ort_inputs)

    # Benchmark FP32
    runs = 30
    start = time.perf_counter()
    for _ in range(runs):
        out_fp32 = session_fp32.run(None, ort_inputs)
    latency_fp32 = ((time.perf_counter() - start) / runs) * 1000

    # Benchmark INT8
    start = time.perf_counter()
    for _ in range(runs):
        out_int8 = session_int8.run(None, ort_inputs)
    latency_int8 = ((time.perf_counter() - start) / runs) * 1000

    print(f"FP32 Batch Latency (batch={len(test_queries)}): {latency_fp32:.2f} ms")
    print(f"INT8 Batch Latency (batch={len(test_queries)}): {latency_int8:.2f} ms")
    speedup = latency_fp32 / latency_int8 if latency_int8 > 0 else 1.0
    print(f"Quantization Speedup: {speedup:.2f}x")

    # 6. Save Metadata
    metadata = {
        "tier": 2,
        "model_id": model_id,
        "architecture": "DistilBertForSequenceClassification",
        "num_classes": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "fp32_model_path": "model_fp32.onnx",
        "int8_model_path": "model_int8.onnx",
        "tokenizer_path": "tokenizer.json",
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_mb": round(int8_size_mb, 2),
        "size_reduction_pct": round(reduction, 2),
        "benchmark_latency_ms": {
            "fp32": round(latency_fp32, 2),
            "int8": round(latency_int8, 2),
            "speedup": round(speedup, 2)
        },
        "inputs": [
            {"name": "input_ids", "type": "int64", "shape": ["batch_size", "sequence_length"]},
            {"name": "attention_mask", "type": "int64", "shape": ["batch_size", "sequence_length"]}
        ],
        "outputs": [
            {"name": "logits", "type": "float32", "shape": ["batch_size", num_labels]}
        ]
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to: {meta_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DistilBERT to ONNX with INT8 dynamic quantization")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--multilingual", action="store_true", help="Use 3-class multilingual model")
    parser.add_argument("--output-dir", type=str, default="models/tier2_distilbert", help="Output directory")
    args = parser.parse_args()

    model_to_export = MULTILINGUAL_3CLASS_MODEL if args.multilingual else args.model_id
    export_distilbert(model_id=model_to_export, output_dir=args.output_dir)
