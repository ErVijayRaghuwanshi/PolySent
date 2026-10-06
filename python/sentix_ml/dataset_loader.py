"""
Dataset Loader for Sentix Sentiment Analysis.
Integrates Hugging Face datasets (`load_dataset`) with support for:
- cardiffnlp/tweet_eval (sentiment: 3-class negative/neutral/positive)
- stanfordnlp/sst2 (Stanford Sentiment Treebank: binary negative/positive)
- synthetic (offline self-contained 3-class baseline)
"""

from typing import List, Tuple, Optional
from collections import Counter


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


def load_sentiment_dataset(
    dataset_name: str = "sst2",
    split: str = "train",
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Loads and standardizes a sentiment dataset into (texts, labels).
    
    Supported datasets:
    - 'sst2' (stanfordnlp/sst2): binary sentiment ('negative', 'positive')
    - 'tweet_eval' (cardiffnlp/tweet_eval): 3-class sentiment ('negative', 'neutral', 'positive')
    - 'synthetic': built-in 3-class offline dataset
    """
    if dataset_name == "synthetic":
        try:
            from sentix_ml.train_tfidf import SYNTHETIC_DATASET, expand_dataset
        except ImportError:
            from train_tfidf import SYNTHETIC_DATASET, expand_dataset
        texts, labels = expand_dataset(SYNTHETIC_DATASET)
        if max_samples:
            texts, labels = texts[:max_samples], labels[:max_samples]
        return texts, labels

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required. Install via `uv pip install datasets`.")

    print(f"Loading '{dataset_name}' dataset (split='{split}')...")

    if dataset_name in DATASET_CATALOG:
        info = DATASET_CATALOG[dataset_name]
        if info["config"]:
            ds = load_dataset(info["repo"], info["config"], split=split)
        else:
            ds = load_dataset(info["repo"], split=split)
        text_col = info["text_col"]
        label_col = info["label_col"]
        label_map = info["label_map"]
    else:
        # Generic direct repository identifier
        ds = load_dataset(dataset_name, split=split)
        text_col = "sentence" if "sentence" in ds.column_names else "text"
        label_col = "label"
        label_map = {0: "negative", 1: "positive"}

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    texts = [row[text_col] for row in ds]
    labels = [label_map.get(row[label_col], str(row[label_col])) for row in ds]

    counts = Counter(labels)
    print(f"Successfully loaded {len(texts)} samples from '{dataset_name}' [{split}]:")
    for lbl, cnt in counts.items():
        print(f"  • {lbl}: {cnt}")

    return texts, labels


if __name__ == "__main__":
    # Test with standard sst2 validation split
    texts, labels = load_sentiment_dataset("sst2", split="validation", max_samples=5)
    for t, l in zip(texts, labels):
        print(f"  [{l}] {t[:60]}")
