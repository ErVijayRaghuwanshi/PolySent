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


# Built-in offline 3-class baseline dataset
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
