import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUTPUT_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    """Clean URLs, HTML tags, mentions, and extra whitespace."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_split(dataset_split) -> pd.DataFrame:
    """Convert a Hugging Face dataset split into a cleaned DataFrame."""
    df = pd.DataFrame(dataset_split)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df[["text", "label"]]


def main():
    print("Loading IMDb dataset...")

    dataset = load_dataset("stanfordnlp/imdb")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = process_split(dataset["train"])
    test_df = process_split(dataset["test"])

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()