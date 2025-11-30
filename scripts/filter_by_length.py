"""
Script to filter articles by word length and dataset quality rules.

- Keeps ALL articles with < max_words words.
- Randomly samples X% of longer articles.
- Computes dataset quality metrics (summary length, article length,
  compression ratio, abstractiveness, alignment).
- Removes bad samples BEFORE splitting.
- Outputs clean train/val/test splits.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from nltk.util import ngrams

# Ensure nltk punkt
nltk.download("punkt")

# Add root directory to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.preprocessing import Preprocessor


def compute_abstractiveness(article, summary):
    article_words = set(str(article).lower().split())
    summary_words = str(summary).lower().split()
    if len(summary_words) == 0:
        return 0
    copied = sum(1 for w in summary_words if w in article_words)
    return copied / len(summary_words)


def compute_alignment(article, summary):
    article_words = str(article).lower().split()
    summary_words = str(summary).lower().split()
    if len(summary_words) < 2:
        return 0
    article_bigrams = set(ngrams(article_words, 2))
    summary_bigrams = list(ngrams(summary_words, 2))
    overlap = sum(1 for bg in summary_bigrams if bg in article_bigrams)
    return overlap / len(summary_bigrams) if len(summary_bigrams) else 0


def filter_by_length(
    raw_file="data/raw/data.csv",
    output_dir="data/filtered_v1",
    max_words=500,
    sample_percent=0.1,
    random_seed=42,
):
    """
    Filters by word length THEN applies dataset-quality cleanup,
    THEN splits train/val/test.
    """

    raw_path = project_root / raw_file
    output_path = project_root / output_dir
    np.random.seed(random_seed)

    if not raw_path.exists():
        print(f"❌ Raw dataset not found: {raw_path}")
        return

    print("\n" + "=" * 60)
    print("Filtering Articles by Length and Dataset Quality")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. Load raw dataset
    # -------------------------------------------------------
    print(f"\n📖 Loading raw dataset: {raw_path}")
    pre = Preprocessor(text_column="Content")
    df_raw = pre.load_dataset(str(raw_path))
    print(f"   Loaded entries: {len(df_raw)}")

    # -------------------------------------------------------
    # 2. Preprocess & clean
    # -------------------------------------------------------
    print("\n🔹 Cleaning dataset...")
    df = pre.preprocess_dataframe(df_raw)
    print(f"   After cleaning: {len(df)} entries")

    # -------------------------------------------------------
    # 3. Compute word counts
    # -------------------------------------------------------
    print("\n🔹 Computing word counts...")
    df["article_len_words"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    df["summary_len_words"] = df["Summary"].apply(lambda x: len(str(x).split()))

    # -------------------------------------------------------
    # 4. Compute dataset quality metrics
    # -------------------------------------------------------
    print("\n🔹 Computing dataset quality metrics...")

    df["compression_ratio"] = (
        df["article_len_words"] / df["summary_len_words"].replace(0, np.nan)
    )
    df["abstractiveness"] = df.apply(
        lambda r: compute_abstractiveness(r["clean_text"], r["Summary"]), axis=1
    )
    df["alignment"] = df.apply(
        lambda r: compute_alignment(r["clean_text"], r["Summary"]), axis=1
    )

    # -------------------------------------------------------
    # 5. Filter short + sampled long articles
    # -------------------------------------------------------
    print("\n🔹 Filtering by article length (main task rule)...")

    short_articles = df[df["article_len_words"] < max_words]
    long_articles = df[df["article_len_words"] >= max_words]

    num_sample = max(1, int(len(long_articles) * sample_percent))
    long_sampled = long_articles.sample(n=num_sample, random_state=random_seed)

    df_filtered = pd.concat([short_articles, long_sampled], ignore_index=True)
    df_filtered = df_filtered.sample(frac=1, random_state=random_seed).reset_index(
        drop=True
    )

    print(f"   Total after length filtering: {len(df_filtered)}")

    # -------------------------------------------------------
    # 6. Apply dataset-quality cleanup BEFORE splitting
    # -------------------------------------------------------
    print("\n🔹 Applying dataset quality filters...")

    before = len(df_filtered)

    df_filtered = df_filtered[df_filtered["summary_len_words"] >= 10]
    df_filtered = df_filtered[df_filtered["article_len_words"] >= 50]
    df_filtered = df_filtered[df_filtered["abstractiveness"] < 0.8]
    df_filtered = df_filtered[df_filtered["alignment"] < 0.5]
    df_filtered = df_filtered[
        (df_filtered["compression_ratio"] >= 5)
        & (df_filtered["compression_ratio"] <= 40)
    ]

    df_filtered = df_filtered.drop_duplicates(subset=["clean_text"])
    df_filtered = df_filtered.drop_duplicates(subset=["Summary"])

    after = len(df_filtered)
    print(f"   Removed {before - after} low-quality samples")
    print(f"   Final dataset size: {after}")

    # -------------------------------------------------------
    # 7. Train/val/test split
    # -------------------------------------------------------
    print("\n🔹 Splitting dataset (train/val/test)...")
    train_df, val_df, test_df = pre.split_dataset(
        df_filtered, test_size=0.1, val_size=0.1
    )

    print(f"   Train: {len(train_df)}")
    print(f"   Val:   {len(val_df)}")
    print(f"   Test:  {len(test_df)}")

    # -------------------------------------------------------
    # 8. Save results
    # -------------------------------------------------------
    print(f"\n💾 Saving filtered splits to {output_path} ...")
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print(f"\n✅ train.csv saved.")
    print(f"✅ val.csv saved.")
    print(f"✅ test.csv saved.")

    print("\n" + "=" * 60)
    print("Filtering Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter dataset with quality rules")
    parser.add_argument("--raw", type=str, default="data/raw/data.csv")
    parser.add_argument("--output", type=str, default="data/filtered_v1")
    parser.add_argument("--max-words", type=int, default=500)
    parser.add_argument("--sample-percent", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    filter_by_length(
        raw_file=args.raw,
        output_dir=args.output,
        max_words=args.max_words,
        sample_percent=args.sample_percent,
        random_seed=args.seed,
    )
