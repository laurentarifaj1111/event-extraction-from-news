import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams

# DOWNLOAD NLTK TOKENIZER IF NOT PRESENT
nltk.download("punkt")

# ============================
# CONFIG — CHANGE THESE
# ============================
CSV_PATH = "../data/filtered/train.csv"
ARTICLE_COL = "clean_text"
SUMMARY_COL = "Summary"
# ============================

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# ------------------------------------
# 1. Missing Values
# ------------------------------------
print("\n==== Missing Values ====")
print(df[[ARTICLE_COL, SUMMARY_COL]].isna().sum())

df = df.dropna(subset=[ARTICLE_COL, SUMMARY_COL])


# ------------------------------------
# 2. Summary & Article Length
# ------------------------------------
print("\n==== Length Statistics ====")

df["summary_len_words"] = df[SUMMARY_COL].apply(lambda x: len(str(x).split()))
df["article_len_words"] = df[ARTICLE_COL].apply(lambda x: len(str(x).split()))

print("\n-- Summary Length (Words) --")
print(df["summary_len_words"].describe())

print("\n-- Article Length (Words) --")
print(df["article_len_words"].describe())

# ------------------------------------
# 3. Compression Ratio
# ------------------------------------
print("\n==== Compression Ratio (article_len / summary_len) ====")

df["compression_ratio"] = df["article_len_words"] / df["summary_len_words"].replace(0, np.nan)
print(df["compression_ratio"].describe())


# ------------------------------------
# 4. Abstractiveness Score
# ------------------------------------
def abstractiveness(article, summary):
    article_words = set(article.lower().split())
    summary_words = summary.lower().split()
    copied = sum(1 for w in summary_words if w in article_words)
    return copied / len(summary_words) if len(summary_words) > 0 else 0

print("\n==== Abstractiveness (0=abstractive, 1=copy-paste) ====")

df["abstractiveness"] = df.apply(
    lambda row: abstractiveness(row[ARTICLE_COL], row[SUMMARY_COL]), axis=1
)

print(df["abstractiveness"].describe())


# ------------------------------------
# 5. Alignment Score (bigram overlap)
# ------------------------------------
def alignment_score(article, summary):
    article_bigrams = set(ngrams(article.lower().split(), 2))
    summary_bigrams = list(ngrams(summary.lower().split(), 2))
    if len(summary_bigrams) == 0:
        return 0
    overlap = sum(1 for bg in summary_bigrams if bg in article_bigrams)
    return overlap / len(summary_bigrams)

print("\n==== Alignment (Bigram Overlap) ====")

df["alignment"] = df.apply(
    lambda row: alignment_score(row[ARTICLE_COL], row[SUMMARY_COL]), axis=1
)

print(df["alignment"].describe())


# ------------------------------------
# 6. Duplicate Detection
# ------------------------------------
print("\n==== Duplicate Articles ====")
dup_articles = df.duplicated(subset=[ARTICLE_COL]).sum()
print("Duplicates in articles:", dup_articles)

print("\n==== Duplicate Summaries ====")
dup_summaries = df.duplicated(subset=[SUMMARY_COL]).sum()
print("Duplicates in summaries:", dup_summaries)


# ------------------------------------
# 7. Outlier Detection
# ------------------------------------
print("\n==== Outliers ====")

very_short_summaries = df[df["summary_len_words"] < 10]
very_long_summaries = df[df["summary_len_words"] > 400]
very_short_articles = df[df["article_len_words"] < 50]
bad_compression = df[df["compression_ratio"] > 50]

print(f"Summaries <10 words: {len(very_short_summaries)}")
print(f"Summaries >400 words: {len(very_long_summaries)}")
print(f"Articles <50 words: {len(very_short_articles)}")
print(f"Compression ratio >50: {len(bad_compression)}")


# ------------------------------------
# 8. Final Summary Report
# ------------------------------------
print("\n==================== DATASET QUALITY REPORT ====================\n")

print("1. SUMMARY LENGTH")
print("   ✓ Ideal range: 30–150 words")
print("   → Avg:", df["summary_len_words"].mean())

print("\n2. ARTICLE LENGTH")
print("   ✓ Ideal: majority <1500–2000 words")
print("   → Avg:", df["article_len_words"].mean())

print("\n3. COMPRESSION RATIO")
print("   ✓ Ideal: 10–30")
print("   → Avg:", df["compression_ratio"].mean())

print("\n4. ABSTRACTIVENESS")
print("   ✓ Ideal: 0.20–0.60")
print("   → Avg:", df["abstractiveness"].mean())

print("\n5. ALIGNMENT")
print("   ✓ Ideal: 0.05–0.25")
print("   → Avg:", df["alignment"].mean())

print("\n6. DUPLICATES")
print("   Articles:", dup_articles)
print("   Summaries:", dup_summaries)

print("\n7. OUTLIERS")
print("   Very short summaries:", len(very_short_summaries))
print("   Very long summaries:", len(very_long_summaries))
print("   Very short articles:", len(very_short_articles))
print("   Bad compression ratio (>50):", len(bad_compression))

print("\n===============================================================\n")
print("Dataset quality analysis completed.")
