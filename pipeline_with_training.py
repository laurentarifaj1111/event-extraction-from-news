"""
Pipeline that uses fine-tuned models for summarization.
This pipeline loads fine-tuned models if they exist, otherwise falls back to pretrained models.
"""
import argparse
import json
import os
from pathlib import Path

import nltk
import pandas as pd

from scripts.preprocessing import Preprocessor

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from scripts.event_extraction import EventExtractor
from scripts.summarizers.summarizer_bert import BertExtractiveSummarizer
from scripts.summarizers.summarizer_pegasus import PegasusSummarizer
from scripts.summarizers.summarizer_t5 import T5Summarizer
from scripts.evaluation import ModelEvaluator
from scripts.utils import print_device_info

def run_pipeline(limit: int = 50, use_finetuned: bool = True):
    print("\n=============================================")
    print("   NLP PIPELINE: WITH FINE-TUNED MODELS")
    print("=============================================\n")
    
    # Print device information
    print_device_info()
    
    # Check if fine-tuned models exist
    pegasus_finetuned = os.path.exists("models/pegasus-finetuned")
    t5_finetuned = os.path.exists("models/t5-finetuned")
    bert_finetuned = os.path.exists("models/bert-finetuned")
    
    if use_finetuned:
        print("\n🔹 Fine-tuned Model Status:")
        print(f"  BERT: {'✓ Found' if bert_finetuned else '✗ Not found (will use pretrained)'}")
        print(f"  Pegasus: {'✓ Found' if pegasus_finetuned else '✗ Not found (will use pretrained)'}")
        print(f"  T5: {'✓ Found' if t5_finetuned else '✗ Not found (will use pretrained)'}")
        if not pegasus_finetuned and not t5_finetuned and not bert_finetuned:
            print("\n  ⚠ No fine-tuned models found!")
            print("  💡 Run: python scripts/train_summarizers.py to train models")
            print("  Continuing with pretrained models...\n")
    else:
        print("\n🔹 Using pretrained models (fine-tuning disabled)\n")

    # -------------------------
    # 1. Load RAW dataset
    # -------------------------
    if not preprocessed_data_exists():
        raw_path = "data/raw/data.csv"

        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw dataset NOT found at {raw_path}\n"
                "Please add the Kaggle dataset in data/raw"
            )

        print("🔹 Loading raw dataset...")
        pre = Preprocessor(text_column="Content")
        df_raw = pre.load_dataset(raw_path)

        # -------------------------
        # 2. Clean and preprocess
        # -------------------------
        print("🔹 Cleaning dataset...")
        df_clean = pre.preprocess_dataframe(df_raw)

        # -------------------------
        # 3. Split dataset
        # -------------------------
        print("🔹 Splitting dataset...")
        train_df, val_df, test_df = pre.split_dataset(df_clean)

        # Ensure output directory
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        pre.save_splits(train_df, val_df, test_df)

        print("✔ Preprocessing complete.\n")
    else:
        print("✔ Preprocessed data found. Skipping preprocessing.\n")

    # Choose dataset to run pipeline on
    test_df = pd.read_csv("data/processed/test.csv")
    df = test_df.copy()
    if limit > 0:
        df = df.head(limit)
        print(f"🔹 Using first {limit} articles.")

    # -------------------------
    # 4. Initialize models
    # -------------------------
    print("\n🔹 Initializing models...")
    event_extractor = EventExtractor()
    
    # Use fine-tuned models if available
    # bert_summ = BertExtractiveSummarizer(use_finetuned=use_finetuned)
    pegasus_summ = PegasusSummarizer(use_finetuned=use_finetuned)
    t5_summ = T5Summarizer(use_finetuned=use_finetuned)
    
    evaluator = ModelEvaluator()
    print("✔ All models initialized.\n")

    results = []

    # -------------------------
    # 5. Process each article
    # -------------------------
    for idx, row in df.iterrows():
        article_id = int(idx)
        text = row["clean_text"]

        print(f"\n========== ARTICLE {article_id} ==========")

        # Event Extraction
        event_info = event_extractor.extract_event(text)

        # Summaries + inference times
        # bert_summary, bert_time = evaluator.measure_inference_time(
        #     bert_summ.summarize, text
        # )
        # peg_summary, peg_time = evaluator.measure_inference_time(
        #     pegasus_summ.summarize, text
        # )
        t5_summary, t5_time = evaluator.measure_inference_time(
            t5_summ.summarize, text
        )

        # Evaluation (ROUGE + BERTScore)
        # bert_scores = evaluator.evaluate_single(text, bert_summary)
        # peg_scores = evaluator.evaluate_single(text, peg_summary)
        t5_scores = evaluator.evaluate_single(text, t5_summary)

        # Structured record
        entry = {
            "article_id": article_id,
            "original_text": text,
            "event_extraction": event_info,
            "summaries": {
                # "bert": bert_summary,
                # "pegasus": peg_summary,
                "t5": t5_summary
            },
            "metrics": {
                # "bert": {**bert_scores, "time_ms": bert_time},
                # "pegasus": {**peg_scores, "time_ms": peg_time},
                "t5": {**t5_scores, "time_ms": t5_time}
            },
            "model_info": {
                # "bert_finetuned": bert_finetuned if use_finetuned else False,
                # "pegasus_finetuned": pegasus_finetuned if use_finetuned else False,
                "t5_finetuned": t5_finetuned if use_finetuned else False
            }
        }

        results.append(entry)

    # -------------------------
    # 6. Save Final Structured Output
    # -------------------------
    Path("results").mkdir(exist_ok=True)
    output_path = "results/structured_results_finetuned.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=============================================")
    print(f" Pipeline complete! Saved results to: {output_path}")
    print("=============================================\n")

def preprocessed_data_exists():
    return (
        os.path.exists("data/processed/train.csv") and
        os.path.exists("data/processed/val.csv") and
        os.path.exists("data/processed/test.csv")
    )


# ============================================================
#              CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Pipeline with Fine-tuned Models")
    parser.add_argument("--limit", type=int, default=50,
                        help="Process first N articles (0 for full dataset)")
    parser.add_argument("--no-finetuned", action="store_true",
                        help="Use pretrained models instead of fine-tuned models")

    args = parser.parse_args()
    run_pipeline(limit=args.limit, use_finetuned=not args.no_finetuned)

