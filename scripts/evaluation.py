# src/evaluation.py

import time
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score


class ModelEvaluator:
    """
    Computes:
    - ROUGE-1, ROUGE-2, ROUGE-L
    - BERTScore (Precision, Recall, F1)
    - Inference time

    Designed for evaluating multiple summarization models.
    """

    def __init__(self):
        # ROUGE scorer
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    # ----------------------------------------------------
    # ROUGE
    # ----------------------------------------------------
    def compute_rouge(self, reference: str, generated: str):
        scores = self.rouge.score(reference, generated)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }

    # ----------------------------------------------------
    # BERTScore
    # ----------------------------------------------------
    def compute_bertscore(self, reference: str, generated: str):
        """
        Returns: precision, recall, F1
        """
        P, R, F1 = bert_score(
            cands=[generated],
            refs=[reference],
            lang="en",
            rescale_with_baseline=True
        )

        return {
            "bertscore_precision": float(P[0]),
            "bertscore_recall": float(R[0]),
            "bertscore_f1": float(F1[0])
        }

    # ----------------------------------------------------
    # Inference Time
    # ----------------------------------------------------
    def measure_inference_time(self, model_fn, text: str):
        """
        model_fn must be a function that accepts text and returns a summary.
        """
        start = time.time()
        summary = model_fn(text)
        end = time.time()

        return summary, (end - start) * 1000.0  # milliseconds

    # ----------------------------------------------------
    # Full Evaluation (One Model on One Article)
    # ----------------------------------------------------
    def evaluate_single(self, reference: str, generated: str):
        rouge_scores = self.compute_rouge(reference, generated)
        bert_scores = self.compute_bertscore(reference, generated)
        return {**rouge_scores, **bert_scores}

    # ----------------------------------------------------
    # Evaluate Results for the Whole Test Set
    # ----------------------------------------------------
    def evaluate_dataset(
        self,
        df,
        model_name: str,
        generated_column: str,
        reference_column: str = "clean_text",
        save_path: str = None
    ):
        """
        df: dataframe with reference text + generated summaries
        model_name: used for filenames
        generated_column: column with model summaries
        reference_column: ground truth / original text
        """

        rows = []

        for idx, row in df.iterrows():
            reference = str(row[reference_column])
            summary = str(row[generated_column])

            # ROUGE + BERTScore
            sc = self.evaluate_single(reference, summary)

            row_out = {
                "id": idx,
                "model": model_name,
                "rouge1": sc["rouge1"],
                "rouge2": sc["rouge2"],
                "rougeL": sc["rougeL"],
                "bertscore_precision": sc["bertscore_precision"],
                "bertscore_recall": sc["bertscore_recall"],
                "bertscore_f1": sc["bertscore_f1"],
            }
            rows.append(row_out)

        results = pd.DataFrame(rows)

        if save_path:
            results.to_csv(save_path, index=False)
            print(f"Saved evaluation results → {save_path}")

        return results
