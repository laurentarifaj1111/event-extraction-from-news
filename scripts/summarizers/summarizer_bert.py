import torch
from summarizer import Summarizer
from transformers import AutoModel, AutoTokenizer
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class BertExtractiveSummarizer:
    """
    Extractive summarizer running BERT on Apple Silicon GPU (MPS).
    """

    def __init__(self, model_name="bert-base-uncased", min_length=40, max_length=180):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load BERT model on MPS
        self.bert_model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(DEVICE)

        # Inject custom GPU-enabled model into Summarizer
        self.model = Summarizer(
            custom_model=self.bert_model,
            custom_tokenizer=self.tokenizer
        )

        self.min_length = min_length
        self.max_length = max_length

    def summarize(self, text: str) -> str:
        if not isinstance(text, str) or len(text) < 50:
            return text

        try:
            summary = self.model(
                text,
                min_length=self.min_length,
                max_length=self.max_length
            )
            return summary.strip()
        except Exception:
            sentences = sent_tokenize(text)
            return " ".join(sentences[:3])
