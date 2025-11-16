from summarizer import Summarizer
import nltk

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

class BertExtractiveSummarizer:
    """
    Wrapper around bert-extractive-summarizer.
    """

    def __init__(self, min_length=40, max_length=180):
        self.model = Summarizer()
        self.min_length = min_length
        self.max_length = max_length

    def summarize(self, text: str) -> str:
        """
        Run extractive summarization.
        """
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
            # Fallback: first 2–3 sentences
            sentences = sent_tokenize(text)
            return " ".join(sentences[:3])
