from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from scripts.utils import chunk_text


class PegasusSummarizer:
    """
    Abstractive summarization using PEGASUS.
    """

    def __init__(self, model_name="google/pegasus-xsum", max_len=256):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.max_len = max_len

    def summarize(self, text: str) -> str:
        """
        Summarize text, handling very long documents.
        """
        if len(text.split()) > 500:
            chunks = chunk_text(text)
            summaries = [self._summarize_chunk(c) for c in chunks]
            return " ".join(summaries)
        else:
            return self._summarize_chunk(text)

    def _summarize_chunk(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=self.max_len,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
