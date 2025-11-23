from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from scripts.utils import chunk_text, get_device
import torch

DEVICE = get_device()

class PegasusSummarizer:
    """
    Abstractive summarization using PEGASUS with Apple MPS GPU support.
    """

    def __init__(self, model_name="google/pegasus-xsum", max_len=256):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.model.eval()  # Set to evaluation mode for inference
        self.max_len = max_len

    def summarize(self, text: str) -> str:
        if len(text.split()) > 500:
            chunks = chunk_text(text)
            summaries = [self._summarize_chunk(c) for c in chunks]
            return " ".join(summaries)
        else:
            return self._summarize_chunk(text)

    def _summarize_chunk(self, text: str) -> str:
        # Encode on CPU
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # MOVE input tensors to MPS
        encoding = {key: value.to(DEVICE) for key, value in encoding.items()}

        # Generate on GPU
        summary_ids = self.model.generate(
            **encoding,
            num_beams=4,
            max_length=self.max_len,
            early_stopping=True
        )

        # Move output back to CPU for decoding
        summary_ids = summary_ids.to("cpu")

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
