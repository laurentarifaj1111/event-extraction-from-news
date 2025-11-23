import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from scripts.utils import chunk_text


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class T5Summarizer:
    """
    Abstractive summarization using FLAN-T5 with Apple GPU (MPS) support.
    """

    def __init__(self, model_name="google/flan-t5-base", max_len=256):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.max_len = max_len

    def summarize(self, text: str) -> str:
        if len(text.split()) > 500:
            chunks = chunk_text(text)
            summaries = [self._summarize_chunk(c) for c in chunks]
            return " ".join(summaries)
        else:
            return self._summarize_chunk(text)

    def _summarize_chunk(self, text):
        prompt = "summarize: " + text

        # Create encoding on CPU
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # MOVE INPUTS TO MPS
        encoding = {key: value.to(DEVICE) for key, value in encoding.items()}

        # Generate summary on GPU
        output = self.model.generate(
            **encoding,
            max_length=self.max_len,
            num_beams=4
        )

        # Move output back to CPU for tokenizer
        output = output.to("cpu")

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
