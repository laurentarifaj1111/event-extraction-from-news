from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from scripts.utils import chunk_text, get_device
import torch
import os

DEVICE = get_device()

class PegasusSummarizer:
    """
    Abstractive summarization using PEGASUS with Apple MPS GPU support.
    """

    def __init__(self, model_name="google/pegasus-xsum", max_len=256, use_finetuned=False):
        # Check if fine-tuned model exists and should be used
        if use_finetuned and os.path.exists("models/pegasus-finetuned"):
            model_name = "models/pegasus-finetuned"
            print(f"✓ Loading fine-tuned Pegasus model from {model_name}")
        elif use_finetuned:
            print(f"⚠ Fine-tuned model not found at models/pegasus-finetuned, using pretrained {model_name}")
        
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

        # summary_ids = self.model.generate(
        #     **encoding,
        #     num_beams=4,
        #     max_length=self.max_len,
        #     early_stopping=True
        # )

        # Generate on GPU
        # Using greedy decoding (num_beams=1) for 3-4x speedup vs beam search
        with torch.inference_mode():  # Faster than no_grad()
            summary_ids = self.model.generate(
                **encoding,
                num_beams=1,  # Greedy decoding - much faster
                max_length=min(self.max_len, 128),  # Cap at 128 for faster generation
                early_stopping=True,
                do_sample=False
            )

        # Move output back to CPU for decoding
        summary_ids = summary_ids.to("cpu")

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
