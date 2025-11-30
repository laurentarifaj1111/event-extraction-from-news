import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from scripts.utils import chunk_text, get_device
import os

DEVICE = get_device()

class T5Summarizer:
    """
    Abstractive summarization using FLAN-T5 with Apple GPU (MPS) support.
    """

    def __init__(self, model_name="google/flan-t5-base", max_len=256, use_finetuned=False):
        # Check if fine-tuned model exists and should be used
        if use_finetuned and os.path.exists("models/t5-finetuned"):
            model_name = "models/t5-finetuned"
            print(f"✓ Loading fine-tuned T5 model from {model_name}")
        elif use_finetuned:
            print(f"⚠ Fine-tuned model not found at models/t5-finetuned, using pretrained {model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.model.eval()  # Set to evaluation mode for inference
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


        # output = self.model.generate(
        #     **encoding,
        #     max_length=self.max_len,
        #     num_beams=4
        # )

        # Generate summary on GPU
        # Using greedy decoding (num_beams=1) for 3-4x speedup vs beam search
        with torch.inference_mode():  # Faster than no_grad()
            output = self.model.generate(
                **encoding,
                max_length=min(self.max_len, 128),  # Cap at 128 for faster generation
                num_beams=1,  # Greedy decoding - much faster
                do_sample=False
            )

        # Move output back to CPU for tokenizer
        output = output.to("cpu")

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
