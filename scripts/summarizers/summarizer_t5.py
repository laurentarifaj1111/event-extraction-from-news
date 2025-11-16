from transformers import T5Tokenizer, T5ForConditionalGeneration

from scripts.utils import chunk_text


class T5Summarizer:
    """
    Abstractive summarization using FLAN-T5.
    """

    def __init__(self, model_name="google/flan-t5-base", max_len=256):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
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

        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output = self.model.generate(
            **encoding,
            max_length=self.max_len,
            num_beams=4
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
