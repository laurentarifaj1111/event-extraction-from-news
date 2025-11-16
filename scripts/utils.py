def chunk_text(text: str, max_tokens=512):
    """
    Split long text into chunks the model can handle.
    """
    tokens = text.split()
    chunks = []
    current = []

    for tok in tokens:
        current.append(tok)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks