import torch


def compile_model(model, device):
    """
    Compile model for faster inference on CUDA.
    Note: torch.compile() may not work well with MPS yet, so we skip it for MPS.
    """
    try:
        # Only compile for CUDA (MPS support for torch.compile is limited)
        if hasattr(torch, 'compile') and device.type == 'cuda':
            # Compile model for faster inference (PyTorch 2.0+)
            return torch.compile(model, mode='reduce-overhead')
        return model
    except Exception as e:
        # Silently fall back to uncompiled model
        return model


def get_device():
    """
    Get the best available device for PyTorch models.
    Prioritizes MPS (Apple Silicon GPU) > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_device_info():
    """
    Print information about the device being used.
    Useful for verifying GPU usage.
    """
    device = get_device()
    print(f"\n{'='*50}")
    print(f"Device Configuration:")
    print(f"  Using device: {device}")

    if device.type == "mps":
        print(f"  ✓ Apple Silicon GPU (MPS) is available")
        print(f"  ✓ Models will run on GPU")
    elif device.type == "cuda":
        print(f"  ✓ CUDA GPU is available")
        print(f"  ✓ Models will run on GPU")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠ No GPU available, using CPU")
        print(f"  ⚠ Performance will be slower")

    print(f"{'='*50}\n")


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