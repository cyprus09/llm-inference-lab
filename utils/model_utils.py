import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_name: str,
    dtype=torch.float16,
    attn_implementation: Optional[str] = None,
):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    model.eval()
    return model, tokenizer, device


def warmup(model, tokenizer, device, n_tokens: int = 8):
    inputs = tokenizer("Warmup.", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model.generate(inputs, max_new_tokens=n_tokens, do_sample=False)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_and_warm(
    model_name: str,
    dtype=torch.float16,
    attn_implementation: Optional[str] = None,
):
    model, tokenizer, device = load_model(
        model_name, dtype=dtype, attn_implementation=attn_implementation
    )
    warmup(model, tokenizer, device)
    return model, tokenizer, device


def assert_tokenizer_compatible(tok_a, tok_b) -> None:
    """Verify two tokenizers share a vocabulary — required for speculative decoding -- Phase 3
    since acceptance/rejection compares token IDs directly across draft and verifier."""
    assert (
        tok_a.vocab_size == tok_b.vocab_size
    ), f"Vocab size mismatch: {tok_a.vocab_size} vs {tok_b.vocab_size}"
    assert (
        tok_a.get_vocab() == tok_b.get_vocab()
    ), "Vocab mappings differ — draft and verifier tokenizers are not compatible"
    # spot-check special tokens explicitly, since these matter most for ChatML formatting
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        a_val, b_val = getattr(tok_a, attr), getattr(tok_b, attr)
        assert a_val == b_val, f"{attr} mismatch: {a_val} vs {b_val}"
