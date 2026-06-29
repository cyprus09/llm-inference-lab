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
