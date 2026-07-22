import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DRAFT_MODEL, MODEL as VERIFIER_MODEL

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading tokenizer for {DRAFT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL)

    print(f"Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, dtype=torch.float16, device_map=None
    ).to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    model.eval()

    prompt = "Explain photosynthesis in simple terms"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    elapsed = time.time() - t0

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\nOutput: {text}")
    print(
        f"\nGenerated {len(generated_tokens)} tokens in {elapsed:.2f}s"
        f"({len(generated_tokens)/elapsed:.1f} tok/s)"
    )


if __name__ == "__main__":
    main()
