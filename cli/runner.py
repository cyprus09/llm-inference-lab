import sys
import time

import torch
from rich.console import Console
from rich.rule import Rule

from utils.model_utils import load_and_warm, assert_tokenizer_compatible
from utils.constants import DRAFT_MODEL, MODEL, MAX_NEW_TOKENS, ATTN_IMPL

from attention_sink_detector.attention_sink import analyze_prefill, context_health_score
from attention_sink_detector.renderer import highlight_prompt, print_health_summary

from speculative_decoding.generate import generate
from speculative_decoding.renderer import replay, print_summary as print_spec_summary

from cli.renderer import print_draft_entropy, print_combined_summary


def run_attention_sink_stage(prompt: str, console: Console) -> float:
    """Prefill-only pass in eager attention mode (required to materialize the
    full attention matrix -- see CLAUDE.md). Model is dropped before stage 2
    so the default-attention verifier never shares memory with it.
    """
    console.print(Rule("Stage 1: attention-sink analysis (eager)"))
    model, tokenizer, device = load_and_warm(MODEL, attn_implementation=ATTN_IMPL)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)

    records = analyze_prefill(model, tokenizer, inputs["input_ids"])
    health = context_health_score(records)
    highlight_prompt(records, tokenizer, console)
    print_health_summary(health, records, tokenizer, console)

    del model, tokenizer, inputs
    if device.type == "mps":
        torch.mps.empty_cache()

    return health


def run_speculative_decoding_stage(prompt: str, console: Console):
    """Default-attention pass: draft + verifier speculative decoding is the
    single generation call for the whole CLI. Draft-side entropy comes along
    for free via RoundResult.entropy_trace -- no separate entropy_tracker pass.
    """
    console.print(Rule("Stage 2: speculative decoding (default attention)"))
    draft_model, draft_tokenizer, draft_device = load_and_warm(DRAFT_MODEL)
    verifier_model, verifier_tokenizer, verifier_device = load_and_warm(MODEL)

    assert_tokenizer_compatible(draft_tokenizer, verifier_tokenizer)
    assert draft_device == verifier_device, "Draft and verifier must be on the same device"

    input_ids = verifier_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(verifier_device)
    prompt_len = input_ids["input_ids"].shape[1]

    t0 = time.time()
    output_ids, stats = generate(
        draft_model,
        verifier_model,
        input_ids["input_ids"],
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=verifier_tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0
    n_new = output_ids.shape[1] - prompt_len
    console.print(f"{n_new} tokens in {elapsed:.1f}s\n")

    replay(stats, output_ids[0, prompt_len:].tolist(), verifier_tokenizer, console)
    print_spec_summary(stats, elapsed, console)
    print_draft_entropy(stats, console)

    return stats


def main(prompt: str):
    console = Console()

    health = run_attention_sink_stage(prompt, console)
    stats = run_speculative_decoding_stage(prompt, console)

    console.print(Rule("Summary"))
    print_combined_summary(health, stats, console)


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "Summarize the following in detail, covering every section: Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process occurs primarily in the chloroplasts of plant cells, specifically within structures called thylakoids, which contain the green pigment chlorophyll. Photosynthesis consists of two main stages: the light-dependent reactions and the light-independent reactions, also known as the Calvin cycle. During the light-dependent reactions, which take place in the thylakoid membranes, light energy is absorbed by chlorophyll and other pigments. This energy is used to split water molecules into oxygen, protons, and electrons in a process called photolysis. The oxygen is released as a byproduct through the stomata of the leaves, while the electrons move through an electron transport chain, generating ATP and NADPH, which are energy-carrying molecules. The light-dependent reactions also involve two photosystems, Photosystem II and Photosystem I, which work together to capture and transfer energy efficiently. In the Calvin cycle, which occurs in the stroma of the chloroplast, the ATP and NADPH produced in the light-dependent reactions are used to convert carbon dioxide into glucose through a series of enzyme-catalyzed reactions. This cycle involves carbon fixation, reduction, and regeneration of the starting molecule ribulose bisphosphate. The enzyme RuBisCO plays a critical role in fixing carbon dioxide onto this five-carbon sugar. The glucose produced can then be used by the plant for energy or converted into other organic compounds like starch and cellulose for structural and storage purposes. Environmental factors such as light intensity, carbon dioxide concentration, and temperature significantly affect the rate of photosynthesis. Understanding these factors is crucial for agriculture and for predicting how plants respond to climate change."
    )
