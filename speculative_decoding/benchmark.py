import sys
import time

import torch
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from utils.constants import MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from speculative_decoding.generate import generate
from speculative_decoding.runner import setup_models
from speculative_decoding.flops import draft_verifier_cost_ratio, theoretical_speedup

# Gamma values to sweep for the speculative run. Edit this list directly to try
# different proposal lengths.
GAMMA_SWEEP = [4, 6]


@torch.no_grad()
def run_verifier_only(verifier_model, tokenizer, input_ids, device):
    t0 = time.time()
    output = verifier_model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0
    n_new = output.shape[1] - input_ids.shape[1]
    return n_new, elapsed


def run_speculative(draft_model, verifier_model, tokenizer, input_ids, gamma):
    t0 = time.time()
    output_ids, stats = generate(
        draft_model,
        verifier_model,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        gamma=gamma,
    )
    elapsed = time.time() - t0
    n_new = output_ids.shape[1] - input_ids.shape[1]
    return n_new, elapsed, stats


def main(prompt: str):
    console = Console()

    console.print(Rule("Loading"))
    draft_model, verifier_model, tokenizer, device = setup_models()

    cost_ratio, draft_flops, verifier_flops = draft_verifier_cost_ratio(
        draft_model, verifier_model
    )
    console.print(
        f"FLOPs/token -- draft: {draft_flops:,}, verifier: {verifier_flops:,}, "
        f"cost ratio c={cost_ratio:.4f}\n"
    )

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)["input_ids"]

    console.print(Rule("Verifier-only (sampling) baseline"))
    base_n, base_elapsed = run_verifier_only(
        verifier_model, tokenizer, input_ids, device
    )
    console.print(
        f"{base_n} tokens in {base_elapsed:.1f}s ({base_n / base_elapsed:.1f} tok/s)\n"
    )

    table = Table(title="Verifier-only vs. speculative decoding (gamma sweep)")
    table.add_column("Mode")
    table.add_column("Gamma")
    table.add_column("Tokens")
    table.add_column("Time (s)")
    table.add_column("Tok/s")
    table.add_column("Acceptance")
    table.add_column("Theoretical Speedup")
    table.add_column("Measured Speedup")
    table.add_row(
        "Verifier-only",
        "-",
        str(base_n),
        f"{base_elapsed:.1f}",
        f"{base_n / base_elapsed:.1f}",
        "-",
        "-",
        "1.00x",
    )

    for gamma in GAMMA_SWEEP:
        console.print(Rule(f"Speculative decoding (sampling-mode, gamma={gamma})"))
        spec_n, spec_elapsed, stats = run_speculative(
            draft_model, verifier_model, tokenizer, input_ids, gamma
        )
        console.print(
            f"{spec_n} tokens in {spec_elapsed:.1f}s ({spec_n / spec_elapsed:.1f} tok/s)\n"
        )
        console.print(f"Acceptance rate: {stats.acceptance_rate:.1%}\n")
        speedup = (base_elapsed / base_n) / (spec_elapsed / spec_n)
        theory = theoretical_speedup(stats.acceptance_rate, gamma, cost_ratio)
        table.add_row(
            "Speculative",
            str(gamma),
            str(spec_n),
            f"{spec_elapsed:.1f}",
            f"{spec_n / spec_elapsed:.1f}",
            f"{stats.acceptance_rate:.1%}",
            f"{theory:.2f}x",
            f"{speedup:.2f}x",
        )

    console.print(Rule("Summary"))
    console.print(table)


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "Summarize the following in detail, covering every section: Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process occurs primarily in the chloroplasts of plant cells, specifically within structures called thylakoids, which contain the green pigment chlorophyll. Photosynthesis consists of two main stages: the light-dependent reactions and the light-independent reactions, also known as the Calvin cycle. During the light-dependent reactions, which take place in the thylakoid membranes, light energy is absorbed by chlorophyll and other pigments. This energy is used to split water molecules into oxygen, protons, and electrons in a process called photolysis. The oxygen is released as a byproduct through the stomata of the leaves, while the electrons move through an electron transport chain, generating ATP and NADPH, which are energy-carrying molecules. The light-dependent reactions also involve two photosystems, Photosystem II and Photosystem I, which work together to capture and transfer energy efficiently. In the Calvin cycle, which occurs in the stroma of the chloroplast, the ATP and NADPH produced in the light-dependent reactions are used to convert carbon dioxide into glucose through a series of enzyme-catalyzed reactions. This cycle involves carbon fixation, reduction, and regeneration of the starting molecule ribulose bisphosphate. The enzyme RuBisCO plays a critical role in fixing carbon dioxide onto this five-carbon sugar. The glucose produced can then be used by the plant for energy or converted into other organic compounds like starch and cellulose for structural and storage purposes. Environmental factors such as light intensity, carbon dioxide concentration, and temperature significantly affect the rate of photosynthesis. Understanding these factors is crucial for agriculture and for predicting how plants respond to climate change."
    )
