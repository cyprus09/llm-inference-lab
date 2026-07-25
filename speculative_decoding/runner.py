import sys
import time

from rich.console import Console
from rich.rule import Rule

from utils.model_utils import load_and_warm, assert_tokenizer_compatible
from utils.constants import DRAFT_MODEL, MODEL as VERIFIER_MODEL, MAX_NEW_TOKENS

from speculative_decoding.generate import generate
from speculative_decoding.renderer import replay, print_summary


def setup_models():
    draft_model, draft_tokenizer, draft_device = load_and_warm(DRAFT_MODEL)
    verifier_model, verifier_tokenizer, verifier_device = load_and_warm(VERIFIER_MODEL)

    assert_tokenizer_compatible(draft_tokenizer, verifier_tokenizer)
    assert (
        draft_device == verifier_device
    ), "Draft and verifier must be on the same device"

    return draft_model, verifier_model, draft_tokenizer, draft_device


def main(prompt: str):
    console = Console()

    console.print(Rule("Loading"))
    draft_model, verifier_model, tokenizer, device = setup_models()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)
    prompt_len = input_ids["input_ids"].shape[1]

    console.print(Rule("Generating (sampling-mode speculative decoding)"))
    t0 = time.time()
    output_ids, stats = generate(
        draft_model,
        verifier_model,
        input_ids["input_ids"],
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0
    n_new = output_ids.shape[1] - prompt_len
    console.print(f"{n_new} tokens in {elapsed:.1f}s\n")

    replay(stats, output_ids[0, prompt_len:].tolist(), tokenizer, console)

    console.print(Rule("Summary"))
    print_summary(stats, elapsed, console)


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "Summarize the following in detail, covering every section: Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process occurs primarily in the chloroplasts of plant cells, specifically within structures called thylakoids, which contain the green pigment chlorophyll. Photosynthesis consists of two main stages: the light-dependent reactions and the light-independent reactions, also known as the Calvin cycle. During the light-dependent reactions, which take place in the thylakoid membranes, light energy is absorbed by chlorophyll and other pigments. This energy is used to split water molecules into oxygen, protons, and electrons in a process called photolysis. The oxygen is released as a byproduct through the stomata of the leaves, while the electrons move through an electron transport chain, generating ATP and NADPH, which are energy-carrying molecules. The light-dependent reactions also involve two photosystems, Photosystem II and Photosystem I, which work together to capture and transfer energy efficiently. In the Calvin cycle, which occurs in the stroma of the chloroplast, the ATP and NADPH produced in the light-dependent reactions are used to convert carbon dioxide into glucose through a series of enzyme-catalyzed reactions. This cycle involves carbon fixation, reduction, and regeneration of the starting molecule ribulose bisphosphate. The enzyme RuBisCO plays a critical role in fixing carbon dioxide onto this five-carbon sugar. The glucose produced can then be used by the plant for energy or converted into other organic compounds like starch and cellulose for structural and storage purposes. Environmental factors such as light intensity, carbon dioxide concentration, and temperature significantly affect the rate of photosynthesis. Understanding these factors is crucial for agriculture and for predicting how plants respond to climate change."
    )
