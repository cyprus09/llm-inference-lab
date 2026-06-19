import sys
import time
import torch
from transformers import LogitsProcessorList
from rich.console import Console
from rich.rule import Rule

from llm_lab.model_utils import load_and_warm
from llm_lab.entropy_tracker import (
    EntropyLogitsProcessor,
    attach_chosen_tokens,
    summarize,
)
from summary_stats.render import replay, print_summary

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 250


def main(prompt: str):
    console = Console()

    console.print(Rule("Loading"))
    model, tokenizer, device = load_and_warm(MODEL)

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)
    prompt_len = input_ids["input_ids"].shape[1]

    processor = EntropyLogitsProcessor()

    console.print(Rule("Generating"))
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.eos_token_id,
        )
    n_new = output.shape[1] - prompt_len
    console.print(f"{n_new} tokens in {time.time() - t0:.1f}s\n")

    attach_chosen_tokens(processor.records, output[0], prompt_len)

    console.print(
        Rule(
            "[green3]■[/green3] confident  [yellow3]■[/yellow3] uncertain  [red3]■[/red3] very uncertain"
        )
    )
    replay(processor.records, tokenizer, console)

    console.print(Rule("Summary"))
    print_summary(summarize(processor.records, tokenizer), console)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Continue this story in exactly 3 sentences: She opened the envelope and laughed, then cried, then laughed again.")
