import sys

from rich.console import Console
from rich.rule import Rule

from utils.model_utils import load_and_warm
from .attention_sink import analyze_prefill, context_health_score
from .renderer import highlight_prompt, print_health_summary

from utils.constants import MODEL, ATTN_IMPL


def main(prompt: str):
    console = Console()

    console.print(Rule("Loading"))
    model, tokenizer, device = load_and_warm(MODEL, attn_implementation=ATTN_IMPL)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)

    console.print(Rule("Analyzing prefill"))
    records = analyze_prefill(model, tokenizer, inputs["input_ids"])
    for r in records:
        if r.position in (2, 278):
            print(r.position, r.token_id, r.is_structural, r.z_score)
    health = context_health_score(records)

    highlight_prompt(records, tokenizer, console)

    console.print(Rule("Summary"))
    print_health_summary(health, records, tokenizer, console)


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else 'Your observation touches on some important aspects of large language model (LLM) training and potential biases. Let\'s break it down: ### Training Data Source 1. **Web Crawled Data**: Traditional LLMs were indeed trained on a wide variety of text scraped from the web, which included a broad range of sources, including books, articles, news, social media posts, etc. 2. **AI-Generated Data**: Modern models like Qwen are often trained on their own outputs, which can be seen as a form of "self-supervised learning." This means the model is essentially using the text it generates during inference as part of its training process. ### Paradoxical Claim: Dead Internet Theory The concept you\'re referring to is somewhat related to the "dead internet theory" in a way. This theory posits that because the internet is dynamic and ever-changing, it\'s not truly representative of human knowledge or experience. However, the data used for training LLMs is usually a snapshot at a particular time, so it may not fully capture the nuances and diversity of contemporary human expression. ### Self-Supervised Learning and Bias. When an LLM trains on its own outputs, several factors come into play regarding potential biases:'
    )
