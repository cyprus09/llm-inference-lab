from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenAttentionRecord:
    position: int
    token_id: int
    mean_attention: float  # causal-normalized, averaged over layers/heads
    z_score: float
    is_sink: bool
    is_structural: bool = False  # e.g. BOS token, which is always a sink


def compute_prefill_attentions(
    model, input_ids: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """Single forward pass over the prompt. Requires the model to be loaded
    with attn_implementation='eager' — SDPA/flash kernels don't materialize
    the full attention matrix and will return None here."""
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None or attentions[0] is None:
        raise RuntimeError(
            "No attentions returned — load the model with "
            "attn_implementation='eager' for output_attentions to work."
        )
    return attentions


def aggregate_attention(attentions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Mean attention matrix across layers and heads.
    Returns (seq_len, seq_len); attn[i, j] = how much query i attends to key j."""
    # each tensor: (batch=1, num_heads, seq_len, seq_len)
    stacked = torch.stack([layer_attn[0].float() for layer_attn in attentions], dim=0)
    return stacked.mean(dim=(0, 1))


def mean_attention_received(attn_matrix: torch.Tensor) -> torch.Tensor:
    """Per-token attention received, normalized by causal visibility.

    Token j is only visible to queries i >= j, so a raw column sum favors early
    tokens purely because they had more chances to be attended to — not because
    they're genuinely absorbing more weight. Dividing by (seq_len - j), the
    number of queries that could actually see token j, gives the average
    attention per visible query — comparable across positions."""
    seq_len = attn_matrix.shape[0]
    col_sums = attn_matrix.sum(dim=0)
    valid_queries = torch.arange(
        seq_len, 0, -1, dtype=torch.float32, device=attn_matrix.device
    )
    return col_sums / valid_queries


def detect_sinks(
    mean_attn: torch.Tensor,
    threshold: float = 6.0,  # see rationale below
    exclude_positions: Tuple[int, ...] = (0,),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust sink detection via median/MAD in log-space.

    Attention mass is strictly positive and right-skewed (most tokens get
    near-zero, a few get more) — it is NOT approximately normal. A raw
    mean/std z-score under-penalizes the skew and flags ordinary
    high-attention content words (topically salient tokens, section
    markers) as if they were sinks. Two fixes:
      1. log-transform before computing spread, since attention ratios
         behave closer to log-normal than normal.
      2. use median/MAD instead of mean/std, since MAD is not itself
         dragged upward by the same skew it's trying to measure.
    """
    mask = torch.ones_like(mean_attn, dtype=torch.bool)
    for p in exclude_positions:
        if 0 <= p < len(mask):
            mask[p] = False

    population = mean_attn[mask]
    log_population = torch.log(population.clamp_min(1e-8))
    log_all = torch.log(mean_attn.clamp_min(1e-8))

    median = log_population.median()
    mad = (log_population - median).abs().median()
    scaled_mad = (
        mad * 1.4826
    )  # makes MAD ≈ std under normality, for comparable thresholds

    z = (
        torch.zeros_like(mean_attn)
        if scaled_mad.item() < 1e-8
        else (log_all - median) / scaled_mad
    )

    is_sink = z > threshold
    return is_sink, z


def find_structural_positions(input_ids: torch.Tensor, tokenizer) -> set[int]:
    """Positions that are content-independent: BOS, every special token
    (<|im_start|>, <|im_end|>), the role-name token right after
    <|im_start|> (e.g. 'assistant'), and the newline immediately
    following either the role name or <|im_end|> — both are fixed by
    the ChatML template, not by what was said."""
    ids = input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    structural = {0}
    for i, tid in enumerate(ids):
        if tid in special_ids:
            structural.add(i)
        if i > 0 and ids[i - 1] in (im_start_id, im_end_id):
            structural.add(i)  # role name (after im_start) or newline (after im_end)
            if ids[i - 1] == im_start_id and i + 1 < len(ids):
                structural.add(i + 1)  # newline after the role name specifically
    return structural


def analyze_prefill(model, tokenizer, input_ids, threshold=2.0, exclude_positions=None):
    attentions = compute_prefill_attentions(model, input_ids)
    assert len(attentions) == model.config.num_hidden_layers
    assert attentions[0].shape[1] == model.config.num_attention_heads

    if exclude_positions is None:
        exclude_positions = find_structural_positions(input_ids, tokenizer)

    attn_matrix = aggregate_attention(attentions)
    mean_attn = mean_attention_received(attn_matrix)
    is_sink, z_scores = detect_sinks(
        mean_attn, threshold=threshold, exclude_positions=exclude_positions
    )

    token_ids = input_ids[0].tolist()
    return [
        TokenAttentionRecord(
            position=i,
            token_id=token_ids[i],
            mean_attention=mean_attn[i].item(),
            z_score=z_scores[i].item(),
            is_sink=bool(is_sink[i].item()),
            is_structural=i in exclude_positions,
        )
        for i in range(len(token_ids))
    ]


def context_health_score(records: List[TokenAttentionRecord]) -> float:
    """Health reflects anomalous sinks only — structural sinks (BOS) are
    expected in every prompt and excluded. Severity is averaged over the
    sinks themselves, not the full sequence length, so it doesn't get
    diluted on longer prompts."""
    candidates = [r for r in records if not r.is_structural]
    if not candidates:
        return 10.0
    sinks = [r for r in candidates if r.is_sink]
    if not sinks:
        return 10.0

    sink_density = len(sinks) / len(candidates)
    mean_severity = sum(r.z_score for r in sinks) / len(sinks)

    score = 10.0 - (sink_density * 10.0) - min(mean_severity, 5.0)
    return max(0.0, min(10.0, score))
