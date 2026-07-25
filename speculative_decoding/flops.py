"""FLOPs-based theoretical speedup for speculative decoding (Leviathan et al., 2023).

Hardware-agnostic by construction: the draft/verifier cost ratio `c` is derived
from model architecture dims (FLOPs-per-token), not measured MPS milliseconds.
This gives an upper-bound speedup under the paper's own cost model, independent
of any particular chip's kernel-dispatch overhead.
"""

from __future__ import annotations


def flops_per_token(config) -> int:
    """Forward-pass FLOPs for one token, derived from architecture dims.
    Uses the standard 2*N approximation (Kaplan et al.) applied separately to
    attention (GQA-aware, via num_key_value_heads) and MLP blocks, plus the
    final unembedding projection.
    """
    h = config.hidden_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = h // n_heads
    inter = config.intermediate_size
    vocab = config.vocab_size

    # Attention: Q proj (h->h), K/V proj (h -> n_kv_heads*head_dim each), O proj (h->h).
    attn_params = h * h + 2 * (h * n_kv_heads * head_dim) + h * h
    # MLP: SwiGLU has gate+up (h->inter each) and down (inter->h).
    mlp_params = 2 * (h * inter) + (inter * h)
    layer_params = attn_params + mlp_params
    total_params = n_layers * layer_params + h * vocab  # + final unembedding

    return 2 * total_params  # 2 FLOPs per param (multiply-add) per token


def draft_verifier_cost_ratio(draft_model, verifier_model) -> tuple[float, int, int]:
    """c = FLOPs-per-token(draft) / FLOPs-per-token(verifier), used in the
    Leviathan et al. theoretical speedup formula. Computed from real model
    configs so it reflects actual architecture, not a guessed param ratio.
    """
    draft_flops = flops_per_token(draft_model.config)
    verifier_flops = flops_per_token(verifier_model.config)
    return draft_flops / verifier_flops, draft_flops, verifier_flops


def theoretical_speedup(acceptance_rate: float, gamma: int, c: float) -> float:
    """Leviathan et al. (2023) expected speedup for speculative decoding:
    E[speedup] = (1 - alpha^(gamma+1)) / ((1 - alpha) * (gamma*c + 1))
    where alpha is acceptance rate and c is the draft/verifier cost ratio.
    """
    alpha = acceptance_rate
    if alpha >= 1.0:
        # Limit as alpha -> 1: every round accepts all gamma tokens for free.
        return (gamma + 1) / (gamma * c + 1)
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


def breakeven_acceptance_rate(gamma: int, c: float, tol: float = 1e-4) -> float | None:
    """Acceptance rate alpha at which theoretical_speedup(alpha, gamma, c) == 1.0,
    found by bisection since the formula has no closed-form inverse in alpha.
    theoretical_speedup is monotonically increasing in alpha, so bisection is safe.
    Returns None if even alpha=1.0 can't reach speedup 1.0 (c too large for this gamma).
    """
    if theoretical_speedup(1.0, gamma, c) < 1.0:
        return None

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if theoretical_speedup(mid, gamma, c) < 1.0:
            lo = mid
        else:
            hi = mid
    return hi
