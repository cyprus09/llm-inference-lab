from __future__ import annotations

import torch
import torch.nn.functional as F

# Matches entropy_tracker/runner.py's sampling params, to keep the lab's decode
# mode consistent across tools.


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the low-probability tail beyond nucleus top_p, renormalize.
    probs: [vocab_size], already softmaxed.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # keep the smallest prefix whose cumulative mass >= top_p; always keep at least 1
    cutoff = cumulative >= top_p
    cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False
    sorted_probs[cutoff] = 0.0
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx] = sorted_probs
    return filtered / filtered.sum()


def sample_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> tuple[int, float]:
    """Sample one token from a single step's logits. Returns (token_id, its probability
    under this same filtered distribution) since accept/reject needs p_draft(x) for that
    exact token under the same sampling procedure that produced it.
    logits: [vocab_size], unbatched, un-normalized.
    """
    probs = F.softmax(logits.float() / temperature, dim=-1)
    probs = top_p_filter(probs, top_p)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id, probs[token_id].item()
