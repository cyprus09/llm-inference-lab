from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from speculative_decoding.sampling import sample_token, top_p_filter

from utils.constants import GAMMA, TEMPERATURE, TOP_P

# Draft proposal length per round. Sampling-mode speculative decoding
# guarantees output == verifier's own sampling distribution only when
# both draft and verifier sample stochastically -- see sampling.py for the
# shared temperature/top_p that both sides use.


@dataclass
class RoundResult:
    accepted_tokens: List[int]  # tokens accepted this round, in order (draft-sampled)
    bonus_token: (
        int  # either the verifier's correction, or an extra verifier-sampled token
    )
    num_proposed: int  # gamma, for acceptance-rate bookkeeping
    num_accepted: int  # len(accepted_tokens)


@torch.no_grad()
def draft_step(
    draft_model,
    input_ids: torch.Tensor,
    gamma: int = GAMMA,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> tuple[torch.Tensor, List[float]]:
    """Autoregressively sample gamma tokens from the draft model using KV cache.
    One full forward pass to build the cache, then gamma incremental passes (one per new token),
    each processing only the last token with position IDs to maintain absolute position awareness.
    Returns (draft_ids [1, gamma], p_draft per sampled token).
    """
    device = input_ids.device
    draft_probs: List[float] = []
    draft_ids_list: List[int] = []

    # Initial full forward pass to build KV cache
    prefix_len = input_ids.shape[1]
    output = draft_model(input_ids, use_cache=True)
    cache = output.past_key_values

    # Sample gamma tokens autoregressively with cache reuse
    for i in range(gamma):
        # Current position in the sequence: where the next token will be predicted
        current_pos = prefix_len + i

        # Only pass the last token (the one we just sampled) with its position ID
        if i == 0:
            # First token: use the last token of input_ids
            model_input = input_ids[:, -1:]
            position_ids = torch.tensor([[current_pos - 1]], device=device, dtype=torch.long)
        else:
            # Subsequent tokens: use the token we just sampled
            model_input = torch.tensor(
                [[draft_ids_list[-1]]], device=device, dtype=torch.long
            )
            position_ids = torch.tensor([[current_pos - 1]], device=device, dtype=torch.long)

        # Incremental forward with explicit position_ids
        output = draft_model(
            model_input, past_key_values=cache, position_ids=position_ids, use_cache=True
        )
        cache = output.past_key_values

        # Sample from the logits of this new position
        logits = output.logits[0, -1]
        token_id, p = sample_token(logits, temperature, top_p)
        draft_ids_list.append(token_id)
        draft_probs.append(p)

    draft_ids = torch.tensor([draft_ids_list], device=device, dtype=torch.long)
    return draft_ids, draft_probs


@torch.no_grad()
def verify_and_accept(
    verifier_model,
    input_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_probs: List[float],
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> RoundResult:
    """Single batched forward pass over [prefix + draft tokens], then sampling-mode
    accept/reject: accept token i with probability min(1, p_verify(x_i)/p_draft(x_i));
    on first rejection, resample from the residual max(0, p_verify - p_draft) and stop.
    If all gamma tokens are accepted, sample one bonus token from the verifier's next
    position for free (the whole point of speculative decoding: gamma+1 tokens for the
    cost of one verifier forward pass when acceptance is perfect).
    """
    gamma = draft_ids.shape[1]
    full_ids = torch.cat([input_ids, draft_ids], dim=1)
    logits = verifier_model(full_ids).logits[0]  # [seq_len, vocab]
    prefix_len = input_ids.shape[1]

    accepted: List[int] = []
    for i in range(gamma):
        step_logits = logits[prefix_len - 1 + i]
        verify_probs = F.softmax(step_logits.float() / temperature, dim=-1)
        verify_probs = top_p_filter(verify_probs, top_p)

        token_id = int(draft_ids[0, i].item())
        p_verify = verify_probs[token_id].item()
        p_draft = draft_probs[i]

        accept_prob = min(1.0, p_verify / p_draft) if p_draft > 0 else 0.0
        if torch.rand(1).item() < accept_prob:
            accepted.append(token_id)
            continue

        # rejection: resample from the residual distribution and stop this round
        draft_dist = torch.zeros_like(verify_probs)
        draft_dist[token_id] = (
            p_draft  # only the sampled token's mass is known/relevant
        )
        residual = torch.clamp(verify_probs - draft_dist, min=0.0)
        if residual.sum() <= 0:
            correction = int(torch.argmax(verify_probs).item())
        else:
            residual = residual / residual.sum()
            correction = int(torch.multinomial(residual, num_samples=1).item())
        return RoundResult(
            accepted_tokens=accepted,
            bonus_token=correction,
            num_proposed=gamma,
            num_accepted=len(accepted),
        )

    # all gamma tokens accepted -- sample a free bonus token from the verifier
    bonus_logits = logits[prefix_len - 1 + gamma]
    bonus_probs = top_p_filter(
        F.softmax(bonus_logits.float() / temperature, dim=-1), top_p
    )
    bonus_token = int(torch.multinomial(bonus_probs, num_samples=1).item())
    return RoundResult(
        accepted_tokens=accepted,
        bonus_token=bonus_token,
        num_proposed=gamma,
        num_accepted=len(accepted),
    )
