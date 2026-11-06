"""Utilities for belief-state SR grounding."""

from __future__ import annotations

import math
from typing import Callable, Dict


SRBelief = Dict[str, float]


def normalize_belief(
    scores: Dict[str, float],
    tau: float = 1.0,
    eps: float = 1e-12,
) -> SRBelief:
    """Convert unnormalized SR scores into a probability distribution."""
    if not scores:
        return {}

    tau = max(float(tau), eps)
    max_score = max(float(score) for score in scores.values())
    exp_scores = {
        sr: math.exp((float(score) - max_score) / tau)
        for sr, score in scores.items()
    }
    total = sum(exp_scores.values())
    if total <= eps:
        return {}
    return {sr: val / total for sr, val in exp_scores.items()}


def prune_belief(
    belief: SRBelief,
    top_m: int = 5,
    min_prob: float = 1e-6,
) -> SRBelief:
    """Keep the top-m SRs above min_prob and renormalize."""
    if not belief:
        return {}

    items = sorted(
        ((sr, float(prob)) for sr, prob in belief.items() if float(prob) >= min_prob),
        key=lambda item: (-item[1], item[0]),
    )[: max(1, int(top_m))]
    total = sum(prob for _sr, prob in items)
    if total <= 0:
        return {}
    return {sr: prob / total for sr, prob in items}


def get_sr_belief_for_semiology(
    semiology: str,
    get_prior_scores_fn: Callable[[str], Dict[str, float]],
    top_m: int = 5,
    tau: float = 1.0,
    min_prob: float = 1e-6,
) -> SRBelief:
    """Build a pruned SR belief distribution from KB/KG prior scores."""
    scores = get_prior_scores_fn(semiology)
    if not scores:
        return {}

    top_scores = dict(
        sorted(
            ((sr, float(score)) for sr, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )[: max(1, int(top_m))]
    )
    return prune_belief(
        normalize_belief(top_scores, tau=tau),
        top_m=top_m,
        min_prob=min_prob,
    )


def sr_transition_score(
    prev_sr: str,
    next_sr: str,
    get_hop_order_fn: Callable[[str, str], int],
    same_score: float = 0.2,
    unreachable_score: float = 0.01,
) -> float:
    """Score SR-to-SR connectivity using the existing hop-order semantics."""
    if prev_sr == next_sr:
        return float(same_score)

    hop_order = int(get_hop_order_fn(prev_sr, next_sr))
    if hop_order == 1:
        return 1.0
    if hop_order == 2:
        return 0.65
    if hop_order == 3:
        return 0.35
    return float(unreachable_score)


def expected_hop_score(
    current_belief: SRBelief,
    next_belief: SRBelief,
    get_hop_order_fn: Callable[[str, str], int],
    same_score: float = 0.2,
    unreachable_score: float = 0.01,
) -> float:
    """Expected SR connectivity score under current and next SR beliefs."""
    if not current_belief or not next_belief:
        return 0.0

    score = 0.0
    for prev_sr, p_prev in current_belief.items():
        for next_sr, p_next in next_belief.items():
            score += (
                float(p_prev)
                * float(p_next)
                * sr_transition_score(
                    prev_sr,
                    next_sr,
                    get_hop_order_fn,
                    same_score=same_score,
                    unreachable_score=unreachable_score,
                )
            )
    return float(score)


def expected_transition_score(
    current_belief: SRBelief,
    next_belief: SRBelief,
    get_transition_score_fn: Callable[[str, str], float],
) -> float:
    """Expected SR connectivity score using a cached transition scorer."""
    if not current_belief or not next_belief:
        return 0.0

    score = 0.0
    for prev_sr, p_prev in current_belief.items():
        for next_sr, p_next in next_belief.items():
            score += (
                float(p_prev)
                * float(p_next)
                * float(get_transition_score_fn(prev_sr, next_sr))
            )
    return float(score)
