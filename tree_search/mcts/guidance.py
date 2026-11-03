"""Guidance score ``G`` of the complexity-distance guidance, and the schedules of its weight ``k``.

``G`` ranks each reaction among its siblings by the complexity and distance metrics, and MCTS
blends it into the PUCT score as ``(1 - k) * PUCT + k * G``. Retro* reuses ``G`` to rank the
sibling reactions of an expansion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

EPSILON = 1e-12

# Guidance modes: PUCT alone (baseline), or PUCT blended with the guidance score
MODEL_PRIOR_PUCT_MODE = "model_prior_puct"
METRIC_HYBRID_PUCT_MODE = "metric_hybrid_puct"
VALID_GUIDANCE_MODES = {
    MODEL_PRIOR_PUCT_MODE,
    METRIC_HYBRID_PUCT_MODE,
}

# Aggregations of the metric ranks into the guidance score
RANK_AGGREGATION_MODE = "rank_aggregation"
SIGNED_RANK_POSITIVE_ONLY_MODE = "signed_rank_positive_only"
VALID_METRIC_AGGREGATION_MODES = {
    RANK_AGGREGATION_MODE,
    SIGNED_RANK_POSITIVE_ONLY_MODE,
}


def parse_metric_names(metric_name: str) -> List[str]:
    """Split a combined metric string into unique metric names."""
    if not metric_name:
        return ["ged"]

    metric_names: List[str] = []
    for item in metric_name.split("&"):
        item = item.strip()
        if item and item not in metric_names:
            metric_names.append(item)
    return metric_names or ["ged"]


def _is_active_metric(
    values: Dict[str, float],
    valid_rxns: List[str],
    epsilon: float = EPSILON,
) -> bool:
    """Whether at least two finite values of the metric differ among the siblings."""
    finite_values = []
    for rxn in valid_rxns:
        value = float(values.get(rxn, np.nan))
        if np.isfinite(value):
            finite_values.append(value)

    if len(finite_values) <= 1:
        return False

    finite_values.sort()
    for index in range(1, len(finite_values)):
        if abs(finite_values[index] - finite_values[index - 1]) > epsilon:
            return True
    return False


def _rank_scores(
    values: Dict[str, float],
    valid_rxns: List[str],
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """Map sibling metric values to [0, 1] ranks; larger values are better."""
    if not valid_rxns:
        return {}

    pairs = []
    for rxn in valid_rxns:
        value = float(values.get(rxn, np.nan))
        if np.isfinite(value):
            pairs.append((rxn, value))

    if not pairs:
        uniform = 1.0 / len(valid_rxns)
        return {rxn: uniform for rxn in valid_rxns}

    pairs.sort(key=lambda item: item[1], reverse=True)
    if len(pairs) == 1:
        only_rxn = pairs[0][0]
        return {rxn: (1.0 if rxn == only_rxn else 0.0) for rxn in valid_rxns}

    scores: Dict[str, float] = {}
    left = 0
    total = len(pairs)
    while left < total:
        right = left + 1
        current_value = pairs[left][1]
        while right < total and abs(pairs[right][1] - current_value) <= epsilon:
            right += 1

        average_rank = (left + right - 1) / 2.0
        normalized = 1.0 - average_rank / max(total - 1, 1)
        for index in range(left, right):
            scores[pairs[index][0]] = float(normalized)
        left = right

    fallback = min(scores.values())
    for rxn in valid_rxns:
        scores.setdefault(rxn, fallback)
    return scores


def _center_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    mean_score = float(np.mean(list(scores.values())))
    return {rxn: float(score - mean_score) for rxn, score in scores.items()}


def compute_guidance_scores(
    rxns: List[str],
    metric_values: Dict[str, Dict[str, float]],
    mode: str = RANK_AGGREGATION_MODE,
) -> Dict[str, float]:
    """Centered guidance score ``G`` of the sibling reactions.

    Only the metrics whose values differ among the siblings are used. ``rank_aggregation`` takes the
    mean of their ranks, and ``signed_rank_positive_only`` the mean of ``rank * 1[value > 0]``.
    """
    if mode not in VALID_METRIC_AGGREGATION_MODES:
        raise ValueError(
            f"Invalid metric aggregation mode: {mode}. "
            f"Expected one of {sorted(VALID_METRIC_AGGREGATION_MODES)}."
        )

    if not rxns:
        return {}

    active_metrics = [
        metric_name
        for metric_name, values in metric_values.items()
        if _is_active_metric(values, rxns)
    ]

    if active_metrics:
        metric_ranks = {
            metric_name: _rank_scores(metric_values[metric_name], rxns)
            for metric_name in active_metrics
        }

        if mode == RANK_AGGREGATION_MODE:
            raw_scores = {
                rxn: float(np.mean([metric_ranks[metric][rxn] for metric in active_metrics]))
                for rxn in rxns
            }
        else:
            raw_scores = {}
            for rxn in rxns:
                signed_terms = []
                for metric_name in active_metrics:
                    metric_value = float(metric_values[metric_name].get(rxn, np.nan))
                    is_positive = float(np.isfinite(metric_value) and metric_value > EPSILON)
                    signed_terms.append(metric_ranks[metric_name][rxn] * is_positive)
                raw_scores[rxn] = float(np.mean(signed_terms))
    else:
        raw_scores = {rxn: 0.0 for rxn in rxns}

    return _center_scores(raw_scores)


# Schedules of the guidance weight k. Each maps the iteration x (0..max_iter) to the weight, from
# start at the first iteration to end at the planned iteration budget max_iter.

def constant(x, start=0.05, end=0.05, max_iter=100):
    """Hold the weight at ``start`` for the whole search."""
    return start


def linear(x, start=0.0, end=2.0, max_iter=100):
    """Interpolate linearly from ``start`` to ``end`` over ``max_iter`` iterations."""
    return start + (end - start) * (x / max_iter)


def exponential(x, start=0.0, end=2.0, max_iter=100, epsilon=1e-8):
    """Ramp slowly at first and accelerate toward ``end`` (convex/exponential)."""
    ratio = (x + epsilon) / (max_iter + epsilon)
    scale = np.expm1(ratio * np.log1p(max_iter))
    return start + (end - start) * scale / np.expm1(np.log1p(max_iter))


def logarithmic(x, start=0.0, end=2.0, max_iter=100, epsilon=1e-8):
    """Ramp quickly at first and flatten toward ``end`` (concave/logarithmic)."""
    return start + (end - start) * np.log1p(x) / np.log1p(max_iter + epsilon)


def sigmoid(x, start=0.0, end=2.0, max_iter=100, scale=10):
    """S-shaped ramp centered at ``max_iter / 2`` with steepness ``scale``."""
    return start + (end - start) / (1 + np.exp(-(x - max_iter / 2) / scale))


# Schedule name -> schedule function, selected by ``--metric_schedule_type``. Retro* uses its own
# fixed logarithmic schedule (``planner_native_rank_fusion.logarithmic_rank_weight``).
guidance_weight_schedule_fns = {
    "constant": constant,
    "linear": linear,
    "exponential": exponential,
    "logarithmic": logarithmic,
    "sigmoid": sigmoid,
}
