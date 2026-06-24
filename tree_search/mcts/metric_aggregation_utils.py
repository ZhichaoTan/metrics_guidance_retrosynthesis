"""Metric aggregation utilities for metric-guided PUCT."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPSILON = 1e-12

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


def _safe_softmax(logits: Dict[str, float], epsilon: float = EPSILON) -> Dict[str, float]:
    if not logits:
        return {}

    rxns = list(logits.keys())
    values = np.asarray([float(logits[rxn]) for rxn in rxns], dtype=float)
    if not np.all(np.isfinite(values)):
        uniform = 1.0 / len(rxns)
        return {rxn: uniform for rxn in rxns}

    shifted = values - np.max(values)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if not np.isfinite(denom) or denom <= epsilon:
        uniform = 1.0 / len(rxns)
        return {rxn: uniform for rxn in rxns}

    probs = exps / denom
    return {rxn: float(probs[index]) for index, rxn in enumerate(rxns)}


def _is_active_metric(
    values: Dict[str, float],
    valid_rxns: List[str],
    epsilon: float = EPSILON,
) -> bool:
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


def compute_action_metric_scores(
    valid_rxns: List[str],
    model_priors: Dict[str, float],
    metric_values: Dict[str, Dict[str, float]],
    mode: str = RANK_AGGREGATION_MODE,
    include_model_prior: bool = True,
    beta: float = 1.0,
    epsilon: float = EPSILON,
    return_details: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, object]]:
    """Aggregate sibling metric values into per-action metric scores.

    Returns a model-prior-fused softmax distribution ``softmax(beta * G + log p)`` (used as the
    one-step prior by Retro*); ``details["centered_scores"]`` holds the centered chemistry score
    ``G`` that the MCTS controller blends into its UCB as a convex term.
    """
    if mode not in VALID_METRIC_AGGREGATION_MODES:
        raise ValueError(
            f"Invalid metric aggregation mode: {mode}. "
            f"Expected one of {sorted(VALID_METRIC_AGGREGATION_MODES)}."
        )

    if not valid_rxns:
        return ({}, {}) if return_details else {}

    active_metrics = [
        metric_name
        for metric_name, values in metric_values.items()
        if _is_active_metric(values, valid_rxns, epsilon=epsilon)
    ]

    metric_component_scores: Dict[str, Dict[str, float]] = {}
    if active_metrics:
        for metric_name in active_metrics:
            metric_component_scores[metric_name] = _rank_scores(
                metric_values[metric_name],
                valid_rxns,
                epsilon=epsilon,
            )

        if mode == RANK_AGGREGATION_MODE:
            raw_scores = {
                rxn: float(
                    np.mean([metric_component_scores[metric_name][rxn] for metric_name in active_metrics])
                )
                for rxn in valid_rxns
            }
        else:
            raw_scores = {}
            for rxn in valid_rxns:
                signed_terms = []
                for metric_name in active_metrics:
                    metric_value = float(metric_values[metric_name].get(rxn, np.nan))
                    is_positive = float(np.isfinite(metric_value) and metric_value > epsilon)
                    signed_terms.append(metric_component_scores[metric_name][rxn] * is_positive)
                raw_scores[rxn] = float(np.mean(signed_terms))
    else:
        raw_scores = {rxn: 0.0 for rxn in valid_rxns}

    centered_scores = _center_scores(raw_scores)
    logits: Dict[str, float] = {}
    for rxn in valid_rxns:
        logit = float(beta) * centered_scores[rxn]
        if include_model_prior:
            logit += float(np.log(max(float(model_priors.get(rxn, 0.0)), epsilon)))
        logits[rxn] = logit

    priors = _safe_softmax(logits, epsilon=epsilon)
    uniform_weight = 1.0 / len(active_metrics) if active_metrics else 0.0
    details = {
        "mode": mode,
        "active_metrics": active_metrics,
        "metric_weights": {
            metric_name: float(uniform_weight) for metric_name in active_metrics
        },
        "metric_component_scores": metric_component_scores,
        "centered_scores": {
            rxn: float(centered_scores[rxn]) for rxn in valid_rxns
        },
        "logits": {rxn: float(logits[rxn]) for rxn in valid_rxns},
        "priors": {rxn: float(priors[rxn]) for rxn in valid_rxns},
        "prior_sum": float(sum(priors.values())),
        "include_model_prior": bool(include_model_prior),
        "beta": float(beta),
    }
    return (priors, details) if return_details else priors
