"""Metric-weight schedules shared by metric-guided search (PUCT and Retro*).

Every schedule maps an iteration index to the metric-term weight ``w`` used to tilt the search
prior toward chemically favorable disconnections. All functions share one signature:

    fn(x, start=..., end=..., max_iter=...) -> float

where ``x`` is the current iteration (0..max_iter), ``start``/``end`` are the weight at the first
and last iteration, and ``max_iter`` is the planned iteration budget. Select one by name through
``metric_weight_schedule_fns``.
"""

from __future__ import annotations

import numpy as np


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


# Public registry: schedule name -> schedule function. Used by both the MCTS controller and the
# Retro* one-step adapter to resolve ``--metric_schedule_type``.
metric_weight_schedule_fns = {
    "constant": constant,
    "linear": linear,
    "exponential": exponential,
    "logarithmic": logarithmic,
    "sigmoid": sigmoid,
}
