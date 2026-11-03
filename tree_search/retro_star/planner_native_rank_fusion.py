"""Fixed SA+SC+GED complexity-distance guidance for Retro*, by decision-rank fusion.

The chemistry signal is defined within each one-step sibling set, transported by
the incoming reaction, and fused only when Retro* selects an open frontier node.
It never changes one-step probabilities, reaction costs, value estimates, or the
native stopping condition.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, Sequence, Tuple

import numpy as np


BINGHONG_UPSTREAM_COMMIT = "60e7a24213810706a7075d113704c9320158d726"
METRIC_NAMES = ("sa_score", "sc_score", "ged")
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def percentile_ranks(
    values: Sequence[float], *, higher_is_better: bool
) -> np.ndarray:
    """Return average-tie percentile ranks (best=1, worst=0, flat=0.5)."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    if not np.all(np.isfinite(array)):
        raise ValueError("Percentile ranks require finite values")
    if array.size == 1 or np.all(array == array[0]):
        return np.full(array.shape, 0.5, dtype=float)

    ordered = array if higher_is_better else -array
    order = ordered.argsort(kind="stable")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(array.size, dtype=float)
    _, inverse = np.unique(ordered, return_inverse=True)
    sums = np.bincount(inverse, weights=ranks)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse] / (array.size - 1)


def logarithmic_rank_weight(selection_index: int) -> float:
    """The released schedule: ``log(1 + t) / log(501)``."""
    selection_index = int(selection_index)
    if selection_index < 0:
        raise ValueError("selection_index must be non-negative")
    return math.log1p(selection_index) / math.log(501.0)


def fuse_frontier_ranks(
    native_costs: Sequence[float],
    chemistry_scores: Sequence[float],
    *,
    selection_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``F=(Rn+k*Rc)/(1+k)`` and its two percentile operands."""
    if len(native_costs) != len(chemistry_scores):
        raise ValueError("Native and chemistry frontier sizes must match")
    native_rank = percentile_ranks(native_costs, higher_is_better=False)
    chemistry_rank = percentile_ranks(chemistry_scores, higher_is_better=True)
    weight = logarithmic_rank_weight(selection_index)
    fused = (native_rank + weight * chemistry_rank) / (1.0 + weight)
    return fused, native_rank, chemistry_rank


def mol_node_lineage_reactions(mol_node: Any) -> Tuple[str, ...]:
    """Return canonical root-to-node reactions from a MolStar parent chain."""
    reactions = []
    current = mol_node
    while getattr(current, "parent", None) is not None:
        reaction = current.parent
        product_node = reaction.parent
        precursors = ".".join(sorted(child.mol for child in reaction.children))
        reactions.append(f"{precursors}>>{product_node.mol}")
        current = product_node
    reactions.reverse()
    return tuple(reactions)


def _ensure_project_on_path() -> None:
    """Prefer this release's top-level metric and chemistry packages."""
    if not sys.path or sys.path[0] != _PROJECT_ROOT:
        if _PROJECT_ROOT in sys.path:
            sys.path.remove(_PROJECT_ROOT)
        sys.path.insert(0, _PROJECT_ROOT)
    loaded_utils = sys.modules.get("utils")
    loaded_path = (
        (getattr(loaded_utils, "__file__", "") or "") if loaded_utils else ""
    )
    if loaded_utils is not None and not os.path.abspath(loaded_path).startswith(
        _PROJECT_ROOT
    ):
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]


def _compute_guidance_scores(**kwargs):
    """Import the shared guidance score without loading MCTS at module import."""
    _ensure_project_on_path()
    from tree_search.mcts.guidance import compute_guidance_scores

    return compute_guidance_scores(**kwargs)


class PlannerNativeRankFusion:
    """Target-local chemistry scoring and open-frontier rank fusion."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset schedule, caches, and diagnostics before one target."""
        self._selection_index = 0
        self._metric_cache: Dict[tuple, float] = {}
        self._lineage_tree_cache: Dict[Tuple[str, ...], Any] = {}
        self._path_metric_cache: Dict[tuple, float] = {}
        self._failure_types: Counter[str] = Counter()
        self.candidate_set_count = 0
        self.candidate_count = 0
        self.lineage_failure_count = 0
        self.path_ged_evaluation_count = 0
        self.selection_set_count = 0
        self.selection_order_change_count = 0
        self.chemistry_noop_count = 0
        self.first_selection_divergence_call = None

    def diagnostics(self) -> Dict[str, object]:
        return {
            "schedule_call_count": self._selection_index,
            "candidate_set_count": self.candidate_set_count,
            "candidate_count": self.candidate_count,
            "selection_set_count": self.selection_set_count,
            "selection_order_change_count": self.selection_order_change_count,
            "chemistry_noop_count": self.chemistry_noop_count,
            "first_selection_divergence_call": self.first_selection_divergence_call,
            "path_ged_evaluation_count": self.path_ged_evaluation_count,
            "lineage_failure_count": self.lineage_failure_count,
            "unmappable_reactions": self.lineage_failure_count,
            "lineage_failure_types": dict(sorted(self._failure_types.items())),
        }

    def _record_failure(self, error: Exception) -> None:
        self.lineage_failure_count += 1
        self._failure_types[type(error).__name__] += 1

    def _lineage_tree(self, lineage: Tuple[str, ...]):
        cached = self._lineage_tree_cache.get(lineage, ...)
        if cached is None:
            raise ValueError("The synthesis lineage could not be mapped")
        if cached is not ...:
            return cached

        from metrics.complexity_distance import ComplexityDistanceSynTree
        from utils.chem_utils import map_one_reaction

        try:
            if len(lineage) == 1:
                tree = ComplexityDistanceSynTree(
                    [map_one_reaction(lineage[0])],
                    track_atom_init=True,
                    add_ged_init=True,
                )
            else:
                parent_tree = self._lineage_tree(lineage[:-1])
                tree = copy.deepcopy(parent_tree)
                tree.add_reaction_to_tree(map_one_reaction(lineage[-1]))
                tree.map_reaction_in_tree()
        except Exception:
            self._lineage_tree_cache[lineage] = None
            raise

        self._lineage_tree_cache[lineage] = tree
        return tree

    def _ged_value(self, reaction: str, lineage: Tuple[str, ...]) -> float:
        from metrics.complexity_distance import cal_reaction_edit_distance
        from utils.chem_utils import map_one_reaction

        if not lineage:
            key = ("ged", reaction)
            if key not in self._metric_cache:
                self._metric_cache[key] = float(
                    cal_reaction_edit_distance(map_one_reaction(reaction))
                )
            return self._metric_cache[key]

        key = (lineage, reaction)
        if key in self._path_metric_cache:
            return self._path_metric_cache[key]

        tree = copy.deepcopy(self._lineage_tree(lineage))
        tree.add_reaction_to_tree(map_one_reaction(reaction))
        new_edge = tree.map_reaction_in_tree()
        edge_data, product_node, reactant_node = next(
            (data, product, reactant)
            for product, reactant, data in tree.tree.edges(data=True)
            if data["reaction"].can_reaction_smiles == new_edge.can_reaction_smiles
        )
        value = float(
            tree.cal_relative_edit_distance(
                edge_data, product_node, reactant_node, "ged"
            )
        )
        self.path_ged_evaluation_count += 1
        self._path_metric_cache[key] = value
        return value

    def _metric_value(
        self, reaction: str, metric_name: str, lineage: Tuple[str, ...]
    ) -> float:
        _ensure_project_on_path()
        if metric_name == "ged":
            return self._ged_value(reaction, lineage)

        key = (metric_name, reaction)
        if key not in self._metric_cache:
            from metrics.complexity_distance import cal_complexity_reduction

            self._metric_cache[key] = float(
                cal_complexity_reduction(reaction, metric_name)
            )
        return self._metric_cache[key]

    def score_expansion(self, mol_node: Any) -> None:
        """Attach a local chemistry percentile to each valid new reaction child."""
        reactions = list(getattr(mol_node, "children", ()))
        if not reactions:
            return

        self.candidate_set_count += 1
        self.candidate_count += len(reactions)
        lineage = mol_node_lineage_reactions(mol_node)
        metric_values: Dict[str, Dict[str, float]] = {
            metric_name: {} for metric_name in METRIC_NAMES
        }
        valid_keys = []
        valid_indices = []
        for index, reaction_node in enumerate(reactions):
            reactants = ".".join(sorted(child.mol for child in reaction_node.children))
            reaction = f"{reactants}>>{mol_node.mol}"
            try:
                values = {
                    metric_name: self._metric_value(
                        reaction, metric_name, lineage
                    )
                    for metric_name in METRIC_NAMES
                }
                if not np.all(np.isfinite(list(values.values()))):
                    raise ValueError("Non-finite chemistry metric")
            except Exception as error:
                reaction_node.chemistry_rank = 0.5
                self._record_failure(error)
                continue
            key = str(index)
            for metric_name, value in values.items():
                metric_values[metric_name][key] = value
            valid_keys.append(key)
            valid_indices.append(index)

        if not valid_keys:
            return
        guidance_scores = _compute_guidance_scores(
            rxns=valid_keys,
            metric_values=metric_values,
            mode="signed_rank_positive_only",
        )
        composites = [guidance_scores[key] for key in valid_keys]
        local_ranks = percentile_ranks(composites, higher_is_better=True)
        for index, rank in zip(valid_indices, local_ranks):
            reactions[index].chemistry_rank = float(rank)

    @staticmethod
    def incoming_chemistry_rank(mol_node: Any) -> float:
        reaction = getattr(mol_node, "parent", None)
        if reaction is None:
            return 0.5
        return float(getattr(reaction, "chemistry_rank", 0.5))

    def blended_metric(
        self, mol_nodes: Sequence[Any], native_scores: Sequence[float]
    ) -> np.ndarray:
        """Return stable selection priorities while leaving closed nodes at ``inf``."""
        scores = np.asarray(native_scores, dtype=float)
        priorities = np.full(scores.shape, np.inf, dtype=float)
        open_indices = np.flatnonzero(np.isfinite(scores)).tolist()
        if not open_indices:
            return priorities

        native = [float(scores[index]) for index in open_indices]
        chemistry = [
            self.incoming_chemistry_rank(mol_nodes[index])
            for index in open_indices
        ]
        fused, native_rank, chemistry_rank = fuse_frontier_ranks(
            native,
            chemistry,
            selection_index=self._selection_index,
        )
        self._selection_index += 1

        if len(open_indices) > 1:
            self.selection_set_count += 1
            self.chemistry_noop_count += int(
                np.all(chemistry_rank == chemistry_rank[0])
            )

        stable_ids = [
            int(getattr(mol_nodes[index], "id", index))
            for index in open_indices
        ]
        order = sorted(
            range(len(open_indices)),
            key=lambda index: (
                -float(fused[index]),
                -float(native_rank[index]),
                float(native[index]),
                stable_ids[index],
            ),
        )
        native_choice = min(
            range(len(open_indices)),
            key=lambda index: (float(native[index]), stable_ids[index]),
        )
        guided_choice = order[0]
        if guided_choice != native_choice:
            self.selection_order_change_count += 1
            if self.first_selection_divergence_call is None:
                self.first_selection_divergence_call = self._selection_index - 1

        for position, local_index in enumerate(order):
            priorities[open_indices[local_index]] = float(position)
        return priorities
