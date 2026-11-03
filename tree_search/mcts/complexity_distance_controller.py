"""MCTS with the complexity-distance guidance for retrosynthetic search."""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from metrics.complexity_distance import (
    ComplexityDistanceSynTree,
    cal_complexity_reduction,
    cal_max_tanimoto_similarity,
    cal_reaction_edit_distance,
)
from tree_search.mcts.guidance import (
    METRIC_HYBRID_PUCT_MODE,
    MODEL_PRIOR_PUCT_MODE,
    RANK_AGGREGATION_MODE,
    VALID_GUIDANCE_MODES,
    VALID_METRIC_AGGREGATION_MODES,
    compute_guidance_scores,
    guidance_weight_schedule_fns,
    parse_metric_names,
)
from tree_search.mcts.mcts_controller import MCTS
from utils.chem_utils import map_one_reaction

os.environ["USE_LOCAL_RETRO"] = "true"

# Metrics calculated on the synthesis tree of the selected path, from the atom-mapped reactions
DISTANCE_METRICS = {"ged", "tanimoto"}


def _cal_one_step_metric(rxn: str, metric_name: str) -> float:
    """Metric of a reaction alone, without the synthesis tree of the selected path.

    Used for the reactions of the target, and at every depth when no distance metric is used.
    Each metric in a combined name (e.g. ``"ged&tanimoto"``) is calculated separately, so
    ``metric_name`` here is always a single metric.
    """
    mapped_reaction = map_one_reaction(rxn)
    metric = 0
    if metric_name == "ged":
        # GED requires atom-mapped reactions; the unmapped input often collapses to zero.
        metric = cal_reaction_edit_distance(mapped_reaction)
    elif metric_name == "tanimoto":
        # The helper gives the reactants' max similarity to the target. The target's similarity to
        # itself is 1, so the reduction is 1 - max similarity, as in the later steps.
        metric = 1.0 - cal_max_tanimoto_similarity(rxn)
    elif metric_name in ["sa_score", "sc_score"]:
        metric = cal_complexity_reduction(rxn, metric_name)
    return metric


class ComplexityDistanceMCTS(MCTS):
    """MCTS with the complexity-distance guidance.

    In ``metric_hybrid_puct`` mode, each reaction option is scored by

        score = (1 - k) * (Q + c * U) + k * G

    where Q + c * U is the PUCT score with the model score of the reaction as the prior, G is the
    centered rank of the reaction among its siblings by how much it reduces the molecular
    complexity (``sa_score``, ``sc_score``) and changes the distance to the target (``ged``,
    ``tanimoto``), and the weight k is scheduled over the iterations. ``model_prior_puct`` mode
    selects by the PUCT score alone.
    """

    def __init__(
        self,
        metric_name: str = "ged",
        metric_weight_start: float = 0.0,
        metric_weight_end: float = 1.0,
        metric_schedule_type: str = "constant",
        guidance_mode: str = METRIC_HYBRID_PUCT_MODE,
        metric_aggregation_mode: str = RANK_AGGREGATION_MODE,
    ):
        super().__init__()
        if guidance_mode not in VALID_GUIDANCE_MODES:
            raise ValueError(
                f"Invalid guidance_mode: {guidance_mode}. "
                f"Expected one of {sorted(VALID_GUIDANCE_MODES)}."
            )
        if metric_aggregation_mode not in VALID_METRIC_AGGREGATION_MODES:
            raise ValueError(
                f"Invalid metric_aggregation_mode: {metric_aggregation_mode}. "
                f"Expected one of {sorted(VALID_METRIC_AGGREGATION_MODES)}."
            )
        if metric_schedule_type not in guidance_weight_schedule_fns:
            raise ValueError(
                f"Invalid metric_schedule_type: {metric_schedule_type}. "
                f"Expected one of {sorted(guidance_weight_schedule_fns)}."
            )

        self.metric_name = metric_name
        self.metric_names = parse_metric_names(metric_name)
        # Only the distance metrics need the synthesis tree of the selected path; the complexity
        # metrics are local to the reaction, so the tree is not built or copied without them.
        self.uses_distance_metrics = bool(DISTANCE_METRICS & set(self.metric_names))
        # Weight k of the guidance score, scheduled from metric_weight_start to metric_weight_end
        self.metric_weight_start = float(metric_weight_start)
        self.metric_weight_end = float(metric_weight_end)
        self.metric_schedule_type = metric_schedule_type
        self.guidance_weight_schedule_fn = guidance_weight_schedule_fns[metric_schedule_type]
        self.guidance_mode = guidance_mode
        self.metric_aggregation_mode = metric_aggregation_mode

        self.raw_metrics: Dict[str, Dict[str, float]] = {}  # reaction -> {metric: raw value}
        # Synthesis tree of the selected path, for the distance metrics
        self.distance_tree: Optional[ComplexityDistanceSynTree] = None

    def _select(self) -> tuple[list[str], list[str]]:
        """Select the next unexpanded leaf node, keeping the synthesis tree of the selected path.

        As ``MCTS._select()``, except that backtracking from a chemical also invalidates the
        reaction leading to it and removes that reaction from the synthesis tree.
        """
        chem_path = [self.target]
        rxn_path: List[str] = []
        invalid_options: Set[str] = set()
        self.distance_tree = None

        while True:
            leaf = chem_path[-1]
            max_depth = self.build_tree_options.max_depth
            self.tree.nodes[leaf]["min_depth"] = min(
                self.tree.nodes[leaf]["min_depth"],
                len(chem_path) - 1,
            )

            if (
                (max_depth is None or len(chem_path) <= max_depth)
                and not self.tree.nodes[leaf]["expanded"]
            ):
                break

            if max_depth is not None and len(chem_path) >= max_depth:
                invalid_options.add(leaf)
                if rxn_path:
                    invalid_options.add(rxn_path[-1])
                del chem_path[-1]
                if self.distance_tree is not None and rxn_path:
                    self.distance_tree.remove_reaction_from_tree(rxn_path[-1])
                try:
                    del rxn_path[-1]
                except IndexError:
                    self.distance_tree = None
                    return [], []
                continue

            options = self.ucb(
                node=leaf,
                chem_path=chem_path,
                invalid_options=invalid_options,
                exploration_weight=self.build_tree_options.exploration_weight,
            )
            if not options:
                invalid_options.add(leaf)
                if chem_path[-1] == self.target:
                    return [], []
                if rxn_path:
                    invalid_options.add(rxn_path[-1])
                del chem_path[-1]
                try:
                    if self.distance_tree is not None and rxn_path:
                        self.distance_tree.remove_reaction_from_tree(rxn_path[-1])
                    del rxn_path[-1]
                except IndexError:
                    self.distance_tree = None
                    return [], []
                continue

            _, reaction = options[0]
            precursor = min(
                (
                    child
                    for child in self.tree.successors(reaction)
                    if not self.tree.nodes[child]["done"] and child not in invalid_options
                ),
                key=lambda node: self.tree.nodes[node]["visit_count"],
                default=None,
            )
            if precursor is None:
                invalid_options.add(reaction)
                continue

            chem_path.append(precursor)
            rxn_path.append(reaction)
            if self.guidance_mode == MODEL_PRIOR_PUCT_MODE or not self.uses_distance_metrics:
                continue

            mapped_reaction = map_one_reaction(reaction)
            if self.distance_tree is None:
                self.distance_tree = ComplexityDistanceSynTree(
                    [mapped_reaction],
                    track_atom_init=True,
                    add_ged_init=True,
                )
            else:
                self.distance_tree.add_reaction_to_tree(mapped_reaction)
                self.distance_tree.map_reaction_in_tree()

        return chem_path, rxn_path

    def ucb(
        self,
        node: str,
        chem_path: List[str],
        invalid_options: Set[str],
        exploration_weight: float,
    ) -> List[Tuple[float, str]]:
        """Score the reaction options from ``node`` by the guidance mode, sorted by score."""
        if self.guidance_mode == MODEL_PRIOR_PUCT_MODE:
            return self._model_prior_puct_ucb(
                node,
                chem_path,
                invalid_options,
                exploration_weight,
            )
        return self._metric_hybrid_puct_ucb(
            node,
            chem_path,
            invalid_options,
            exploration_weight,
        )

    def _model_prior_puct_ucb(
        self,
        node: str,
        chem_path: List[str],
        invalid_options: Set[str],
        exploration_weight: float,
    ) -> List[Tuple[float, str]]:
        product_visits = max(self.tree.nodes[node]["visit_count"], 1)
        options: List[Tuple[float, str]] = []

        for rxn in self.tree.successors(node):
            rxn_data = self.tree.nodes[rxn]
            if (
                rxn in invalid_options
                or all(self.tree.nodes[c]["done"] for c in self.tree.successors(rxn))
                or len(set(self.tree.successors(rxn)) & set(chem_path)) > 0
            ):
                continue

            node_visits = rxn_data["visit_count"]
            q_sa = rxn_data["est_value"] / node_visits
            u_sa = rxn_data["rxn_score_from_model"] * np.sqrt(product_visits) / (1 + node_visits)
            options.append((float(q_sa + exploration_weight * u_sa), rxn))

        options.sort(key=lambda item: item[0], reverse=True)
        return options

    def _metric_hybrid_puct_ucb(
        self,
        node: str,
        chem_path: List[str],
        invalid_options: Set[str],
        exploration_weight: float,
    ) -> List[Tuple[float, str]]:
        candidate_rxns: List[str] = []
        for rxn in self.tree.successors(node):
            if (
                rxn in invalid_options
                or all(self.tree.nodes[c]["done"] for c in self.tree.successors(rxn))
                or len(set(self.tree.successors(rxn)) & set(chem_path)) > 0
            ):
                continue
            candidate_rxns.append(rxn)

        if not candidate_rxns:
            return []

        metric_values: Dict[str, Dict[str, float]] = {
            metric_name: {} for metric_name in self.metric_names
        }
        for rxn in candidate_rxns:
            raw_metrics = self._get_raw_metrics(rxn)
            for metric_name in self.metric_names:
                metric_values[metric_name][rxn] = raw_metrics[metric_name]

        guidance_scores = compute_guidance_scores(
            candidate_rxns,
            metric_values,
            mode=self.metric_aggregation_mode,
        )
        guidance_weight = self._get_guidance_weight()
        product_visits = max(self.tree.nodes[node]["visit_count"], 1)
        options: List[Tuple[float, str]] = []

        for rxn in candidate_rxns:
            rxn_data = self.tree.nodes[rxn]
            node_visits = rxn_data["visit_count"]
            rxn_probability = float(rxn_data["rxn_score_from_model"])
            q_sa = rxn_data["est_value"] / node_visits
            u_sa = rxn_probability * np.sqrt(product_visits) / (1 + node_visits)
            puct_score = q_sa + exploration_weight * u_sa
            score = (1.0 - guidance_weight) * puct_score + guidance_weight * guidance_scores[rxn]
            options.append((float(score), rxn))

        options.sort(key=lambda item: item[0], reverse=True)
        return options

    def _get_raw_metrics(self, rxn: str) -> Dict[str, float]:
        """Raw metrics of a reaction, cached from the path it is first scored on."""
        if rxn not in self.raw_metrics:
            self.raw_metrics[rxn] = self._cal_raw_metrics(rxn)
        return self.raw_metrics[rxn]

    def _cal_raw_metrics(self, rxn: str) -> Dict[str, float]:
        """Raw metrics of a reaction, on a copy of the synthesis tree of the path if it has one."""
        if self.distance_tree is None:
            return {
                metric_name: float(_cal_one_step_metric(rxn, metric_name))
                for metric_name in self.metric_names
            }

        mapped_reaction = map_one_reaction(rxn)
        temp_tree = copy.deepcopy(self.distance_tree)
        temp_tree.add_reaction_to_tree(mapped_reaction)
        new_reaction = temp_tree.map_reaction_in_tree()
        product_node, reactant_node, edge_data = next(
            edge
            for edge in temp_tree.tree.edges(data=True)
            if edge[2]["reaction"].can_reaction_smiles == new_reaction.can_reaction_smiles
        )
        return {
            metric_name: float(
                self._cal_path_metric(
                    temp_tree,
                    edge_data,
                    product_node,
                    reactant_node,
                    metric_name,
                )
            )
            for metric_name in self.metric_names
        }

    @staticmethod
    def _cal_path_metric(
        temp_tree: ComplexityDistanceSynTree,
        edge_data,
        product_node,
        reactant_node,
        metric_name: str,
    ) -> float:
        """Metric of a deeper reaction, in the synthesis tree of the path with it added."""
        if metric_name == "ged":
            return temp_tree.cal_relative_edit_distance(
                edge_data,
                product_node,
                reactant_node,
                metric_name,
            )
        if metric_name == "tanimoto":
            return temp_tree.cal_similarity_reduction(
                edge_data,
                product_node,
                reactant_node,
                metric_name,
            )
        if metric_name in {"sa_score", "sc_score"}:
            return temp_tree.cal_one_step_complexity_reduction(edge_data, metric_name)
        raise ValueError(f"Unsupported metric: {metric_name}")

    def _get_guidance_weight(self) -> float:
        """Weight k of the guidance score, scheduled over the iterations."""
        return float(
            self.guidance_weight_schedule_fn(
                self.iterations,
                start=self.metric_weight_start,
                end=self.metric_weight_end,
                max_iter=self.build_tree_options.max_iterations,
            )
        )
