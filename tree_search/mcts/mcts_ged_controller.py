"""Metric-guided PUCT controller for retrosynthetic search."""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from metrics.syn_tree_ged import SynTreeGED
from tree_search.mcts.metric_weight_schedule import metric_weight_schedule_fns
from tree_search.mcts.mcts_controller import MCTS
from tree_search.mcts.metric_aggregation_utils import (
    RANK_AGGREGATION_MODE,
    VALID_METRIC_AGGREGATION_MODES,
    compute_action_metric_scores,
    parse_metric_names,
)
from tree_search.mcts.utils_mcts import _cal_delta_metric_in_ucb
from utils.chem_utils import map_one_reaction

os.environ["USE_LOCAL_RETRO"] = "true"

MODEL_PRIOR_PUCT_MODE = "model_prior_puct"
METRIC_HYBRID_PUCT_MODE = "metric_hybrid_puct"
VALID_GUIDANCE_MODES = {
    MODEL_PRIOR_PUCT_MODE,
    METRIC_HYBRID_PUCT_MODE,
}


class MCTSGED(MCTS):
    """Monte Carlo tree search with optional metric-guided PUCT selection."""

    def __init__(
        self,
        metric_name: str = "ged",
        metric_weight_start: float = 0.0,
        metric_weight_end: float = 1.0,
        metric_schedule_type: str = "constant",
        guidance_mode: str = METRIC_HYBRID_PUCT_MODE,
        metric_aggregation_mode: str = RANK_AGGREGATION_MODE,
        include_model_prior: bool = True,
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
        if metric_schedule_type not in metric_weight_schedule_fns:
            raise ValueError(
                f"Invalid metric_schedule_type: {metric_schedule_type}. "
                f"Expected one of {sorted(metric_weight_schedule_fns)}."
            )

        self.metric_name = metric_name
        self.metric_names = parse_metric_names(metric_name)
        # Only ged/tanimoto need the pairwise SynTreeGED; sa_score/sc_score are reaction-local. Skip
        # building/copying the GED tree entirely when no tree-based metric is requested (avoids redundant
        # atom-mapping + deepcopy work and the matching memory growth).
        self._needs_ged_tree = bool({"ged", "tanimoto"} & set(self.metric_names))
        # Metric-term weight schedule: weight ramps from metric_weight_start to metric_weight_end.
        self.metric_weight_start = float(metric_weight_start)
        self.metric_weight_end = float(metric_weight_end)
        self.metric_schedule_type = metric_schedule_type
        self.metric_weight_schedule_fn = metric_weight_schedule_fns[metric_schedule_type]
        self.guidance_mode = guidance_mode
        self.metric_aggregation_mode = metric_aggregation_mode
        self.include_model_prior = bool(include_model_prior)

        self.reaction_delta_metric_history: Dict[str, Dict[str, object]] = {}
        self.tree_for_ged: Optional[SynTreeGED] = None

    def _select(self) -> tuple[list[str], list[str]]:
        chem_path = [self.target]
        rxn_path: List[str] = []
        invalid_options: Set[str] = set()
        self.tree_for_ged = None

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
                if self.tree_for_ged is not None and rxn_path:
                    self.tree_for_ged.remove_reaction_from_tree(rxn_path[-1])
                try:
                    del rxn_path[-1]
                except IndexError:
                    self.tree_for_ged = None
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
                    if self.tree_for_ged is not None and rxn_path:
                        self.tree_for_ged.remove_reaction_from_tree(rxn_path[-1])
                    del rxn_path[-1]
                except IndexError:
                    self.tree_for_ged = None
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
            if self.guidance_mode == MODEL_PRIOR_PUCT_MODE or not self._needs_ged_tree:
                continue

            if self.tree_for_ged is None:
                mapped_reaction = map_one_reaction(reaction)
                self.tree_for_ged = SynTreeGED(
                    [mapped_reaction],
                    track_atom_init=True,
                    add_ged_init=True,
                )
            else:
                self.tree_for_ged = self._tree_after_reaction(reaction)

        return chem_path, rxn_path

    def ucb(
        self,
        node: str,
        chem_path: List[str],
        invalid_options: Set[str],
        exploration_weight: float,
    ) -> List[Tuple[float, str]]:
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
            model_u_sa = (
                rxn_data["rxn_score_from_model"]
                * np.sqrt(product_visits)
                / (1 + node_visits)
            )
            options.append((float(q_sa + exploration_weight * model_u_sa), rxn))

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
        model_priors: Dict[str, float] = {}

        for rxn in candidate_rxns:
            cached_entry = self._get_or_create_metric_entry(rxn)
            for metric_name in self.metric_names:
                metric_values[metric_name][rxn] = float(cached_entry["raw_metrics"][metric_name])
            model_priors[rxn] = float(self.tree.nodes[rxn]["rxn_score_from_model"])

        _, metric_details = compute_action_metric_scores(
            valid_rxns=candidate_rxns,
            model_priors=model_priors,
            metric_values=metric_values,
            mode=self.metric_aggregation_mode,
            include_model_prior=self.include_model_prior,
            beta=1.0,
            return_details=True,
        )

        metric_term_weight = self._get_metric_term_weight()
        product_visits = max(self.tree.nodes[node]["visit_count"], 1)
        options: List[Tuple[float, str]] = []

        for rxn in candidate_rxns:
            rxn_data = self.tree.nodes[rxn]
            node_visits = rxn_data["visit_count"]
            q_sa = rxn_data["est_value"] / node_visits
            model_u_sa = model_priors[rxn] * np.sqrt(product_visits) / (1 + node_visits)
            base_policy_score = q_sa + exploration_weight * model_u_sa
            metric_term_score = float(metric_details["centered_scores"][rxn])
            score = (
                (1.0 - metric_term_weight) * base_policy_score
                + metric_term_weight * metric_term_score
            )

            rxn_data["chemistry_score"] = metric_term_score
            rxn_data["metric_term_weight"] = float(metric_term_weight)
            options.append((float(score), rxn))

        options.sort(key=lambda item: item[0], reverse=True)
        return options

    def _get_or_create_metric_entry(self, rxn: str) -> Dict[str, object]:
        cached_entry = self.reaction_delta_metric_history.get(rxn)
        if cached_entry is None or "raw_metrics" not in cached_entry:
            raw_metrics = self._compute_raw_metric_bundle(rxn)
            cached_entry = self.reaction_delta_metric_history.setdefault(rxn, {})
            cached_entry["raw_metrics"] = raw_metrics
        return cached_entry

    def _tree_after_reaction(self, rxn: str) -> Optional[SynTreeGED]:
        """The GED tree the descent carries once this reaction is taken.

        Rebuilt from the tree currently held rather than kept from when the reaction was first
        scored.  Keeping it cost one full SynTreeGED per distinct reaction -- thousands per target
        -- which is what made a single worker's footprint grow with the search rather than with
        the route.  Rebuilding costs one tree per step actually taken, not one per candidate
        scored.  Ported from the main repository's mcts_ged_controller_codex_revised.
        """
        if self.tree_for_ged is None:
            return None
        temp_tree = copy.deepcopy(self.tree_for_ged)
        temp_tree.add_reaction_to_tree(map_one_reaction(rxn))
        temp_tree.map_reaction_in_tree()
        return temp_tree

    def _compute_raw_metric_bundle(
        self,
        rxn: str,
    ) -> Dict[str, float]:
        if self.tree_for_ged is None:
            raw_metrics = {}
            for metric_name in self.metric_names:
                raw_metrics[metric_name] = float(
                    _cal_delta_metric_in_ucb(rxn, metric_name=metric_name)
                )
            return raw_metrics

        mapped_reaction = map_one_reaction(rxn)
        temp_tree = copy.deepcopy(self.tree_for_ged)
        temp_tree.add_reaction_to_tree(mapped_reaction)
        new_edge_data_tagged = temp_tree.map_reaction_in_tree()
        edge_data, product_node, reactant_node = next(
            (
                edge_data,
                product_node,
                reactant_node,
            )
            for product_node, reactant_node, edge_data in temp_tree.tree.edges(data=True)
            if edge_data["reaction"].can_reaction_smiles
            == new_edge_data_tagged.can_reaction_smiles
        )

        raw_metrics: Dict[str, float] = {}
        for metric_name in self.metric_names:
            raw_metrics[metric_name] = float(
                self._compute_metric_from_tree(
                    temp_tree,
                    edge_data,
                    product_node,
                    reactant_node,
                    metric_name,
                )
            )
        return raw_metrics

    def _compute_metric_from_tree(
        self,
        temp_tree: SynTreeGED,
        edge_data,
        product_node,
        reactant_node,
        metric_name: str,
    ) -> float:
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
            return temp_tree.cal_one_step_mol_clx_reduction(edge_data, metric_name)
        raise ValueError(f"Unsupported metric: {metric_name}")

    def _get_metric_term_weight(self) -> float:
        return float(
            self.metric_weight_schedule_fn(
                self.iterations,
                start=self.metric_weight_start,
                end=self.metric_weight_end,
                max_iter=self.build_tree_options.max_iterations,
            )
        )
