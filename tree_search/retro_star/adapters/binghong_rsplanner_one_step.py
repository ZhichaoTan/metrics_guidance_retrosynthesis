"""MLP-compatible one-step wrapper for official Binghong RSPlanner."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from tree_search.retro_star.modules.local_expand_adapter import LocalExpandOneAdapter
from tree_search.retro_star.options import ExpandOneOptions, RetroBackendOption

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _ensure_project_on_path() -> None:
    """Make this project's packages win over the binghong fork's sys.path.

    Under the RSPlanner worker the binghong repo is on PYTHONPATH, which can shadow the project's
    top-level ``utils``/``datastructs`` packages with a namespace package. Force the project root
    to the front and evict any wrongly-resolved ``utils`` so the metric stack imports cleanly.
    """
    if not sys.path or sys.path[0] != _PROJECT_ROOT:
        if _PROJECT_ROOT in sys.path:
            sys.path.remove(_PROJECT_ROOT)
        sys.path.insert(0, _PROJECT_ROOT)
    mod = sys.modules.get("utils")
    if mod is not None:
        mod_file = getattr(mod, "__file__", None) or ""
        if not os.path.abspath(mod_file).startswith(_PROJECT_ROOT):
            for name in [n for n in list(sys.modules) if n == "utils" or n.startswith("utils.")]:
                del sys.modules[name]


def _canonicalize_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)


# Process-level RXNMapper singleton, local to this adapter (the MCTS path keeps using
# chem_utils.map_one_reaction unchanged). chem_utils.map_one_reaction instantiates a fresh
# RXNMapper() BERT transformer on EVERY call; under metric guidance that reloads the model once per
# candidate per expansion and dominates runtime. Reusing one instance (created lazily post-fork in
# each worker) makes mapping cheap while producing byte-identical output.
_RXN_MAPPER = None


def _get_rxn_mapper():
    global _RXN_MAPPER
    if _RXN_MAPPER is None:
        from rxnmapper import RXNMapper

        _RXN_MAPPER = RXNMapper()
    return _RXN_MAPPER


def _map_one_reaction(rxn_smi: str) -> str:
    """Mirror of utils.chem_utils.map_one_reaction but reuses the singleton mapper."""
    reactants, products = rxn_smi.split(">>")
    can_reactants = ".".join(Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reactants.split("."))
    can_products = ".".join(Chem.MolToSmiles(Chem.MolFromSmiles(p)) for p in products.split("."))
    can_rsmi = f"{can_reactants}>>{can_products}"
    return _get_rxn_mapper().get_attention_guided_atom_maps([can_rsmi])[0]["mapped_rxn"]


class TemplateRelevanceOneStep:
    """Expose template_relevance (Roh) in the shape expected by RSPlanner.expand_fn.

    Optional metric guidance: when ``metric_name`` is set, the candidate model scores are reweighted
    with the SAME fusion law as the MCTS controller -- the log-additive softmax
    ``priors = softmax(w_i * G + log(p))`` (``compute_action_metric_scores`` with
    ``include_model_prior=True`` and ``beta = w_i``). The model prior ``p_j`` is preserved in
    log-space and the signed-rank chemistry score ``G`` *tilts* it toward chemically favorable
    disconnections; it is never discarded as ``w_i -> 1`` (a convex blend would, and regressed
    USPTO-190). At ``w_i = 0`` it reduces to ``softmax(log p) = p`` (pure model prior). The result
    stays a valid prior (RSPlanner turns it into cost ``-log(prior) >= 0``), biasing the A*-style
    search toward chemically favorable disconnections. ``w_i`` follows an iteration schedule
    (best-first does one expansion per iteration, tracked by ``self._expansion_count``). Default
    (metric_name=None or beta_start==beta_end==0) leaves output byte-for-byte unchanged. See
    ``_reweight_scores`` for the exact fusion.
    """

    def __init__(
        self,
        retro_model_name: str = "uspto_original_consol_Roh",
        metric_name: Optional[str] = None,
        metric_weight_start: float = 0.0,
        metric_weight_end: float = 0.0,
        metric_schedule_type: str = "constant",
        max_iterations: int = 500,
    ):
        self.retro_model_name = retro_model_name
        self._adapter = LocalExpandOneAdapter()
        self._expand_one_options = ExpandOneOptions(
            retro_backend_options=[
                RetroBackendOption(
                    retro_backend="template_relevance",
                    retro_model_name=retro_model_name,
                    max_cum_prob=1.0,
                )
            ]
        )

        self.metric_name = metric_name or None
        self.metric_weight_start = float(metric_weight_start)
        self.metric_weight_end = float(metric_weight_end)
        self.metric_schedule_type = str(metric_schedule_type)
        self.max_iterations = int(max_iterations)
        self._metric_names: List[str] = (
            [m for m in self.metric_name.split("&") if m] if self.metric_name else []
        )
        self._expansion_count = 0
        # Reaction-level metric cache keyed by (metric_name, canonical reaction SMILES). Mirrors the
        # MCTS controller's reaction_delta_metric_history. Persists across every target this worker
        # handles (one expand_fn per worker), so repeated/shared disconnections skip recomputation.
        self._metric_cache: Dict[Tuple[str, str], float] = {}

    def _metric_value(self, rxn: str, metric: str) -> float:
        """Cached per-reaction value for a single (atomic) metric.

        Mirrors the no-tree branch of the MCTS ``_cal_delta_metric_in_ucb`` (using the singleton
        mapper for the ged path). ``metric`` is always a single metric; combined names are split by
        the caller and each sub-metric is signed-ranked separately.
        """
        key = (metric, rxn)
        cached = self._metric_cache.get(key)
        if cached is not None:
            return cached

        from metrics.syn_tree_ged import (
            _cal_one_step_edit_distance,
            _cal_one_step_mol_clx_reduction,
            _cal_one_step_similarity_reduction,
        )

        if metric == "ged":
            value = _cal_one_step_edit_distance(_map_one_reaction(rxn), metric)
        elif metric == "tanimoto":
            value = _cal_one_step_similarity_reduction(rxn, metric)
        elif metric in ["sa_score", "sc_score"]:
            value = _cal_one_step_mol_clx_reduction(rxn, metric)
        else:
            value = 0.0

        value = float(value)
        self._metric_cache[key] = value
        return value

    def _metric_enabled(self) -> bool:
        return bool(self._metric_names) and not (
            self.metric_weight_start == 0.0 and self.metric_weight_end == 0.0
        )

    def _reweight_scores(self, scores: List[float], reactants: List[str], product: str) -> List[float]:
        """Return the MCTS-aligned log-additive softmax prior softmax(w_i * G + log(p)).

        This reuses the shared metric aggregation: ``compute_action_metric_scores`` with
        ``include_model_prior=True`` and ``beta = w_i``. The model prior ``p`` (normalized one-step
        score) is preserved in log-space and the signed-rank chemistry score ``G`` *tilts* it toward
        chemically favorable disconnections -- it is never discarded as ``w_i -> 1`` (unlike a convex
        blend). At ``w_i = 0`` it reduces to ``softmax(log p) = p`` (pure model prior). The returned
        distribution stays a valid prior for RSPlanner (cost ``-log(prior) >= 0``). ``w_i`` follows the
        iteration schedule (best-first does one expansion per iteration, tracked by
        ``self._expansion_count``).
        """
        # Lazy imports to avoid pulling MCTS modules unless metric guidance is requested.
        _ensure_project_on_path()
        from tree_search.mcts.metric_weight_schedule import metric_weight_schedule_fns
        from tree_search.mcts.metric_aggregation_utils import compute_action_metric_scores

        rxn_keys = [f"{prec}>>{product}" for prec in reactants]
        if len(set(rxn_keys)) != len(rxn_keys):  # duplicate candidates -> skip reweight, keep scores
            return scores

        metric_values: Dict[str, Dict[str, float]] = {m: {} for m in self._metric_names}
        for rxn in rxn_keys:
            for metric in self._metric_names:
                metric_values[metric][rxn] = self._metric_value(rxn, metric)

        schedule_fn = metric_weight_schedule_fns[self.metric_schedule_type]
        w_i = float(
            schedule_fn(
                self._expansion_count,
                start=self.metric_weight_start,
                end=self.metric_weight_end,
                max_iter=self.max_iterations,
            )
        )

        # Log-additive softmax: priors = softmax(w_i * G + log(p)). beta = w_i is the temperature on
        # the chemistry score; include_model_prior=True keeps log(p) in the logit so the policy signal
        # is never replaced. Identical to the metric-hybrid one-step fusion.
        model_priors = {rxn: scores[i] for i, rxn in enumerate(rxn_keys)}
        priors = compute_action_metric_scores(
            valid_rxns=rxn_keys,
            model_priors=model_priors,
            metric_values=metric_values,
            mode="signed_rank_positive_only",
            include_model_prior=True,
            beta=w_i,
        )
        return [float(priors[rxn]) for rxn in rxn_keys]

    def run(self, smiles: str, topk: int = 10) -> Optional[Dict]:
        self._expand_one_options.retro_backend_options[0].max_num_templates = int(topk)
        results = self._adapter(smiles=smiles, expand_one_options=self._expand_one_options)
        if not results:
            return None

        cano_product = _canonicalize_smiles(smiles)
        reactants: List[str] = []
        scores: List[float] = []
        templates: List[Optional[str]] = []

        for result in results:
            if len(scores) >= topk:
                break

            precursor_smiles = _canonicalize_smiles(result["outcome"])
            if precursor_smiles == cano_product:
                continue

            reactants.append(precursor_smiles)
            scores.append(float(result["normalized_model_score"]))

            template = result.get("template")
            if isinstance(template, dict):
                templates.append(template.get("reaction_smarts"))
            else:
                templates.append(template)

        if not scores:
            return None

        total = sum(scores)
        if total > 0:
            scores = [score / total for score in scores]

        if self._metric_enabled():
            if len(scores) >= 2:
                scores = self._reweight_scores(scores, reactants, cano_product)
            # Count every expansion so the iteration schedule advances over the search.
            self._expansion_count += 1

        return {
            "reactants": reactants,
            "scores": scores,
            "template": templates,
        }
