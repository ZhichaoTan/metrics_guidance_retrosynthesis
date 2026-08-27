"""PDVN-SPRS rollout policy with unguided zero-value Retro* search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from tree_search.binghong_mlp_one_step import ensure_binghong_on_path


def _load_mol_tree(binghong_root: str | Path):
    ensure_binghong_on_path(binghong_root)
    from retro_star.alg.mol_tree import MolTree

    return MolTree


def _route_evidence(route: Any) -> Optional[dict[str, Any]]:
    if route is None:
        return None
    depths = [0] * len(route.mols)
    for index in range(1, len(route.mols)):
        depths[index] = depths[int(route.parents[index])] + 1
    steps = []
    for index, children in enumerate(route.children):
        if children is None:
            continue
        cost = float(route.costs[index])
        steps.append(
            {
                "product": route.mols[index],
                "reactants": [route.mols[child] for child in children],
                "score": float(np.exp(-cost)),
                "template": route.templates[index],
            }
        )
    return {
        "target": route.target_mol,
        "max_molecule_depth": max(depths, default=0),
        "reaction_count": len(steps),
        "route_cost": float(route.total_cost),
        "steps": steps,
    }


@dataclass(frozen=True)
class PDVNPlanResult:
    solved: bool
    model_calls: int
    termination_reason: str
    molecule_nodes: int
    reaction_nodes: int
    route: Optional[dict[str, Any]]


class PDVNRetroStarPlanner:
    """Best-first Retro*-0 with no metric signal or learned value function."""

    def __init__(
        self,
        expand_fn: Callable[[str], Optional[dict[str, Any]]],
        is_known_fn: Callable[[str], bool],
        binghong_root: str | Path,
        max_model_calls: int = 500,
        search_max_depth: Optional[int] = None,
    ) -> None:
        if int(max_model_calls) < 0:
            raise ValueError("max_model_calls must be non-negative")
        if search_max_depth is not None and int(search_max_depth) < 0:
            raise ValueError("search_max_depth must be non-negative or None")
        self.expand_fn = expand_fn
        self.is_known_fn = is_known_fn
        self.binghong_root = Path(binghong_root).expanduser().resolve()
        self.max_model_calls = int(max_model_calls)
        self.search_max_depth = (
            None if search_max_depth is None else int(search_max_depth)
        )

    def plan(self, target: str) -> PDVNPlanResult:
        MolTree = _load_mol_tree(self.binghong_root)

        known_cache: dict[str, bool] = {}

        def cached_known(molecule: str) -> bool:
            if molecule not in known_cache:
                known_cache[molecule] = bool(self.is_known_fn(molecule))
            return known_cache[molecule]

        class KnownMolecules:
            def __contains__(_, molecule: str) -> bool:
                return cached_known(molecule)

        tree = MolTree(
            target_mol=target,
            known_mols=KnownMolecules(),
            value_fn=lambda _: 0.0,
            max_depth=self.search_max_depth,
        )
        model_calls = 0
        reason = "target_in_stock" if tree.succ else "budget_exhausted"

        while not tree.succ and model_calls < self.max_model_calls:
            frontier = np.asarray(
                [node.v_target() if node.open else np.inf for node in tree.mol_nodes],
                dtype=float,
            )
            if np.min(frontier) == np.inf:
                reason = "exhausted"
                break
            tree.search_status = float(np.min(frontier))
            molecule = tree.mol_nodes[int(np.argmin(frontier))]
            result = self.expand_fn(molecule.mol)
            model_calls += 1

            if result and result.get("scores"):
                reactants = result.get("reactants", [])
                scores = np.asarray(result.get("scores", []), dtype=float)
                templates = result.get("templates", result.get("template", []))
                if not (len(reactants) == len(scores) == len(templates)):
                    raise ValueError("One-step result fields have inconsistent lengths")
                costs = -np.log(np.clip(scores, 1e-3, 1.0))
                reactant_lists = [
                    list(dict.fromkeys(str(value).split("."))) for value in reactants
                ]
                tree.expand(molecule, reactant_lists, costs, templates)
            else:
                tree.expand(molecule, None, None, None)

            if tree.succ:
                reason = "solved"

        route = tree.get_best_route() if tree.succ else None
        return PDVNPlanResult(
            solved=bool(tree.succ),
            model_calls=model_calls,
            termination_reason=reason,
            molecule_nodes=len(tree.mol_nodes),
            reaction_nodes=len(tree.reaction_nodes),
            route=_route_evidence(route),
        )
