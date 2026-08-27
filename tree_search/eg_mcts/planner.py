"""Minimal EG-MCTS search adapted from the official implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

ExpandFn = Callable[[str], Optional[dict[str, Any]]]
IsKnownFn = Callable[[str], bool]
ValueFn = Callable[[str, str], float]
UNSCORED_TEMPLATE_VALUE = 0.5


@dataclass
class MoleculeNode:
    mol: str
    parent: Optional["ReactionNode"]
    depth: int
    is_known: bool
    value: float = field(init=False)
    open: bool = field(init=False)
    solved: bool = field(init=False)
    children: list["ReactionNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.solved = self.is_known
        self.open = not self.is_known
        self.value = 10.0 if self.is_known else np.inf

    def valid_children(self) -> list["ReactionNode"]:
        return [child for child in self.children if child.valid]

    def ancestors(self) -> set[str]:
        if self.parent is None:
            return {self.mol}
        return self.parent.parent.ancestors() | {self.mol}


@dataclass
class ReactionNode:
    parent: MoleculeNode
    prior: float
    template: str
    value: float
    children: list[MoleculeNode] = field(default_factory=list)
    count: int = 1
    valid: bool = True
    solved: bool = False

    def set_invalid(self) -> None:
        self.valid = False
        self.value = -10.0
        self.count += 1

    def set_solved(self) -> None:
        self.solved = True
        self.value = 10.0

    def select_child(self, rng: np.random.Generator) -> MoleculeNode:
        pending = [child for child in self.children if child.open and not child.solved]
        if pending:
            return pending[0]
        unfinished = [child for child in self.children if not child.solved]
        if not unfinished:
            raise RuntimeError("Solved reaction has no unfinished child")
        return unfinished[int(rng.integers(0, len(unfinished)))]

    def update_q(self) -> None:
        if all(child.solved for child in self.children):
            self.set_solved()
            self.count += 1
            return
        expanded = [child.value for child in self.children if not child.open]
        if not expanded:
            raise RuntimeError("Cannot update a reaction before a child is expanded")
        new_q = float(sum(expanded) / len(expanded))
        self.value = (self.value * self.count + new_q) / (self.count + 1)
        self.count += 1


class SearchTree:
    def __init__(
        self,
        target: str,
        is_known_fn: IsKnownFn,
        value_fn: ValueFn,
        max_depth: Optional[int],
    ) -> None:
        self.is_known_fn = is_known_fn
        self.value_fn = value_fn
        self.max_depth = max_depth
        self.molecule_nodes: list[MoleculeNode] = []
        self.reaction_nodes: list[ReactionNode] = []
        self.value_calls = 0
        self.unscored_templates = 0
        self.root = self._new_molecule(target, parent=None, depth=0)
        self.solved = self.root.solved
        if not self.solved and self.max_depth == 0:
            self.root.open = False
            self.root.value = -np.inf

    def _new_molecule(
        self,
        mol: str,
        parent: Optional[ReactionNode],
        depth: int,
    ) -> MoleculeNode:
        node = MoleculeNode(
            mol=mol,
            parent=parent,
            depth=depth,
            is_known=bool(self.is_known_fn(mol)),
        )
        self.molecule_nodes.append(node)
        if parent is not None:
            parent.children.append(node)
        return node

    def _new_reaction(
        self,
        parent: MoleculeNode,
        reactants: list[str],
        prior: float,
        template: str,
        value: float,
    ) -> Optional[ReactionNode]:
        if any(reactant in parent.ancestors() for reactant in reactants):
            return None
        child_depth = parent.depth + 1
        known = [bool(self.is_known_fn(reactant)) for reactant in reactants]
        if (
            self.max_depth is not None
            and child_depth >= self.max_depth
            and not all(known)
        ):
            return None

        reaction = ReactionNode(parent, float(prior), str(template), float(value))
        parent.children.append(reaction)
        self.reaction_nodes.append(reaction)
        for reactant in reactants:
            self._new_molecule(reactant, reaction, child_depth)
        if all(child.solved for child in reaction.children):
            reaction.set_solved()
        return reaction

    def expand(self, node: MoleculeNode, result: Optional[dict[str, Any]]) -> None:
        if not node.open or node.children:
            raise RuntimeError("Only an open, unexpanded molecule can be expanded")
        if not result or not result.get("scores"):
            node.value = -np.inf
            node.open = False
            return

        reactants = result.get("reactants", [])
        scores = result.get("scores", [])
        templates = result.get("templates", result.get("template", []))
        if not (len(reactants) == len(scores) == len(templates)):
            raise ValueError("One-step result fields have inconsistent lengths")

        any_solved = False
        for precursor_smiles, prior, template in zip(reactants, scores, templates):
            if float(prior) < 0:
                raise ValueError("EG-MCTS priors must be non-negative")
            precursors = list(dict.fromkeys(str(precursor_smiles).split(".")))
            if not precursors or any(not precursor for precursor in precursors):
                raise ValueError(f"Invalid reactant list: {precursor_smiles}")
            try:
                value = float(self.value_fn(node.mol, str(template)))
            except ValueError:
                value = UNSCORED_TEMPLATE_VALUE
                self.unscored_templates += 1
            self.value_calls += 1
            reaction = self._new_reaction(
                node, precursors, float(prior), str(template), value
            )
            if reaction is not None:
                any_solved = any_solved or reaction.solved

        node.open = False
        valid = node.valid_children()
        if not valid:
            node.value = -np.inf
        elif any_solved:
            node.solved = True
            node.value = 10.0
        else:
            node.value = max(child.value for child in valid)

    def update(self, node: MoleculeNode) -> bool:
        current = node
        while True:
            if current is self.root:
                self.solved = current.solved
                break
            parent_reaction = current.parent
            if parent_reaction is None:
                raise RuntimeError("Non-root molecule has no parent reaction")

            if np.isneginf(current.value):
                parent_reaction.set_invalid()
                current = parent_reaction.parent
                valid = current.valid_children()
                if not valid:
                    current.value = -np.inf
                else:
                    old_value = current.value
                    current.value = max(child.value for child in valid)
                    if current.value == old_value:
                        break
                continue

            parent_reaction.update_q()
            current = parent_reaction.parent
            valid = current.valid_children()
            if any(child.solved for child in valid):
                current.value = 10.0
                current.solved = True
                continue
            old_value = current.value
            current.value = max(child.value for child in valid)
            if current.value == old_value:
                break
        return self.solved

    def solution_route(self) -> dict[str, Any]:
        if not self.solved:
            raise RuntimeError("Cannot export an unsolved tree")

        def build(node: MoleculeNode) -> dict[str, Any]:
            result: dict[str, Any] = {
                "smiles": node.mol,
                "depth": node.depth,
                "in_stock": node.is_known,
            }
            if node.is_known:
                return result
            reactions = [child for child in node.valid_children() if child.solved]
            if not reactions:
                raise RuntimeError(f"Solved molecule has no solved reaction: {node.mol}")
            reaction = reactions[0]
            result["reaction"] = {
                "template": reaction.template,
                "score": reaction.prior,
                "q_value": reaction.value,
                "reactants": [build(child) for child in reaction.children],
            }
            return result

        return build(self.root)


@dataclass(frozen=True)
class EGMCTSResult:
    solved: bool
    model_calls: int
    value_calls: int
    unscored_templates: int
    termination_reason: str
    route: Optional[dict[str, Any]]
    reaction_nodes: int
    molecule_nodes: int


class EGMCTSPlanner:
    """Official PUCT and backup rules with an optional molecule-depth cap."""

    def __init__(
        self,
        expand_fn: ExpandFn,
        is_known_fn: IsKnownFn,
        value_fn: ValueFn,
        max_model_calls: int = 500,
        search_max_depth: Optional[int] = None,
        exploration_constant: float = 0.5,
    ) -> None:
        if int(max_model_calls) < 0:
            raise ValueError("max_model_calls must be non-negative")
        if search_max_depth is not None and int(search_max_depth) < 0:
            raise ValueError("search_max_depth must be non-negative or None")
        self.expand_fn = expand_fn
        self.is_known_fn = is_known_fn
        self.value_fn = value_fn
        self.max_model_calls = int(max_model_calls)
        self.search_max_depth = (
            None if search_max_depth is None else int(search_max_depth)
        )
        self.exploration_constant = float(exploration_constant)

    def _select_reaction(
        self,
        reactions: list[ReactionNode],
        parent_visits: int,
    ) -> ReactionNode:
        scores = [
            reaction.value / reaction.count
            + self.exploration_constant
            * reaction.prior
            * np.sqrt(parent_visits)
            / (1 + reaction.count)
            for reaction in reactions
        ]
        return reactions[int(np.argmax(scores))]

    def plan(self, target: str, seed: int = 1234) -> EGMCTSResult:
        rng = np.random.default_rng(seed)
        tree = SearchTree(
            target,
            self.is_known_fn,
            self.value_fn,
            self.search_max_depth,
        )
        model_calls = 0

        if tree.solved:
            reason = "target_in_stock"
        else:
            reason = "budget_exhausted"
            for iteration in range(self.max_model_calls):
                next_molecule = tree.root
                parent_visits = iteration + 1
                while next_molecule.valid_children():
                    reaction = self._select_reaction(
                        next_molecule.valid_children(), parent_visits
                    )
                    parent_visits = reaction.count
                    next_molecule = reaction.select_child(rng)

                if next_molecule is tree.root and not next_molecule.open:
                    reason = "exhausted"
                    break
                if not next_molecule.open:
                    raise RuntimeError("Tree policy selected a closed molecule")
                result = self.expand_fn(next_molecule.mol)
                model_calls += 1
                tree.expand(next_molecule, result)
                if tree.update(next_molecule):
                    reason = "solved"
                    break

        return EGMCTSResult(
            solved=tree.solved,
            model_calls=model_calls,
            value_calls=tree.value_calls,
            unscored_templates=tree.unscored_templates,
            termination_reason=reason,
            route=tree.solution_route() if tree.solved else None,
            reaction_nodes=len(tree.reaction_nodes),
            molecule_nodes=len(tree.molecule_nodes),
        )
