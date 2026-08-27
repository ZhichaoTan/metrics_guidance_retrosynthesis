"""Released SPRS reproduction of the PDVN two-branch rollout policy."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from tree_search.binghong_mlp_one_step import ensure_binghong_on_path


def select_two_branch_candidates(
    reference_logits: torch.Tensor,
    learned_logits: torch.Tensor,
    topk: int,
) -> tuple[list[int], list[float]]:
    """Select global top-k with the reference branch, then rerank those indices."""
    if reference_logits.ndim != 2 or reference_logits.shape[0] != 1:
        raise ValueError("Expected one two-dimensional reference-logit row")
    if learned_logits.shape != reference_logits.shape:
        raise ValueError("Reference and learned logits must have the same shape")
    if not 0 < int(topk) <= reference_logits.shape[1]:
        raise ValueError("topk must be within the shared template vocabulary")

    _, indices = torch.topk(reference_logits, k=int(topk), dim=1)
    scores = F.softmax(learned_logits.gather(1, indices), dim=1)
    return (
        [int(index) for index in indices[0].detach().cpu().tolist()],
        [float(score) for score in scores[0].detach().cpu().tolist()],
    )


def canonical_reactants(reactant_smiles: str) -> Optional[str]:
    parts = []
    for part in reactant_smiles.split("."):
        molecule = Chem.MolFromSmiles(part)
        if molecule is None:
            return None
        parts.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
    return ".".join(sorted(parts))


def apply_and_merge_templates(
    product_smiles: str,
    candidates: list[tuple[float, str]],
    apply_template: Callable[[str, str], list[str]],
) -> Optional[dict[str, list[Any]]]:
    """Apply templates, canonicalize and merge outcomes, normalize, then drop identity."""
    product = canonical_reactants(product_smiles)
    if product is None:
        raise ValueError(f"Invalid product SMILES: {product_smiles}")

    outcomes: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for score, rule in candidates:
        try:
            applied = sorted(apply_template(rule, product_smiles))
        except (IndexError, KeyError, RuntimeError, ValueError):
            continue
        if not applied:
            continue
        divided_score = float(score) / len(applied)
        for raw_reactants in applied:
            key = canonical_reactants(raw_reactants)
            if key is not None:
                outcomes[key].append((divided_score, rule))

    if not outcomes:
        return None
    merged = [
        (reactants, sum(score for score, _ in entries), entries[0][1])
        for reactants, entries in outcomes.items()
    ]
    merged.sort(key=lambda item: item[1], reverse=True)
    total = float(np.sum([score for _, score, _ in merged]))
    if total <= 0:
        return None
    normalized = [
        (reactants, score / total, rule)
        for reactants, score, rule in merged
    ]
    kept = [item for item in normalized if item[0] != product]
    if not kept:
        return None
    return {
        "reactants": [reactants for reactants, _, _ in kept],
        "scores": [float(score) for _, score, _ in kept],
        "templates": [rule for _, _, rule in kept],
    }


class PDVNSPRSOneStep:
    """Fixed SPRS learned branch with the fixed Retro* realistic filter."""

    def __init__(
        self,
        checkpoint: str | Path,
        reference_checkpoint: str | Path,
        templates_path: str | Path,
        binghong_root: str | Path,
        expansion_topk: int = 50,
        device: int = -1,
    ) -> None:
        if int(expansion_topk) <= 0:
            raise ValueError("expansion_topk must be positive")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.reference_checkpoint = Path(reference_checkpoint).expanduser().resolve()
        self.templates_path = Path(templates_path).expanduser().resolve()
        self.expansion_topk = int(expansion_topk)
        self._model = _RolloutModel(
            checkpoint=self.checkpoint,
            reference_checkpoint=self.reference_checkpoint,
            templates_path=self.templates_path,
            binghong_root=binghong_root,
            device=int(device),
        )

    def __call__(self, smiles: str) -> Optional[dict[str, list[Any]]]:
        return self._model.run(smiles, topk=self.expansion_topk)


class _RolloutModel:
    def __init__(
        self,
        checkpoint: Path,
        reference_checkpoint: Path,
        templates_path: Path,
        binghong_root: str | Path,
        device: int,
    ) -> None:
        ensure_binghong_on_path(binghong_root)
        from mlp_retrosyn.mlp_policies import RolloutPolicyNet

        rules = templates_path.read_text(encoding="utf-8").splitlines()
        if not rules or any(not rule for rule in rules):
            raise ValueError(f"Invalid template vocabulary: {templates_path}")
        self.rules = rules

        def load_network(path: Path):
            network = RolloutPolicyNet(len(rules), fp_dim=2048)
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = {
                key.removeprefix("module."): value for key, value in state.items()
            }
            network.load_state_dict(state)
            network.eval()
            return network

        self.network = load_network(checkpoint)
        self.reference_network = load_network(reference_checkpoint)
        self.device = torch.device(f"cuda:{device}" if device >= 0 else "cpu")
        self.network.to(self.device)
        self.reference_network.to(self.device)

    def run(self, smiles: str, topk: int) -> Optional[dict[str, list[Any]]]:
        from mlp_retrosyn.mlp_policies import preprocess
        from rdchiral.main import rdchiralRunText

        fingerprint = preprocess(smiles, 2048).reshape(1, -1)
        tensor = torch.tensor(
            fingerprint, dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            learned_logits = self.network(tensor)
            reference_logits = self.reference_network(tensor)
        indices, scores = select_two_branch_candidates(
            reference_logits, learned_logits, topk
        )
        candidates = [
            (score, self.rules[index]) for index, score in zip(indices, scores)
        ]
        return apply_and_merge_templates(
            smiles,
            candidates,
            lambda rule, product: list(rdchiralRunText(rule, product)),
        )
