"""Adapter for the public Retro* (binghong-ml) one-step MLP policy and stock.

The adapter is planner-neutral: MCTS, EG-MCTS, Retro*, and PDVN all consume the
same ``reactants/scores/templates`` result dictionary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from rdkit import Chem


def default_binghong_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return Path(
        os.environ.get(
            "BINGHONG_RETRO_STAR_ROOT",
            project_root / "external" / "binghong_retro_star",
        )
    ).expanduser().resolve()


def ensure_binghong_on_path(binghong_root: str | Path) -> Path:
    """Expose the pinned checkout and its two local Python packages."""
    root = Path(binghong_root).expanduser().resolve()
    candidates = (
        root,
        root / "retro_star" / "packages" / "mlp_retrosyn",
        root / "retro_star" / "packages" / "rdchiral",
    )
    for candidate in reversed(candidates):
        text = str(candidate)
        if candidate.is_dir() and text not in sys.path:
            sys.path.insert(0, text)
    return root


def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def normalize_binghong_result(
    product_smiles: str,
    raw: Optional[dict[str, Any]],
) -> Optional[dict[str, list[Any]]]:
    """Canonicalize and merge model outcomes, then remove the identity outcome."""
    if not raw:
        return None
    reactants = raw.get("reactants", [])
    scores = raw.get("scores", [])
    templates = raw.get("template", raw.get("templates", []))
    if not (len(reactants) == len(scores) == len(templates)):
        raise ValueError("Binghong one-step result fields have inconsistent lengths")

    product = canonicalize_smiles(product_smiles)
    merged: dict[str, tuple[float, str]] = {}
    for outcome, score, template in zip(reactants, scores, templates):
        precursor = canonicalize_smiles(str(outcome))
        current_score, current_template = merged.get(
            precursor, (0.0, str(template))
        )
        merged[precursor] = (current_score + float(score), current_template)

    ordered = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
    kept = [item for item in ordered if item[0] != product]
    if not kept:
        return None
    return {
        "reactants": [precursor for precursor, _ in kept],
        "scores": [score for _, (score, _) in kept],
        "templates": [template for _, (_, template) in kept],
    }


class BinghongMLPOneStep:
    """Load and run the public Retro* MLP one-step policy."""

    def __init__(
        self,
        binghong_root: str | Path | None = None,
        model_path: str | Path | None = None,
        templates_path: str | Path | None = None,
        expansion_topk: int = 50,
        device: int = -1,
    ) -> None:
        if int(expansion_topk) <= 0:
            raise ValueError("expansion_topk must be positive")
        root = ensure_binghong_on_path(binghong_root or default_binghong_root())
        asset_root = root / "retro_star" / "one_step_model"
        self.model_path = Path(model_path or asset_root / "saved_rollout_state_1_2048.ckpt")
        self.templates_path = Path(templates_path or asset_root / "template_rules_1.dat")
        self.expansion_topk = int(expansion_topk)

        from mlp_retrosyn.mlp_inference import MLPModel

        self._model = MLPModel(
            str(self.model_path),
            str(self.templates_path),
            device=int(device),
        )

    def __call__(self, smiles: str) -> Optional[dict[str, list[Any]]]:
        raw = self._model.run(smiles, topk=self.expansion_topk)
        return normalize_binghong_result(smiles, raw)


class BinghongStock:
    """Exact SMILES membership in the Retro* ``origin_dict.csv`` stock."""

    def __init__(self, stock_path: str | Path) -> None:
        import pandas as pd

        self.stock_path = Path(stock_path).expanduser().resolve()
        frame = pd.read_csv(self.stock_path, usecols=["mol"])
        self._molecules = set(frame["mol"].dropna().astype(str))

    def __contains__(self, smiles: str) -> bool:
        return smiles in self._molecules

    def __call__(self, smiles: str) -> bool:
        return smiles in self
