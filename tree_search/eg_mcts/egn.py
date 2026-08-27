"""Experience Guidance Network from the official EG-MCTS release."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import nn


class EGMLP(nn.Module):
    """Official 4096-256-1 EGN architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(fingerprints))


def smiles_to_fp(smiles: str, fp_dim: int = 2048) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid molecule SMILES for EGN: {smiles}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, 2, nBits=int(fp_dim)
    )
    result = np.zeros(int(fp_dim), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def reaction_smarts_to_fp(template: str, fp_dim: int = 2048) -> np.ndarray:
    reaction = AllChem.ReactionFromSmarts(template)
    if reaction is None:
        raise ValueError(f"Invalid reaction SMARTS for EGN: {template}")
    fingerprint = AllChem.CreateStructuralFingerprintForReaction(reaction)
    if fingerprint.GetNumBits() % int(fp_dim) != 0:
        raise ValueError("Reaction fingerprint cannot be folded to the EGN dimension")
    fold_factor = fingerprint.GetNumBits() // int(fp_dim)
    if fold_factor > 1:
        fingerprint = DataStructs.FoldFingerprint(fingerprint, fold_factor)
    result = np.zeros(int(fp_dim), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


class ExperienceGuidance:
    """Load the fixed official EGN and score ``(product, template)`` pairs."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.model = EGMLP().to(self.device)
        state = torch.load(
            self.checkpoint, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def __call__(self, molecule: str, template: str) -> float:
        combined = np.concatenate(
            (smiles_to_fp(molecule), reaction_smarts_to_fp(template)), axis=0
        ).reshape(1, -1)
        tensor = torch.from_numpy(combined).to(self.device)
        with torch.inference_mode():
            return float(self.model(tensor).item())
