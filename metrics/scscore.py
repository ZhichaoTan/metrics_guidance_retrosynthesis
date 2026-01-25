"""SCScore (Synthetic Complexity Score) calculator.
"""

import askcos_pickle as pickle
import math
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from typing import Dict, List

class SCScorePrecursorPrioritizer:
    """Standalone, importable SCScorer model.

    It does not have tensorflow as a dependency and is a more attractive option
    for deployment. The calculations are fast enough that there is no real
    reason to use GPUs (via tf) instead of CPUs (via np).

    Attributes:
        vars (list of np.ndarry of np.ndarray of np.float32): Weights and bias
            of given model.
        FP_rad (int): Fingerprint radius.
        FP_len (int): Fingerprint length.
        score_scale (float): Upper-bound of scale for scoring.
    """

    def __init__(self, score_scale=5.0):
        """Initializes SCScorePrecursorPrioritizer.

        Args:
            score_scale (float, optional): Upper-bound of scale for scoring.
                (default: {5.0})
        """
        with open("/home/zhichaotan/Workspace/ZhichaoTan/tanzc/pathway_ged/sc_score_model/model_1024bool.pickle", "rb") as fid:
            self.vars = pickle.load(fid)
        self.FP_rad = 2
        self.FP_len = 1024
        self.score_scale = score_scale

    def mol_to_fp(self, mol):
        """Returns fingerprint of molecule for bool model.

        Args:
            mol (Chem.rdchem.Mol or None): Molecule to get fingerprint
                of.

        Returns:
            np.ndarray of np.bool or np.float32: Fingerprint of given
                molecule.
        """
        if mol is None:
            return np.zeros((self.FP_len,), dtype=np.float32)

        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, self.FP_rad, nBits=self.FP_len, useChirality=True
            ),
            dtype=bool
        )

    def smi_to_fp(self, smi: str) -> np.ndarray:
        """Returns fingerprint of molecule from given SMILES string.

        Args:
            smi (str): SMILES string of given molecule.
        """
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)

        return self.mol_to_fp(Chem.MolFromSmiles(str(smi)))

    def apply(self, x: np.ndarray) -> float:
        """Applies model to a fingerprint to calculate score.

        Args:
            x (np.ndarray): Fingerprint of molecule to apply model to.

        Returns:
            float: Score of molecule.
        """
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = i == (len(self.vars) - 2)
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.dot(W.T, x) + b
            if not last_layer:
                x = x * (x > 0)             # ReLU
        x = 1 + (self.score_scale - 1) * sigmoid(x)

        return x

    def _get_one_score(self, smiles: str):
        """Returns score of molecule from given single-molecule SMILES string.

        Args:
            smiles (str): SMILES string of molecule; can be multi-component.
        """
        fp = np.array((self.smi_to_fp(smiles)), dtype=np.float32)
        if sum(fp) == 0:
            cur_score = 0.0
        else:
            # Run
            cur_score = self.apply(fp)

        return cur_score

    def get_score_from_smiles(self, smiles: str):
        """Returns score of molecule from given SMILES string.

        Args:
            smiles (str): SMILES string of molecule; can be multi-component.
        """
        if "." in smiles:
            cur_score = np.max([self._get_one_score(smi) for smi in smiles.split(".")])
        else:
            cur_score = self._get_one_score(smiles)

        return cur_score

    def _get_scores_from_smiles_list(self, smiles_list: List[str]) -> Dict[str, float]:
        smiles_to_scores = {}
        if not smiles_list:
            return {}

        fps = [self.smi_to_fp(smi) for smi in smiles_list]
        fps = np.stack(fps, axis=0)             # (b, fp_size)
        x = fps

        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = i == (len(self.vars) - 2)
            W = self.vars[i]
            b = self.vars[i + 1]
            # x = np.dot(W.T, x) + b
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU

        x = np.squeeze(x, axis=1)               # (b, 1) -> (b,)
        x = [
            1 + (self.score_scale - 1) * sigmoid(logit)
            for logit in x
        ]

        scores = x
        for smi, fp, score in zip(smiles_list, fps, scores):
            if np.sum(fp) == 0:
                smiles_to_scores[smi] = 0.0
            else:
                smiles_to_scores[smi] = float(score)

        return smiles_to_scores

    def get_batch_scores(self, smiles_list: List[str]) -> List[float]:
        # first loop to get all smiles
        all_smiles = set()
        for smiles in smiles_list:
            for s in smiles.split("."):
                if s.strip() not in all_smiles:
                    all_smiles.add(s)

        all_smiles = list(all_smiles)
        smiles_to_scores = self._get_scores_from_smiles_list(all_smiles)

        # second loop to get max scores
        scores = []
        for smiles in smiles_list:
            score = max(smiles_to_scores[s] for s in smiles.split("."))
            scores.append(score)

        del all_smiles
        del smiles_to_scores

        return scores

def sigmoid(x):
    """Returns sigmoid of x.

    Args:
        x (float): Input value.
    """
    if x < -10:
        return 0
    if x > 10:
        return 1
    return 1.0 / (1 + math.exp(-x))

def test():
    scscorer = SCScorePrecursorPrioritizer()
    x = np.random.random((2, 1024))

    for i in range(0, len(scscorer.vars), 2):
        last_layer = i == (len(scscorer.vars) - 2)
        W = scscorer.vars[i]
        b = scscorer.vars[i + 1]
        print(f"W{i}: {W.shape}")
        print(f"b{i+1}: {b.shape}")
        # x = np.dot(W.T, x) + b
        x = np.matmul(x, W) + b

        if not last_layer:
            x = x * (x > 0)  # ReLU
    print(f"x: {x.shape}")
    x = np.squeeze(x).tolist()
    x = [
        1 + (scscorer.score_scale - 1) * sigmoid(logit)
        for logit in x
    ]

    return x

if __name__ == "__main__":
    test()
