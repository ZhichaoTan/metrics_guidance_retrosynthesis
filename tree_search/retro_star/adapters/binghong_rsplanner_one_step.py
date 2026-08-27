"""Template-relevance one-step adapter for official Binghong RSPlanner."""

from __future__ import annotations

from typing import List, Optional

from rdkit import Chem

from tree_search.retro_star.modules.local_expand_adapter import LocalExpandOneAdapter
from tree_search.retro_star.options import ExpandOneOptions, RetroBackendOption


def _canonicalize_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)


class TemplateRelevanceOneStep:
    """Expose template relevance in the shape expected by ``RSPlanner``."""

    def __init__(
        self,
        retro_model_name: str = "uspto_original_consol_Roh",
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

    def run(self, smiles: str, topk: int = 10) -> Optional[dict]:
        self._expand_one_options.retro_backend_options[0].max_num_templates = int(topk)
        results = self._adapter(
            smiles=smiles,
            expand_one_options=self._expand_one_options,
        )
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

        return {
            "reactants": reactants,
            "scores": scores,
            "template": templates,
        }
