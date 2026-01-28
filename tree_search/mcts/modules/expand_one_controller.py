import numpy as np
import os
import time
from multiprocessing import Pool
from pydantic import BaseModel, Field
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from typing import Any, Dict, List, Optional, Tuple

USE_LOCAL_RETRO = True


class RetroBackendOption(BaseModel):
    retro_backend: str = "template_relevance"
    retro_model_name: str = "uspto_original_consol_Roh"
    max_num_templates: int = 100
    max_cum_prob: float = 0.995
    attribute_filter: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    threshold: float = 0.3
    top_k: int = 10


def print_if_debug(m: str, debug: bool = False):
    if debug:
        print(m)


class ExpandOneController:
    def __init__(self):
        # Lazy initialization to avoid creating objects at module level
        if USE_LOCAL_RETRO:
            from .local_retro_controller import LocalRetroController
            self.retro_controller = LocalRetroController(
                model_base_path="tree_search/uspto_original_consol_Roh",
                default_backend="template_relevance"
            )
        else:
            raise ValueError("Please deploy ASKCOS server first.")
        # Don't create Pool at initialization - create on demand if needed
        self.p = None

    def merge_data(self, 
                result_1: Dict[str, Any],
                result_2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges the data from two result dictionaries.

        Args:
            result_1: The first result dictionary.
            result_2: The second result dictionary.

        Returns:
            A dictionary containing the merged data.
        """
        if result_1 == result_2:
            return result_1

        if result_1["normalized_model_score"] > result_2["normalized_model_score"]:
            new_result, old_result = result_1, result_2
        else:
            new_result, old_result = result_2, result_1

        # Currently keeping the highest score (score, template score, etc)
        new_result = self.merge_template_data(new_result, old_result)
        new_result = self.merge_retrosim_data(new_result, old_result)
        
        new_result["models_predicted_by"] += old_result["models_predicted_by"]
        new_result["models_predicted_by"].sort(key=lambda x: x[2], reverse=True)

        return new_result

    def merge_template_data(self, 
                new_result: Dict[str, Any],
                old_result: Dict[str, Any]) -> Dict[str, Any]:
        # Merge template information
        if old_result.get("template"):
            if new_result.get("template"):
                new_result["template"]["tforms"].extend(old_result["template"]["tforms"])
                new_result["template"]["tsources"].extend(old_result["template"]["tsources"])
                new_result["template"]["num_examples"] += old_result["template"]["num_examples"]
            else:
                new_result["template"] = old_result["template"]

        return new_result
        
    def merge_retrosim_data(self,
                new_result: Dict[str, Any],
                old_result: Dict[str, Any]) -> Dict[str, Any]:
        
        # Merge reaction data information
        if old_result.get("reaction_data"):
            if new_result.get("reaction_data"):

                for field in ["reference_url", "patent_number"]:
                    new_result["reaction_data"][field] = (
                        new_result["reaction_data"].get(field) or 
                        old_result["reaction_data"].get(field)
                    )
            else:
                new_result["reaction_data"] = old_result["reaction_data"]

        return new_result

    def get_outcomes(
        self,
        smiles: str,
        retro_backend_options: List[RetroBackendOption],
        banned_chemicals: List[str] = None,
        banned_reactions: List[str] = None,
        use_fast_filter: bool = True,
        fast_filter_threshold: float = 0.75,
        retro_rerank_backend: str = "relevance_heuristic",
        cluster_precursors: bool = True,
        extract_template: bool = False,
        return_reacting_atoms: bool = True,
        selectivity_check: bool = False,
        debug: bool = False
    ) -> List[Dict[str, any]]:
        if not banned_chemicals:
            banned_chemicals = []
        if not banned_reactions:
            banned_reactions = []

        # retro_controller takes in list[str], here we only pass in one smiles
        # retro_results is list[dict]
        start = time.time()
        retro_results = []
        for option in retro_backend_options:
            retro_result = self.retro_controller(
                smiles=[smiles],
                backend=option.retro_backend,
                model_name=option.retro_model_name,
                max_num_templates=option.max_num_templates,
                max_cum_prob=option.max_cum_prob,
                attribute_filter=option.attribute_filter,
                threshold=option.threshold,
                top_k=option.top_k
            )[0]
            for result in retro_result:
                result["retro_backend"] = option.retro_backend
                result["retro_model_name"] = option.retro_model_name
                result["models_predicted_by"] = \
                    [(option.retro_backend, option.retro_model_name, result["normalized_model_score"])]

                if result.get("template"):
                    result["template"]["tsources"] = [option.retro_model_name]*len(result["template"]["tforms"])


            retro_results.extend(retro_result)
        print_if_debug(f"retro: {time.time() - start}", debug)

        # A number of postprocessing steps
        # <deduplication>
        start = time.time()
        mol = Chem.MolFromSmiles(smiles)
        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        unique_results_dict = {}

        for result in retro_results:
            reactants_split = result["outcome"].split(".")
            if any(smi in banned_chemicals for smi in reactants_split):
                continue

            reaction_smi = result["outcome"] + ">>" + smiles
            cano_outcome = Chem.MolToSmiles(
                Chem.MolFromSmiles(result["outcome"]), isomericSmiles=True
            )
            cano_rxn_smi = f"{cano_outcome}>>{cano_smiles}"
            if reaction_smi in banned_reactions or cano_rxn_smi in banned_reactions:
                continue

            if cano_outcome == cano_smiles:
                continue

            if cano_outcome in unique_results_dict:
                
                result = self.merge_data(result, unique_results_dict[cano_outcome])

            #cano_outcomes.append(cano_outcome)
            #reaction_smis.append(reaction_smi)
            unique_results_dict[cano_outcome] = result

        unique_results = list(unique_results_dict.values())
        reaction_smis = [result["outcome"] + ">>" + smiles for result in unique_results]
        print_if_debug(f"dedup: {time.time() - start}", debug)
        # </deduplication>


        # <template extraction>
        start = time.time()
        if extract_template or selectivity_check:
            for result in unique_results:
                if "template" in result and result["template"]:
                    continue

                if "mapped_smiles" not in result:
                    rxn_smi = result["outcome"] + ">>" + smiles
                    res_atom_mapper = self.atom_mapper(smiles=[rxn_smi])
                    mapped_rxn_smi = res_atom_mapper[0] if res_atom_mapper else ""
                    result["mapped_smiles"] = mapped_rxn_smi

                reactants, _, products = result["mapped_smiles"].split(">")
                reaction = {
                    '_id': -1,
                    'reactants': reactants,
                    'products': products
                }
                try:
                    template = extract_from_reaction(reaction)
                except:
                    template = {}
                if (
                    "reaction_smarts" not in template
                    or not template["reaction_smarts"]
                ):
                    template["reaction_smarts"] = "failed_extraction"

                for k in [
                    "reactants_smarts",
                    "products_smarts",
                    "reaction_smarts_forward",
                    "reaction_smarts_retro",
                    "reactants",
                    "products"
                ]:
                    template.pop(k, None)
                result["template"] = template
        print_if_debug(f"extract: {time.time() - start}", debug)
        # </template extraction>

        return unique_results

    @staticmethod
    def _rerank_default(filtered_results: List[Dict[str, Any]]
                           ) -> List[Dict[str, Any]]:
        for result in filtered_results:
            result["score"] = result["normalized_model_score"]

        reranked_results = sorted(
            filtered_results,
            key=lambda d: d["score"],
            reverse=True
        )

        for rank, result in enumerate(reranked_results, start=1):
            result["rank"] = rank

        return reranked_results

    def _rerank_by_relevance_heuristic(self, filtered_results: List[Dict[str, Any]]
                                       ) -> List[Dict[str, Any]]:
        tasks = []
        for result in filtered_results:
            try:
                necessary_reagent = result["template"]["necessary_reagent"]
            except (KeyError, TypeError):
                necessary_reagent = ""

            try:
                template_score = result["template"]["template_score"]
            except (KeyError, TypeError):
                template_score = result["normalized_model_score"]

            tasks.append((result["outcome"], necessary_reagent, template_score))

        scores = self.p.imap(_get_relevance, tasks)
        for result, score in zip(filtered_results, scores):
            result["score"] = score

        reranked_results = sorted(
            filtered_results,
            key=lambda d: d["score"],
            reverse=True
        )

        for rank, result in enumerate(reranked_results, start=1):
            result["rank"] = rank

        return reranked_results

    @staticmethod
    def _rerank_by_scscore(filtered_results: List[Dict[str, Any]]
                           ) -> List[Dict[str, Any]]:
        for result in filtered_results:
            result["score"] = result["scscore"]

        reranked_results = sorted(
            filtered_results,
            key=lambda d: d["score"],
            reverse=False
        )

        for rank, result in enumerate(reranked_results, start=1):
            result["rank"] = rank

        return reranked_results

if __name__ == "__main__":
    controller = ExpandOneController()
    smiles = "O=C(NCc1cccc(Cl)c1)c1ccc2c(c1)OCO2"
    results = controller.get_outcomes(smiles, [RetroBackendOption()])
    print(results)