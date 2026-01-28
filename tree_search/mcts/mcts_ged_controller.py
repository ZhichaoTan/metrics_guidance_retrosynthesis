"""
Metrics-guided Monte Carlo Tree Search controller.

This module extends the base MCTS controller with metrics-based guidance for retrosynthesis planning.
"""

import os
import json
import copy
from typing import List, Set, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem

# Local imports
from tree_search.mcts.mcts_controller import MCTS
from tree_search.mcts.options import ExpandOneOptions, BuildTreeOptions, RetroBackendOption
from tree_search.mcts.ged_weight_change import ged_weight_change_fun_dict
from utils.chem_utils import map_one_reaction
from tree_search.mcts.utils_mcts import _cal_delta_metric_in_ucb
from metrics.syn_tree_ged import SynTree_ged

os.environ['USE_LOCAL_RETRO'] = "true"

class MCTS_ged(MCTS):
    """
    Metrics-guided Monte Carlo Tree Search controller.
    
    Extends the base MCTS controller with metrics-based guidance for retrosynthesis planning using GED,
    Tanimoto similarity, SA_Score, SC_Score, or hybrid metrics.
    """
    
    def __init__(
        self,
        metric_name: str = "ged",
        ged_weight: float = 0,
        ged_weight_start: float = 0,
        ged_weight_end: float = 1,
        ged_change_type: str = "constant",
        track_ged_change: bool = False,
        ged_weight_in_metric: float = 1,
        tanimoto_weight_in_metric: float = 1,
        ged_change_type_in_metric: str = "hybrid_exponential_change"
    ):
        """
        Initialize metrics-guided MCTS controller.
        
        Args:
            metric_name: Metric to use ('ged', 'tanimoto', 'SA_Score', 'SC_Score', or hybrid)
            ged_weight: Weight for GED metric in UCB scoring
            ged_weight_start: Starting weight for dynamic weight adjustment
            ged_weight_end: Ending weight for dynamic weight adjustment
            ged_change_type: Type of weight change function ('constant', 'linear', etc.)
            track_ged_change: Whether to track and visualize GED changes
            ged_weight_in_metric: Weight for GED in hybrid metrics
            tanimoto_weight_in_metric: Weight for Tanimoto in hybrid metrics
            ged_change_type_in_metric: Change type for metric weights
        """
        super().__init__()
        self.ged_weight = ged_weight
        self.track_ged_change = track_ged_change
        self.reaction_delta_metric_history = {}
        
        self.ged_weight_start = ged_weight_start
        self.ged_weight_end = ged_weight_end
        self.delta_metric_change_fun = ged_weight_change_fun_dict[ged_change_type]
        self.metric_name = metric_name
        self.ged_weight_in_metric = ged_weight_in_metric
        self.tanimoto_weight_in_metric = tanimoto_weight_in_metric
        
        # Limit history size to prevent memory issues
        self.max_history_size = 10000  # Maximum number of reactions to keep in history
        
    def _select(self) -> tuple[list[str], list[str]]:
        chem_path = [self.target]
        rxn_path = []
        invalid_options = set()
        self.tree_for_ged = None
        self.previous_route_ged_list = []
        self.trival_ged_count = 0
        while True:
            leaf = chem_path[-1]
                            
            self.tree.nodes[leaf]["min_depth"] = min(self.tree.nodes[leaf]["min_depth"], len(chem_path)-1)

            if len(chem_path) <= self.build_tree_options.max_depth and not self.tree.nodes[leaf]["expanded"]:
                break
            
            elif len(chem_path) >= self.build_tree_options.max_depth:
                # print(f"len(chem_path) >= self.build_tree_options.max_depth: remove {rxn_path[-1]}")
                invalid_options.add(leaf)
                invalid_options.add(rxn_path[-1])
                del chem_path[-1]
                self.tree_for_ged.remove_reaction_from_tree(rxn_path[-1])
                try:
                    # print("len(chem_path) >= self.build_tree_options.max_depth")
                    del rxn_path[-1]
                # except (IndexError):
                except:
                    self.tree_for_ged = None
                    return [], []
                continue
            
            options = self.ucb(
                node=leaf,
                chem_path=chem_path,
                invalid_options=invalid_options,
                exploration_weight=self.build_tree_options.exploration_weight,
                ged_weight=self.ged_weight
            )
            if not options:
                invalid_options.add(leaf)
                invalid_options.add(rxn_path[-1])
                if chem_path[-1] == self.target:
                    return [], []
                del chem_path[-1]
                # print(f"not options: remove {rxn_path[-1]}")
                try:
                    # print("not options")
                    self.tree_for_ged.remove_reaction_from_tree(rxn_path[-1])
                    del rxn_path[-1]
                # except (IndexError):
                except:
                    self.tree_for_ged = None
                    return [], []
                continue

            score, reaction = options[0]
            # print(f"prioritized reaction: {reaction}")
            precursor = min(
                (
                    c for c in self.tree.successors(reaction)
                    if not self.tree.nodes[c]["done"] and c not in invalid_options
                ),
                key=lambda _node: self.tree.nodes[_node]["visit_count"],
                default=None
            )
            if precursor is None:
                # print("precursor is None: continue")
                invalid_options.add(reaction)
                continue
            else:
                chem_path.append(precursor)
                rxn_path.append(reaction)
                if self.tree_for_ged is None:
                    reaction = map_one_reaction(reaction)
                    self.tree_for_ged = SynTree_ged([reaction], track_atom_init=True, add_ged_init=True)
                else:
                    # Use shallow copy to save memory - tree_for_ged will be modified but that's OK
                    _, temp_tree = self.reaction_delta_metric_history[reaction]
                    self.tree_for_ged = copy.copy(temp_tree)  # Shallow copy is much faster
        return chem_path, rxn_path

    def ucb(
        self,
        node: str,
        chem_path: List[str],
        invalid_options: Set[str],
        exploration_weight: float,
        ged_weight: float = None,
    ) -> List[Tuple[float, str]]:
        options = []
        product_visits = self.tree.nodes[node]["visit_count"]
        
        # if self.use_strategy is None:
        #     self.use_strategy = len(self.expand_one_options.retro_backend_options) > 1
        
        for rxn in self.tree.successors(node):
            rxn_is_strategy = False
            rxn_data = self.tree.nodes[rxn]
            
            # print(rxn_data)
            if (
                rxn in invalid_options
                or all(self.tree.nodes[c]["done"] for c in self.tree.successors(rxn))
                or len(set(self.tree.successors(rxn)) & set(chem_path)) > 0
            ):
                continue
            est_value = rxn_data["est_value"]
            node_visits = rxn_data["visit_count"]
            rxn_probability = rxn_data["rxn_score_from_model"]

            q_sa = est_value / node_visits
            u_sa = rxn_probability * np.sqrt(np.log(product_visits) / node_visits) # corrected ucb in askcos
            if rxn not in self.reaction_delta_metric_history.keys():
                delta_metric, temp_tree, rxn_smi_mapped = _cal_delta_metric_in_ucb(self.tree_for_ged, rxn, metric_name=self.metric_name,
                                                                                   ged_weight=self.ged_weight_in_metric, tanimoto_weight=self.tanimoto_weight_in_metric)
                self.reaction_delta_metric_history[rxn] = (delta_metric, temp_tree)
                
                # Limit history size - remove oldest entries if exceeded
                if len(self.reaction_delta_metric_history) > self.max_history_size:
                    # Remove oldest entries (first N entries)
                    keys_to_remove = list(self.reaction_delta_metric_history.keys())[:len(self.reaction_delta_metric_history) - self.max_history_size + 1000]
                    for key in keys_to_remove:
                        del self.reaction_delta_metric_history[key]
            else:
                # Use shallow copy - we only need to read, not modify
                delta_metric, temp_tree = self.reaction_delta_metric_history[rxn]
                # No need to copy if we're just reading
                
            metric_weight = self.delta_metric_change_fun(self.iterations, start=self.ged_weight_start, end=self.ged_weight_end,
                                             max_iter=self.build_tree_options.max_iterations)

            score = (1-metric_weight) * (q_sa + exploration_weight * u_sa) + metric_weight * delta_metric
            
            options.append((score, rxn))

        options.sort(key=lambda x: x[0], reverse=True)
        return options
    
    def _expand(self, chem_path: List[str]) -> None:
        """
        Expand the tree by running one-step retro prediction to a chemical node
        """
        leaf = chem_path[-1]
        retro_results = self.expand_one(
            smiles=leaf,
            expand_one_options=self.expand_one_options
        )
        
        self.tree.nodes[leaf]["expanded"] = True
        if not retro_results:
            self.tree.nodes[leaf]["done"] = True
            return

        for result in retro_results:
            precursor_smiles = result["outcome"]
            reaction_smiles = precursor_smiles + ">>" + leaf
            reactant_list = precursor_smiles.split(".")

            try:
                template = result["template"]
                template_tuple = (template["index"], template["template_set"])
                num_examples = template["num_examples"]
            except (KeyError, TypeError):
                # try-except just for backward compatibility
                template = None
                template_tuple = None
                num_examples = 0

            if reaction_smiles in self.reactions:
                # This Reaction node already exists
                rxn_data = self.tree.nodes[reaction_smiles]
                if (
                    template_tuple is not None
                    and template_tuple not in rxn_data["template_tuples"]
                ):
                    # import ipdb; ipdb.set_trace();
                    rxn_data["template_tuples"].append(template_tuple)
                    rxn_data["num_examples"] += num_examples

                # retro controller now computes normalized_model_score
                rxn_data["rxn_score_from_model"] = max(
                    rxn_data["rxn_score_from_model"], result["normalized_model_score"]
                )

            else:
                # This is new, so create a Reaction node
                tforms = template.get(
                    "tforms") if isinstance(template, dict) else None
                tsources = template.get(
                    "tsources") if isinstance(template, dict) else None

                necessary_reagent = template.get(
                    "necessary_reagent") if isinstance(template, dict) else None
                self.create_reaction_node(
                    smiles=reaction_smiles,
                    precursor_smiles=precursor_smiles,
                    tforms=tforms,
                    tsources=tsources,
                    necessary_reagent=necessary_reagent,
                    template_tuple=template_tuple,
                    rxn_score_from_model=result["normalized_model_score"],
                    num_examples=num_examples,
                    template=template,
                    retro_backend=result.get("retro_backend"),
                    retro_model_name=result.get("retro_model_name"),
                    models_predicted_by=result.get("models_predicted_by")
                )

            # Add edges to connect target -> reaction -> precursors
            self.tree.add_edge(leaf, reaction_smiles)
            for reactant in reactant_list:
                if reactant not in self.chemicals:
                    # This is new, so create a Chemical node
                    self.create_chemical_node(smiles=reactant)
                self.tree.add_edge(reaction_smiles, reactant)
                # initialize/change min_depth for reactants 
                self.tree.nodes[reactant]['min_depth'] = int(nx.shortest_path_length(self.tree, source=self.target, target=reactant)/2)
            # This _update_value only updates reactions *below* leaf
            self._update_value(reaction_smiles)


if __name__ == "__main__":
    target =  "O=C(O1)C2C=CC[C@]2([H])C1=O"
    cano_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(target))
    controller = MCTS_ged(metric_name="SA_Score&SC_Score&tanimoto&ged", track_ged_change=False,
                          ged_weight_start=0, ged_weight_end=1, ged_change_type="logarithmic_change")
    expand_one_options= ExpandOneOptions()
    retro_one_option_1 = RetroBackendOption()
    retro_one_option_1.retro_model_name = "uspto_original_consol_Roh"
    expand_one_options.retro_backend_options = [retro_one_option_1]
    build_tree_options = BuildTreeOptions()
    build_tree_options.max_depth = 10
    build_tree_options.max_iterations = 100 
    paths, stats, graph = controller.get_buyable_paths(target=cano_smiles, expand_one_options=expand_one_options, build_tree_options=build_tree_options)
    print(stats)