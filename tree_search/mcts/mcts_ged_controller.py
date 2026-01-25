"""
GED-guided Monte Carlo Tree Search controller for retrosynthetic planning.

This module extends the base MCTS controller with Graph Edit Distance (GED)
and other metric-based guidance for pathway evaluation.
"""

import os
import json
import pickle
import copy
from typing import List, Set, Tuple, Optional

import networkx as nx
import numpy as np
from rdkit import Chem
from rxnmapper import RXNMapper
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

# Local imports
from .mcts_controller import MCTS
from .options import ExpandOneOptions, BuildTreeOptions, RetroBackendOption
from .ged_weight_change import ged_weight_change_fun_dict
from ...metrics.syn_tree_ged import (
    SynTree_ged,
    _cal_one_step_edit_distance,
    _cal_one_step_similarity_reduction,
    _cal_one_step_mol_clx_reduction,
    _cal_hybrid_distance
)
from ...utils.chem_utils import has_mapping

# Set environment variables for local retro model (if not already set)
if 'USE_LOCAL_RETRO' not in os.environ:
    os.environ['USE_LOCAL_RETRO'] = 'true'
if 'LOCAL_RETRO_MODEL_BASE_PATH' not in os.environ:
    # Update this path according to your local setup
    os.environ['LOCAL_RETRO_MODEL_BASE_PATH'] = '/path/to/retro/template_relevance/mars_ged'

class MCTS_ged(MCTS):
    """
    GED-guided Monte Carlo Tree Search controller.
    
    Extends the base MCTS controller with metric-based guidance for
    retrosynthetic pathway evaluation using GED, Tanimoto similarity,
    SA_Score, SC_Score, or hybrid metrics.
    """
    def __init__(
        self,
        metric_name: str = "ged",
        ged_weight: float = 0,
        ged_weight_start: float = 0,
        ged_weight_end: float = 1,
        ged_change_type: str = "constant",
        track_ged_change: bool = False,
        ged_output_folder: Optional[str] = None,
        ged_folder_prefix: Optional[str] = None,
        max_ged: float = 999,
        strategy_cut: int = 2,
        use_strategy_lib: bool = False,
        deterministic_strategies_path: Optional[str] = None,
        use_external_temps: bool = False,
        external_templates_path: Optional[str] = None,
        ged_weight_in_metric: float = 1,
        tanimoto_weight_in_metric: float = 1,
        ged_change_type_in_metric: str = "hybrid_exponential_change",
        target_smiles: Optional[str] = None,
        gateway_url: Optional[str] = None
    ):
        """
        Initialize GED-guided MCTS controller.
        
        Args:
            metric_name: Metric to use ('ged', 'tanimoto', 'SA_Score', 'SC_Score', or hybrid)
            ged_weight: Weight for GED metric in UCB scoring
            ged_weight_start: Starting weight for dynamic weight adjustment
            ged_weight_end: Ending weight for dynamic weight adjustment
            ged_change_type: Type of weight change function ('constant', 'linear', etc.)
            track_ged_change: Whether to track and visualize GED changes
            ged_output_folder: Folder for GED visualization outputs
            ged_folder_prefix: Prefix for GED output folder names
            max_ged: Maximum allowed GED value
            strategy_cut: Strategy extraction threshold
            use_strategy_lib: Whether to use strategy library
            deterministic_strategies_path: Path to deterministic strategies pickle file
            use_external_temps: Whether to use external templates
            external_templates_path: Path to external templates file
            ged_weight_in_metric: Weight for GED in hybrid metrics
            tanimoto_weight_in_metric: Weight for Tanimoto in hybrid metrics
            ged_change_type_in_metric: Change type for metric weights
            target_smiles: Target molecule SMILES (for external templates)
            gateway_url: URL for ASKCOS gateway (optional)
        """
        if deterministic_strategies_path is None:
            deterministic_strategies_path = "/path/to/deterministic_strategies_consol_uspto_full.pickle"
        super().__init__(gateway_url=gateway_url)
        self.ged_weight = ged_weight
        self.track_ged_change = track_ged_change
        self.ged_output_folder = ged_output_folder
        self.ged_folder_prefix = ged_folder_prefix
        self.reaction_delta_metric_history = {}
        
        self.ged_weight_start = ged_weight_start
        self.ged_weight_end = ged_weight_end
        self.delta_metric_change_fun = ged_weight_change_fun_dict[ged_change_type]
        self.max_ged = max_ged
        self.use_strategy_lib = use_strategy_lib
        self.strategy_cut = strategy_cut
        self.use_external_temps = use_external_temps
        self.metric_name = metric_name
        if self.use_strategy_lib:
            with open(deterministic_strategies_path, "rb") as f:
                self.deterministic_strategies = pickle.load(f)
            self.strategy_grouped_by_triggering_reaction = {}
            for strategy in self.deterministic_strategies:
                # strategy = strategy_list[0]
                triggering_template = strategy["triggering_reaction_template"]
                if triggering_template not in self.strategy_grouped_by_triggering_reaction.keys():
                    self.strategy_grouped_by_triggering_reaction[triggering_template] = [strategy]
                else:
                    self.strategy_grouped_by_triggering_reaction[triggering_template].append(strategy)
        
        if self.use_external_temps:
            with open(external_templates_path, "rb") as f:
                external_templates_dict = pickle.load(f)

            if isinstance(external_templates_dict, dict):
                if target_smiles in external_templates_dict.keys():
                    self.external_templates = external_templates_dict[target_smiles]["templates"]
                    print(f"Found {len(self.external_templates)} external templates for {target_smiles}")
                else:
                    self.external_templates = []
                    print(f"No external templates found for {target_smiles}")
            else:
                self.external_templates = external_templates_dict
                print(f"Found {len(self.external_templates)} external templates")
        self.ged_weight_in_metric = ged_weight_in_metric
        self.tanimoto_weight_in_metric = tanimoto_weight_in_metric
        
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
                # print(f"append: {reaction}")
                if self.tree_for_ged is None:
                    # print("Initialize the tree for ged")
                    reaction = map_one_reaction(reaction)
                    self.tree_for_ged = SynTree_ged([reaction], track_atom_init=True, add_ged_init=True)
                else:
                    _, self.tree_for_ged = copy.deepcopy(self.reaction_delta_metric_history[reaction])
                    if self.tree_for_ged and self.track_ged_change:
                        total_folder = self.ged_output_folder + "/" + f"{self.ged_folder_prefix}_ged_folder"
                        if not os.path.exists(total_folder):
                            os.mkdir(total_folder)
                        path_name = self.target.replace('/', '_').replace("\\", "__")
                        output_folder = os.path.join(total_folder, path_name)
                        if not os.path.exists(output_folder):
                            os.mkdir(output_folder)
                        pathway_ged_plot_path = os.path.join(output_folder, f"{self.iterations}.png")
                        self.vis_ged_in_iterations(pathway_ged_plot_path)
        self.previous_route_ged_list = [self.reaction_delta_metric_history[reaction][0] for reaction in rxn_path]
        self.trival_ged_count = sum([1 for ged in self.previous_route_ged_list if ged <=0])
        self.latest_rxn_path = rxn_path
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
            # try:
            if rxn not in self.reaction_delta_metric_history.keys():
                # ged_weight_in_metric = self.ged_weight_in_metric_fun(self.iterations, start=self.ged_weight_start, end=self.ged_weight_end,
                #                     max_iter=self.build_tree_options.max_iterations)
                delta_metric, temp_tree, rxn_smi_mapped = _cal_delta_metric_in_ucb(self.tree_for_ged, rxn, metric_name=self.metric_name,
                                                                                   ged_weight=self.ged_weight_in_metric, tanimoto_weight=self.tanimoto_weight_in_metric)
                if self.use_strategy_lib:
                    if rxn_data["template"]["template_set"] == "strategy":
                        # ged = ged/self.rxn_depth_dict[strategy_template]
                        delta_metric = delta_metric/rxn_data["template"]["strategy_depth"]
                    
                self.reaction_delta_metric_history[rxn] = (delta_metric, temp_tree)
            else:
                delta_metric, temp_tree = copy.deepcopy(self.reaction_delta_metric_history[rxn])
                
            metric_weight = self.delta_metric_change_fun(self.iterations, start=self.ged_weight_start, end=self.ged_weight_end,
                                             max_iter=self.build_tree_options.max_iterations)
            # ged = min(ged, self.max_ged)
            # external_template_bonus = 100 if rxn_data["template"]["template_set"] == "external_template" else 0
            external_template_bonus = 0
            score = (1-metric_weight) * (q_sa + exploration_weight * u_sa) + metric_weight * delta_metric + external_template_bonus
            
            options.append((score, rxn))

        options.sort(key=lambda x: x[0], reverse=True)
        # self.previous_best_ged = self.reaction_delta_metric_history[options[0][1]][0] if options else 999
        # import ipdb; ipdb.set_trace();
        return options
    
    def vis_ged_in_iterations(self, pathway_ged_plot_path):
        try:
            self.tree_for_ged.pathway_ged_plot_path = pathway_ged_plot_path
            self.tree_for_ged.visualize_pathway_ged_changes()
        except:
            print("vis_ged_in_iterations failed")
    
    def _expand(self, chem_path: List[str]) -> None:
        """
        Expand the tree by running one-step retro prediction to a chemical node
        """
        leaf = chem_path[-1]
        # if self.iterations < self.strategy_stage:
        #     self.expand_one_options.retro_backend_options = [self.expand_one_options.retro_backend_options[0]]
        
        retro_results = self.expand_one(
            smiles=leaf,
            expand_one_options=self.expand_one_options
        )
        
        external_temps_results =[]
        if self.use_external_temps:
            external_temp_result = {'outcome': None, 'model_score': 1, 'normalized_model_score': 1,
                                    'template': {'index': 0, 'reaction_smarts': None,
                                                 'count': 0, 'num_references': 0, 'necessary_reagent': '', 'intra_only': True, 'dimer_only': False, 'template_set': 'external_template',
                                                 'attributes': {'ring_delta': 1.0, 'chiral_delta': 0}, '_id': 0, 'template_score': 1, 'template_rank': 0,
                                                 'tforms': ['uspto_original_consol_57'], 'num_examples': 0, 'tsources': ['external_template']}, 'reaction_data': None, 'retro_backend': 'template_relevance',
                                    'retro_model_name': 'external_template', 'models_predicted_by': [['template_relevance', 'uspto_original_consol_Roh', 0]], 'plausibility': 1.0, 'rms_molwt': 0, 'num_rings': 0,
                                    'scscore': 0, 'group_id': None, 'group_name': None, 'mapped_smiles': None, 'reacting_atoms': [6, 7],
                                    'selec_error': None, 'outcomes': None, 'mapped_outcomes': None, 'mapped_precursors': None, 'score': 0, 'rank': 1}
            for template in self.external_templates:
                external_temp_result["template"]["reaction_smarts"] = template
                external_temp_result["template"]["template_set"] = "external_template"
                outcomes = gen_precs(leaf, template)
                if outcomes:
                    # print(outcomes[0])
                    external_temp_result["outcome"] = outcomes[0]
                    external_temps_results.append(external_temp_result)
                else:
                    continue
        # import ipdb; ipdb.set_trace();
        # moved so that expanded nodes without retro_results are also marked "expanded" 
        self.tree.nodes[leaf]["expanded"] = True
        if not retro_results:
            # if no retro_results, then this node is done
            self.tree.nodes[leaf]["done"] = True
            return
        
        retro_results += external_temps_results
        if self.use_strategy_lib:
            strategy_results = []
            for result in retro_results:
                template_smarts = result["template"]["reaction_smarts"]
                if template_smarts in self.strategy_grouped_by_triggering_reaction.keys():
                    strategy_result = copy.deepcopy(result)
                    strategy_template = self.strategy_grouped_by_triggering_reaction[template_smarts][0]["strategy_template"]
                    # outcome, template["reaction_smarts", "template_set"], mapped_smiles
                    strategy_result["template"]["reaction_smarts"] = strategy_template
                    strategy_result["template"]["template_set"] = "strategy"
                    strategy_result["template"]["strategy_depth"] = self.strategy_grouped_by_triggering_reaction[template_smarts][0]["strategy_depth"]
                    # strategy_result["normalized_model_score"] = min(strategy_result["normalized_model_score"] * 1.01, 1)
                    strategy_result["normalized_model_score"] = result["normalized_model_score"]
                    
                    # strategy_result["normalized_model_score"] = strategy_result["normalized_model_score"] ** strategy_result["template"]["strategy_depth"]
                    # strategy_result["normalized_model_score"] = 1
                    outcomes = gen_precs(leaf, strategy_template)
                    if outcomes:
                        strategy_result["outcome"] = outcomes[0]
                    else:
                        continue
                    if strategy_result not in strategy_results:
                        strategy_results.append(strategy_result)
            retro_results += strategy_results[:int(len(strategy_results)/self.strategy_cut)]

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
                    plausibility=result["plausibility"],
                    num_examples=num_examples,
                    forward_score=result.get("forward_score"),
                    rms_molwt=result.get("rms_molwt"),
                    num_rings=result.get("num_rings"),
                    scscore=result.get("scscore"),
                    rank=result.get("rank"),
                    score=result.get("score", result["normalized_model_score"]),
                    class_num=result.get("class_num"),
                    class_name=result.get("class_name"),
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


def gen_precs(prod_smi_nomap: str, template: str) -> Optional[List[str]]:
    """
    Generate precursors using rdchiral template matching.
    
    Args:
        prod_smi_nomap: Product SMILES without atom mapping
        template: Reaction template SMARTS
        
    Returns:
        List of precursor SMILES or None if generation fails
    """
    try:
        # template_one = "(" + template.replace(">>", ")>>(") + ")"
        # rxn = rdchiralReaction(str(template_one))
        rxn = rdchiralReaction(template)
        prod = rdchiralReactants(prod_smi_nomap)
        precs = rdchiralRun(rxn, prod, combine_enantiomers=False)
        return precs
    except:
        return None

def map_one_reaction(reaction: str) -> str:
    """
    Map a reaction SMILES using RXNMapper if not already mapped.
    
    Args:
        reaction: Reaction SMILES string
        
    Returns:
        Mapped reaction SMILES string
    """
    if not has_mapping(reaction):
        rxn_mapper = RXNMapper()
        reactants, products = reaction.split('>>')
        try:
            can_reactants = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reactants.split('.')])
            can_products = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(p)) for p in products.split('.')])
            can_rsmi = f"{can_reactants}>>{can_products}"
            reaction = rxn_mapper.get_attention_guided_atom_maps([can_rsmi])[0]["mapped_rxn"]
        except:
            print(f"{reaction} failed")
    return reaction
    

def _cal_delta_metric_in_ucb(
    tree_for_ged: Optional[SynTree_ged],
    rxn: str,
    metric_name: str = "ged",
    ged_weight: float = 1,
    tanimoto_weight: float = 1
) -> Tuple[float, Optional[SynTree_ged], str]:
    """
    Calculate delta metric for a reaction in UCB scoring.
    
    The reaction has not been added into the tree yet.
    
    Args:
        tree_for_ged: Current synthesis tree (can be None for first reaction)
        rxn: Reaction SMILES string
        metric_name: Metric name to calculate
        ged_weight: Weight for GED in hybrid metrics
        tanimoto_weight: Weight for Tanimoto in hybrid metrics
        
    Returns:
        Tuple of (delta_metric, updated_tree, mapped_reaction_smiles)
    """
    rxn_smi = map_one_reaction(rxn)
    delta_metric = 0
    temp_tree = copy.deepcopy(tree_for_ged)
    if tree_for_ged:
        # print(f"add {rxn} in ucb")
        temp_tree.add_reaction_to_tree(rxn_smi)
        # temp_tree.add_subtrees()
        # try:
        new_edge_data_tagged = temp_tree.map_reaction_in_tree()
        # except:
        #     reaction_list = set()
        #     for product_node, reactant_node, edge_data in tree_for_ged.tree.edges(data=True):
        #         reaction_list.add(edge_data["reaction"].tagged_reaction_smiles)
            # print(reaction_list)
            # print(rxn in reaction_list)
            # import ipdb; ipdb.set_trace();
        edge_data, product_node, reactant_node = [(edge_data, product_node, reactant_node) for product_node, reactant_node, edge_data  \
                                        in temp_tree.tree.edges(data=True) if edge_data["reaction"].can_reaction_smiles == new_edge_data_tagged.can_reaction_smiles][0]
        if metric_name == "ged":
            delta_metric = temp_tree.cal_relative_edit_distance(edge_data, product_node, reactant_node, metric_name)
        elif metric_name == "tanimoto":
            delta_metric = temp_tree.cal_similarity_reduction(edge_data, product_node, reactant_node, metric_name)
        elif metric_name in ["SA_Score", "SC_Score"]:
            delta_metric = temp_tree.cal_one_step_mol_clx_reduction(edge_data, metric_name)
        elif "&" in metric_name:
            delta_metric = temp_tree.cal_hybrid_distance(edge_data, product_node, reactant_node, metric_name,
                                                         ged_weight=ged_weight, tanimoto_weight=tanimoto_weight)
            
        temp_tree.add_pairwise_attributes(metric_name=metric_name, ged_weight=ged_weight, tanimoto_weight=tanimoto_weight)
    else:
        if metric_name == "ged":
            delta_metric = _cal_one_step_edit_distance(rxn, metric_name)
        elif metric_name == "tanimoto":
            delta_metric = _cal_one_step_similarity_reduction(rxn, metric_name)
        elif metric_name in ["SA_Score", "SC_Score"]:
            delta_metric = _cal_one_step_mol_clx_reduction(rxn, metric_name)
        elif "&" in metric_name:
            delta_metric = _cal_hybrid_distance(rxn, metric_name, ged_weight=ged_weight, tanimoto_weight=tanimoto_weight)

    return delta_metric, temp_tree, rxn_smi
        

def map_all_reactions(route: List[str]) -> List[str]:
    """
    Map all reactions in a route using RXNMapper.
    
    Args:
        route: List of reaction SMILES strings
        
    Returns:
        List of mapped reaction SMILES strings
    """
    rxn_mapper = RXNMapper()
    route_with_mapping = []
    for rsmi in route:
        if not has_mapping(rsmi):
            reactants, products = rsmi.split('>>')
            try:
                can_reactants = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reactants.split('.')])
                can_products = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(p)) for p in products.split('.')])
                can_rsmi = f"{can_reactants}>>{can_products}"
                rsmi = rxn_mapper.get_attention_guided_atom_maps([can_rsmi])[0]["mapped_rxn"]
            except:
                print(f"{rsmi} failed")
                continue
        route_with_mapping.append(rsmi)
    
    return route_with_mapping

def extract_unique_smiles(data) -> Set[str]:
    """
    Extract unique SMILES strings from nested pathway data structure.
    
    Args:
        data: Nested dictionary or list containing pathway data
        
    Returns:
        Set of unique SMILES strings
    """
    unique_smiles = set()
    
    def traverse(node):
        if "smiles" in node:
            unique_smiles.add(node["smiles"])
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                traverse(child)
    
    if isinstance(data, list):
        for node in data:
            traverse(node)
    elif isinstance(data, dict):
        traverse(data)
    
    return unique_smiles

if __name__ == "__main__":
    targets = ["C1=CCCCC1"]
    for smiles in targets:
        prefix = smiles.replace('/', '_').replace("\\", "__")
        # controller = MCTS_ged(metric_name="SA_Score", track_ged_change=False, ged_folder_prefix=prefix, ged_weight_start=0, ged_weight_end=1,
        #                       ged_change_type="logarithmic_change", use_strategy_lib=True, use_external_temps=True)
        
        cano_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        controller = MCTS_ged(metric_name="SA_Score&SC_Score&tanimoto&ged", track_ged_change=False, ged_folder_prefix=prefix,
                              ged_weight_start=0, ged_weight_end=1, ged_change_type="logarithmic_change", use_external_temps=False)
        expand_one_options= ExpandOneOptions()
        retro_one_option_1 = RetroBackendOption()
        # For local mode, use an available model from mars_ged directory
        # Options: original_ged_threshold_0, simplified_ged_threshold_0, etc.
        # Set USE_LOCAL_RETRO=true to enable local mode (no Docker containers needed)
        retro_one_option_1.retro_model_name = "uspto_original_consol_Roh"  # or "uspto_original_consol_Roh" if extracted
        # retro_one_option_1 = RetroBackendOption(); retro_one_option_2 = RetroBackendOption()
        # retro_one_option_2.retro_model_name = "uspto_full_filtered_5_strategy_reactions_all_ged_consol"
        expand_one_options.retro_backend_options = [retro_one_option_1]
        # Disable return_reacting_atoms to avoid needing Atom Map API (which requires gateway)
        # This is optional - if gateway is running, you can set it to True
        expand_one_options.return_reacting_atoms = False
        # expand_one_options.retro_backend_options = [retro_one_option_2]
        build_tree_options = BuildTreeOptions()
        build_tree_options.max_depth = 10
        build_tree_options.max_iterations = 100
        # build_tree_options.custom_buyables = ["CC1(C)OCC(/C=C/I)CO1"]
        custom_buyables = ["Oc1c(C(N)=O)c(OC2=CC(O)=C(C([C@]23C)=O)C(C)=O)c3c(O)c1"]
        build_tree_options.custom_buyables = []
        for smiles in custom_buyables:
            cano_smiles_buyable = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            build_tree_options.custom_buyables.append(cano_smiles_buyable)
        paths, stats, graph = controller.get_buyable_paths(
            target=cano_smiles,
            expand_one_options=expand_one_options,
            build_tree_options=build_tree_options
        )
        print(stats)
    
        output_folder = "pathway_folder_Dec_27_combined"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not os.path.exists(os.path.join(output_folder, prefix)):
            os.mkdir(os.path.join(output_folder, prefix))
        with open(f"{output_folder}/{prefix}/paths.json", "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=4, ensure_ascii=False)
        with open(f"{output_folder}/{prefix}/stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        with open(f"{output_folder}/{prefix}/graph.json", "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=4, ensure_ascii=False)
        if paths:
            reactions = list(extract_unique_smiles(paths[0]))
            reactions = [smarts for smarts in reactions if ">>" in smarts]
            reactions = map_all_reactions(reactions)
            with open(f"{output_folder}/{prefix}/unique_smiles.json", "w", encoding="utf-8") as f:
                json.dump(reactions, f, indent=4, ensure_ascii=False)
        else:
            reactions = controller.latest_rxn_path
            reactions = map_all_reactions(reactions)
            with open(os.path.join(f"{output_folder}/{prefix}/failed_smiles.json"), "w", encoding="utf-8") as f:
                json.dump(reactions, f, indent=4, ensure_ascii=False)
        