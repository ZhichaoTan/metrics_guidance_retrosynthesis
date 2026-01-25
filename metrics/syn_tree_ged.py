"""
Synthesis Tree with Graph Edit Distance (GED) metrics.

This module extends SynTree with various distance and similarity metrics
for retrosynthetic pathways.
"""

import sys
import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Local imports
from ..datastructs.syn_tree import SynTree
from ..utils.chem_utils import has_mapping, get_atom_maps, get_largest_chemical, canonicalize, canonicalize_rsmi
from .GED_scorer import GED_scorer
from .linear_tree_dfs import extract_linear_trees_dfs
from .scscore import SCScorePrecursorPrioritizer

# External imports

# SA_Score import (try to find it in common locations)
try:
    import sascorer
except ImportError:
    # Try to add common SA_Score locations
    import sys
    sa_score_paths = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'RDKit', 'Contrib', 'SA_Score'),
        os.path.join(sys.prefix, 'share', 'RDKit', 'Contrib', 'SA_Score'),
    ]
    for path in sa_score_paths:
        if os.path.exists(path):
            sys.path.append(path)
            try:
                import sascorer
                break
            except ImportError:
                continue

logging.getLogger("transformers").setLevel(logging.ERROR)

class SynTree_ged(SynTree):
    """
    Extended synthesis tree with Graph Edit Distance and other metrics.

    Supports calculation of various metrics including:
    - GED (Graph Edit Distance)
    - Tanimoto similarity
    - SA_Score (Synthetic Accessibility Score)
    - SC_Score (Synthetic Complexity Score)
    - Hybrid metrics combining multiple measures
    """

    def __init__(self, route=None, pathway_ged_plot_path=None, track_atom_init=False, add_ged_init=False, one_step_ged_plot_path=None):
        if not all([has_mapping(rsmi) for rsmi in route]):
            route = map_all_reactions(route)
        super().__init__(route)
        if track_atom_init:
            self.track_atoms()
        self.linear_tree_list = None
        self.branch_in_tree = False
        if add_ged_init:
            self.add_pairwise_attributes()
        self.pathway_ged_plot_path = pathway_ged_plot_path
        self.one_step_ged_plot_path = one_step_ged_plot_path

    def _extract_linear_tree(self):
        if self._branch_in_tree():
            self.linear_tree_list = extract_linear_trees_dfs(self.tree)

    def _branch_in_tree(self):
        if not self.branch_in_tree:
            for node, out_deg in self.tree.out_degree():
                if out_deg > 1:
                    children = list(self.tree.successors(node))
                    if all(self.tree.out_degree(child) > 0 for child in children):
                        self.branch_in_tree = True
                        return True
        return self.branch_in_tree

    def map_reaction_in_tree(self): # the latest reaction has not been mapped to the target
        new_edge_data_tagged = None
        new_edge = None
        if not self.is_route_mapped():
            raise ValueError('All reactions must be atom-mapped')
        for node in list(nx.topological_sort(self.tree)):
            if self.tree.out_degree(node) == 0:
                self.leaf_tags.append(
                    tuple(get_atom_maps(
                        self.tree.nodes[node]['molecule'].tagged_smiles
                    ))
                )
                continue

            if self.tree.in_degree(node) == 0 and node != self.root:
                raise ValueError(f'Non-root node {node} is isolated')

            if node == self.root:
                self.tag_root_mol()

            if not all(hasattr(self.tree.nodes[child]['molecule'], "tagged_smiles") for child in self.tree.successors(node)):
                map_to_tag = self.get_map_to_tag(node)
                for child in self.tree.successors(node):
                    edge = (node, child)
                    if not hasattr(self.tree[edge[0]][edge[1]]['reaction'], "tagged_reaction_smiles"):
                        new_edge = edge
                    self.update_child_with_parent_tag(child, map_to_tag)
            self.update_edges_with_tagged_reaction_smiles(node)

        new_edge_data_tagged = self.tree[new_edge[0]][new_edge[1]]['reaction']

        return new_edge_data_tagged

    def remove_reaction_from_tree(self, reaction):
        can_reaction_smiles = canonicalize_rsmi(reaction)

        if can_reaction_smiles in self.can_route:
            self.can_route.remove(can_reaction_smiles)
        reactants, _, products = reaction.split(">")
        largest_prod = get_largest_chemical(products)
        can_product = canonicalize(largest_prod)
        product_node = (can_product, 1)
        reactant_nodes = []
        for reactant in reactants.split("."):
            can_reactant = canonicalize(reactant)

            if can_reactant in self.node_counter:
                current_idx = self.node_counter[can_reactant]
                reactant_node = (can_reactant, current_idx)

                if self.tree.has_node(reactant_node) and self.tree.has_edge(product_node, reactant_node):
                    reactant_nodes.append(reactant_node)

        for reactant_node in reactant_nodes:
            self.tree.remove_edge(product_node, reactant_node)
            self.tree.remove_node(reactant_node)

            reactant_name = reactant_node[0]
            if reactant_name in self.node_counter:
                self.node_counter[reactant_name] -= 1
                if self.node_counter[reactant_name] == 0:
                    del self.node_counter[reactant_name]

        nodes_to_remove = [node for node in self.tree.nodes if self.tree.in_degree(node) == 0 and node != self.root]
        self.tree.remove_nodes_from(nodes_to_remove)

    def add_pairwise_attributes(self, metric_name="ged", ged_weight=0, tanimoto_weight=0):
        self.one_step_delta_metric_list = []
        self.pairwise_delta_metric_list = []
        visited_reaction_dict = {}
        if metric_name == "ged":
            one_step_fun = self.cal_one_step_edit_distance
            pairwise_fun = self.cal_relative_edit_distance
        elif metric_name == "tanimoto":
            one_step_fun = self.cal_one_step_similarity_reduction
            pairwise_fun = self.cal_similarity_reduction
        elif metric_name in ["SA_Score", "SC_Score"]:
            one_step_fun = self.cal_one_step_mol_clx_reduction
        elif "&" in metric_name:
            one_step_fun = None
            pairwise_fun = self.cal_hybrid_distance
            ged_weight = ged_weight
            tanimoto_weight = tanimoto_weight
        else:
            return
        for product_node, reactant_node, edge_data in self.tree.edges(data=True):
            if not all(key in edge_data.keys() for key in ('one_step_delta_metric', 'pairwise_delta_metric')):
                tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
                if tagged_reaction_smiles not in visited_reaction_dict.keys():
                    edge_data['one_step_delta_metric'] = one_step_fun(edge_data, metric_name=metric_name) if one_step_fun else 0

                    if (ged_weight or tanimoto_weight) and ("ged" in metric_name or "tanimoto" in metric_name):
                        edge_data['pairwise_delta_metric'] = pairwise_fun(edge_data, product_node, reactant_node, metric_name=metric_name,
                                                                          ged_weight=ged_weight, tanimoto_weight=tanimoto_weight)
                    else: edge_data['pairwise_delta_metric'] = pairwise_fun(edge_data, product_node, reactant_node, metric_name=metric_name) if \
                                                         metric_name not in ["SA_Score", "SC_Score"] else edge_data['one_step_delta_metric']

                    self.one_step_delta_metric_list.append(edge_data['one_step_delta_metric'])
                    self.pairwise_delta_metric_list.append(edge_data['pairwise_delta_metric'])
                    visited_reaction_dict[tagged_reaction_smiles] = {"one_step_delta_metric": edge_data['one_step_delta_metric'],
                                                                    "pairwise_delta_metric": edge_data["pairwise_delta_metric"]}
                else:
                    edge_data['one_step_delta_metric'] = visited_reaction_dict[tagged_reaction_smiles]["one_step_delta_metric"]
                    edge_data['pairwise_delta_metric'] = visited_reaction_dict[tagged_reaction_smiles]["pairwise_delta_metric"] # Same reaction usually has only one pathway_ged

    # add the metric as an attribute of node
    def add_node_attributes(self, metric_name="tanimoto"):
        self.node_metric_list = []
        visited_molecule_dict = {}

        if metric_name == "tanimoto":
            node_fun = self.cal_node_similarity
            sc_score_model = None
        elif metric_name in ["SA_Score", "SC_Score"]:
            node_fun = self.cal_node_complexity
            # Pre-instantiate SC_Score model to avoid repeated instantiation
            if metric_name == "SC_Score":
                sc_score_model = SCScorePrecursorPrioritizer()
            else:
                sc_score_model = None
        else:
            return

        root_node = self.root
        root_smiles = self.tree.nodes[root_node]["molecule"].tagged_smiles

        for node in self.tree.nodes:
            if node == root_node:
                # Set default values for root node based on metric type
                if metric_name == "tanimoto":
                    self.tree.nodes[node][f"node_{metric_name}"] = 1.0  # Root node similarity to itself is 1
                elif metric_name in ["SA_Score", "SC_Score"]:
                    node_smiles = self.tree.nodes[node]["molecule"].tagged_smiles
                    self.tree.nodes[node][f"node_{metric_name}"] = node_fun(node_smiles, root_smiles, metric_name, sc_score_model)
                self.node_metric_list.append(self.tree.nodes[node][f"node_{metric_name}"])
                continue

            node_smiles = self.tree.nodes[node]["molecule"].tagged_smiles

            # Use canonical smiles as key to avoid duplicate calculations
            can_node_smiles = canonicalize(node_smiles)
            if can_node_smiles not in visited_molecule_dict:
                node_metric = node_fun(node_smiles, root_smiles, metric_name, sc_score_model)
                self.tree.nodes[node][f"node_{metric_name}"] = node_metric
                self.node_metric_list.append(node_metric)
                visited_molecule_dict[can_node_smiles] = node_metric
            else:
                self.tree.nodes[node][f"node_{metric_name}"] = visited_molecule_dict[can_node_smiles]
                self.node_metric_list.append(visited_molecule_dict[can_node_smiles])

    def cal_node_similarity(self, node_smiles, root_smiles, metric_name, sc_score_model=None):
        """Calculate Tanimoto similarity between node and root molecule."""
        node_mol = Chem.MolFromSmiles(node_smiles)
        root_mol = Chem.MolFromSmiles(root_smiles)

        if node_mol is None or root_mol is None:
            return 0.0

        node_fp = get_fingerprint(node_mol)
        root_fp = get_fingerprint(root_mol)

        similarity = DataStructs.TanimotoSimilarity(node_fp, root_fp)
        return similarity

    def cal_node_complexity(self, node_smiles, root_smiles, metric_name, sc_score_model=None):
        """Calculate molecular complexity score (SA_Score or SC_Score)."""
        node_mol = Chem.MolFromSmiles(node_smiles)

        if node_mol is None:
            return 0.0

        if metric_name == "SA_Score":
            complexity = sascorer.calculateScore(node_mol)
        elif metric_name == "SC_Score":
            if sc_score_model is None:
                sc_score_model = SCScorePrecursorPrioritizer()
            complexity = sc_score_model._get_one_score(Chem.MolToSmiles(node_mol))
        else:
            return 0.0

        return complexity

    def cal_one_step_edit_distance(self, edge_data, metric_name):
        tagged_reaction_smiles = edge_data["reaction"].reaction_smiles
        one_step_ged = _cal_one_step_edit_distance(tagged_reaction_smiles, metric_name)
        return one_step_ged

    def cal_one_step_similarity_reduction(self, edge_data, metric_name):
        tagged_reaction_smiles = edge_data["reaction"].reaction_smiles
        similarity_reduction = _cal_one_step_similarity_reduction(tagged_reaction_smiles, metric_name)
        return similarity_reduction

    def cal_one_step_mol_clx_reduction(self, edge_data, metric_name):
        tagged_reaction_smiles = edge_data["reaction"].reaction_smiles
        similarity_reduction = _cal_one_step_mol_clx_reduction(tagged_reaction_smiles, metric_name)
        return similarity_reduction

    def cal_relative_edit_distance(self, edge_data, product_node, reactant_node, metric_name, ged_weight=0, tanimoto_weight=0):
        tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
        reactants, products = tagged_reaction_smiles.split(">") # Reagents are generally absent
        reference_molecule = self.get_reference_mol(product_node, reactant_node)

        virtual_rxn_react = reactants + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        virtual_rxn_prod = products + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        absolute_diff = _cal_one_step_edit_distance(virtual_rxn_react, metric_name)
        pathway_ged = absolute_diff - _cal_one_step_edit_distance(virtual_rxn_prod, metric_name)
        return pathway_ged

    def cal_similarity_reduction(self, edge_data, product_node, reactant_node, metric_name, ged_weight=1, tanimoto_weight=1):
        tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
        reactants, products = tagged_reaction_smiles.split(">")
        reference_molecule = self.get_reference_mol(product_node, reactant_node)
        virtual_rxn_react = reactants + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        virtual_rxn_prod = products + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        absolute_diff = _cal_one_step_similarity_reduction(virtual_rxn_react, metric_name)
        pathway_ged = _cal_one_step_similarity_reduction(virtual_rxn_prod, metric_name) - absolute_diff
        return pathway_ged

    def cal_hybrid_distance(self, edge_data, product_node, reactant_node, metric_name, ged_weight=1, tanimoto_weight=1):
        sub_metric_list = metric_name.split("&")
        hybrid_distance = 0
        for sub_metric_name in sub_metric_list:
            if sub_metric_name == "ged":
                sub_metric = self.cal_relative_edit_distance(edge_data, product_node, reactant_node, sub_metric_name) * ged_weight
            elif sub_metric_name == "tanimoto":
                sub_metric = self.cal_similarity_reduction(edge_data, product_node, reactant_node, sub_metric_name) * tanimoto_weight
            elif sub_metric_name in ["SA_Score", "SC_Score"]:
                sub_metric = self.cal_one_step_mol_clx_reduction(edge_data, sub_metric_name)
            hybrid_distance += sub_metric

        return hybrid_distance

    def compute_absolute_ged(self, ged_key="pairwise_delta_metric"):
        absolute_ged = {}
        for node in self.tree.nodes:
            if node == self.root:
                absolute_ged[node] = 0
            else:
                path = nx.shortest_path(self.tree, source=self.root, target=node)
                absolute_ged[node] = sum(self.tree[u][v][ged_key] for u, v in zip(path[:-1], path[1:]))

        nx.set_node_attributes(self.tree, absolute_ged, f"absolute_{ged_key}")

    def compute_deepest_path_ged(self, ged_key="pairwise_delta_metric", metric_name=None):
        longest_path = []
        max_depth = -1

        for node in self.tree.nodes:
            if node == self.root:
                continue
            path = nx.shortest_path(self.tree, source=self.root, target=node)
            if len(path) > max_depth:
                max_depth = len(path)
                longest_path = path

        # For node-based metrics, return node attributes directly
        if metric_name in ["tanimoto", "SA_Score", "SC_Score"]:
            node_attr_key = f"node_{metric_name}"
            # Check if node attributes are computed, compute if not
            if node_attr_key not in next(iter(self.tree.nodes(data=True)))[1]:
                self.add_node_attributes(metric_name=metric_name)

            node_values = [self.tree.nodes[n][node_attr_key] for n in longest_path]
            return node_values

        # Otherwise use original logic, return absolute GED values of edges
        absolute_ged_key = f"absolute_{ged_key}"

        if absolute_ged_key not in next(iter(self.tree.nodes(data=True)))[1]:
            self.compute_absolute_ged(ged_key)

        ged_values = [self.tree.nodes[n][absolute_ged_key] for n in longest_path]

        return ged_values

    def get_reference_mol(self, product_node, reactant_node):
        if self._branch_in_tree():
            self._extract_linear_tree()
            linear_tree_index = _get_linear_tree_index(self.linear_tree_list, product_node, reactant_node)
            reference_molecule = [node for node, in_deg in self.linear_tree_list[linear_tree_index].in_degree() if in_deg == 0][0]
        else:
            reference_molecule = [node for node, in_deg in self.tree.in_degree() if in_deg == 0][0]
        return reference_molecule

    def visualize_pathway_ged_changes(self, output_number=False):
        ged_key = "pairwise_delta_metric"
        self.compute_absolute_ged(ged_key=ged_key)
        _plot_absolute_ged(self.tree, self.root, ged_key, self.pathway_ged_plot_path, output_number=output_number)

    def visualize_one_step_ged_changes(self):
        ged_key = "pairwise_delta_metric"
        self.compute_absolute_ged(ged_key=ged_key)
        _plot_absolute_ged(self.tree, self.root, ged_key, self.one_step_ged_plot_path)

def _plot_absolute_ged(tree, root, ged_key, ged_plot_path, output_number=False, y_name="Graph Edit Distance"):
    depths = nx.single_source_shortest_path_length(tree, root)

    mks = 6
    fs = 7
    label_fs = 7
    synthia_orange = np.array([255, 160, 0, 255]) / 255
    cdraw_orange_face = np.array([252, 236, 207, 255]) / 255
    cdraw_orange_edge = np.array([245, 191, 94, 255]) / 255
    grid_grey = np.array([225, 225, 225, 255]) / 255

    sorted_depths = sorted(set(depths.values()), reverse=True)
    node_key = f"absolute_{ged_key}" if ged_key in ["pairwise_delta_metric", "one_step_delta_metric"] else ged_key
    node_positions = {node: (depths[node], tree.nodes[node][node_key]) for node in tree.nodes if node_key in tree.nodes[node] and tree.nodes[node][node_key] != 999}

    plt.figure(figsize=(8, 5))

    for node, (depth, absolute_ged) in node_positions.items():
        plt.scatter(depth, absolute_ged, color='black', marker="o", s=mks, facecolors="k", edgecolors="k")
        if output_number:
            print(depth, absolute_ged)
    for parent, child in tree.edges():
        if parent in node_positions and child in node_positions:
            parent_depth, parent_ged = node_positions[parent]
            child_depth, child_ged = node_positions[child]
            plt.plot([parent_depth, child_depth], [parent_ged, child_ged], linestyle='-', color="grey")

    ax = plt.gca()
    plt.gca().invert_xaxis()

    ax.set_xticks(range(len(sorted_depths)))
    ax.set_xticklabels(range(0,len(sorted_depths)), fontsize=fs)

    ax.set_ylabel(y_name, fontsize=label_fs)
    ax.set_xlabel("Depth", fontsize=label_fs)
    if ged_plot_path:
        plt.savefig(ged_plot_path, bbox_inches="tight")
        plt.close()

def _get_linear_tree_index(linear_tree_list, product_node, reactant_node):
    for tree_index, tree in enumerate(linear_tree_list):
        if product_node in tree.nodes() and reactant_node in tree.nodes:
            return tree_index

    raise ValueError("The edge is not in any linear tree")

def _cal_one_step_edit_distance(template, metric_name=None):
    scorer = GED_scorer(template)
    one_step_ged = scorer.calculate_score()
    return one_step_ged

def get_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def _cal_one_step_similarity_reduction(reaction, metric_name=None):
    reactants, products = reaction.split(">>")
    reactant_list = reactants.split(".")
    product_list = products.split(".")
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_list if Chem.MolFromSmiles(s)]
    product_mols = [Chem.MolFromSmiles(s) for s in product_list if Chem.MolFromSmiles(s)]
    reactant_fps = [get_fingerprint(mol) for mol in reactant_mols]
    product_fps = [get_fingerprint(mol) for mol in product_mols]
    tanimoto_matrix = np.array([
        [DataStructs.TanimotoSimilarity(p_fp, r_fp) for r_fp in reactant_fps]
        for p_fp in product_fps
    ])
    max_tanimoto = np.max(tanimoto_matrix)

    return max_tanimoto

def _cal_one_step_mol_clx_reduction(reaction, metric_name):
    reactants, products = reaction.split(">>")
    reactant_list = reactants.split("."); product_list = products.split(".")
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_list if Chem.MolFromSmiles(s)]
    product_mols = [Chem.MolFromSmiles(s) for s in product_list if Chem.MolFromSmiles(s)]
    if metric_name == "SA_Score":
        reactant_sa = [sascorer.calculateScore(mol) for mol in reactant_mols]
        product_sa = [sascorer.calculateScore(mol) for mol in product_mols]
        delta_sa_matrix = np.array([[p - r for r in reactant_sa] for p in product_sa])
        delta_metric = np.min(delta_sa_matrix)

    elif metric_name == "SC_Score":
        sc_score_model = SCScorePrecursorPrioritizer()
        reactant_sc = [sc_score_model._get_one_score(Chem.MolToSmiles(mol)) for mol in reactant_mols]
        product_sc = [sc_score_model._get_one_score(Chem.MolToSmiles(mol)) for mol in product_mols]
        delta_sc_matrix = np.array([[p - r for r in reactant_sc] for p in product_sc])
        delta_metric = np.min(delta_sc_matrix)

    return delta_metric

def map_all_reactions(route):
    from rxnmapper import RXNMapper
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
