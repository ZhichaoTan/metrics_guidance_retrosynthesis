"""
Synthesis Tree with Complexity and/or Distance metrics, suppoorting both static pathway analysis and dynamic metric calculation in MCTS.
This module extends SynTree with distance and similarity metrics for synthetic pathways.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from datastructs.syn_tree import SynTree
from utils.chem_utils import has_mapping, get_atom_maps, get_largest_chemical, canonicalize, canonicalize_rsmi
from metrics.GED_scorer import GED_scorer
from metrics.linear_tree_dfs import extract_linear_trees_dfs
from metrics.scscore import SCScorePrecursorPrioritizer
from utils.chem_utils import map_all_reactions
from metrics.sascore import calculateScore


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

    def __init__(self, route=None, track_atom_init=False, add_ged_init=False):
        if not all([has_mapping(rsmi) for rsmi in route]):
            route = map_all_reactions(route)
        super().__init__(route)
        if track_atom_init:
            self.track_atoms()
            
        self.linear_tree_list = None
        self.branch_in_tree = False
        if add_ged_init:
            self.add_pairwise_attributes(metric_name="ged", ged_weight=1, tanimoto_weight=1)

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

    def add_pairwise_attributes(self, metric_name="ged", ged_weight=0, tanimoto_weight=0):
        """
        Add metric attributes to tree edges.
        
        Args:
            metric_name: One of "ged", "tanimoto", "SA_Score", "SC_Score", or hybrid metrics like "SA_Score&ged"
            ged_weight: Weight of the GED metric in a hybrid metric (only used for hybrid metrics)
            tanimoto_weight: Weight of the Tanimoto similarity metric in a hybrid metric (only used for hybrid metrics)
        """
        self.one_step_delta_metric_list = []
        self.pairwise_delta_metric_list = []
        visited_reaction_dict = {}
        
        # Select appropriate calculation functions based on metric type
        one_step_fun, pairwise_fun = self._select_metric_functions(metric_name)
        
        # Process each edge in the tree
        for product_node, reactant_node, edge_data in self.tree.edges(data=True):
            # Skip if metrics already computed
            if all(key in edge_data.keys() for key in ('one_step_delta_metric', 'pairwise_delta_metric')):
                continue
            
            tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
            
            # Use cached values if reaction was already processed, because the same reaction usually has the same metric value in a pathway
            if tagged_reaction_smiles in visited_reaction_dict:
                cached_metrics = visited_reaction_dict[tagged_reaction_smiles]
                edge_data['one_step_delta_metric'] = cached_metrics['one_step_delta_metric']
                edge_data['pairwise_delta_metric'] = cached_metrics['pairwise_delta_metric']
            else:
                # Calculate one-step delta metric
                one_step_metric = self._calculate_one_step_metric(
                    one_step_fun, edge_data, metric_name
                )
                
                # Calculate pairwise delta metric
                pairwise_metric = self._calculate_pairwise_metric(
                    pairwise_fun, one_step_fun, edge_data, product_node, 
                    reactant_node, metric_name, ged_weight, tanimoto_weight
                )
                
                # Store metrics in edge data
                edge_data['one_step_delta_metric'] = one_step_metric
                edge_data['pairwise_delta_metric'] = pairwise_metric
                
                # Cache results for reuse
                visited_reaction_dict[tagged_reaction_smiles] = {
                    'one_step_delta_metric': one_step_metric,
                    'pairwise_delta_metric': pairwise_metric
                }
                
                # Append to lists for tracking the metric trajectory
                self.one_step_delta_metric_list.append(one_step_metric)
                self.pairwise_delta_metric_list.append(pairwise_metric)
    
    def _select_metric_functions(self, metric_name):
        """
        Select appropriate calculation functions based on metric name.
        
        Returns:
            tuple: (one_step_fun, pairwise_fun) - calculation functions or None
        """
        if metric_name == "ged":
            return self.cal_one_step_edit_distance, self.cal_relative_edit_distance
        elif metric_name == "tanimoto":
            return self.cal_one_step_similarity_reduction, self.cal_similarity_reduction
        elif metric_name in ["SA_Score", "SC_Score"]:
            return self.cal_one_step_mol_clx_reduction, None
        elif "&" in metric_name:
            return None, self.cal_hybrid_distance
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")
    
    def _calculate_one_step_metric(self, one_step_fun, edge_data, metric_name):
        """
        Calculate one-step delta metric for an edge.
        
        Returns:
            float: One-step metric value, or -999 if function is None
        """
        if one_step_fun is None:
            return -999
        return one_step_fun(edge_data, metric_name=metric_name)
    
    def _calculate_pairwise_metric(self, pairwise_fun, one_step_fun, edge_data, 
                                   product_node, reactant_node, metric_name,
                                   ged_weight, tanimoto_weight):
        """
        Calculate pairwise delta metric for an edge.
        
        Returns:
            float: Pairwise metric value
        """
        # For complexity metrics (SA_Score, SC_Score), use one-step metric as pairwise metric
        if metric_name in ["SA_Score", "SC_Score"]:
            return self._calculate_one_step_metric(one_step_fun, edge_data, metric_name)
        
        # For hybrid metrics or metrics with weights, use pairwise function with weights
        if pairwise_fun is not None:
            # Check if weights are specified for hybrid metrics
            has_weights = (ged_weight != 0 or tanimoto_weight != 0)
            is_weighted_metric = ("ged" in metric_name or "tanimoto" in metric_name)
            
            if has_weights and is_weighted_metric:
                return pairwise_fun(
                    edge_data, product_node, reactant_node, 
                    metric_name=metric_name,
                    ged_weight=ged_weight, 
                    tanimoto_weight=tanimoto_weight
                )
            else:
                return pairwise_fun(
                    edge_data, product_node, reactant_node, 
                    metric_name=metric_name
                )
        
        # Fallback: should not reach here for valid metric names
        return -999

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
        reactants, products = tagged_reaction_smiles.split(">>") # Reagents are generally absent
        reference_molecule = self.get_reference_mol(product_node, reactant_node)

        virtual_rxn_react = reactants + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        virtual_rxn_prod = products + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        absolute_diff = _cal_one_step_edit_distance(virtual_rxn_react, metric_name)
        pathway_ged = absolute_diff - _cal_one_step_edit_distance(virtual_rxn_prod, metric_name)
        return pathway_ged

    def cal_similarity_reduction(self, edge_data, product_node, reactant_node, metric_name, ged_weight=1, tanimoto_weight=1):
        tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
        reactants, products = tagged_reaction_smiles.split(">>")
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

    def get_reference_mol(self, product_node, reactant_node):
        if self._branch_in_tree():
            self._extract_linear_tree()
            linear_tree_index = _get_linear_tree_index(self.linear_tree_list, product_node, reactant_node)
            reference_molecule = [node for node, in_deg in self.linear_tree_list[linear_tree_index].in_degree() if in_deg == 0][0]
        else:
            reference_molecule = [node for node, in_deg in self.tree.in_degree() if in_deg == 0][0]
        return reference_molecule

    def visualize_pathway_metric_changes(self, metric_trajectory_plot_path="metric_trajectory_plot.png", output_number=False):
        ged_key = "pairwise_delta_metric"
        self.compute_absolute_ged(ged_key=ged_key)
        _plot_absolute_metric(self.tree, self.root, ged_key, metric_trajectory_plot_path, output_number=output_number)

def _plot_absolute_metric(tree, root, plot_key, plot_path, output_number=False):
    depths = nx.single_source_shortest_path_length(tree, root)

    mks = 6
    fs = 7
    label_fs = 7
    
    sorted_depths = sorted(set(depths.values()), reverse=True)
    node_key = f"absolute_{plot_key}"
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

    ax.set_ylabel(plot_key, fontsize=label_fs)
    ax.set_xlabel("Depth", fontsize=label_fs)
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
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
        reactant_sa = [calculateScore(mol) for mol in reactant_mols]
        product_sa = [calculateScore(mol) for mol in product_mols]
        delta_sa_matrix = np.array([[p - r for r in reactant_sa] for p in product_sa])
        delta_metric = np.min(delta_sa_matrix)

    elif metric_name == "SC_Score":
        sc_score_model = SCScorePrecursorPrioritizer()
        reactant_sc = [sc_score_model._get_one_score(Chem.MolToSmiles(mol)) for mol in reactant_mols]
        product_sc = [sc_score_model._get_one_score(Chem.MolToSmiles(mol)) for mol in product_mols]
        delta_sc_matrix = np.array([[p - r for r in reactant_sc] for p in product_sc])
        delta_metric = np.min(delta_sc_matrix)

    return delta_metric

def _cal_hybrid_distance(reaction, metric_name, ged_weight=1, tanimoto_weight=1):
    sub_metric_list = metric_name.split("&")
    hybrid_distance = 0
    for sub_metric_name in sub_metric_list:
        if sub_metric_name in ["SA_Score", "SC_Score"]:
            sub_metric = _cal_one_step_mol_clx_reduction(reaction, sub_metric_name)
        elif sub_metric_name == "ged":
            sub_metric = _cal_one_step_edit_distance(reaction, sub_metric_name) * ged_weight
        elif sub_metric_name == "tanimoto":
            sub_metric = _cal_one_step_similarity_reduction(reaction, sub_metric_name) * tanimoto_weight
        hybrid_distance += sub_metric
    return hybrid_distance