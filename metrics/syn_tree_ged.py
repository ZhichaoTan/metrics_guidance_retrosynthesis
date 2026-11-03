"""
Synthesis Tree with Complexity and/or Distance metrics, supporting both static pathway analysis and dynamic metric calculation in MCTS.
This module extends SynTree with complexity and distance metrics for synthetic pathways.
"""

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from datastructs.syn_tree import SynTree
from utils.chem_utils import has_mapping, get_atom_maps, canonicalize, canonicalize_rsmi
from metrics.ged_scorer import GEDScorer
from metrics.linear_tree_dfs import extract_linear_trees_dfs
from metrics.scscore import SCScorePrecursorPrioritizer
from utils.chem_utils import map_all_reactions
from metrics.sascore import calculateScore


# Process-level SCScore model singleton. SCScorePrecursorPrioritizer() loads a pickled model on
# construction; instantiating it per call (as the sc_score branches previously did) reloads the model
# on every candidate and dominates sc_score cost. One lazily-created instance gives identical output.
_SC_SCORE_MODEL = None


def _get_sc_score_model():
    global _SC_SCORE_MODEL
    if _SC_SCORE_MODEL is None:
        _SC_SCORE_MODEL = SCScorePrecursorPrioritizer()
    return _SC_SCORE_MODEL


class SynTreeGED(SynTree):
    """
    Extended synthesis tree with Graph Edit Distance and other metrics.

    Supports calculation of various metrics including:
    - GED (Graph Edit Distance)
    - Tanimoto similarity
    - sa_score (Synthetic Accessibility Score)
    - sc_score (Synthetic Complexity Score)
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
            self.add_pairwise_attributes(metric_name="ged")

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

    def add_pairwise_attributes(self, metric_name="ged"):
        """
        Add metric attributes to tree edges.

        Args:
            metric_name: One of "ged", "tanimoto", "sa_score", "sc_score".
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
                    reactant_node, metric_name
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
            return self.cal_one_step_max_similarity, self.cal_similarity_reduction
        elif metric_name in ["sa_score", "sc_score"]:
            return self.cal_one_step_complexity_reduction, None
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
                                   product_node, reactant_node, metric_name):
        """
        Calculate pairwise delta metric for an edge.

        Returns:
            float: Pairwise metric value
        """
        # For complexity metrics (sa_score, sc_score), use the one-step metric as the pairwise metric.
        if metric_name in ["sa_score", "sc_score"]:
            return self._calculate_one_step_metric(one_step_fun, edge_data, metric_name)

        if pairwise_fun is not None:
            return pairwise_fun(edge_data, product_node, reactant_node, metric_name=metric_name)

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

        for product_node, reactant_node, edge_data in list(self.tree.edges(data=True)):
            rxn = edge_data.get("reaction")
            if rxn is None or rxn.can_reaction_smiles != can_reaction_smiles:
                continue
            if not self.tree.has_edge(product_node, reactant_node):
                continue

            self.tree.remove_edge(product_node, reactant_node)

            if self.tree.out_degree(reactant_node) == 0 and self.tree.in_degree(reactant_node) == 0:
                self.tree.remove_node(reactant_node)
                reactant_name = reactant_node[0]
                if reactant_name in self.node_counter:
                    self.node_counter[reactant_name] -= 1
                    if self.node_counter[reactant_name] <= 0:
                        del self.node_counter[reactant_name]

        nodes_to_remove = [
            node for node in self.tree.nodes
            if node != self.root and self.tree.in_degree(node) == 0
        ]
        self.tree.remove_nodes_from(nodes_to_remove)
        
    # add the metric as an attribute of node
    def add_node_attributes(self, metric_name="tanimoto"):
        self.node_metric_list = []
        visited_molecule_dict = {}

        if metric_name == "tanimoto":
            node_fun = self.cal_node_similarity
            sc_score_model = None
        elif metric_name in ["sa_score", "sc_score"]:
            node_fun = self.cal_node_complexity
            if metric_name == "sc_score":
                sc_score_model = _get_sc_score_model()
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
                elif metric_name in ["sa_score", "sc_score"]:
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
        """Calculate molecular complexity score (sa_score or sc_score)."""
        node_mol = Chem.MolFromSmiles(node_smiles)

        if node_mol is None:
            return 0.0

        if metric_name == "sa_score":
            complexity = calculateScore(node_mol)
        elif metric_name == "sc_score":
            if sc_score_model is None:
                sc_score_model = _get_sc_score_model()
            complexity = sc_score_model._get_one_score(Chem.MolToSmiles(node_mol))
        else:
            return 0.0

        return complexity

    def cal_one_step_edit_distance(self, edge_data, metric_name):
        reaction_smiles = edge_data["reaction"].reaction_smiles
        one_step_ged = cal_reaction_edit_distance(reaction_smiles)
        return one_step_ged

    def cal_one_step_max_similarity(self, edge_data, metric_name):
        reaction_smiles = edge_data["reaction"].reaction_smiles
        max_similarity = cal_max_tanimoto_similarity(reaction_smiles)
        return max_similarity

    def cal_one_step_complexity_reduction(self, edge_data, metric_name):
        reaction_smiles = edge_data["reaction"].reaction_smiles
        complexity_reduction = cal_complexity_reduction(reaction_smiles, metric_name)
        return complexity_reduction

    def cal_relative_edit_distance(self, edge_data, product_node, reactant_node, metric_name):
        tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
        reactants, products = tagged_reaction_smiles.split(">>") # Reagents are generally absent
        reference_molecule = self.get_reference_mol(product_node, reactant_node)

        virtual_rxn_react = reactants + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        virtual_rxn_prod = products + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        reactant_ged = cal_reaction_edit_distance(virtual_rxn_react)
        pathway_ged = reactant_ged - cal_reaction_edit_distance(virtual_rxn_prod)
        return pathway_ged

    def cal_similarity_reduction(self, edge_data, product_node, reactant_node, metric_name):
        tagged_reaction_smiles = edge_data["reaction"].tagged_reaction_smiles
        reactants, products = tagged_reaction_smiles.split(">>")
        reference_molecule = self.get_reference_mol(product_node, reactant_node)
        virtual_rxn_react = reactants + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        virtual_rxn_prod = products + ">>" + self.tree.nodes[reference_molecule]["molecule"].tagged_smiles
        reactant_similarity = cal_max_tanimoto_similarity(virtual_rxn_react)
        similarity_reduction = cal_max_tanimoto_similarity(virtual_rxn_prod) - reactant_similarity
        return similarity_reduction

    def get_reference_mol(self, product_node, reactant_node):
        if self._branch_in_tree():
            self._extract_linear_tree()
            linear_tree_index = _get_linear_tree_index(self.linear_tree_list, product_node, reactant_node)
            reference_molecule = [node for node, in_deg in self.linear_tree_list[linear_tree_index].in_degree() if in_deg == 0][0]
        else:
            reference_molecule = [node for node, in_deg in self.tree.in_degree() if in_deg == 0][0]
        return reference_molecule

def _get_linear_tree_index(linear_tree_list, product_node, reactant_node):
    for tree_index, tree in enumerate(linear_tree_list):
        if product_node in tree.nodes() and reactant_node in tree.nodes:
            return tree_index

    raise ValueError("The edge is not in any linear tree")

def cal_reaction_edit_distance(reaction_smiles):
    """Graph edit distance between the reactants and the products of an atom-mapped reaction."""
    scorer = GEDScorer(reaction_smiles)
    one_step_ged = scorer.calculate_score()
    return one_step_ged

def get_fingerprint(mol):
    """Return the 2048-bit Morgan (radius 2) fingerprint for ``mol``."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def cal_max_tanimoto_similarity(reaction_smiles):
    """Largest Tanimoto similarity between a product and a reactant of the reaction."""
    reactants, products = reaction_smiles.split(">>")
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

def cal_complexity_reduction(reaction_smiles, metric_name):
    """Smallest decrease of ``metric_name`` (sa_score or sc_score) from a product to a reactant."""
    reactants, products = reaction_smiles.split(">>")
    reactant_list = reactants.split("."); product_list = products.split(".")
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_list if Chem.MolFromSmiles(s)]
    product_mols = [Chem.MolFromSmiles(s) for s in product_list if Chem.MolFromSmiles(s)]
    if metric_name == "sa_score":
        reactant_sa = [calculateScore(mol) for mol in reactant_mols]
        product_sa = [calculateScore(mol) for mol in product_mols]
        delta_sa_matrix = np.array([[p - r for r in reactant_sa] for p in product_sa])
        complexity_reduction = np.min(delta_sa_matrix)

    elif metric_name == "sc_score":
        sc_score_model = _get_sc_score_model()
        reactant_sc = [sc_score_model._get_one_score(Chem.MolToSmiles(mol)) for mol in reactant_mols]
        product_sc = [sc_score_model._get_one_score(Chem.MolToSmiles(mol)) for mol in product_mols]
        delta_sc_matrix = np.array([[p - r for r in reactant_sc] for p in product_sc])
        complexity_reduction = np.min(delta_sc_matrix)

    return complexity_reduction
