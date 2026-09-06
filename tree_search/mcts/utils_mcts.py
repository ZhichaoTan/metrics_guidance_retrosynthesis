"""
The following implementation is based on the code from: https://gitlab.com/mlpds_mit/askcosv2
"""

import itertools
import networkx as nx
import numpy as np
import uuid
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from metrics.syn_tree_ged import (
    _cal_one_step_edit_distance,
    _cal_one_step_similarity_reduction,
    _cal_one_step_mol_clx_reduction,
)
from utils.chem_utils import map_one_reaction

NIL_UUID = "00000000-0000-0000-0000-000000000000"
NODE_LINK_ATTRS = {
    "source": "from",
    "target": "to",
    "name": "id",
    "key": "key",
    "link": "edges",
}

OUTPUT_KEYS = {
    "chemical": [
        "smiles",
        "id",
        "as_reactant",
        "as_product",
        "ppg",
        "properties",
        "terminal",
    ],
    "reaction": [
        "smiles",
        "id",
        "plausibility",
        "forward_score",
        "template_score",
        "template_tuples",
        "tforms",
        "tsources",
        "num_examples",
        "necessary_reagent",
        "precursor_smiles",
        "rms_molwt",
        "num_rings",
        "scscore",
        "rank",
        "class_num",
        "class_name",
    ],
}

# Map from keys used by tree builder graph to name used in pathways
PATH_KEY_DICT = {
    "smiles": "smiles",
    "type": "type",
    "id": "id",
    "as_reactant": "as_reactant",
    "as_product": "as_product",
    "plausibility": "plausibility",
    "forward_score": "forward_score",   # From graph optimization
    "ppg": "ppg",               # If calling clean_json on a previously cleaned tree
    "purchase_price": "ppg",
    "properties": "properties",
    "rxn_score_from_model": "rxn_score_from_model",
    "terminal": "terminal",
    "tforms": "tforms",
    "tsources": "tsources",
    "num_examples": "num_examples",
    "template_tuples": "template_tuples",
    "necessary_reagent": "necessary_reagent",
    "precursor_smiles": "precursor_smiles",
    "rms_molwt": "rms_molwt",
    "num_rings": "num_rings",
    "scscore": "scscore",
    "rank": "rank",
    "class_num": "class_num",
    "class_name": "class_name",
    "template": "template",
    "retro_backend": "retro_backend",
    "retro_model_name": "retro_model_name",
    "models_predicted_by": "models_predicted_by",
}

def get_graph_from_tree(tree: nx.DiGraph) -> nx.DiGraph:
    """Return cleaned version of original graph."""
    graph = tree.copy(as_view=False)

    # Remove unnecessary attributes
    for node, node_data in graph.nodes.items():
        attr_to_remove = [attr for attr in node_data if attr not in PATH_KEY_DICT]
        for attr in attr_to_remove:
            del node_data[attr]

    return graph

def chunk_by_comma(string):
    """Split a string on top-level commas, ignoring commas inside ``{...}`` groups."""
    # Find the indices of commas from right to left
    comma_indices = []
    balance = 0
    for i in range(len(string)-1, -1, -1):
        if string[i] == ',' and balance == 0: comma_indices.append(i)
        elif string[i] == '}': balance -= 1
        elif string[i] == '{': balance += 1

    # Reverse the indices list to be in the order they appear from left to right
    comma_indices.reverse()
    comma_indices = comma_indices + [len(string)]

    chunks = []
    start = 0
    for idx in comma_indices:
        chunks.append(string[start:idx])
        start = idx + 1  # Move start to the next position after comma

    return chunks

def is_terminal(
    smiles: str,
    build_tree_options=None,
    ppg: float = None,
    hist: dict = None,
    properties: list = None
) -> bool:
    """
    Determine if the specified chemical is a terminal node in the tree based
    on pre-specified criteria.

    Criteria to be considered are specified via ``self.termination_logic``,
    and the thresholds for each criterion are specified separately.

    If no criteria are specified, will always return ``False``.

    Args:
        smiles (str): smiles string of the chemical
        build_tree_options (BuildTreeOptions object): options for tree builder
        scscorer (SCScorerAPI): API to be used as an scscorer
        ppg (float): cost of the chemical
        hist (dict): historian data for the chemical
        properties (list): properties of the chemical
    """

    def buyable() -> bool:
        return bool(ppg) or (build_tree_options.custom_buyables and
                             smiles in build_tree_options.custom_buyables)

    def max_ppg() -> bool:
        if build_tree_options.max_ppg is not None:
            # ppg of 0 means not buyable
            return ppg is not None and 0 < ppg <= build_tree_options.max_ppg
        return True

    local_dict = locals()
    or_criteria = build_tree_options.termination_logic.get("or")
    and_criteria = build_tree_options.termination_logic.get("and")

    return (
        bool(or_criteria)
        and any(local_dict[criteria]() for criteria in or_criteria)
        or bool(and_criteria)
        and all(local_dict[criteria]() for criteria in and_criteria)
    )

def prune(tree: nx.DiGraph, root: str, max_prunes: int = 100):
    """
    Returns a pruned networkx graph. Iteratively removes non-"terminal" leaf nodes
    and their associated parent reaction nodes.

    Args:
        tree (nx.DiGraph): full results graph from tree builder expansion
        root (str): node ID of the root node (i.e. target chemical)
        max_prunes (int): maximum number of pruning iterations.

    Returns:
        nx.DiGraph with non-terminal leaf nodes and parent reaction nodes removed
    """
    pruned_tree = tree.copy()
    num_nodes = pruned_tree.number_of_nodes()

    parent_reactions = list(pruned_tree.predecessors(root))
    pruned_tree.remove_nodes_from(parent_reactions)

    for _ in range(max_prunes):
        non_terminal_leaves = [
            node
            for node, degree in pruned_tree.out_degree()
            if degree == 0 and not pruned_tree.nodes[node]["terminal"] and node != root
        ]
        for leaf in non_terminal_leaves:
            pruned_tree.remove_nodes_from(list(pruned_tree.predecessors(leaf)))
            pruned_tree.remove_node(leaf)

        if pruned_tree.number_of_nodes() == num_nodes:
            break
        num_nodes = pruned_tree.number_of_nodes()

    if not nx.is_weakly_connected(pruned_tree):
        for component in nx.weakly_connected_components(pruned_tree):
            if root in component:
                pruned_tree.remove_nodes_from(
                    [node for node in pruned_tree if node not in component]
                )
                break

    return pruned_tree


def _prune_productive(tree: nx.DiGraph, root: str):
    """Return root-reachable nodes that can form a finite terminal route."""
    pruned_tree = tree.copy()

    parent_reactions = list(pruned_tree.predecessors(root))
    pruned_tree.remove_nodes_from(parent_reactions)

    productive = {
        node
        for node, data in pruned_tree.nodes(data=True)
        if data.get("type") == "chemical" and data.get("terminal")
    }
    route_depth = {node: 0 for node in productive}
    while True:
        newly_productive = {}
        for node, data in pruned_tree.nodes(data=True):
            if node in productive:
                continue
            successors = list(pruned_tree.successors(node))
            if data.get("type") == "reaction":
                if successors and all(child in productive for child in successors):
                    newly_productive[node] = max(route_depth[child] for child in successors)
            elif data.get("type") == "chemical" and any(
                reaction in productive for reaction in successors
            ):
                newly_productive[node] = 1 + min(
                    route_depth[reaction]
                    for reaction in successors
                    if reaction in productive
                )
        if not newly_productive:
            break
        productive.update(newly_productive)
        route_depth.update(newly_productive)

    productive.add(root)
    pruned_tree.remove_nodes_from(set(pruned_tree) - productive)
    reachable = {root, *nx.descendants(pruned_tree, root)}
    pruned_tree.remove_nodes_from(set(pruned_tree) - reachable)
    pruned_tree.graph["minimum_route_depth"] = route_depth.get(root)

    return pruned_tree

def generate_unique_node():
    """
    Generate a unique node label using the UUID specification.

    Use UUIDv4 to generate random UUIDs instead of UUIDv1 which is used by
    ``networkx.utils.generate_unique_node``.
    """
    return str(uuid.uuid4())

def get_paths(
    tree: nx.DiGraph,
    root: str,
    root_uuid: str,
    max_depth: Optional[int] = None,
    max_trees: int = None,
    validate_paths: bool = True,
    depth_truncated: Optional[List[Any]] = None,
) -> Iterator[nx.DiGraph]:
    """
    Generate all paths from the root node as `nx.DiGraph` objects.

    All node attributes are copied to the output paths.
    ``depth_truncated`` records whether a finite depth stopped a live branch;
    iterative deepening uses that signal to stop once it has covered the finite graph.

    Returns:
        generator of paths
    """

    def get_uuid(smiles: str):
        if smiles == root:
            return root_uuid
        else:
            return generate_unique_node()

    def get_chem_paths(_node: str, chem_path: List[str]):
        """
        Return generator of paths with current node as the root.
        """
        at_depth_cap = max_depth is not None and len(chem_path) >= max_depth
        if tree.out_degree(_node) == 0 or at_depth_cap:
            if at_depth_cap and tree.out_degree(_node) > 0 and depth_truncated is not None:
                depth_truncated.append(_node)
            if tree.nodes[_node]["terminal"] or not validate_paths:
                yield _node
            else:
                return
        else:
            _subpath_count = 0
            for rxn in tree.successors(_node):
                for sub_path in get_rxn_paths(rxn, chem_path + [_node]):
                    if max_trees is not None and _subpath_count >= max_trees:
                        break
                    _subpath_count += 1
                    yield f"{{{sub_path}}}{_node}"

                else:
                    continue
                break

    def get_rxn_paths(_node: str, chem_path: List[str]):
        """
        Return generator of paths with current node as root.
        """
        precursors = list(tree.successors(_node))
        if set(precursors) & set(chem_path):
            # Adding this reaction would create a cycle
            return
        # itertools.product eagerly materializes each input generator before yielding, which on a
        # large/dense graph defeats the lazy max_trees cut-off below and can blow up combinatorially.
        # Cap each precursor's sub-paths so the materialization (and recursion) stays bounded; the
        # output is still cut at max_trees, so no surviving path is lost.
        sub_path_gens = [
            itertools.islice(get_chem_paths(c, chem_path), max_trees)
            if max_trees is not None
            else get_chem_paths(c, chem_path)
            for c in precursors
        ]
        for j, path_combo in enumerate(itertools.product(*sub_path_gens)):
            if max_trees is not None and j >= max_trees:
                break
            path_list = ",".join(
                sorted(path_combo, key=lambda x: len(x) - len(x.lstrip('{')), reverse=True)
            )
            sub_path = f"{{{path_list}}}{_node}"
            yield sub_path

    def postfix_recurse(postfix, path):

        if '{' not in postfix and '}' not in postfix and ',' not in postfix:
            smiles = postfix
            node = get_uuid(smiles)
            path.add_node(
                node,
                smiles = smiles,
                **tree.nodes[smiles],
            )
            return node
        else:
            end = len(postfix) - postfix[::-1].index('}')
            smiles = postfix[end:]
            postfix = postfix[1:end-1]
            #smiles = postfix[len(postfix)-j:]
            node = get_uuid(smiles)
            path.add_node(
                node,
                smiles = smiles,
                **tree.nodes[smiles],
            )
            for subpostfix in chunk_by_comma(postfix[:]):
                if '{' not in subpostfix and '}' not in subpostfix:
                    children = postfix_recurse(subpostfix, path)
                    path.add_edge(node, children)
                else:
                    children = postfix_recurse(subpostfix, path)
                    path.add_edge(node, children)

            return node

    num_paths = 0
    for postfix in get_chem_paths(root, []):
        if max_trees is not None and num_paths >= max_trees:
            break

        path = nx.DiGraph()
        postfix_recurse(postfix, path) # reconstruct networkx graph from postfix
        # Calculate depth of this path, i.e. number of reactions in the longest branch
        path.graph["depth"] = [
            path.nodes[v]["type"] for v in nx.dag_longest_path(path)
        ].count("reaction")
        # Calculate starting material cost for this path, None if any starting materials aren't buyable
        prices = [
            path.nodes[v]["purchase_price"] for v, d in path.out_degree() if d == 0
        ]
        path.graph["precursor_cost"] = (
            None if any(p == 0 for p in prices) else sum(prices)
        )
        # Initialize empty values for pathway score and cluster_id
        path.graph["score"] = None
        path.graph["cluster_id"] = None
        num_paths += 1
        yield path


def sort_paths(paths: List, metric: str) -> List:
    """
    Sort paths by some metric.
    """

    def number_of_starting_materials(tree):
        return len([v for v, d in tree.out_degree() if d == 0])

    def overall_plausibility(tree):
        return np.prod(
            [
                d["rxn_score_from_model"]
                for v, d in tree.nodes(data=True)
                if d["type"] == "reaction"
            ]
        )

    if metric == "plausibility":
        paths = sorted(paths, key=lambda x: overall_plausibility(x), reverse=True)
    elif metric == "number_of_starting_materials":
        paths = sorted(paths, key=lambda x: number_of_starting_materials(x))
    elif metric == "number_of_reactions":
        paths = sorted(paths, key=lambda x: x.graph["depth"])
    elif metric == "score":
        paths = sorted(paths, key=lambda x: x.graph["score"])
    else:
        raise ValueError(f"Need something to sort by! "
                         f"Invalid option provided: {metric}")

    return paths

def nx_graph_to_paths(
    tree: nx.DiGraph,
    root: str,
    max_depth: Optional[int] = None,
    max_trees: int = None,
    sorting_metric: str = "plausibility",
    validate_paths: bool = True,
    max_paths: Optional[int] = None,
    iterative_deepening: bool = False,
) -> Tuple[List, str]:
    """
    Return list of paths to buyables starting from the target node.

    Args:
        tree (nx.DiGraph): full graph to resolve pathways from
        root (str): node ID of the root node (i.e. target chemical)
        max_depth (int, optional): max tree depth (i.e., number of reaction steps)
        max_trees (int, optional): max number of trees to return
        sorting_metric (str, optional): how pathways are sorted, supports 'plausibility',
            'number_of_starting_materials', 'number_of_reactions', or 'score'
        validate_paths (bool, optional): require all leaves to meet terminal criteria
        max_paths (int, optional): maximum paths materialized before sorting
        iterative_deepening (bool, optional): increase the enumeration depth until
            a route is found or the finite graph is exhausted
    """
    root_uuid = NIL_UUID
    # Productive pruning is independent of the depth cap. The minimum route depth
    # must still satisfy the caller's finite budget before any path is materialized.
    productive_fast_path = iterative_deepening and validate_paths
    tree = (
        _prune_productive(tree=tree, root=root)
        if productive_fast_path
        else prune(tree=tree, root=root)
    )

    def enumerate_to(depth, truncated=None):
        paths = get_paths(
            tree,
            root=root,
            root_uuid=root_uuid,
            max_depth=depth,
            max_trees=max_trees,
            validate_paths=validate_paths,
            depth_truncated=truncated,
        )
        if max_paths is not None:
            paths = itertools.islice(paths, max_paths)
        return list(paths)

    if productive_fast_path:
        minimum_depth = tree.graph.get("minimum_route_depth")
        paths = (
            []
            if minimum_depth is None or (max_depth is not None and minimum_depth > max_depth)
            else enumerate_to(int(minimum_depth))
        )
    elif iterative_deepening:
        depth = 0
        paths = []
        while max_depth is None or depth < max_depth:
            depth += 1
            truncated = []
            paths = enumerate_to(depth, truncated)
            if paths or not truncated:
                break
    else:
        paths = enumerate_to(max_depth)

    paths = sort_paths(paths, sorting_metric)  # returns list

    return paths, root_uuid

def clean_json(path: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up json representation of a pathway. Accepts paths from either
    tree builder version.

    Note about chemical/reaction node identification:
        * For treedata format, chemical nodes have an ``is_chemical`` attribute,
          while reaction nodes have an ``is_reaction`` attribute
        * For nodelink format, all nodes have a ``type`` attribute, whose value
          is either ``chemical`` or ``reaction``

    This distinction in the JSON schema is for historical reasons and is not
    an official aspect of the treedata/nodelink formats.
    """

    def _add_missing_fields(_node, _type):
        for _key in OUTPUT_KEYS[_type]:
            if _key not in _node:
                _node[_key] = None

    if "nodes" in path:
        # Node link format
        nodes = []
        for node in path["nodes"]:
            new_node = {
                PATH_KEY_DICT[key]: value
                for key, value in node.items()
                if key in PATH_KEY_DICT
            }
            # Set any output keys not present to None
            _add_missing_fields(new_node, node["type"])
            nodes.append(new_node)
        path["nodes"] = nodes
        output = path
    else:
        # Tree data format
        output = {}
        for key, value in path.items():
            if key == "type":
                if value == "chemical":
                    output["is_chemical"] = True
                elif value == "reaction":
                    output["is_reaction"] = True
            elif key == "children":
                output["children"] = [clean_json(c) for c in value]
            elif key in PATH_KEY_DICT:
                output[PATH_KEY_DICT[key]] = value

        # Set any output keys not present to None
        _add_missing_fields(output, path["type"])

        if "children" not in output:
            output["children"] = []

    return output

def nx_paths_to_json(
    paths: List,
    root_uuid: str,
    json_format: str = "treedata"
) -> List[Dict[str, Any]]:
    """
    Convert list of paths from networkx graphs to json.
    """
    if json_format == "treedata":
        # Include graph attributes at top level of resulting json
        return [
            {"attributes": path.graph, **clean_json(nx.tree_data(path, root_uuid))}
            for path in paths
        ]
    elif json_format == "nodelink":
        return [
            clean_json(nx.node_link_data(path, attrs=NODE_LINK_ATTRS)) for path in paths
        ]
    else:
        raise ValueError(f"Unsupported value for json_format: {json_format}")

def _cal_delta_metric_in_ucb(rxn: str, metric_name: str = "ged") -> float:
    """One-step delta for a single (atomic) metric, computed without tree context.

    Used for the first reaction on a selection path, before a ``SynTreeGED`` exists. Each metric in
    a combined name (e.g. ``"ged&tanimoto"``) is computed separately by the caller, so ``metric_name``
    here is always a single metric.
    """
    rxn_smi = map_one_reaction(rxn)
    delta_metric = 0
    if metric_name == "ged":
        # GED requires atom-mapped reactions; the unmapped input often collapses to zero.
        delta_metric = _cal_one_step_edit_distance(rxn_smi, metric_name)
    elif metric_name == "tanimoto":
        delta_metric = _cal_one_step_similarity_reduction(rxn, metric_name)
    elif metric_name in ["sa_score", "sc_score"]:
        delta_metric = _cal_one_step_mol_clx_reduction(rxn, metric_name)
    return delta_metric
