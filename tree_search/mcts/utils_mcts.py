"""utils_mcts module.
"""

import itertools
import networkx as nx
import numpy as np
import operator
import time
import uuid
from api.pathway_ranker_api import PathwayRankerAPI
from api.scscorer_api import SCScorerAPI
from collections import defaultdict
from collections.abc import Iterator
from rdkit import Chem
from typing import Any, Dict, List, Tuple

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

OPERATOR_MAP = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
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

class CanonicalizationError(ValueError):
    """Exception class for failure to canonicalize SMILES."""

def canonicalize(
    smiles, isomeric_smiles=True, raise_exception=False, keep_agents=False
):
    """Canonicalize the input SMILES."""
    if ">" in smiles:
        # Reaction
        try:
            reactants, agents, products = smiles.split(">")
        except ValueError:
            if raise_exception:
                raise CanonicalizationError(smiles)
            return smiles

        reactants = ".".join(
            sorted(
                canonicalize(
                    smi,
                    isomeric_smiles=isomeric_smiles,
                    raise_exception=raise_exception,
                )
                for smi in reactants.split(".")
            )
        )
        products = ".".join(
            sorted(
                canonicalize(
                    smi,
                    isomeric_smiles=isomeric_smiles,
                    raise_exception=raise_exception,
                )
                for smi in products.split(".")
            )
        )
        if keep_agents:
            agents = ".".join(
                sorted(
                    canonicalize(
                        smi,
                        isomeric_smiles=isomeric_smiles,
                        raise_exception=raise_exception,
                    )
                    for smi in agents.split(".")
                )
            )
        else:
            agents = ""
        return reactants + ">" + agents + ">" + products
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        if raise_exception:
            raise CanonicalizationError(smiles)
        return smiles

def check_property_criteria(properties: list, criteria: list) -> List[bool]:
    """
    Check if the provided properties meet the specified criteria.

    Properties should be dictionaries with 'name' and 'value' keys.
    Criteria should be dictionaries with 'name', 'value', and 'logic keys.

    If a property is in the criteria list but not in the properties list, then
    it is considered not meeting the criteria, i.e. ``False``.

    Args:
        properties (list): list of properties for a particular precursor
        criteria (list): list of criteria to check

    Returns:
        list: True or False for each of the specified criteria
    """
    if properties:
        property_map = {item["name"]: item["value"] for item in properties}
    else:
        property_map = {}
    results = []
    for crit in criteria:
        value = property_map.get(crit["name"])
        if value is not None:
            results.append(OPERATOR_MAP[crit["logic"]](value, crit["value"]))
        else:
            results.append(False)
    return results

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
    scscorer: SCScorerAPI = None,
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

    def max_scscore() -> bool:
        if build_tree_options.max_scscore is not None:
            scscore = scscorer(smiles=smiles)
            return scscore <= build_tree_options.max_scscore
        return True

    def max_elements() -> bool:
        if build_tree_options.max_elements is not None:
            # Get structural properties
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                elem_dict = defaultdict(int)
                for a in mol.GetAtoms():
                    elem_dict[a.GetSymbol()] += 1
                elem_dict["H"] = sum(a.GetTotalNumHs() for a in mol.GetAtoms())

                return all(elem_dict[k] <= v for k, v
                           in build_tree_options.max_elements.items())
        return True

    def min_history() -> bool:
        if build_tree_options.min_history is not None:
            return hist is not None and (
                hist["as_reactant"] >=
                build_tree_options.min_history["as_reactant"]
                or hist["as_product"] >=
                build_tree_options.min_history["as_product"]
            )
        return True

    def property_criteria() -> bool:
        if build_tree_options.property_criteria:
            results = check_property_criteria(
                properties=properties,
                criteria=build_tree_options.property_criteria
            )
            if "property_criteria" in or_criteria:
                return any(results)
            elif "property_criteria" in and_criteria:
                return all(results)
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

    for i in range(max_prunes):
        non_terminal_leaves = [
            v
            for v, d in pruned_tree.out_degree()
            if d == 0 and not pruned_tree.nodes[v]["terminal"] and v != root
        ]

        for leaf in non_terminal_leaves:
            pruned_tree.remove_nodes_from(list(pruned_tree.predecessors(leaf)))
            pruned_tree.remove_node(leaf)

        if pruned_tree.number_of_nodes() == num_nodes:
            break
        else:
            num_nodes = pruned_tree.number_of_nodes()

    # If pruning resulted in a disconnected graph, remove nodes not connected to
    # the root subgraph
    if not nx.is_weakly_connected(pruned_tree):
        for c in nx.weakly_connected_components(pruned_tree):
            if root in c:
                pruned_tree.remove_nodes_from([n for n in pruned_tree if n not in c])
                break

    return pruned_tree

def full_update(
    tree: nx.DiGraph,
    chem_smi: str,
    max_depth: int = None,
    depth: int = 0,
    path: List = None
):
    """Update estimated pathway counts and min price for chemical nodes.

    Estimates pathway count as combinations of paths to terminal chemicals.
    Estimates min price as sum of all precursors leading to a chemical.

    Args:
        tree (nx.DiGraph): full reaction network from tree builder job
        chem_smi (str): SMILES string of root chemical node to evaluate
        max_depth (int, optional): maximum tree depth to evaluate to
        depth (int, optional): current tree depth
        path (list): list of chemical nodes traversed (to avoid loops)
    """
    path = path or []
    chem = tree.nodes[chem_smi]
    chem["pathway_count"] = 0
    chem["ppg"] = chem["purchase_price"]
    if chem["terminal"]:
        chem["pathway_count"] = 1
        return tree

    if max_depth is not None and depth > max_depth:
        return tree

    for reaction_node in tree.successors(chem_smi):
        # Reaction node
        rxn = tree.nodes[reaction_node]
        # Successor chemical nodes
        precursors = list(tree.successors(reaction_node))
        rxn["pathway_count"] = 0
        if len(set(precursors) & set(path)) > 0:
            # This reaction creates a loop
            continue
        for smi in precursors:
            full_update(
                tree, smi, max_depth=max_depth, depth=depth + 1, path=path + [chem_smi]
            )
        price_list = [tree.nodes[smi]["ppg"] for smi in precursors]
        # Price of 0 indicates that the chemical is not buyable
        if all([price > 0 for price in price_list]):
            price = sum(price_list)
            rxn["ppg"] = price
            if rxn["ppg"] < chem["ppg"] or chem["ppg"] <= 0:
                chem["ppg"] = rxn["ppg"]

            rxn["pathway_count"] = np.prod(
                [tree.nodes[smi]["pathway_count"] for smi in precursors]
            )

    chem["pathway_count"] = 0
    for reaction_node in tree.successors(chem_smi):
        chem["pathway_count"] += tree.nodes[reaction_node]["pathway_count"]

    return tree

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
    max_depth: int = None,
    max_trees: int = None,
    validate_paths: bool = True
) -> Iterator[nx.DiGraph]:
    """
    Generate all paths from the root node as `nx.DiGraph` objects.

    All node attributes are copied to the output paths.

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
        if (
            tree.out_degree(_node) == 0
            or max_depth is not None
            and len(chem_path) >= max_depth
        ):
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
        for j, path_combo in enumerate(itertools.product(
            *(get_chem_paths(c, chem_path) for c in precursors)
        )):
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

# import networkx as nx
# import itertools
# from functools import lru_cache

# def get_paths(
#     tree: nx.DiGraph,
#     root: str,
#     root_uuid: str,
#     max_depth: int = None,
#     max_trees: int = None,
#     validate_paths: bool = True
# ):
#     """
#     Optimized path generation from the root node as `nx.DiGraph` objects.
#     """

#     def get_uuid(smiles: str):
#         return root_uuid if smiles == root else generate_unique_node()

#     @lru_cache(maxsize=None)  # Cache results to avoid recomputation
#     def get_chem_paths(_node: str, depth: int):
#         """
#         Optimized generator of paths with current node as the root.
#         Uses depth tracking and memoization to reduce redundant computation.
#         """
#         if tree.out_degree(_node) == 0 or (max_depth is not None and depth >= max_depth):
#             if tree.nodes[_node]["terminal"] or not validate_paths:
#                 yield _node
#             return  # Early exit for invalid paths

#         _subpath_count = 0
#         for rxn in tree.successors(_node):
#             for sub_path in get_rxn_paths(rxn, depth + 1):
#                 if max_trees is not None and _subpath_count >= max_trees:
#                     return  # Stop searching if max trees exceeded
#                 _subpath_count += 1
#                 yield f"{{{sub_path}}}{_node}"

#     def get_rxn_paths(_node: str, depth: int):
#         """
#         Optimized generator of paths with reaction node as root.
#         """
#         precursors = list(tree.successors(_node))
#         if set(precursors) & set(get_chem_paths.cache_info()):  # Prevent cycles
#             return

#         for j, path_combo in enumerate(itertools.product(
#             *(get_chem_paths(c, depth) for c in precursors)
#         )):
#             if max_trees is not None and j >= max_trees:
#                 return  # Stop if max trees exceeded
#             path_list = ",".join(sorted(path_combo, key=lambda x: len(x) - len(x.lstrip('{')), reverse=True))
#             yield f"{{{path_list}}}{_node}"

#     def postfix_recurse(postfix, path):
#         """
#         Iterative reconstruction of networkx graph from postfix expression.
#         """
#         stack = []
#         while postfix:
#             if '{' in postfix and '}' in postfix:
#                 end = postfix.index('}')
#                 smiles = postfix[end + 1:]
#                 postfix = postfix[:end].strip('{}')
#             else:
#                 smiles = postfix
#                 postfix = ""

#             node = get_uuid(smiles)
#             path.add_node(node, smiles=smiles, **tree.nodes[smiles])

#             if stack:
#                 parent = stack[-1]
#                 path.add_edge(parent, node)

#             stack.append(node)

#         return stack[0] if stack else None

#     num_paths = 0
#     for postfix in get_chem_paths(root, 0):
#         if max_trees is not None and num_paths >= max_trees:
#             break

#         path = nx.DiGraph()
#         postfix_recurse(postfix, path)  # Construct graph iteratively

#         # Calculate path properties
#         path.graph["depth"] = sum(1 for v in nx.dag_longest_path(path) if path.nodes[v]["type"] == "reaction")
#         path.graph["precursor_cost"] = (
#             None if any(path.nodes[v]["purchase_price"] == 0 for v, d in path.out_degree() if d == 0)
#             else sum(path.nodes[v]["purchase_price"] for v, d in path.out_degree() if d == 0)
#         )

#         path.graph["score"] = None
#         path.graph["cluster_id"] = None
#         num_paths += 1
#         yield path

def score_paths(
    paths: Iterator[nx.DiGraph],
    cluster_trees: bool,
    pathway_ranker: PathwayRankerAPI,
    cluster_method: str,
    min_samples: int,
    min_cluster_size: int
) -> List:
    """
    Score paths using the provided settings.

    Scores and cluster IDs (if requested) are saved as graph attributes.
    """
    # Turn paths generator into list of graphs and list of json
    graph_paths, json_paths = [], []
    for path in paths:
        graph_paths.append(path)
        json_paths.append(clean_json(nx.tree_data(path, NIL_UUID)))

    # Count how many scorable paths there are
    num_paths = sum(1 for path in graph_paths if path.graph["depth"] > 1)
    if num_paths <= 1:
        # There's nothing to score
        return graph_paths

    # Pathway ranker requires json paths
    results = pathway_ranker(
        trees=json_paths,
        clustering=cluster_trees,
        cluster_method=cluster_method,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size
    )

    for i, path in enumerate(graph_paths):
        path.graph["score"] = results["scores"][i]
        if cluster_trees:
            path.graph["cluster_id"] = results["clusters"][i]

    return graph_paths

def sort_paths(paths: List, metric: str) -> List:
    """
    Sort paths by some metric.
    """

    def number_of_starting_materials(tree):
        return len([v for v, d in tree.out_degree() if d == 0])

    def overall_plausibility(tree):
        return np.prod(
            [
                d["plausibility"]
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
    max_depth: int = None,
    max_trees: int = None,
    sorting_metric: str = "plausibility",
    validate_paths: bool = True,
    score_trees: bool = False,
    cluster_trees: bool = False,
    pathway_ranker=None,
    update: bool = False,
    cluster_method: str = "hdbscan",
    min_samples: int = 5,
    min_cluster_size: int = 5
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
        score_trees (bool, optional): whether to score trees
        cluster_trees (bool, optional): whether to cluster trees
        pathway_ranker (method, optional): method used to score and cluster trees
        update (bool, optional): whether to update min price and pathway counts
            for entire tree (up to max_depth)
        cluster_method (str, optional): hdbscan or kmeans
        min_samples (int, optional): min samples for hdbscan
        min_cluster_size (bool, optional): min cluster_size for hdbscan

    Returns:
        list of paths in specified format
    """
    # Use NIL UUID for root, so we can easily identify it
    root_uuid = NIL_UUID
    tree = prune(tree=tree, root=root)

    if update:
        start = time.time()
        tree = full_update(tree=tree, chem_smi=root, max_depth=max_depth)
        update_time = time.time() - start
        print(f"Full update complete after {update_time:.2f} seconds")
        print(f"Estimated pathway count (over-counting duplicate templates): "
              f"{tree.nodes[root]['pathway_count']}")
        print(f"Estimated minimum price: {tree.nodes[root]['ppg']:.1f}")

    paths = get_paths(
        tree,
        root=root,
        root_uuid=root_uuid,
        max_depth=max_depth,
        max_trees=max_trees,
        validate_paths=validate_paths,
    )       # returns generator

    if score_trees or sorting_metric == "score":
        paths = score_paths(
            paths,
            cluster_trees=cluster_trees,
            pathway_ranker=pathway_ranker,
            cluster_method=cluster_method,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size
        )  # returns list

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

def check_retron_reached(prod_smiles, precursor_smiles, retron_smarts):
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in [prod_smiles]+precursor_smiles.split(".")]
    retron_mol = Chem.MolFromSmarts(retron_smarts)
    for mol in mol_list:
        if mol.HasSubstructMatch(retron_mol):
            return True
    return False