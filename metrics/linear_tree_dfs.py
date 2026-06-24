"""Depth-first search utilities for extracting linear trees.
"""

import networkx as nx

def dfs_split_tree(tree: nx.DiGraph, node: int, visited: set, current_tree_nodes: list, linear_trees: list):
    """Recursively split tree into linear subtrees using DFS."""
    visited.add(node)
    current_tree_nodes.append(node)

    children = list(tree.successors(node))
    non_leaf_children = [child for child in children if tree.out_degree(child) > 0]
    leaf_children = [child for child in children if child not in non_leaf_children]

    visited.update(leaf_children)
    current_tree_nodes.extend(leaf_children)

    if len(non_leaf_children) > 1:
        if len(current_tree_nodes) > 1:
            linear_trees.append(tree.subgraph(current_tree_nodes).copy())

        for child in non_leaf_children:
            if child not in visited:
                # Add the starting node of the branch as a new root
                dfs_split_tree(tree, child, visited, [node], linear_trees)

    if len(non_leaf_children) == 1:
        next_node = non_leaf_children[0]
        if next_node not in visited:
            dfs_split_tree(tree, next_node, visited, current_tree_nodes, linear_trees)

    if len(non_leaf_children) == 0:
        if len(current_tree_nodes) > 1:
            linear_trees.append(tree.subgraph(current_tree_nodes).copy())
        return

def extract_linear_trees_dfs(tree: nx.DiGraph) -> list:
    """Extract all linear trees from a branched synthesis tree."""
    linear_trees = []
    visited = set()

    nodes_to_process = list(nx.topological_sort(tree))

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        if node not in visited:
            current_tree_nodes = []
            dfs_split_tree(tree, node, visited, current_tree_nodes, linear_trees)

    return linear_trees

if __name__ == "__main__":
    tree = nx.DiGraph()
    tree.add_edges_from([(7,8), (9,14), (1, 2), (2, 3), (3, 4), (4,5), (5,6), (6,7), (2, 9), (9, 10), (3, 11), (4, 12), (5, 13), (11,15)])

    linear_trees = extract_linear_trees_dfs(tree)

    for idx, linear_tree in enumerate(linear_trees):
        print(f"Linear Tree {idx + 1}:")
        print("Nodes:", list(linear_tree.nodes()))
        print("Edges:", list(linear_tree.edges()))
