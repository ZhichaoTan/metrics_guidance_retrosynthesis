"""
MCTS utility functions.
"""

def extract_unique_smiles(path):
    """
    Extract unique SMILES from a path (tree data structure).

    Args:
        path: Path data structure (typically a dict with 'smiles' or nested structure)

    Returns:
        set: Set of unique SMILES strings
    """
    unique_smiles = set()

    def extract_from_node(node):
        """Recursively extract SMILES from a node."""
        if isinstance(node, dict):
            # Check for 'smiles' key
            if 'smiles' in node:
                smiles = node['smiles']
                if isinstance(smiles, str) and smiles:
                    unique_smiles.add(smiles)

            # Check for 'id' key (sometimes used for SMILES)
            if 'id' in node:
                node_id = node['id']
                if isinstance(node_id, str) and node_id and '>>' not in node_id:
                    unique_smiles.add(node_id)

            # Recursively process children
            if 'children' in node:
                for child in node['children']:
                    extract_from_node(child)

            # Process all values recursively
            for value in node.values():
                if isinstance(value, (dict, list)):
                    extract_from_node(value)

        elif isinstance(node, list):
            for item in node:
                extract_from_node(item)

    extract_from_node(path)
    return unique_smiles
