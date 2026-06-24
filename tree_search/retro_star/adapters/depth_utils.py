"""Chemical-depth helpers for Retro* route validation (aligned with MCTS chem_path depth)."""

from __future__ import annotations

from typing import Dict, List, Optional


def route_depth_within_limit(
    max_mol_depth: Optional[int], enumerate_max_depth: Optional[int]
) -> bool:
    """True when chemical depth satisfies MCTS-style max_depth (chem_path length cap)."""
    if enumerate_max_depth is None:
        return True
    if max_mol_depth is None:
        return False
    return max_mol_depth < enumerate_max_depth


def binghong_route_max_mol_depth(route_str: str) -> int:
    """Parse Binghong serialized route and return maximum chemical depth (root = 0)."""
    if not route_str:
        return 0

    children_map: Dict[str, List[str]] = {}
    root = None
    for segment in route_str.split("|"):
        parts = segment.split(">")
        if len(parts) < 3:
            if root is None and parts:
                root = parts[0]
            continue
        product, precursors_part = parts[0], parts[-1]
        if root is None:
            root = product
        children_map[product] = [p for p in precursors_part.split(".") if p]

    if root is None:
        return 0

    def dfs(smiles: str, depth: int) -> int:
        children = children_map.get(smiles)
        if not children:
            return depth
        return max(dfs(child, depth + 1) for child in children)

    return dfs(root, 0)
