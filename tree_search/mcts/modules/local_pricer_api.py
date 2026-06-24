"""
Local Pricer API - Direct lookup in buyables.json.gz or Binghong origin_dict.csv.
"""
import gzip
import json
import os
from functools import lru_cache
from typing import Dict, Optional, Set, Tuple, Union
from rdkit import Chem

_BUYABLES_CACHE = {}


def default_local_buyables_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(os.path.dirname(current_dir)),
        "buyables.json.gz",
    )


def default_binghong_buyables_path() -> str:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    binghong_root = os.environ.get(
        "BINGHONG_RETRO_STAR_ROOT",
        os.path.join(project_root, "external", "binghong_retro_star"),
    )
    return os.path.join(binghong_root, "retro_star", "dataset", "origin_dict.csv")


def resolve_buyables_path(buyables_path: Optional[str] = None) -> str:
    """Resolve the buyables path used by the local pricer."""
    if buyables_path is None:
        buyables_path = os.environ.get("LOCAL_BUYABLES_PATH")
    if buyables_path is None:
        buyables_path = default_local_buyables_path()
    return os.path.abspath(buyables_path)


def _load_binghong_csv_buyables(buyables_path: str) -> Set[str]:
    """Load Binghong starting molecules (exact SMILES match, like RSPlanner)."""
    import pandas as pd

    print(f"Loading Binghong buyables from {buyables_path}...")
    df = pd.read_csv(buyables_path, usecols=["mol"])
    exact = set(df["mol"].dropna().astype(str))
    print(f"Loaded {len(exact)} starting molecules from Binghong CSV.")
    return exact


def _load_json_gz_buyables(buyables_path: str) -> Dict[str, Dict[str, dict]]:
    with gzip.open(buyables_path, "rt", encoding="utf-8") as f:
        buyables_list = json.load(f)

    exact_buyables = {}
    canonical_buyables = {}
    for item in buyables_list:
        smiles = item.get("smiles", "")
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    buyable_entry = {
                        "ppg": item.get("ppg", 0.0),
                        "source": item.get("source", ""),
                        "properties": item.get("properties", None),
                        "smiles": smiles,
                    }
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if smiles not in exact_buyables:
                        exact_buyables[smiles] = buyable_entry
                    if canonical_smiles not in canonical_buyables:
                        canonical_buyables[canonical_smiles] = buyable_entry
            except Exception:
                continue

    print(
        "Loaded "
        f"{len(exact_buyables)} exact SMILES and "
        f"{len(canonical_buyables)} unique isomeric canonical SMILES from buyables."
    )
    return {"exact": exact_buyables, "canonical": canonical_buyables}


def _load_buyables_store(buyables_path: str) -> Union[Set[str], dict]:
    """Load buyables exactly once per process and path."""
    if buyables_path in _BUYABLES_CACHE:
        return _BUYABLES_CACHE[buyables_path]

    print(f"Loading buyables from {buyables_path}...")
    if not os.path.exists(buyables_path):
        raise FileNotFoundError(f"Buyables file not found: {buyables_path}")

    if buyables_path.endswith(".csv"):
        store = _load_binghong_csv_buyables(buyables_path)
    else:
        store = _load_json_gz_buyables(buyables_path)

    _BUYABLES_CACHE[buyables_path] = store
    return store


def preload_buyables(buyables_path: Optional[str] = None) -> Union[Set[str], dict]:
    """Preload buyables in the current process for reuse or copy-on-write fork sharing."""
    return _load_buyables_store(resolve_buyables_path(buyables_path))


class LocalPricerAPI:
    """Local pricer for buyables.json.gz or Binghong CSV."""

    def __init__(self, buyables_path: Optional[str] = None):
        self.buyables_path = resolve_buyables_path(buyables_path)
        store = _load_buyables_store(self.buyables_path)
        self._binghong_exact = store if isinstance(store, set) else None
        self._buyables_exact = None
        self._buyables_canonical = None
        if isinstance(store, dict):
            self._buyables_exact = store["exact"]
            self._buyables_canonical = store["canonical"]
    
    @lru_cache(maxsize=10000)
    def _canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Canonicalize a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return None
    
    def __call__(
        self,
        smiles: str,
        source: Optional[str] = None,
        canonicalize: bool = True
    ) -> Tuple[float, Optional[dict]]:
        """
        Look up a molecule in buyables database.
        
        Args:
            smiles: SMILES string to look up
            source: Optional source filter (not used in local implementation)
            canonicalize: Whether to canonicalize the input SMILES before lookup
            
        Returns:
            Tuple of (purchase_price, properties)
            - purchase_price: Price per gram (ppg) if found, 0.0 otherwise
            - properties: Properties dict if found, None otherwise
        """
        if not smiles:
            return 0.0, None

        if self._binghong_exact is not None:
            if smiles in self._binghong_exact:
                return 1.0, {"source": "binghong_origin_dict"}
            return 0.0, None

        if not canonicalize:
            item = self._buyables_exact.get(smiles)
            if item is not None:
                return item["ppg"], item["properties"]
            # Align with higherlev_retro's effective behavior: inputs reaching the
            # pricer are often already canonicalized with stereochemistry preserved.
            # For local lookup we therefore fall back to isomeric canonical matching
            # when the raw SMILES string is non-canonical but chemically equivalent.
            canonical_smiles = self._canonicalize_smiles(smiles)
            if canonical_smiles is None:
                return 0.0, None
            item = self._buyables_canonical.get(canonical_smiles)
            if item is not None:
                return item["ppg"], item["properties"]
            return 0.0, None

        canonical_smiles = self._canonicalize_smiles(smiles)
        if canonical_smiles is None:
            return 0.0, None

        if canonical_smiles in self._buyables_canonical:
            item = self._buyables_canonical[canonical_smiles]
            return item["ppg"], item["properties"]

        return 0.0, None
