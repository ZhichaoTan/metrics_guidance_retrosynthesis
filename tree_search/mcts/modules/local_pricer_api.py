"""
Local Pricer API - Direct lookup in buyables.json.gz without HTTP
"""
import gzip
import json
import os
from functools import lru_cache
from typing import Optional, Tuple
from rdkit import Chem


class LocalPricerAPI:
    """
    Local pricer that looks up molecules in buyables.json.gz file.
    
    Uses canonical SMILES for matching and includes caching for performance.
    """
    
    def __init__(self, buyables_path: Optional[str] = None):
        """
        Initialize the local pricer.
        
        Args:
            buyables_path: Path to buyables.json.gz file. If None, uses default path.
        """
        if buyables_path is None:
            # Default path relative to this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            buyables_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),
                "buyables.json.gz"
            )
        
        self.buyables_path = buyables_path
        self._buyables_dict = None
        self._load_buyables()
    
    def _load_buyables(self):
        """Load buyables from JSON.gz file and create a dictionary indexed by canonical SMILES."""
        print(f"Loading buyables from {self.buyables_path}...")
        
        if not os.path.exists(self.buyables_path):
            raise FileNotFoundError(f"Buyables file not found: {self.buyables_path}")
        
        # Load buyables list
        with gzip.open(self.buyables_path, 'rt', encoding='utf-8') as f:
            buyables_list = json.load(f)
        
        # Create dictionary indexed by canonical SMILES
        self._buyables_dict = {}
        for item in buyables_list:
            smiles = item.get("smiles", "")
            if smiles:
                # Canonicalize the SMILES from buyables file
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                        # Store the first occurrence (or you could merge properties)
                        if canonical_smiles not in self._buyables_dict:
                            self._buyables_dict[canonical_smiles] = {
                                "ppg": item.get("ppg", 0.0),
                                "source": item.get("source", ""),
                                "properties": item.get("properties", None),
                                "smiles": smiles  # original SMILES
                            }
                except Exception as e:
                    # Skip invalid SMILES
                    continue
        
        print(f"Loaded {len(self._buyables_dict)} unique canonical SMILES from buyables.")
    
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
            return Chem.MolToSmiles(mol, isomericSmiles=False)
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
        
        # Always use canonical SMILES for lookup since buyables_dict is indexed by canonical SMILES
        # This ensures accurate matching regardless of the canonicalize parameter
        canonical_smiles = self._canonicalize_smiles(smiles)
        if canonical_smiles is None:
            return 0.0, None
        
        # Look up in buyables dictionary using canonical SMILES
        if canonical_smiles in self._buyables_dict:
            item = self._buyables_dict[canonical_smiles]
            return item["ppg"], item["properties"]
        
        # Not found
        return 0.0, None
