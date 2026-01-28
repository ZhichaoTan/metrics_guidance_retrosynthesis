"""
Local Template Relevance Predictor - Direct model inference without TorchServe
"""
import argparse
import numpy as np
import os
import sys
from . import templ_rel_parser
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun


class LocalTemplRelPredictor:
    """Local Template Relevance Predictor for direct model inference without TorchServe"""

    def __init__(self, model_dir: str, device: str = "auto"):
        """
        Initialize the local predictor.
        
        Args:
            model_dir: Path to model directory containing model_latest.pt and templates.jsonl
            device: Device to use ("auto", "cuda", "cpu", or "cuda:0", etc.)
        """
        self.model_dir = os.path.abspath(model_dir)
        self.device = self._determine_device(device)
        self.args = None
        self.templates = None
        self.template_attributes = None
        self.model = None
        self.initialized = False
        
        # Initialize immediately
        self.initialize()
    
    def _determine_device(self, device: str) -> torch.device:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def initialize(self):
        """Initialize the model and templates."""
        if self.initialized:
            return
        
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Add model directory to path FIRST, so model's utils.py can import its dependencies
        if self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)
        
        # Import from model directory's utils.py
        # Since model_dir is now first in sys.path, 'import utils' will get the model's utils.py
        # Temporarily remove global utils from sys.modules if it exists to avoid conflicts
        utils_path = os.path.join(self.model_dir, 'utils.py')
        if os.path.exists(utils_path):
            # Save and remove global utils if it's already imported
            global_utils_backup = None
            if 'utils' in sys.modules:
                # Check if it's the global utils package (has __path__)
                old_utils = sys.modules['utils']
                if hasattr(old_utils, '__path__'):
                    global_utils_backup = old_utils
                    del sys.modules['utils']
            
            # Now import utils - it will load from model directory since it's first in sys.path
            import utils as model_utils
            
            # Get canonicalize_smiles from model utils, fallback to global utils if not available
            if hasattr(model_utils, 'canonicalize_smiles'):
                canonicalize_smiles = model_utils.canonicalize_smiles
            else:
                # Fallback to global utils (restore it first if we backed it up)
                if global_utils_backup is not None:
                    sys.modules['utils'] = global_utils_backup
                from utils import canonicalize_smiles as global_canonicalize_smiles
                canonicalize_smiles = global_canonicalize_smiles
            
            # Restore global utils if we backed it up (after we're done with imports)
            if global_utils_backup is not None:
                sys.modules['utils'] = global_utils_backup
            get_model = model_utils.get_model
            load_templates_as_list = model_utils.load_templates_as_list
            mol_smi_to_count_fp = model_utils.mol_smi_to_count_fp
        else:
            # Fallback to global utils if model utils.py doesn't exist
            from utils import canonicalize_smiles, get_model, load_templates_as_list, mol_smi_to_count_fp
        self.canonicalize_smiles = canonicalize_smiles
        self.get_model = get_model
        self.load_templates_as_list = load_templates_as_list
        self.mol_smi_to_count_fp = mol_smi_to_count_fp
        
        serve_parser = argparse.ArgumentParser("local_predict")
        templ_rel_parser.add_predict_opts(serve_parser)
        self.args, _ = serve_parser.parse_known_args()
        
        # Load templates
        template_file = os.path.join(self.model_dir, "templates.jsonl")
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"Templates file not found: {template_file}")
        
        self.templates, self.template_attributes = self.load_templates_as_list(template_file=template_file)
        
        # Load model
        checkpoint_file = os.path.join(self.model_dir, "model_latest.pt")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
        
        self.args.load_from = checkpoint_file
        self.args.local_rank = -1
        
        # Load model
        self.model, _ = self.get_model(self.args, device=self.device)
        self.model.eval()
        self.initialized = True
    
    def predict(
        self,
        smiles: List[str],
        max_num_templates: int = 1000,
        max_cum_prob: float = 0.999,
        attribute_filter: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict retrosynthesis outcomes for given SMILES.
        
        Args:
            smiles: List of product SMILES strings
            max_num_templates: Maximum number of templates to consider
            max_cum_prob: Maximum cumulative probability threshold
            attribute_filter: List of attribute filters (optional)
        
        Returns:
            List of result dictionaries, each containing:
            - templates: List of template dictionaries
            - reactants: List of reactant SMILES strings
            - scores: List of scores
        """
        if not self.initialized:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        
        if attribute_filter is None:
            attribute_filter = []
        
        # Canonicalize SMILES
        canonical_smiles = [self.canonicalize_smiles(smi) for smi in smiles]
        
        # Process attribute filters
        filters = []
        if self.template_attributes is not None:
            filters = [x for x in attribute_filter
                      if x.get("name") in self.template_attributes.columns]
        
        filtered_indices = None
        if filters:
            filter_query = " and ".join(
                [f"({q['name']} {q['logic']} {q['value']})" for q in filters]
            )
            filtered_indices = self.template_attributes.query(filter_query).index.values
        
        results = []
        
        # Inference loop (same logic as templ_rel_handler.inference)
        with torch.no_grad():
            for smi in canonical_smiles:
                # Generate fingerprint
                prod_fp = self.mol_smi_to_count_fp(smi, self.args.radius, self.args.fp_size)
                final_fp = torch.as_tensor(prod_fp.toarray()).float().to(self.device)
                
                # Model prediction
                logits = self.model(final_fp)
                scores = F.softmax(logits, dim=1)
                scores = scores.squeeze(dim=0).cpu().numpy()
                indices = np.argsort(-scores)
                scores = scores[indices]
                
                # Filter by attributes
                if filtered_indices is not None:
                    bool_mask = np.isin(indices, filtered_indices)
                    indices = indices[bool_mask]
                    scores = scores[bool_mask]
                
                # Truncate by max_num_templates
                if max_num_templates:
                    indices = indices[:max_num_templates]
                    scores = scores[:max_num_templates]
                
                # Truncate by max_cum_prob
                if max_cum_prob:
                    exceeds = np.nonzero(np.cumsum(scores) >= max_cum_prob)[0]
                    if exceeds.size:
                        max_index = exceeds[0] + 1
                        scores = scores[:max_index]
                        indices = indices[:max_index]
                
                # Generate results
                smiles_to_index = {}
                result = {
                    "templates": [],
                    "reactants": [],
                    "scores": []
                }
                
                for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                    template = self.templates[idx]
                    # IMPORTANT MAGIC from v1. DO NOT TOUCH
                    # Force reactants and products to be one pseudo-molecule (bookkeeping)
                    reaction_smarts = template["reaction_smarts"]
                    reaction_smarts_one = "(" + reaction_smarts.replace(">>", ")>>(") + ")"
                    rxn = rdchiralReaction(str(reaction_smarts_one))
                    prod = rdchiralReactants(smi)
                    try:
                        # New: set return_mapped to False. Remap after in expand_one
                        reactants = rdchiralRun(rxn, prod, return_mapped=False)
                    except:
                        continue  # unknown error in rdchiral
                    
                    if not reactants:  # empty precursors
                        continue
                    
                    template_dict = {k: v for k, v in template.items()
                                    if k not in ["references", "rxn"]}
                    template_dict["template_score"] = score.item()
                    template_dict["template_rank"] = rank
                    
                    for reactant in reactants:
                        smiles_list = reactant.split(".")
                        if template.get("intra_only") and len(smiles_list) > 1:
                            # Disallowed intermolecular reaction
                            continue
                        if template.get("dimer_only") and (
                            len(set(smiles_list)) != 1 or len(smiles_list) != 2
                        ):
                            # Not a dimer
                            continue
                        if smi in smiles_list:
                            # Skip if no transformation happened
                            continue
                        
                        joined_smiles = ".".join(sorted(smiles_list))
                        if joined_smiles in smiles_to_index:
                            # Precursor was already generated by another template
                            res = result["templates"][smiles_to_index[joined_smiles]]
                            if template_dict["_id"] not in res["tforms"]:
                                res["tforms"].append(template_dict["_id"])
                            res["num_examples"] += template_dict.get("count", 0)
                        else:
                            # New precursor -> generate metadata
                            template_dict_copy = template_dict.copy()
                            template_dict_copy["tforms"] = [template_dict["_id"]]
                            template_dict_copy["num_examples"] = template_dict.get("count", 0)
                            smiles_to_index[joined_smiles] = len(result["templates"])
                            
                            result["templates"].append(template_dict_copy)
                            result["reactants"].append(reactant)
                            result["scores"].append(score.item())
                    
                    del rxn, prod, reactants
                
                # Clean up GPU memory after each SMILES prediction
                if self.device.type == 'cuda':
                    del final_fp, logits
                    torch.cuda.empty_cache()
                
                results.append(result)
        
        return results

