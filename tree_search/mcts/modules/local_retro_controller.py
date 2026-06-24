"""
Local Retro Controller - Direct model inference without HTTP API
"""
import os
import sys
from typing import Any, Dict, List, Optional
from .local_templ_rel_predictor import LocalTemplRelPredictor


class LocalRetroController:
    """Local Retro Controller for direct model inference without HTTP API"""
    
    def __init__(
        self,
        model_base_path: str,
        default_backend: str = "template_relevance",
        device: str = "auto"
    ):
        """
        Initialize the local retro controller.
        
        Args:
            model_base_path: Base path to model directories (e.g., tree_search/uspto_original_consol_Roh)
            default_backend: Default backend (currently only "template_relevance" supported)
            device: Device to use ("auto", "cuda", "cpu", etc.)
        """
        self.model_base_path = os.path.abspath(model_base_path)
        self.default_backend = default_backend
        self.device = device
        if not os.path.exists(self.model_base_path):
            raise FileNotFoundError(
                f"Model base path not found: {self.model_base_path}\n"
                f"Please set LOCAL_RETRO_MODEL_BASE_PATH environment variable correctly."
            )
        
        # Cache for predictors to avoid reloading models
        self._predictor_cache = {}
    
    def _get_predictor(self, model_name: str = None) -> LocalTemplRelPredictor:
        """
        Get or create a predictor for the given model name.
        Uses caching to avoid reloading the same model.
        
        Args:
            model_name: Model name (subdirectory name). If None or empty and model_base_path 
                       is itself a model directory, use it directly.
        """
        # Create cache key from model_dir and device
        cache_key = (self.model_base_path, str(self.device))
        
        # Return cached predictor if available
        if cache_key in self._predictor_cache:
            return self._predictor_cache[cache_key]
        
        # Create new predictor and cache it
        predictor = LocalTemplRelPredictor(model_dir=self.model_base_path, device=self.device)
        self._predictor_cache[cache_key] = predictor
        
        return predictor
    
    def __call__(
        self,
        smiles: List[str],
        backend: str = None,
        model_name: str = None,
        max_num_templates: int = 1000,
        max_cum_prob: float = 0.999,
        attribute_filter: List[Dict[str, Any]] = None,
        threshold: float = 0.3,
        top_k: int = 10
    ) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Predict retrosynthesis outcomes.
        
        Args:
            smiles: List of product SMILES strings
            backend: Backend to use (currently only "template_relevance" supported)
            model_name: Model name (optional if model_base_path is a direct model directory)
            max_num_templates: Maximum number of templates to consider
            max_cum_prob: Maximum cumulative probability threshold
            attribute_filter: List of attribute filters
            threshold: Threshold (for retrosim backend, not used for template_relevance)
            top_k: Top k (for retrosim backend, not used for template_relevance)
        
        Returns:
            List[List[Dict]] where each inner list contains results for one SMILES.
            Each result dict contains:
            - outcome: str (reactant SMILES)
            - model_score: float
            - normalized_model_score: float
            - template: Dict[str, Any]
        """
        if not backend:
            backend = self.default_backend
        
        if backend != "template_relevance":
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Only 'template_relevance' is supported in local mode."
            )
        
        if not smiles:
            return [[]]
        
        try:
            # Get predictor for this model (model_name can be None if using direct model dir)
            predictor = self._get_predictor(model_name)
            
            # Convert attribute_filter format if needed
            # The predictor expects a list of dicts with "name", "logic", "value"
            processed_attribute_filter = []
            if attribute_filter:
                for filt in attribute_filter:
                    if isinstance(filt, dict) and "name" in filt:
                        processed_attribute_filter.append(filt)
            
            # Run prediction
            predictor_results = predictor.predict(
                smiles=smiles,
                max_num_templates=max_num_templates,
                max_cum_prob=max_cum_prob,
                attribute_filter=processed_attribute_filter
            )
            
            # Convert to RetroAPI format
            # Format: List[List[Dict]] where each inner list is for one SMILES
            result = []
            for result_per_smi in predictor_results:
                if not result_per_smi.get("scores"):
                    result.append([])
                    continue
                
                # Convert to RetroAPI format
                # Reference: askcos2_core/wrappers/retro/controller.py convert_response()
                converted_results = []
                for outcome, score, template in zip(
                    result_per_smi["reactants"],
                    result_per_smi["scores"],
                    result_per_smi["templates"]
                ):
                    converted_results.append({
                        "outcome": outcome,
                        "model_score": score,
                        "normalized_model_score": score,  # template_relevance uses score as-is
                        "template": template
                    })
                
                result.append(converted_results)
            
            return result
            
        except Exception as e:
            import traceback
            import logging
            logging.error(f"Error in LocalRetroController: {e}")
            traceback.print_exc()
            return None
