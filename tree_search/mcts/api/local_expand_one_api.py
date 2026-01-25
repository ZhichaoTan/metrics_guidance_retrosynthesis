"""
Local ExpandOne API - Direct call to ExpandOneController without HTTP
"""
import os
import sys
from typing import Any, Dict, List, Optional

# Add expand_one to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Current: .../tree_search/mcts/api/local_expand_one_api.py
# Go up: api -> mcts -> tree_search
tree_search_dir = os.path.dirname(os.path.dirname(current_file_dir))
expand_one_path = os.path.join(tree_search_dir, "expand_one")
if expand_one_path not in sys.path:
    sys.path.insert(0, expand_one_path)

# Note: expand_one_controller may need to be available externally or imported differently
# from expand_one_controller import ExpandOneController
from ..options import ExpandOneOptions, RetroBackendOption

class LocalExpandOneAPI:
    """Local ExpandOne API that directly calls ExpandOneController without HTTP"""

    def __init__(self, default_url: str = None, timeout: int = 60, max_retries: int = 3):
        """
        Initialize local expand one API.

        Args:
            default_url: Not used in local mode, kept for compatibility
            timeout: Not used in local mode, kept for compatibility
            max_retries: Not used in local mode, kept for compatibility
        """
        # Initialize the controller (it will use local mode if USE_LOCAL_RETRO is set)
        self.controller = ExpandOneController()

    def __call__(
        self,
        smiles: str,
        expand_one_options: ExpandOneOptions,
        url: str = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Call expand_one directly using ExpandOneController.

        Args:
            smiles: Product SMILES string
            expand_one_options: ExpandOneOptions object
            url: Not used in local mode, kept for compatibility

        Returns:
            List of retro results, same format as ExpandOneAPI
        """
        try:
            # Call get_outcomes directly
            results = self.controller.get_outcomes(
                smiles=smiles,
                retro_backend_options=expand_one_options.retro_backend_options,
                banned_chemicals=expand_one_options.banned_chemicals,
                banned_reactions=expand_one_options.banned_reactions,
                use_fast_filter=expand_one_options.use_fast_filter,
                fast_filter_threshold=expand_one_options.filter_threshold,
                retro_rerank_backend=expand_one_options.retro_rerank_backend,
                cluster_precursors=expand_one_options.cluster_precursors,
                cluster_setting=expand_one_options.cluster_setting,
                extract_template=expand_one_options.extract_template,
                return_reacting_atoms=expand_one_options.return_reacting_atoms,
                selectivity_check=expand_one_options.selectivity_check
            )

            return results

        except Exception as e:
            import traceback
            import logging
            logging.error(f"Error in LocalExpandOneAPI: {e}")
            traceback.print_exc()
            return None

