"""Local ExpandOne API with a process-local shared controller."""

from typing import Any, Dict, List, Optional

from .expand_one_controller import ExpandOneController
from tree_search.mcts.options import ExpandOneOptions

_SHARED_CONTROLLER: Optional[ExpandOneController] = None


def _get_shared_controller() -> ExpandOneController:
    global _SHARED_CONTROLLER
    if _SHARED_CONTROLLER is None:
        _SHARED_CONTROLLER = ExpandOneController()
    return _SHARED_CONTROLLER

class LocalExpandOneAPI:
    def __init__(self):
        self.controller = _get_shared_controller()

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
                extract_template=expand_one_options.extract_template,
                return_reacting_atoms=expand_one_options.return_reacting_atoms,
                selectivity_check=expand_one_options.selectivity_check
            )

            return results

        except Exception as e:
            import traceback
            import logging
            logging.error(f"Error in LocalExpandOneAPI: {e}")
            if any(
                option.retro_backend == "binghong_mlp"
                for option in expand_one_options.retro_backend_options
            ):
                raise
            traceback.print_exc()
            return None
