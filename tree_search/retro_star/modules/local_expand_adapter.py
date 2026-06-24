"""Adapter: Retro* ExpandOneOptions -> MCTS LocalExpandOneAPI (read-only use of mcts modules)."""
from typing import Any, Dict, List, Optional

import importlib.util
import os

from tree_search.retro_star.modules.mcts_module_loader import load_local_expand_one_api_class
from tree_search.retro_star.options import ExpandOneOptions


def _load_mcts_options_classes():
    mcts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "mcts",
    )
    spec = importlib.util.spec_from_file_location(
        "retro_mcts_options", os.path.join(mcts_dir, "options.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExpandOneOptions, mod.RetroBackendOption


def _to_mcts_options(expand_one_options: ExpandOneOptions):
    MCTSExpandOneOptions, MCTSRetroBackendOption = _load_mcts_options_classes()
    retro_opts = expand_one_options.retro_backend_options
    mcts = MCTSExpandOneOptions(
        banned_chemicals=list(expand_one_options.banned_chemicals),
        banned_reactions=list(expand_one_options.banned_reactions),
        use_fast_filter=expand_one_options.use_fast_filter,
        filter_threshold=expand_one_options.filter_threshold,
        retro_rerank_backend=expand_one_options.retro_rerank_backend,
        cluster_precursors=expand_one_options.cluster_precursors,
        extract_template=expand_one_options.extract_template,
        return_reacting_atoms=expand_one_options.return_reacting_atoms,
        selectivity_check=expand_one_options.selectivity_check,
    )
    mcts.retro_backend_options = []
    for opt in retro_opts:
        mcts.retro_backend_options.append(
            MCTSRetroBackendOption(
                retro_backend=opt.retro_backend,
                retro_model_name=opt.retro_model_name,
                max_num_templates=opt.max_num_templates,
                max_cum_prob=opt.max_cum_prob,
                attribute_filter=list(opt.attribute_filter),
                threshold=opt.threshold,
                top_k=opt.top_k,
            )
        )
    return mcts


class LocalExpandOneAdapter:
    """Run a one-step expansion through the MCTS ``LocalExpandOneAPI`` from Retro* options."""

    def __init__(self):
        LocalExpandOneAPI = load_local_expand_one_api_class()
        self._api = LocalExpandOneAPI()

    def __call__(
        self,
        smiles: str,
        expand_one_options: ExpandOneOptions,
        url: str = None,
    ) -> Optional[List[Dict[str, Any]]]:
        return self._api(smiles, _to_mcts_options(expand_one_options), url=url)


class LocalHistorianStub:
    """No-op historian: Retro* does not use reaction-count history, so report zeros."""

    def __call__(
        self,
        smiles: str,
        template_sets=None,
        canonicalize: bool = False,
    ) -> dict:
        return {"as_reactant": 0, "as_product": 0}
