"""Standalone option dataclasses for the Retro* one-step adapter.

Kept separate from ``tree_search.mcts.options`` on purpose: the adapter must not import the
full ``tree_search.mcts`` package at top level. Under the binghong RSPlanner worker the binghong
fork is on ``PYTHONPATH`` and shadows this project's top-level ``utils``/``datastructs`` packages;
``LocalExpandOneAdapter`` therefore loads the MCTS option classes lazily (see
``modules/mcts_module_loader.py``) and converts these lightweight options into MCTS options only
inside the worker.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List


class RetroBackendOption(BaseModel):
    """One-step retro backend selection and its per-call expansion limits."""
    retro_backend: str = "template_relevance"
    retro_model_name: str = "uspto_original_consol_Roh"
    max_num_templates: int = 25
    max_cum_prob: float = 1.0
    attribute_filter: List[Dict[str, Any]] = []

    # For retrosim only
    threshold: float = 0.3
    top_k: int = 10


class ExpandOneOptions(BaseModel):
    """Options for a single one-step expansion (backends, filtering, post-processing)."""
    # aliasing to v1 fields
    template_max_count: int = Field(default=100, alias="template_count")
    template_max_cum_prob: int = Field(default=0.995, alias="max_cum_template_prob")
    banned_chemicals: List[str] = Field(default=[], alias="forbidden_molecules")
    banned_reactions: List[str] = Field(default=[], alias="known_bad_reactions")

    retro_backend_options: List[RetroBackendOption] = [RetroBackendOption()]
    use_fast_filter: bool = False
    filter_threshold: float = 0.0
    retro_rerank_backend: str | None = None
    cluster_precursors: bool = False
    extract_template: bool = False
    return_reacting_atoms: bool = True
    selectivity_check: bool = False

    class Config:
        allow_population_by_field_name = True
