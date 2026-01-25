"""options module.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union

class RetroBackendOption(BaseModel):
    retro_backend: str = "template_relevance"
    # retro_model_name: str = "reaxys"
    # retro_model_name: str = "simplified_ged_threshold_0"
    # retro_model_name: str = "uspto_full_ori"
    retro_model_name: str = "uspto_original_consol_Roh"
    max_num_templates: int = 25
    max_cum_prob: float = 1
    attribute_filter: List[Dict[str, Any]] = []

    # For retrosim only
    threshold: float = 0.3
    top_k: int = 10

    # "original_ged_threshold_0",
    # "uspto_full_ori",
    # "uspto_full_filtered_5_strategy_reactions_all_ged"

class ClusterSetting(BaseModel):
    feature: str = "original"
    cluster_method: str = "rxn_class"
    fp_type: str = "morgan"
    fp_length: int = 512
    fp_radius: int = 1
    classification_threshold: float = 0.2

class ExpandOneOptions(BaseModel):
    # aliasing to v1 fields
    template_max_count: int = Field(default=100, alias="template_count")
    template_max_cum_prob: int = Field(default=0.995, alias="max_cum_template_prob")
    banned_chemicals: List[str] = Field(default=[], alias="forbidden_molecules")
    banned_reactions: List[str] = Field(default=[], alias="known_bad_reactions")

    retro_backend_options: List[RetroBackendOption] = [RetroBackendOption()]
    use_fast_filter: bool = False
    filter_threshold: float = 0
    retro_rerank_backend: str | None = None
    cluster_precursors: bool = False
    cluster_setting: ClusterSetting = None
    extract_template: bool = False
    return_reacting_atoms: bool = True
    selectivity_check: bool = False

    # Bond-based expand_one options
    use_bond_based_expand: bool = False
    top_k_bonds: int = 10
    max_feasible_templates: int = 50
    template_library_path: Optional[str] = None
    use_mcs_matching: bool = True
    max_missing_atoms: int = 2
    complexity_metric: str = "bertz"

    class Config:
        allow_population_by_field_name = True

class BuildTreeOptions(BaseModel):
    expansion_time: int = 12000
    max_iterations: Optional[int] = 500
    max_chemicals: Optional[int] = None
    max_reactions: Optional[int] = None
    max_templates: Optional[int] = None
    max_branching: int = 25
    max_depth: int = 10
    exploration_weight: float = 1.0
    return_first: bool = True
    max_trees: int = 500
    max_ppg: Optional[float] = None
    max_scscore: Optional[float] = None
    max_elements: Optional[Dict[str, int]] = None
    min_history: Optional[Dict[str, int]] = None
    property_criteria: Optional[List[Dict[str, Any]]] = None
    termination_logic: Optional[Dict[str, List[str]]] = {"and": ["buyable"]}
    buyables_source: Optional[Union[str, List[str]]] = None
    custom_buyables: Optional[List[str]] = None

class EnumeratePathsOptions(BaseModel):
    path_format: Literal["json", "graph"] = "json"
    json_format: Literal["treedata", "nodelink"] = "treedata"
    sorting_metric: Literal[
        "plausibility",
        "number_of_starting_materials",
        "number_of_reactions",
        "score"
    ] = "plausibility"
    validate_paths: bool = True
    score_trees: bool = False
    cluster_trees: bool = False
    cluster_method: Literal["hdbscan", "kmeans"] = "hdbscan"
    min_samples: int = 5
    min_cluster_size: int = 5
    paths_only: bool = False
    max_paths: int = 200
