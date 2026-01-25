"""expand_one_api module.
"""

import requests
import traceback as tb
import sys
import os
import time
import logging
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from ..options import ClusterSetting, ExpandOneOptions, RetroBackendOption
from pydantic import BaseModel, error_wrappers
from typing import Any, Dict, List, Optional

class RetroResult(BaseModel):
    # from retro_controller
    outcome: str
    model_score: float
    normalized_model_score: float
    template: Optional[Dict[str, Any]]
    reaction_id: Optional[str]
    reaction_set: Optional[str]

    # extended from postprocessing in expand_one_controller
    retro_backend: str
    retro_model_name: str
    models_predicted_by: list[tuple[str, str, float]]
    plausibility: Optional[float]
    rms_molwt: float
    num_rings: int
    scscore: float
    group_id: Optional[int]
    group_name: Optional[str]
    mapped_smiles: Optional[str]
    reacting_atoms: Optional[List[int]]
    selec_error: Optional[bool]
    mapped_outcomes: Optional[str]
    mapped_precursors: Optional[str]

    score: float
    rank: int

class ExpandOneInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: str
    retro_backend_options: List[RetroBackendOption] = [RetroBackendOption()]
    banned_chemicals: List[str] = None
    banned_reactions: List[str] = None
    use_fast_filter: bool = True
    fast_filter_threshold: float = 0.75
    retro_rerank_backend: Optional[str] = None
    cluster_precursors: bool = False
    cluster_setting: Optional[ClusterSetting] = None
    extract_template: bool = False
    return_reacting_atoms: bool = True
    selectivity_check: bool = False

class ExpandOneResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    status_code: int
    message: str
    result: Optional[List[RetroResult]]  # Allow None for cases where API returns None

class ExpandOneAPI:
    """ExpandOne API to be used as a one-step expansion engine"""
    def __init__(self, default_url: str, timeout: int = 60, max_retries: int = 3):
        self.default_url = default_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._create_session()

    def _create_session(self):
        """Create a new session with retry strategy and connection pooling"""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False
        )

        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def __call__(
        self,
        smiles: str,
        expand_one_options: ExpandOneOptions,
        url: str = None
    ) -> Optional[List[Dict[str, any]]]:
        if not url:
            url = self.default_url

        # Overriding backend option fields if provided in expand_one_options
        retro_backend_options = expand_one_options.retro_backend_options
        # if expand_one_options.template_max_count:
        #     for option in expand_one_options.retro_backend_options:
        #         option.max_num_templates = expand_one_options.template_max_count
        # if expand_one_options.template_max_cum_prob:
        #     for option in expand_one_options.retro_backend_options:
        #         option.max_cum_prob = expand_one_options.template_max_cum_prob

        # Modify the overriding logic to apply only if not provided for each option
        for option in expand_one_options.retro_backend_options:
            if not option.max_num_templates and expand_one_options.template_max_count:
                option.max_num_templates = expand_one_options.template_max_count
            if not option.max_cum_prob and expand_one_options.template_max_cum_prob:
                option.max_cum_prob = expand_one_options.template_max_cum_prob

        cluster_setting = expand_one_options.cluster_setting
        if cluster_setting is not None:
            cluster_setting = cluster_setting.dict()

        input = {
            "smiles": smiles,
            "retro_backend_options": [
                option.dict() for option in retro_backend_options
            ],
            "banned_chemicals": expand_one_options.banned_chemicals,
            "banned_reactions": expand_one_options.banned_reactions,
            "use_fast_filter": expand_one_options.use_fast_filter,
            "fast_filter_threshold": expand_one_options.filter_threshold,
            "retro_rerank_backend": expand_one_options.retro_rerank_backend,
            "cluster_precursors": expand_one_options.cluster_precursors,
            "cluster_setting": cluster_setting,
            "extract_template": expand_one_options.extract_template,
            "return_reacting_atoms": expand_one_options.return_reacting_atoms,
            "selectivity_check": expand_one_options.selectivity_check
        }
        # additional validation. Sending null/none value to FastAPI seems to
        # fail the validation check and break the defaulting mechanism
        input = {k: v for k, v in input.items() if v is not None}

        ExpandOneInput(**input)                     # merely validate the input

        # Retry logic with session refresh on connection errors
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.session.post(
                    url=url,
                    json=input,
                    timeout=self.timeout
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                response_data = response.json()

                # Extract result first, before validation
                # Handle cases where result might be None or missing
                result = response_data.get("result")
                if result is None:
                    # Try alternative field name
                    result = response_data.get("results")

                # If result is still None, return empty list (not None)
                if result is None:
                    logging.warning(f"ExpandOneAPI: result field is None in response for {smiles}")
                    return []

                # Validate response structure (but be lenient about result being None)
                try:
                    # Create a copy for validation with None replaced by empty list
                    validation_data = response_data.copy()
                    if validation_data.get("result") is None:
                        validation_data["result"] = []
                    ExpandOneResponse(**validation_data)  # validate the response structure
                except error_wrappers.ValidationError as e:
                    # Log warning but continue - we already extracted the result
                    logging.warning(f"ExpandOneAPI validation error (attempt {attempt + 1}/{max_attempts}): {e}")
                    # If validation fails but we have a result, use it anyway
                    if result is not None:
                        return result if isinstance(result, list) else []

                # Ensure result is a list
                if not isinstance(result, list):
                    logging.warning(f"ExpandOneAPI: result is not a list, got {type(result)}")
                    return []

                return result

            except requests.exceptions.ConnectionError as e:
                logging.warning(f"ExpandOneAPI connection error (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    # Recreate session on connection error
                    self._create_session()
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    logging.error(f"ExpandOneAPI connection failed after {max_attempts} attempts")
                    tb.print_exc()
                    return None

            except requests.exceptions.Timeout as e:
                logging.warning(f"ExpandOneAPI timeout error (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    # Recreate session on timeout
                    self._create_session()
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    logging.error(f"ExpandOneAPI timeout after {max_attempts} attempts")
                    tb.print_exc()
                    return None

            except requests.exceptions.HTTPError as e:
                logging.warning(f"ExpandOneAPI HTTP error (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    logging.error(f"ExpandOneAPI HTTP error after {max_attempts} attempts: {e}")
                    tb.print_exc()
                    return None

            except Exception as e:
                logging.error(f"ExpandOneAPI unexpected error (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    self._create_session()  # Recreate session on unexpected error
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    logging.error(f"ExpandOneAPI error after {max_attempts} attempts: {e}")
                    tb.print_exc()
                    return None

        return None

if __name__ == "__main__":
    # default_url = "http://0.0.0.0:9410/predictions/reaxys"
    default_url = "http://0.0.0.0:9301/get_outcomes"
    expand_one_api = ExpandOneAPI(default_url=default_url)
    expand_one_options = ExpandOneOptions()
    retro_results = expand_one_api(smiles="CC(C)Sc1ncccc1F", expand_one_options=expand_one_options)
    print(retro_results)
    # "CC(C)Sc1ncccc1F",