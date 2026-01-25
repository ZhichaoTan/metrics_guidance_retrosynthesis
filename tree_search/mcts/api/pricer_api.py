"""pricer_api module.
"""

import requests
import traceback as tb
from pydantic import BaseModel
from typing import List, Optional, Tuple

class PricerInput(BaseModel):
    # mirroring the (default) wrapper; convenient to turn into a client library
    smiles: str
    source: str | list[str] | None = None
    canonicalize: bool

class PricerResponse(BaseModel):
    # mirroring the (default) wrapper, but without BaseResponse (semi-hardcode)
    _id: str
    smiles: str
    ppg: float
    source: str
    properties: Optional[List[dict]]

class PricerAPI:
    """Pricer API to be used as a Pricer"""
    def __init__(self, default_url: str):
        self.default_url = default_url
        self.session = requests.Session()

    def __call__(
        self,
        smiles: str,
        source: str | list[str] | None = None,
        canonicalize: bool = False,
        url: str = None
    ) -> Tuple[float, Optional[dict]]:
        if not url:
            url = self.default_url

        input = {
            "smiles": smiles,
            "source": source,
            "canonicalize": canonicalize
        }

        PricerInput(**input)                        # merely validate the input
        try:
            response = self.session.post(url=url, params=input).json()
            if not response:                        # not found
                return 0.0, None

            PricerResponse(**response)              # merely validate the response
        except requests.ConnectionError as e:
            # Handle the connection error appropriately
            print("Connection error for PricerAPI:")
            tb.print_exc()

            return 0.0, None
        except Exception as e:
            # Handle any other exception that might occur
            print("An error occurred for PricerAPI:")
            tb.print_exc()

            return 0.0, None

        purchase_price = response.get("ppg", 0.0)
        properties = response.get("properties", None)

        return purchase_price, properties
