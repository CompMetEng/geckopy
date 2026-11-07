"""Submit/fetch kcat predictions via the OpenKineticsPredictor REST API.

Replaces the manual write-CSV -> upload -> download -> parse workflow.
"""
from __future__ import annotations

from .build_input import build_okp_input_csv
from .client import OKPClient, OKPError
from .parse_output import parse_okp_output
from .run import (
    fetch_open_kinetics_predictor,
    submit_open_kinetics_predictor,
)

__all__ = [
    "OKPClient",
    "OKPError",
    "build_okp_input_csv",
    "fetch_open_kinetics_predictor",
    "parse_okp_output",
    "submit_open_kinetics_predictor",
]
