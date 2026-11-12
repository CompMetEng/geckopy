"""Submit/fetch kcat predictions via the OpenKineticsPredictor REST API.

The engine behind ``kcats_from_okp`` and ``kcats_from_dlkcat(via="okp")``:
its submit, fetch, input and output functions are private.
"""
from __future__ import annotations

from .client import OKPClient, OKPError

__all__ = [
    "OKPClient",
    "OKPError",
]
