"""Submit and fetch OpenKineticsPredictor jobs.

Two entry points mirroring the MATLAB submit/fetchOpenKineticsPredictor:

* ``submit_open_kinetics_predictor`` builds the input CSV and submits a
  kcat-prediction job, persisting the job id to ``data/OKP_job.txt``.
* ``fetch_open_kinetics_predictor`` checks status; on completion it
  downloads ``data/OKP_output.csv`` and parses it into a kcat_list
  DataFrame. ``use_stored`` re-parses the saved file without an API call.

Obtaining an API key (free, no registration):
    1. Open https://predictor.openkinetics.org/api-docs in a browser.
    2. In the API key generator, click "Generate" and copy the key
       (shown only once; revoke + regenerate if lost).
The key is a secret: provide it via the ``api_key`` argument, the
``OKP_API_KEY`` environment variable, or ``data/okpApiKey.txt`` (the
latter is git-ignored). It is never read from the model adapter.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd
import requests

from ...databases.dlkcat_ignore_lists import load_dlkcat_ignore_lists
from .build_input import build_okp_input_csv
from .client import OKPClient, OKPError
from .parse_output import parse_okp_output

if TYPE_CHECKING:
    from ...adapter import ModelAdapter
    from ...databases.dlkcat_ignore_lists import DLKcatIgnoreLists
    from ...ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

_BASE_URL = "https://predictor.openkinetics.org/api/v1"


def submit_open_kinetics_predictor(
    model: "EcModel",
    *,
    overwrite: bool = False,
    api_key: Optional[str] = None,
    method: Optional[str] = None,
    ec_rxns: Optional[Iterable[str]] = None,
    only_with_smiles: bool = True,
    ignore_lists: Optional["DLKcatIgnoreLists"] = None,
    adapter: Optional["ModelAdapter"] = None,
    base_url: str = _BASE_URL,
    session: Optional[requests.Session] = None,
) -> str:
    """Build the OKP input and submit a kcat-prediction job.

    Parameters
    ----------
    model
        EcModel with ``ec.sequence`` and per-metabolite SMILES.
    overwrite
        Rebuild ``data/OKP.csv`` even if it exists (default False
        reuses an existing file).
    api_key
        OKP API key. If omitted, resolved from ``OKP_API_KEY`` or
        ``data/okpApiKey.txt`` (see the module docstring).
    method
        Predictor method. Resolved as: this arg -> ``params.okp.method``
        -> ``'CataPro'``.
    ec_rxns
        Optional iterable of reaction IDs to include (default all).
    only_with_smiles
        Drop entries without a SMILES (default True).
    ignore_lists
        Pre-loaded ``DLKcatIgnoreLists``; loaded from the project data
        folder (with bundled fallback) if omitted.
    adapter
        Model adapter; falls back to ``model.adapter``.
    base_url, session
        For testing / custom deployments.

    Returns
    -------
    str
        The OKP job id (also written to ``data/OKP_job.txt``).
    """
    from ...adapter import resolve_adapter
    adapter = resolve_adapter(
        model, adapter, purpose="submit_open_kinetics_predictor needs "
        "params.path and params.okp",
    )
    data_dir = Path(adapter.params.path) / "data"
    api_key = _resolve_api_key(api_key, adapter)
    method = method or adapter.params.okp.method
    okp = adapter.params.okp

    okp_csv = data_dir / "OKP.csv"
    if okp_csv.is_file() and not overwrite:
        csv_text = okp_csv.read_text(encoding="utf-8")
        logger.info("Using existing %s (set overwrite=True to rebuild).", okp_csv)
    else:
        if ignore_lists is None:
            ignore_lists = load_dlkcat_ignore_lists(data_dir)
        csv_text = build_okp_input_csv(
            model, ignore_lists, ec_rxns=ec_rxns,
            only_with_smiles=only_with_smiles,
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        okp_csv.write_text(csv_text, encoding="utf-8")

    client = OKPClient(api_key, base_url=base_url, session=session)
    resp = client.submit(
        csv_text.encode("utf-8"),
        method=method,
        handle_long_sequences=okp.handle_long_sequences,
        include_similarity_columns=okp.include_similarity_columns,
        canonicalize_substrates=okp.canonicalize_substrates,
    )
    job_id = resp.get("jobId")
    if not job_id:
        raise OKPError(f"OKP submit returned no jobId: {resp!r}")

    _write_job_metadata(data_dir / "OKP_job.txt", job_id, method)
    logger.info("Submitted OKP job %s (method: %s).", job_id, method)
    return job_id


def fetch_open_kinetics_predictor(
    model: "EcModel",
    use_stored: bool = False,
    *,
    job_id: Optional[str] = None,
    wait: bool = False,
    poll_interval: float = 30.0,
    timeout: float = 3600.0,
    api_key: Optional[str] = None,
    adapter: Optional["ModelAdapter"] = None,
    base_url: str = _BASE_URL,
    session: Optional[requests.Session] = None,
    out_file: Optional[str | Path] = None,
) -> tuple[bool, Optional[pd.DataFrame]]:
    """Check an OKP job and, when finished, download + parse the result.

    Parameters
    ----------
    model
        EcModel used to map results back to reactions.
    use_stored
        If True, skip the API and parse ``data/OKP_output.csv``.
    job_id
        OKP job id. If omitted, read from ``data/OKP_job.txt``.
    wait
        If True, poll until the job finishes (or ``timeout`` elapses);
        if False, report once and return.
    poll_interval, timeout
        Polling cadence and overall deadline (seconds) when ``wait``.
    api_key, adapter, base_url, session
        Key resolution / adapter / testing knobs (see submit).
    out_file
        Where the result is stored, and read from with ``use_stored``
        (default: ``data/OKP_output.csv``).

    Returns
    -------
    (done, kcat_list)
        ``done`` is True when a result was obtained; ``kcat_list`` is the
        parsed DataFrame then, otherwise ``None``. The DataFrame matches
        the ``read_dlkcat_output`` / ``fuzzy_kcat_matching`` schema and
        feeds ``apply_kcat_list``.
    """
    from ...adapter import resolve_adapter
    adapter = resolve_adapter(
        model, adapter, purpose="fetch_open_kinetics_predictor needs params.path",
    )
    data_dir = Path(adapter.params.path) / "data"
    out_file = Path(out_file) if out_file is not None else data_dir / "OKP_output.csv"

    if use_stored:
        if not out_file.is_file():
            raise FileNotFoundError(
                f"use_stored=True but no stored result at {out_file}. "
                f"Run fetch_open_kinetics_predictor(use_stored=False) first."
            )
        return True, parse_okp_output(model, out_file)

    api_key = _resolve_api_key(api_key, adapter)
    if job_id is None:
        job_id = _read_job_id(data_dir / "OKP_job.txt")
    client = OKPClient(api_key, base_url=base_url, session=session)

    deadline = time.monotonic() + timeout
    while True:
        status = client.status(job_id)
        state = str(status.get("status", "")).lower()

        if state == "completed":
            csv_text = client.result_csv(job_id)
            data_dir.mkdir(parents=True, exist_ok=True)
            out_file.write_text(csv_text, encoding="utf-8")
            logger.info("OKP job %s completed; result stored at %s.",
                        job_id, out_file)
            return True, parse_okp_output(model, csv_text)

        if state == "failed":
            raise OKPError(
                f"OKP job {job_id} failed. Check "
                f"https://predictor.openkinetics.org/ for details."
            )

        progress = _progress_text(status)
        if wait and time.monotonic() < deadline:
            logger.info("OKP job %s status: %s%s. Waiting %.0f s...",
                        job_id, status.get("status"), progress, poll_interval)
            time.sleep(poll_interval)
            continue

        logger.info(
            "OKP job %s not finished (status: %s%s). Try again later, or "
            "check https://predictor.openkinetics.org/.",
            job_id, status.get("status"), progress,
        )
        return False, None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_api_key(api_key: Optional[str], adapter: "ModelAdapter") -> str:
    if api_key:
        return api_key.strip()
    env_key = os.environ.get("OKP_API_KEY")
    if env_key:
        return env_key.strip()
    key_file = Path(adapter.params.path) / "data" / "okpApiKey.txt"
    if key_file.is_file():
        return key_file.read_text(encoding="utf-8").strip()
    raise ValueError(
        "No OpenKineticsPredictor API key found. Pass api_key=..., set the "
        "OKP_API_KEY environment variable, or place the key in "
        f"{key_file}. Generate one at https://predictor.openkinetics.org/."
    )


def _read_job_id(meta_file: Path) -> str:
    if not meta_file.is_file():
        raise FileNotFoundError(
            f"No job_id given and no metadata file at {meta_file}. Run "
            f"submit_open_kinetics_predictor first, or pass job_id explicitly."
        )
    for line in meta_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("jobId:"):
            return stripped.split(":", 1)[1].strip()
    raise ValueError(f"Could not read a jobId from {meta_file}.")


def _write_job_metadata(meta_file: Path, job_id: str, method: str) -> None:
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    submitted = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_file.write_text(
        f"jobId: {job_id}\nmethod: {method}\nsubmittedAt: {submitted}\n",
        encoding="utf-8",
    )


def _progress_text(status: dict) -> str:
    progress = status.get("progress") or {}
    total = progress.get("predictionsTotal")
    made = progress.get("predictionsMade")
    if total:
        return f" ({made}/{total} predictions)"
    return ""
