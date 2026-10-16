"""Gather the kcats of one source and keep them as a kcat source file.

One function per source, each with its own options, each writing
``data/kcats/<source>.tsv`` in the project folder by default (see
:mod:`geckopy.gather_kcats.kcat_sources`). GECKO (MATLAB) has the same
functions as ``kcatsFromBrenda``, ``kcatsFromDLKcat`` and ``kcatsFromOKP``.
"""
from __future__ import annotations

import logging
import lzma
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd

from ..databases import load_brenda_data, load_dlkcat_ignore_lists, load_phyl_dist
from .fuzzy_kcat_matching import fuzzy_kcat_matching
from .kcat_sources import kcat_source_file, save_kcat_source
from .open_kinetics_predictor import fetch_open_kinetics_predictor, submit_open_kinetics_predictor
from .open_kinetics_predictor.parse_output import parse_okp_output
from .read_dlkcat_output import read_dlkcat_output
from .run_dlkcat import run_dlkcat
from .write_dlkcat_input import write_dlkcat_input

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..databases import BrendaData, PhylDist
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


def _adapter(model: "EcModel", adapter: Optional["ModelAdapter"], purpose: str) -> "ModelAdapter":
    from ..adapter import resolve_adapter

    return resolve_adapter(model, adapter, purpose=purpose)


def _rxn_subset(ec_rxns: Optional[Iterable[str]]) -> Optional[list[str]]:
    if ec_rxns is None:
        return None
    ec_rxns = list(ec_rxns)
    if not ec_rxns:
        raise ValueError("ec_rxns is empty; leave it out to gather for all reactions.")
    return ec_rxns


def _warn_subset(ec_rxns: Optional[list[str]], model: "EcModel", path: Path) -> None:
    if ec_rxns is not None and len(set(ec_rxns)) < len(model.ec.rxns) and path.is_file():
        logger.warning("%s is replaced by kcats for %d reactions only. Give save_as to keep it.",
                       path, len(ec_rxns))


def _brenda_release(folder: Path) -> str:
    """The release line at the top of kcat.tsv, e.g. 'BRENDA release 2026.1'."""
    for path, opener in ((folder / "kcat.tsv.xz", lzma.open), (folder / "kcat.tsv", open)):
        if path.is_file():
            with opener(path, "rt", encoding="utf-8") as f:
                m = re.match(r"^#\s*(BRENDA release \S+)", f.readline())
            return m.group(1) if m else ""
    return ""


def kcats_from_brenda(
    model: "EcModel",
    *,
    aggregation: Optional[str] = None,
    force_wildcard_level: int = 0,
    ec_rxns: Optional[Iterable[str]] = None,
    brenda: Optional["BrendaData"] = None,
    phyl_dist: Optional["PhylDist"] = None,
    save: bool = True,
    save_as: str = "brenda",
    adapter: Optional["ModelAdapter"] = None,
) -> pd.DataFrame:
    """Gather kcats from BRENDA by fuzzy matching and keep them as ``brenda.tsv``.

    Runs locally on the BRENDA files of geckopy (see
    :func:`~geckopy.gather_kcats.fuzzy_kcat_matching`).

    Parameters
    ----------
    model
        EcModel with EC numbers.
    aggregation
        Which BRENDA value per EC number, substrate and organism is used:
        ``"max"`` or ``"median"`` (default: ``params.kcat_aggregate_brenda``).
    force_wildcard_level
        Force a minimum wildcard level in the EC numbers.
    ec_rxns
        ``model.ec.rxns`` to gather kcats for (default: all). With the
        default ``save_as`` the file then only holds those reactions.
    brenda, phyl_dist
        Loaded BRENDA data and KEGG phylogenetic distances (default: loaded
        from the adapter's folders).
    save, save_as
        Write the source file, and its name without ``.tsv``.
    adapter
        Model adapter (default: ``model.adapter``).

    Returns
    -------
    pandas.DataFrame
        The kcat list.
    """
    adapter = _adapter(model, adapter, "kcats_from_brenda needs the BRENDA folder and params")
    if aggregation is None:
        aggregation = adapter.params.kcat_aggregate_brenda
    aggregation = aggregation.lower()
    if aggregation not in ("max", "median"):
        raise ValueError("aggregation must be 'max' or 'median'")
    ec_rxns = _rxn_subset(ec_rxns)
    if save:
        path = kcat_source_file(save_as, model=model, adapter=adapter)  # fails before the matching
        _warn_subset(ec_rxns, model, path)
    folder = Path(adapter.get_brenda_db_folder())
    if brenda is None:
        brenda = load_brenda_data(folder)
    if phyl_dist is None:
        phyl_dist = load_phyl_dist(adapter.get_phyl_dist_path())
    kcat_list = fuzzy_kcat_matching(
        model, brenda, phyl_dist, ec_rxns=ec_rxns,
        force_wildcard_level=force_wildcard_level, aggregate=aggregation, adapter=adapter,
    )
    if save:
        parameters = {"aggregation": aggregation, "force_wildcard_level": force_wildcard_level}
        if ec_rxns is not None:
            parameters["ec_rxns"] = len(ec_rxns)
        path = save_kcat_source(
            model, kcat_list, save_as, adapter=adapter, generator="kcats_from_brenda",
            parameters=parameters, data=_brenda_release(folder),
        )
        logger.info("BRENDA kcats for %d reactions written to %s",
                    int((kcat_list["kcat"] > 0).sum()), path)
    return kcat_list


def kcats_from_dlkcat(
    model: "EcModel",
    *,
    run: bool = False,
    ec_rxns: Optional[Iterable[str]] = None,
    save: bool = True,
    save_as: str = "dlkcat",
    adapter: Optional["ModelAdapter"] = None,
) -> pd.DataFrame:
    """Gather kcats from DLKcat and keep them as ``dlkcat.tsv``.

    With ``run=True`` the DLKcat input is written and DLKcat runs through
    its Docker image (which needs Docker); otherwise the output of an
    earlier run in ``data/DLKcat.tsv`` is read. A run records how many
    reactions it covered in ``data/DLKcat_run.txt``, so that the source file
    says so also when the output is read later.

    Parameters
    ----------
    model
        EcModel with metabolite SMILES and protein sequences.
    run
        Write the input and run DLKcat (default False).
    ec_rxns
        ``model.ec.rxns`` to run DLKcat for (default: all); needs ``run``.
    save, save_as, adapter
        As in :func:`kcats_from_brenda`.

    Returns
    -------
    pandas.DataFrame
        The kcat list.
    """
    adapter = _adapter(model, adapter, "kcats_from_dlkcat needs params.path")
    ec_rxns = _rxn_subset(ec_rxns)
    if ec_rxns is not None and not run:
        raise ValueError("ec_rxns selects the reactions DLKcat is run for, so it needs run=True.")
    if save:
        path = kcat_source_file(save_as, model=model, adapter=adapter)
        if run:
            _warn_subset(ec_rxns, model, path)
    data_dir = Path(adapter.params.path) / "data"
    dlkcat_file = data_dir / "DLKcat.tsv"
    run_file = data_dir / "DLKcat_run.txt"
    if run:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "DLKcat runs in a Docker image, but Docker was not found. Install it from "
                "https://www.docker.com, or call kcats_from_dlkcat without run=True to read "
                "an earlier DLKcat output."
            )
        write_dlkcat_input(model, dlkcat_file, load_dlkcat_ignore_lists(data_dir),
                           ec_rxns=ec_rxns, overwrite=True)
        run_dlkcat(dlkcat_file)
        run_file.write_text(f"ecRxns: {len(set(ec_rxns))}\n" if ec_rxns is not None else "",
                            encoding="utf-8", newline="\n")
    elif not dlkcat_file.is_file():
        raise FileNotFoundError(
            f"No DLKcat output at {dlkcat_file}. Run kcats_from_dlkcat(model, run=True) first."
        )
    kcat_list = read_dlkcat_output(model, dlkcat_file)
    if save:
        parameters = {"run": bool(run)}
        if run_file.is_file():
            m = re.search(r"ecRxns:\s*(\d+)", run_file.read_text(encoding="utf-8"))
            if m:
                parameters["ec_rxns"] = int(m.group(1))
        path = save_kcat_source(
            model, kcat_list, save_as, adapter=adapter, generator="kcats_from_dlkcat",
            parameters=parameters,
        )
        logger.info("DLKcat kcats written to %s", path)
    return kcat_list


_JOB_KEYS = ("jobId", "method", "submittedAt", "ecRxns", "resultOf")
_DATABASE_SOURCES = {"okp-brenda", "okp-sabio-rk", "okp-uniprot"}


def _read_job_file(path: Path) -> dict[str, str]:
    job: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z]\w*):\s*(.*?)\s*$", line)
            if m:
                job[m.group(1)] = m.group(2)
    return job


def _write_job_file(path: Path, job: dict[str, str]) -> None:
    path.write_text("".join(f"{k}: {job[k]}\n" for k in _JOB_KEYS if k in job),
                    encoding="utf-8", newline="\n")


def _adopt_shared_result(model: "EcModel", data_dir: Path, method: str, job_file: Path, out_file: Path) -> None:
    """Take over a result that ``fetch_open_kinetics_predictor`` stored by
    itself, in ``data/OKP_output.csv`` with ``data/OKP_job.txt``, only when
    its job is for this method and its predictions are labelled with this
    method."""
    shared = data_dir / "OKP_output.csv"
    job = _read_job_file(data_dir / "OKP_job.txt")
    ok = shared.is_file() and "jobId" in job and job.get("method", "").lower() == method.lower()
    if ok:
        sources = [str(x) for x in parse_okp_output(model, shared)["source"]]
        predicted = [x for x in sources if x.lower().startswith("okp-") and x.lower() not in _DATABASE_SOURCES]
        ok = bool(predicted) and all(x.lower() == f"okp-{method}".lower() for x in predicted)
    if not ok:
        raise FileNotFoundError(
            f"No stored OKP result for {method} ({out_file}). Call kcats_from_okp without "
            f"use_stored to submit or fetch a job."
        )
    shutil.copyfile(shared, out_file)
    _write_job_file(job_file, {**job, "method": method, "resultOf": job["jobId"]})


def kcats_from_okp(
    model: "EcModel",
    *,
    method: Optional[str] = None,
    wait: bool = False,
    resubmit: bool = False,
    use_stored: bool = False,
    ec_rxns: Optional[Iterable[str]] = None,
    save: bool = True,
    save_as: Optional[str] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> tuple[Optional[pd.DataFrame], bool]:
    """Gather kcats from OpenKineticsPredictor and keep them as ``okp_<method>.tsv``.

    OKP is a web service with a job queue, so this can be called repeatedly:
    without a job for ``method`` it submits one; with a job it fetches the
    result, and writes the file once the job is done. Each method keeps its
    own job in ``data/OKP_job_<method>.txt`` and its own result in
    ``data/OKP_output_<method>.csv``, so jobs of several methods can run at
    the same time. The job file records which job the stored result came
    from; ``use_stored`` only uses a result of the current job. An API key
    is needed.

    Parameters
    ----------
    model
        EcModel with metabolite SMILES and protein sequences.
    method
        OKP method, e.g. ``"CataPro"``, ``"CatRange"``, ``"UniKP"`` (default:
        ``params.okp.method``).
    wait
        Wait until the job is done.
    resubmit
        Submit a new job even if one exists for this method; its earlier
        result is removed.
    use_stored
        Read the stored result of this method's job without contacting OKP.
    ec_rxns
        ``model.ec.rxns`` to predict for (default: all); only used when a
        job is submitted.
    save, save_as, adapter
        As in :func:`kcats_from_brenda`; ``save_as`` defaults to
        ``okp_<method>``, in lower case, with other characters than letters
        and digits as ``_``.

    Returns
    -------
    (kcat_list, done)
        The kcat list (``None`` while the job is not done), and whether the
        result was available.
    """
    adapter = _adapter(model, adapter, "kcats_from_okp needs params.path")
    if not method:
        method = adapter.params.okp.method
    ec_rxns = _rxn_subset(ec_rxns)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", method).lower()
    data_dir = Path(adapter.params.path) / "data"
    job_file = data_dir / f"OKP_job_{tag}.txt"
    out_file = data_dir / f"OKP_output_{tag}.csv"
    name = save_as or f"okp_{tag}"
    job = _read_job_file(job_file)
    submitting = not use_stored and (resubmit or "jobId" not in job)
    if ec_rxns is not None and not submitting:
        raise ValueError(
            f"ec_rxns selects the reactions of a new job, but this call submits none (a job for "
            f"{method} exists; use resubmit=True for a new one)."
        )
    if save:
        path = kcat_source_file(name, model=model, adapter=adapter)  # fails before submitting
        if submitting:
            _warn_subset(ec_rxns, model, path)

    if use_stored:
        if "jobId" not in job:
            _adopt_shared_result(model, data_dir, method, job_file, out_file)
            job = _read_job_file(job_file)
        if not out_file.is_file() or job.get("resultOf") != job.get("jobId"):
            raise FileNotFoundError(
                f"No stored result of the current {method} job ({job_file}). Call kcats_from_okp "
                f"without use_stored to fetch it."
            )
        kcat_list, done = parse_okp_output(model, out_file), True
    else:
        if submitting:
            out_file.unlink(missing_ok=True)
            job_id = submit_open_kinetics_predictor(model, overwrite=True, method=method,
                                                    ec_rxns=ec_rxns, adapter=adapter)
            job = {"jobId": job_id, "method": method,
                   "submittedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            if ec_rxns is not None:
                job["ecRxns"] = str(len(set(ec_rxns)))
            _write_job_file(job_file, job)
        done, kcat_list = fetch_open_kinetics_predictor(model, job_id=job["jobId"], wait=wait,
                                                        adapter=adapter, out_file=out_file)
        if done:
            job["resultOf"] = job["jobId"]
            _write_job_file(job_file, job)

    if not done:
        logger.info("Call kcats_from_okp(model, method=%r) again later to collect the result.", method)
        return None, False
    if save:
        parameters: dict = {"method": method}
        if "ecRxns" in job:
            parameters["ec_rxns"] = int(job["ecRxns"])
        path = save_kcat_source(model, kcat_list, name, adapter=adapter, generator="kcats_from_okp",
                                parameters=parameters, data=f"OKP job {job['jobId']}")
        logger.info("OKP (%s) kcats written to %s", method, path)
    return kcat_list, True
