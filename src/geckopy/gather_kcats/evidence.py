"""The kcats that every source gives for selected reactions.

:func:`kcat_evidence` puts the kcat of each source file of the project next
to the kcat of the model and, for a tuned kcat, the kcat before tuning.
:func:`kcats_for_reactions` adds the EC-tree estimate and predictions of
OpenKineticsPredictor for a few reactions. GECKO has the same functions as
``kcatEvidence`` and ``kcatsForReactions``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from .ec_tree import ECTree
from .kcat_source_format import parse_kcat_source
from .kcat_sources import OKP_METHODS, ec_rxn_keys, load_kcat_source, okp_file

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

# Methods that predict from one enzyme and one substrate: kcats_from_okp
# rejects CatPred and TurNup.
_PAIR_METHODS = tuple(OKP_METHODS)


class KcatEvidence(NamedTuple):
    """Result of :func:`kcat_evidence`: ``evidence``, ``summary`` and ``rows``."""

    evidence: pd.DataFrame
    summary: pd.DataFrame
    rows: np.ndarray


def _names_to_rows(model: "EcModel", base: list[str], names: Iterable[str], label: str) -> np.ndarray:
    names = list(dict.fromkeys(names))
    known = set(model.ec.rxns) | set(base)
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(f"Not reactions of model.ec ({label}): {', '.join(missing)}.")
    wanted = set(names)
    return np.array([r in wanted or b in wanted for r, b in zip(model.ec.rxns, base)], dtype=bool)


def _select_rows(model: "EcModel", rxns, enzymes, tuned) -> np.ndarray:
    n_ec = len(model.ec.rxns)
    if rxns is None and enzymes is None and tuned is None:
        return np.arange(n_ec)
    selected = np.zeros(n_ec, dtype=bool)
    base, _ = ec_rxn_keys(model)
    if rxns is not None:
        if isinstance(rxns, str):
            rxns = [rxns]
        rxns = list(rxns) if not isinstance(rxns, np.ndarray) else rxns
        arr = np.asarray(rxns)
        if arr.dtype == bool:
            selected |= arr
        elif np.issubdtype(arr.dtype, np.integer):
            selected[arr] = True
        else:
            selected |= _names_to_rows(model, base, [str(r) for r in rxns], "rxns")
    if tuned is not None:
        names = getattr(tuned, "rxns", tuned)
        selected |= _names_to_rows(model, base, [str(r) for r in names], "tuned")
    if enzymes is not None:
        enzymes = [enzymes] if isinstance(enzymes, str) else list(enzymes)
        missing = sorted(set(enzymes) - set(model.ec.enzymes))
        if missing:
            raise ValueError(f"Not in model.ec.enzymes: {', '.join(missing)}.")
        cols = [j for j, e in enumerate(model.ec.enzymes) if e in set(enzymes)]
        mat = model.ec.rxn_enz_mat.tocsc()[:, cols]
        selected |= np.asarray((mat != 0).sum(axis=1)).ravel() > 0
    return np.flatnonzero(selected)


def kcat_evidence(
    model: "EcModel",
    *,
    rxns=None,
    enzymes: Optional[Iterable[str]] = None,
    tuned=None,
    sources: Optional[Sequence[str]] = None,
    compute: Sequence[str] = (),
    measurements: Optional[pd.DataFrame] = None,
    save_as: Optional[str | Path] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> KcatEvidence:
    """Show the kcats that every source gives for selected reactions.

    For each selected ``model.ec`` reaction, reads the kcat of every source
    file in ``data/kcats`` (see :func:`~geckopy.gather_kcats.load_kcat_source`)
    and puts it next to the kcat that the model has now and, for a tuned
    kcat, the kcat before tuning. A reaction that ``rxns``, ``enzymes`` or
    ``tuned`` selects is included; without a selection, all reactions are. The
    source files are only read.

    Parameters
    ----------
    model
        The ecModel.
    rxns
        Reactions to include: names of ``model.ec.rxns`` or conventional
        reaction names (a reverse direction ends with ``_REV``), integer
        indices, or a boolean array over ``model.ec.rxns``.
    enzymes
        UniProt accessions; every reaction catalysed by one of them is
        included, for example the enzymes that
        :func:`~geckopy.limit_proteins.flexibilize_enz_concs` relaxed.
    tuned
        The result of :func:`~geckopy.kcat_tuning.sensitivity_tuning`, or a
        list of reaction names.
    sources
        Names of the source files, without ``.tsv`` (default: all files in
        ``data/kcats``; an empty list for none).
    compute
        Sources to compute for the selected reactions when the project has no
        file of that name; ``"ec_tree"`` is supported.
    measurements
        BRENDA measurement table for the EC-tree estimate (default: the
        adapter's table).
    save_as
        Path of a tab-separated file to write ``summary`` to.
    adapter
        Model adapter (default: ``model.adapter``).

    Returns
    -------
    KcatEvidence
        ``evidence``: one row per kcat of a source, with ``ec_rxn``,
        ``source``, ``kcat``, ``substrate``, ``detail`` and ``assigned`` (the
        kcat is the one assigned to the reaction: the kcat before tuning for
        a tuned reaction, the current kcat otherwise). ``summary``: one row
        per selected reaction, with ``ec_rxn``, ``enzymes``, ``eccodes``,
        ``kcat`` (current), ``origin``, ``tuned``, ``prior`` (NaN when
        untuned), ``fold`` (``kcat / prior``) and one column ``src_<source>``
        with the median kcat of that source (NaN without one). ``rows``: the
        selected indices of ``model.ec.rxns``.
    """
    from .kcats_from import _adapter, _measurements

    unknown = sorted(set(compute) - {"ec_tree"})
    if unknown:
        raise ValueError(f"Cannot compute {', '.join(unknown)}; supported: ec_tree.")
    adapter = _adapter(model, adapter, "kcat_evidence needs params.path")
    rows = _select_rows(model, rxns, enzymes, tuned)
    ec_rxns = [model.ec.rxns[i] for i in rows]

    if sources is None:
        folder = Path(adapter.params.path) / "data" / "kcats"
        sources = sorted(p.stem for p in folder.glob("*.tsv"))
    sources = list(sources)

    parts = []
    for name in sources:
        kcat_list, _ = load_kcat_source(model, name, adapter=adapter)
        keep = kcat_list["rxn_id"].isin(ec_rxns)
        sub = kcat_list[keep]
        parts.append(pd.DataFrame({
            "ec_rxn": sub["rxn_id"].to_numpy(),
            "source": name,
            "kcat": sub["kcat"].to_numpy(dtype=float),
            "substrate": ["; ".join(s) for s in sub["substrates"]],
            "detail": sub["detail"].to_numpy(),
        }))
    if "ec_tree" in compute and "ec_tree" not in sources and len(rows):
        from .kcats_from import kcats_from_ec_tree

        tree = kcats_from_ec_tree(model, ec_rxns=ec_rxns, save=False, measurements=measurements, adapter=adapter)
        parts.append(pd.DataFrame({
            "ec_rxn": tree["rxn_id"].to_numpy(), "source": "ec_tree",
            "kcat": tree["kcat"].to_numpy(dtype=float), "substrate": "", "detail": tree["detail"].to_numpy(),
        }))
        sources.append("ec_tree")
    columns = ["ec_rxn", "source", "kcat", "substrate", "detail"]
    evidence = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    info = [parse_kcat_source(model.ec.source[i]) for i in rows]
    current = np.asarray(model.ec.kcat, dtype=float)[rows]
    prior = np.array([np.nan if x.prior is None else x.prior for x in info], dtype=float)
    tuned_mask = ~np.isnan(prior)
    reference = np.where(tuned_mask, prior, current)
    ref_of = dict(zip(ec_rxns, reference))
    ref = evidence["ec_rxn"].map(ref_of).to_numpy(dtype=float)
    evidence["assigned"] = np.isclose(evidence["kcat"].to_numpy(dtype=float), ref, rtol=1e-6, atol=0)
    evidence = evidence[columns + ["assigned"]]

    mat = model.ec.rxn_enz_mat.tocsr()
    enzyme_text = []
    for i in rows:
        row = mat[i]
        enzyme_text.append(";".join(model.ec.enzymes[j] for j in row.indices[row.data != 0]))
    summary = pd.DataFrame({
        "ec_rxn": ec_rxns,
        "enzymes": enzyme_text,
        "eccodes": [";".join(c) if isinstance(c, (list, tuple)) else (c if isinstance(c, str) else "")
                    for c in (model.ec.eccodes[i] for i in rows)],
        "kcat": current,
        "origin": [x.origin for x in info],
        "tuned": tuned_mask,
        "prior": prior,
        "fold": current / prior,
    })
    medians = evidence.groupby(["source", "ec_rxn"])["kcat"].median()
    for name in sources:
        summary[f"src_{name}"] = [medians.get((name, r), np.nan) for r in ec_rxns]
    if save_as is not None:
        summary.to_csv(save_as, sep="\t", index=False)
    return KcatEvidence(evidence, summary, rows)


def _okp_methods(okp, adapter: "ModelAdapter") -> list[str]:
    if okp is None or okp is False:
        return []
    if okp is True:
        from .kcat_assignment import default_okp_method

        return [default_okp_method(adapter)]
    if isinstance(okp, str):
        if okp.lower() == "all":
            return list(_PAIR_METHODS)
        okp = [okp]
    methods = list(okp)
    unknown = [m for m in methods if m not in _PAIR_METHODS]
    if unknown:
        raise ValueError(f"Unknown OKP methods: {', '.join(unknown)}. Methods: {', '.join(_PAIR_METHODS)}.")
    return methods


def kcats_for_reactions(
    model: "EcModel",
    rxns,
    *,
    okp=False,
    sources: Optional[Sequence[str]] = None,
    compute: Sequence[str] = ("ec_tree",),
    measurements: Optional[pd.DataFrame] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> KcatEvidence:
    """Get the kcats of every source for a few reactions.

    Reads the kcats that the source files of the project give for the
    reactions, computes the EC-tree estimate for them, and optionally predicts
    their kcats with one or several OpenKineticsPredictor (OKP) methods; the
    result is shown as :func:`kcat_evidence` does.

    The OKP jobs have their own tag (``job_tag`` of
    :func:`~geckopy.gather_kcats.kcats_from_okp`): they submit the
    enzyme-substrate pairs of the selected reactions only, and leave the job
    and the result that ``kcats_from_okp`` keeps for all reactions as they
    are. The predictions are written to ``data/kcats/okp_<method>_query.tsv``
    and read back as a source. The function waits for each job to finish and
    submits a new job on every call. A method that fails is logged as a
    warning and the others continue. Each OKP method submits a job and counts
    against the daily prediction quota of the API key.

    Parameters
    ----------
    model
        EcModel with metabolite SMILES and protein sequences when OKP is used.
    rxns
        The reactions, as in :func:`kcat_evidence`.
    okp
        OKP methods to predict with: ``False`` for none (default), ``True``
        for the first method of the adapter, ``"all"`` for every method
        that :func:`~geckopy.gather_kcats.kcats_from_okp` accepts, or a method
        name or a list of names.
    sources, compute, measurements, adapter
        As in :func:`kcat_evidence`; ``compute`` is ``("ec_tree",)`` by default.

    Returns
    -------
    KcatEvidence
    """
    from .kcats_from import _adapter, kcats_from_okp

    adapter = _adapter(model, adapter, "kcats_for_reactions needs params.path")
    rows = kcat_evidence(model, rxns=rxns, sources=[], adapter=adapter).rows
    if len(rows) == 0:
        raise ValueError("No reaction of model.ec is selected.")
    ec_rxns = [model.ec.rxns[i] for i in rows]
    queried = []
    for method in _okp_methods(okp, adapter):
        try:
            _, done = kcats_from_okp(model, method=method, ec_rxns=ec_rxns, job_tag="query",
                                     resubmit=True, wait=True, adapter=adapter)
        except Exception as err:  # one method failing does not stop the others
            logger.warning("OKP %s failed: %s", method, err)
            continue
        if done:
            queried.append(f"{okp_file(method)}_query")
    if sources is None:
        folder = Path(adapter.params.path) / "data" / "kcats"
        sources = [p.stem for p in sorted(folder.glob("*.tsv"))
                   if not (re.fullmatch(r"okp_.*_query", p.stem) and p.stem not in queried)]
    return kcat_evidence(model, rxns=rows, sources=sources, compute=compute,
                         measurements=measurements, adapter=adapter)
