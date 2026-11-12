"""Kcats from a WILDkCAT run.

WILDkCAT (https://github.com/sysbiolux/WILDkCAT) retrieves kcat values from
BRENDA and SABIO-RK for each reaction, EC number, enzyme and direction of a
conventional GEM, and fills gaps with CataPro predictions. It keeps one value
per row, the measurement with the lowest penalty score (0 is an exact match
of organism, enzyme, substrate, temperature and pH; higher scores are less
close matches). Run WILDkCAT on the conventional GEM the ecModel was made
from, then give its output table (``kcat_full.tsv``, or ``kcat_retrieved.tsv``
without predictions) to :func:`kcats_from_wildkcat`. GECKO has the same in
``kcatsFromWILDkCAT``.
"""
from __future__ import annotations

import csv
import io
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

_REQUIRED = ["rxn", "direction", "genes", "kcat", "db", "penalty_score"]
#: Detail key and WILDkCAT column, in the order of the detail text.
_DETAIL_FIELDS = [
    ("db", "db"),
    ("penalty", "penalty_score"),
    ("organism", "kcat_organism"),
    ("enzyme", "kcat_enzyme"),
    ("temperature", "kcat_temperature"),
    ("pH", "kcat_ph"),
    ("identity", "kcat_id_percent"),
]


def _read_table(path: Path) -> dict[str, list[str]]:
    """The tab-separated table as columns of text; blank lines are skipped and
    short rows are padded with empty fields."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in csv.reader(io.StringIO(text, newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
             if any(f.strip() for f in ln)]
    if not lines:
        raise ValueError(f"{path} is empty.")
    header = [h.strip() for h in lines[0]]
    columns: dict[str, list[str]] = {h: [] for h in header}
    for ln in lines[1:]:
        for j, h in enumerate(header):
            columns[h].append(ln[j] if j < len(ln) else "")
    return columns


def _number(text: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return math.nan


def _gene_key(text: str) -> str:
    parts = sorted({g.strip() for g in text.split(";")})
    return re.sub(r"^;|;$", "", ";".join(parts))


def kcats_from_wildkcat(
    model: "EcModel",
    filename: str | Path,
    *,
    max_penalty: float = math.inf,
    predictions: bool = True,
    save: bool = True,
    save_as: str = "wildkcat",
    adapter: Optional["ModelAdapter"] = None,
) -> tuple[pd.DataFrame, dict]:
    """Read the kcats of a WILDkCAT run and keep them as ``wildkcat.tsv``.

    A WILDkCAT row is matched to the ``model.ec`` rows of the same reaction
    (its ``_REV`` for direction ``reverse``) whose genes (``model.ec.genes`` of
    the enzymes in ``model.ec.rxn_enz_mat``) are the genes of the row. When
    several WILDkCAT rows match the same ``model.ec`` row, the one with the
    lowest penalty score is kept, and of those the highest kcat, as WILDkCAT
    itself chooses. Rows without a kcat, and rows that match no ``model.ec``
    row, are counted in the report. The result is written to
    ``data/kcats/wildkcat.tsv`` (see :func:`~geckopy.gather_kcats.save_kcat_source`),
    which a stage of ``[kcat_assignment]`` in ``model_adapter.toml`` can name
    as a source. MATLAB GECKO has the same (``kcatsFromWILDkCAT``).

    Parameters
    ----------
    model
        An ecModel.
    filename
        The WILDkCAT output table (``kcat_full.tsv`` or ``kcat_retrieved.tsv``).
    max_penalty
        Keep only values with this penalty score or lower (default: all
        values). A predicted value has the penalty score of the measurement
        it replaced.
    predictions
        Keep the CataPro predictions, the rows whose ``db`` is ``catapro``.
    save, save_as
        Write the source file, and its name without ``.tsv``.
    adapter
        Model adapter (default: ``model.adapter``).

    Returns
    -------
    (kcat_list, report)
        ``kcat_list`` is the kcat list with columns ``rxn_id``, ``source``,
        ``substrates``, ``kcat``, ``eccode`` and ``detail`` (db, penalty
        score, and the organism, enzyme, temperature, pH and sequence identity
        of the measurement, as far as WILDkCAT gives them). ``report`` is a
        dict with the numbers of WILDkCAT rows: ``rows``, ``no_value`` (no
        kcat), ``excluded`` (by ``max_penalty`` or ``predictions``),
        ``matched``, and ``unmatched``, the sorted ``<rxn>|<direction>|<genes>``
        of the rows that match no ``model.ec`` row.

    Examples
    --------
    >>> kcats_from_wildkcat(ec_model, "wildkcat_output/kcat_full.tsv")  # doctest: +SKIP
    >>> kcats_from_wildkcat(ec_model, "kcat_full.tsv", max_penalty=4, predictions=False)  # doctest: +SKIP
    """
    from .kcat_sources import ec_rxn_keys, save_kcat_source

    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"WILDkCAT output {path} does not exist.")
    t = _read_table(path)
    missing = [c for c in _REQUIRED if c not in t]
    if missing:
        raise ValueError(f"{path} lacks the WILDkCAT column(s) {', '.join(missing)}.")

    def column(name: str, idx: int) -> str:
        return t[name][idx] if name in t else ""

    n = len(t["rxn"])
    kcat = [_number(v) for v in t["kcat"]]
    penalty = [_number(v) for v in t["penalty_score"]]
    db = [v.strip().lower() for v in t["db"]]

    has_value = [math.isfinite(k) and k > 0 for k in kcat]
    use = [has_value[i] and not penalty[i] > max_penalty and (predictions or db[i] != "catapro") for i in range(n)]
    report: dict = {"rows": n, "no_value": has_value.count(False),
                    "excluded": sum(has_value[i] and not use[i] for i in range(n))}

    # Keys of the model.ec rows: conventional reaction and sorted gene set.
    ec = model.ec
    base_rxns, _ = ec_rxn_keys(model)
    mat = ec.rxn_enz_mat.tocsr()
    gene_sets = []
    for i in range(len(ec.rxns)):
        cols = mat[i].indices[mat[i].data != 0] if mat.shape[1] else []
        gene_sets.append(";".join(sorted({ec.genes[j] for j in cols})))
    ec_keys: dict[str, list[int]] = {}
    for i, (b, g) in enumerate(zip(base_rxns, gene_sets)):
        if g:
            ec_keys.setdefault(f"{b}|{g}", []).append(i)

    # Keys of the WILDkCAT rows.
    row_keys = []
    for i in range(n):
        rxn = t["rxn"][i].strip()
        if t["direction"][i].strip().lower() == "reverse":
            rxn += "_REV"
        row_keys.append(f"{rxn}|{_gene_key(t['genes'][i])}")

    matched = [False] * n
    best: dict[int, int] = {}  # chosen WILDkCAT row per model.ec row
    for i in range(n):
        if not use[i]:
            continue
        hits = ec_keys.get(row_keys[i], [])
        if not hits:
            continue
        matched[i] = True
        for h in hits:
            b = best.get(h)
            if b is None or penalty[i] < penalty[b] or (penalty[i] == penalty[b] and kcat[i] > kcat[b]):
                best[h] = i
    report["matched"] = sum(matched)
    unmatched = [use[i] and not matched[i] for i in range(n)]
    report["unmatched"] = sorted({f"{t['rxn'][i]}|{t['direction'][i]}|{t['genes'][i]}"
                                  for i in range(n) if unmatched[i]})

    rows = []
    for h in sorted(best):
        src = best[h]
        parts = []
        for key, col in _DETAIL_FIELDS:
            v = column(col, src).strip()
            if v and v.lower() != "nan":
                parts.append(f"{key}={v.replace(';', ',')}")
        substrate = column("kcat_substrate", src)
        rows.append({"rxn_id": ec.rxns[h], "source": "wildkcat", "substrates": [substrate] if substrate else [],
                     "kcat": kcat[src], "eccode": column("ec_code", src), "detail": "; ".join(parts)})
    kcat_list = pd.DataFrame(rows, columns=["rxn_id", "source", "substrates", "kcat", "eccode", "detail"])

    logger.info("WILDkCAT: %d rows, %d without a kcat, %d excluded, %d matched to %d model.ec reactions, "
                "%d unmatched.", report["rows"], report["no_value"], report["excluded"], report["matched"],
                len(rows), sum(unmatched))
    if save:
        from ..adapter import resolve_adapter

        adapter = resolve_adapter(model, adapter, purpose="kcats_from_wildkcat needs the project folder")
        parameters: dict = {"file": path.name, "predictions": bool(predictions)}
        if math.isfinite(max_penalty):
            parameters["max_penalty"] = max_penalty
        out = save_kcat_source(model, kcat_list, save_as, adapter=adapter, generator="kcats_from_wildkcat",
                               parameters=parameters)
        logger.info("WILDkCAT kcats for %d reactions written to %s", kcat_list["rxn_id"].nunique(), out)
    return kcat_list, report
