"""Parse an OpenKineticsPredictor result CSV into a kcat_list DataFrame.

OKP returns only (kcat, Source kcat, Protein Sequence, Substrate) per
row, so each prediction is mapped back to the matching reaction(s) via
sequence -> ec.genes and SMILES -> metabolite -> ec.rxns, mirroring the
MATLAB readOpenKineticsPredictorOutput. The output schema matches
``read_dlkcat_output`` / ``fuzzy_kcat_matching`` so it feeds
``apply_kcat_list`` directly.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ...databases.pubchem import met_smiles as read_met_smiles

if TYPE_CHECKING:
    from ...ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

_KCAT_COLUMN_SUFFIX = "kcat (1/s)"
_RANGE_RE = re.compile(
    r"^\s*([0-9.eE+-]+)\s+to\s+([0-9.eE+-]+)\s*(?:s\^-1)?\s*$"
)


def _kcat_value(text: str) -> float | None:
    """A prediction as a number in 1/s.

    Plain numbers are returned as they are. CatRange predicts a class,
    which OKP returns as its range (``"10 to 100 s^-1"``); it becomes the
    geometric midpoint, or half the upper bound for the class starting at
    0, the values OKP returned for these classes before. Anything else is
    None.
    """
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    match = _RANGE_RE.match(text or "")
    if match is None:
        return None
    try:
        low, high = float(match.group(1)), float(match.group(2))
    except ValueError:
        return None
    return (low * high) ** 0.5 if low > 0 else high / 2

_OUTPUT_COLUMNS = [
    "rxn_id", "source", "eccode", "substrates", "genes",
    "kcat", "wildcard_level", "origin",
]
_PREDICTION_PREFIX_RE = re.compile(r"^Prediction from\s+")


def parse_okp_output(
    model: "EcModel",
    result: str | Path,
) -> pd.DataFrame:
    """Parse an OKP result CSV (text or path) into a kcat_list DataFrame.

    Parameters
    ----------
    model
        EcModel with ``ec.rxns``, ``ec.genes``, ``ec.sequence``,
        ``ec.rxn_enz_mat`` and per-metabolite SMILES populated.
    result
        Either the CSV text or a path to the result file. Expected
        columns include ``kcat (1/s)`` (CatRange: ``Predicted kcat
        range: kcat (1/s)``, a number or a class range such as
        ``10 to 100 s^-1``), ``Source kcat``, ``Protein Sequence`` and
        ``Substrate`` (extra similarity columns are ignored).

    Returns
    -------
    pandas.DataFrame
        Columns ``rxn_id, source, eccode, substrates, genes, kcat,
        wildcard_level, origin``. ``source`` is the per-row provenance
        from the ``Source kcat`` column (the prediction method such as
        ``CataPro``, or a database such as ``BRENDA`` / ``Sabio-RK`` /
        ``UniProt``), prefixed with ``OKP-`` to mark it as having come
        through this pipeline stage (e.g. ``OKP-CataPro``,
        ``OKP-BRENDA``). ``merge_kcats`` recognizes the prefix (after
        ``normalize_source`` folds it to ``okp_``) and still routes an
        ``OKP-BRENDA`` / ``OKP-Sabio-RK`` row to the database tiers.

    Raises
    ------
    ValueError
        If the file has no numeric kcat values, lacks the required
        columns, or none of the rows can be mapped to ``model.ec.rxns``.
    """
    text = _read_text(result)
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    kcat_columns = [c for c in df.columns if c.endswith(_KCAT_COLUMN_SUFFIX)]
    required = ["Source kcat", "Protein Sequence", "Substrate"]
    missing = [c for c in required if c not in df.columns]
    if len(kcat_columns) != 1:
        missing.insert(0, _KCAT_COLUMN_SUFFIX)
    if missing:
        raise ValueError(
            f"OKP result is missing expected column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )

    numeric_kcat = pd.to_numeric(df[kcat_columns[0]].map(_kcat_value), errors="coerce")
    valid_mask = numeric_kcat.notna()
    if not valid_mask.any():
        raise ValueError(
            "OKP result contains no numeric kcat values. Has the job "
            "finished successfully?"
        )

    seq_to_proteins = _index(model.ec.sequence)
    smiles_to_mets: dict[str, list] = {}
    for met in model.metabolites:
        smiles = read_met_smiles(met)
        if smiles:
            smiles_to_mets.setdefault(smiles, []).append(met)

    if model.ec.gecko_light:
        ec_to_cobra = [r[4:] for r in model.ec.rxns]
    else:
        ec_to_cobra = list(model.ec.rxns)

    # metabolite id -> cobra reaction ids that consume it (coeff < 0)
    met_consumers: dict[str, set[str]] = {}
    for rxn in model.reactions:
        for met, coeff in rxn.metabolites.items():
            if coeff < 0:
                met_consumers.setdefault(met.id, set()).add(rxn.id)

    rxn_enz_csr = model.ec.rxn_enz_mat.tocsr()
    rxn_enz_csc = model.ec.rxn_enz_mat.tocsc()
    n_proteins = rxn_enz_csc.shape[1]
    protein_to_ecrxns = {
        p: set(rxn_enz_csc[:, p].indices) for p in range(n_proteins)
    }

    rows: list[tuple] = []
    unmatched = 0
    for idx in df.index[valid_mask]:
        sequence = df.at[idx, "Protein Sequence"]
        smiles = df.at[idx, "Substrate"]
        kcat = float(numeric_kcat[idx])
        bare_source = _PREDICTION_PREFIX_RE.sub("", df.at[idx, "Source kcat"]).strip()
        source = f"OKP-{bare_source}"

        protein_idxs = seq_to_proteins.get(sequence, [])
        mets = smiles_to_mets.get(smiles, [])
        if not protein_idxs or not mets:
            unmatched += 1
            continue

        consumer_cobra: set[str] = set()
        for met in mets:
            consumer_cobra |= met_consumers.get(met.id, set())
        catalyzed_ec: set[int] = set()
        for p in protein_idxs:
            catalyzed_ec |= protein_to_ecrxns.get(p, set())

        candidate_ec = [
            e for e in catalyzed_ec if ec_to_cobra[e] in consumer_cobra
        ]
        if not candidate_ec:
            unmatched += 1
            continue

        for ec_idx in sorted(candidate_ec):
            catalysts = set(rxn_enz_csr[ec_idx].indices)
            matching = [p for p in protein_idxs if p in catalysts]
            gene = (
                model.ec.genes[matching[0]]
                if matching and matching[0] < len(model.ec.genes) else ""
            )
            cobra_id = ec_to_cobra[ec_idx]
            sub_name = ""
            for met in mets:
                if cobra_id in met_consumers.get(met.id, set()):
                    sub_name = met.name
                    break
            rows.append((model.ec.rxns[ec_idx], source, sub_name, gene, kcat))

    if unmatched:
        logger.warning(
            "parse_okp_output: %d result row(s) could not be mapped to an "
            "ec.rxn (sequence and/or SMILES not in the model).", unmatched,
        )
    if not rows:
        raise ValueError(
            "No OKP result rows could be mapped to model.ec.rxns. The result "
            "was likely generated from a different ecModel."
        )

    n = len(rows)
    out = pd.DataFrame({
        "rxn_id": [r[0] for r in rows],
        "source": [r[1] for r in rows],
        "eccode": [""] * n,
        "substrates": [[r[2]] for r in rows],
        "genes": [[r[3]] for r in rows],
        "kcat": [r[4] for r in rows],
        "wildcard_level": pd.array([pd.NA] * n, dtype="Int64"),
        "origin": pd.array([pd.NA] * n, dtype="Int64"),
    })
    logger.info("parse_okp_output: mapped %d kcat row(s).", n)
    return out


def _read_text(result: str | Path) -> str:
    """Return CSV text from either a path or a raw CSV string."""
    if isinstance(result, Path):
        return result.read_text(encoding="utf-8")
    # A short string with a newline and commas is treated as CSV text;
    # otherwise treat it as a path.
    if "\n" in result or "," in result:
        return result
    path = Path(result)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return result


def _index(values: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, value in enumerate(values):
        if value:
            out.setdefault(value, []).append(i)
    return out
