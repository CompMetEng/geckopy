"""Read a DLKcat predictions TSV into a kcat_list DataFrame.

Ported from GECKO MATLAB:
src/geckomat/gather_kcats/readDLKcatOutput.m.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


_OUTPUT_COLUMNS = [
    "rxn_id",
    "source",
    "eccode",
    "substrates",
    "genes",
    "kcat",
    "wildcard_level",
    "origin",
]


def read_dlkcat_output(
    model: "EcModel",
    file_path: str | Path,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Parse DLKcat's output TSV into a kcat_list DataFrame.

    Ported from GECKO MATLAB:
    src/geckomat/gather_kcats/readDLKcatOutput.m.

    The input file is the 6-column TSV that DLKcat writes (see
    ``run_dlkcat``), with one header line and columns
    ``rxn_id, gene, substrate, smiles, sequence, kcat``. Rows whose
    kcat column is not a finite number (``"NA"``, ``"None"``, empty,
    ``"inf"``, ``"1.2.3"``) are dropped before the reactions are validated
    against the model.

    The output DataFrame matches the schema produced by
    ``fuzzy_kcat_matching`` so downstream functions
    (``apply_kcat_list``, ``merge_kcats``) can
    consume both interchangeably. Columns not produced by DLKcat
    (``eccode``, ``wildcard_level``, ``origin``) are filled with
    ``""`` / ``<NA>``.

    Substrate names are matched against ``model.metabolites``
    case-insensitively, so differences in capitalization between the
    DLKcat output and the SBML-loaded model do not cause spurious
    failures. A row whose substrate isn't recognized is dropped (with
    a warning) rather than aborting the whole read, unless
    ``strict=True``. Either way, the drop counts for non-numeric kcat
    and unrecognized substrate are logged separately (a row can be
    counted under both).

    Parameters
    ----------
    model
        EcModel with populated ``ec.rxns`` and ``metabolites``.
    file_path
        Path to the DLKcat output TSV.
    strict
        If True, raise ``ValueError`` on any unrecognized substrate
        name instead of dropping just that row. Default False.

    Returns
    -------
    pandas.DataFrame
        Rows where the kcat field was numeric and, unless
        ``strict=True`` let an unrecognized substrate through, the
        substrate was recognized. Columns as listed above; ``source``
        is ``"DLKcat"`` for every row.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If the file contains no numeric kcat values; if
        ``strict=True`` and any substrate name from the file is not
        in ``model.metabolites.<name>`` (case-insensitive); or if the
        ``rxn_id`` of a row that is kept is not in ``model.ec.rxns``.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"DLKcat output file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=0,
        names=["rxn_id", "gene", "substrate", "smiles", "sequence", "kcat"],
        dtype=str,
        keep_default_na=False,
    )

    numeric_kcat = pd.to_numeric(df["kcat"], errors="coerce")
    numeric_kcat = numeric_kcat.where(np.isfinite(numeric_kcat))
    valid_mask = numeric_kcat.notna()

    if not valid_mask.any():
        raise ValueError(
            f"{file_path} contains no numeric kcat values. Did you forget "
            f"to run run_dlkcat() first?"
        )

    model_met_names_lower = {m.name.lower() for m in model.metabolites}
    file_subs_lower = df["substrate"].str.lower()
    unknown_subs_mask = ~file_subs_lower.isin(model_met_names_lower)
    if unknown_subs_mask.any():
        unknown = sorted(df.loc[unknown_subs_mask, "substrate"].unique())
        if strict:
            raise ValueError(
                f"DLKcat output references {len(unknown)} substrate "
                f"name(s) not in model.metabolites (case-insensitive). "
                f"Examples: {unknown[:5]}."
            )
        # A few renamed/normalised substrate names should not discard the
        # whole prediction file; drop those rows and keep the rest (mirrors
        # how non-numeric kcats are handled).
        logger.warning(
            "DLKcat output references %d substrate name(s) not in "
            "model.metabolites (case-insensitive); those rows are skipped. "
            "Examples: %s",
            len(unknown), unknown[:5],
        )
        valid_mask = valid_mask & ~unknown_subs_mask

    ec_rxn_ids = set(model.ec.rxns)
    unknown_rxns_mask = (~df["rxn_id"].isin(ec_rxn_ids)) & valid_mask
    if unknown_rxns_mask.any():
        unknown = sorted(df.loc[unknown_rxns_mask, "rxn_id"].unique())
        preview = unknown[:5]
        raise ValueError(
            f"DLKcat output references {len(unknown)} reaction ID(s) "
            f"not present in model.ec.rxns. Examples: {preview}. The "
            f"output was likely generated from a different ecModel."
        )

    kept = df[valid_mask].reset_index(drop=True)
    kept_kcats = numeric_kcat[valid_mask].reset_index(drop=True).astype(float)
    n = len(kept)

    out = pd.DataFrame({
        "rxn_id": kept["rxn_id"].astype(str).tolist(),
        "source": ["DLKcat"] * n,
        "eccode": [""] * n,
        "substrates": [[s] for s in kept["substrate"].astype(str)],
        "genes": [[g] for g in kept["gene"].astype(str)],
        "kcat": kept_kcats.tolist(),
        "wildcard_level": pd.array([pd.NA] * n, dtype="Int64"),
        "origin": pd.array([pd.NA] * n, dtype="Int64"),
    })

    non_numeric_dropped = int((~numeric_kcat.notna()).sum())
    unknown_subs_dropped = int(unknown_subs_mask.sum())
    if non_numeric_dropped or unknown_subs_dropped:
        logger.info(
            "read_dlkcat_output: read %d row(s) from %s; dropped %d row(s) "
            "with non-numeric kcat, %d row(s) with an unrecognized "
            "substrate (a row failing both is counted in each).",
            n, file_path, non_numeric_dropped, unknown_subs_dropped,
        )
    else:
        logger.info("read_dlkcat_output: read %d row(s) from %s.", n, file_path)

    return out
