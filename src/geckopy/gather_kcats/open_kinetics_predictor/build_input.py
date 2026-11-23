"""Build the OpenKineticsPredictor input CSV from an ec model.

The CSV has the header ``Protein Sequence,Substrate`` and one row per
unique (enzyme sequence, substrate SMILES) pair. Substrate/SMILES
extraction and filtering are shared with the DLKcat input builder via
``extract_enzyme_substrate_pairs``.
"""
from __future__ import annotations

import csv
import io
import logging
from typing import TYPE_CHECKING, Iterable, Optional

from ..write_dlkcat_input import extract_enzyme_substrate_pairs

if TYPE_CHECKING:
    from ...databases.dlkcat_ignore_lists import DLKcatIgnoreLists
    from ...ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


def _build_okp_input_csv(
    model: "EcModel",
    ignore_lists: "DLKcatIgnoreLists",
    *,
    ec_rxns: Optional[Iterable[str]] = None,
    only_with_smiles: bool = True,
) -> str:
    """Return the OKP input CSV text for the selected reactions.

    Parameters
    ----------
    model
        EcModel with ``ec.sequence`` and per-metabolite SMILES.
    ignore_lists
        Loaded ``DLKcatIgnoreLists`` (ignored/currency metabolites).
    ec_rxns
        Optional iterable of reaction IDs (must be in ``model.ec.rxns``).
        ``None`` means all.
    only_with_smiles
        Drop rows lacking a SMILES when True; otherwise emit ``"None"``.

    Returns
    -------
    str
        CSV text with header ``Protein Sequence,Substrate``. Duplicate
        (sequence, SMILES) pairs are collapsed to one row to save quota;
        the output parser re-maps each prediction back to every matching
        reaction.
    """
    pairs = extract_enzyme_substrate_pairs(
        model, ignore_lists, ec_rxns=ec_rxns, only_with_smiles=only_with_smiles,
    )
    pairs = pairs[pairs["sequence"].astype(str) != ""]
    unique = pairs[["sequence", "smiles"]].drop_duplicates()

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Protein Sequence", "Substrate"])
    for sequence, smiles in zip(unique["sequence"], unique["smiles"]):
        writer.writerow([sequence, smiles])

    logger.info("_build_okp_input_csv: %d unique (sequence, SMILES) pair(s).",
                len(unique))
    return buffer.getvalue()
