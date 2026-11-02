"""Compute the f-factor for an ecModel.

The f-factor is the fraction of the cell's total protein mass
that the enzymes in the model account for. (The rest is ribosomes,
structural proteins, transcription factors, etc. — proteins that
don't catalyse a metabolic reaction in the model.)

The protein pool's upper bound is set to
``P_tot * f * sigma``, so getting f right matters: too low and
the model is under-constrained (artificially easy to grow); too
high and it's over-constrained (artificially hard).

geckopy computes f from proteomics data (a ``ProtData`` table
loaded by ``load_pax_db`` or similar): for each measured protein,
check whether it's in the model, sum the masses, divide by the
total. A default value of 0.5 is a reasonable starting point when
no proteomics data is available.

Ported from GECKO MATLAB:
src/geckomat/limit_proteins/calculateFfactor.m.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..databases.pax_db_loader import ProtData
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


def calculate_f_factor(
    model: "EcModel",
    prot_data: "ProtData",
    *,
    enzymes: Optional[list[str]] = None,
) -> float:
    """Compute the f-factor for an ec model.

    The f-factor is the mass fraction of total cellular protein that
    is accounted for by enzymes in the model. It is computed as

        f = sum(abundances of in-model enzymes) / sum(all abundances)

    where abundances are read from a pre-loaded ``ProtData`` (typically
    populated via ``load_pax_db``).

    For 2-D ``abundances`` (multiple proteomics samples), the
    per-protein average is taken first via ``np.nanmean(axis=1)``.

    Ported from GECKO MATLAB:
    src/geckomat/limit_proteins/calculateFfactor.m.

    Parameters
    ----------
    model
        EcModel; only used to default ``enzymes`` to
        ``model.ec.enzymes``.
    prot_data
        Pre-loaded proteomics data (typically from ``load_pax_db``).
        There is no implicit fallback when proteomics data is
        unavailable -- callers without measurements should use a
        default value such as ``f = 0.5`` directly instead of calling
        this function.
    enzymes
        UniProt IDs counted as "in the model". Defaults to
        ``model.ec.enzymes``.

    Returns
    -------
    float
        The f-factor in [0, 1]. Returns NaN, as MATLAB does, if the total
        proteome abundance is zero (the factor is 0/0), and 0.0 if none of
        the proteomics UniProt IDs match a model enzyme.
    """
    if enzymes is None:
        enzymes = model.ec.enzymes

    abundances = prot_data.abundances
    if abundances.ndim == 2:
        avg = np.nanmean(abundances, axis=1)
    else:
        avg = abundances

    total_prot = float(np.nansum(avg))
    if total_prot == 0.0:
        logger.warning(
            "calculate_f_factor: total proteome abundance is zero; "
            "returning f = NaN."
        )
        return float("nan")

    enzyme_set = set(enzymes)
    in_model = np.array(
        [uid in enzyme_set for uid in prot_data.uniprot_ids],
        dtype=bool,
    )
    if not in_model.any():
        # None of the proteomics IDs matched a model enzyme -- almost always
        # an ID-namespace mismatch, not a real f of 0. Flag it loudly rather
        # than silently zeroing the protein budget.
        logger.warning(
            "calculate_f_factor: none of the %d proteomics UniProt IDs match "
            "a model enzyme; returning f = 0. Check that proteomics IDs and "
            "ec.enzymes use the same namespace.",
            len(prot_data.uniprot_ids),
        )
        return 0.0

    total_enz = float(np.nansum(avg[in_model]))
    return total_enz / total_prot
