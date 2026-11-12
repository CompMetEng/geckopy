"""Limit unmeasured isozymes that have a measured sibling.

Ported from GECKO MATLAB:
src/limit_proteins/zeroUnmeasuredIsozymes.m.
"""
from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..ec_model.constants import USAGE_PREFIX
from .get_isozyme_coverage import get_isozyme_coverage

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

_METHODS = ("zero", "detection_limit")


def zero_unmeasured_isozymes(
    model: "EcModel",
    *,
    method: str = "zero",
    limit: Optional[float] = None,
) -> pd.DataFrame:
    """Limit unmeasured isozymes that have a measured sibling.

    For a reaction with a measured and an unmeasured isozyme (see
    ``get_isozyme_coverage``), the usage upper bound of the enzymes of the
    unmeasured isozyme is lowered, instead of leaving them to draw from the
    protein pool without a limit. The function lowers the usage upper bound
    in the model and does not change ``model.ec.concs``. An enzyme that also
    catalyses a reaction without a measured sibling is left unchanged,
    because limiting it would limit that reaction as well.

    Not detecting a protein is not the same as the protein being absent: a
    protein below the detection limit of the measurement is missing from the
    data as well. Method ``"zero"`` treats the unmeasured isozyme as absent
    and sets its usage upper bound to 0, which removes every flux through
    the enzyme and can lower the growth rate or make the model infeasible.
    Method ``"detection_limit"`` treats it as present at or below the
    detection limit and sets the upper bound to the detection limit. An
    upper bound that is already below the value to set is not raised. A
    lower bound above the upper bound to set is lowered to it.

    An enzyme lowered by the function has a NaN concentration and is a
    candidate in ``relax_proteomics_greedy`` like any other enzyme with a
    bound below the default.

    Run after ``constrain_enz_concs``, which resets the usage upper bounds of
    all enzymes. The function is not called by other functions of geckopy.

    Ported from GECKO MATLAB:
    src/limit_proteins/zeroUnmeasuredIsozymes.m (method
    ``'detectionLimit'`` there is ``"detection_limit"`` here).

    Parameters
    ----------
    model
        Full (not gecko-light) EcModel with concentrations in
        ``model.ec.concs``, constrained by ``constrain_enz_concs``. The usage
        upper bounds are lowered in place.
    method
        ``"zero"`` or ``"detection_limit"``.
    limit
        The detection limit in mg/gDCW for ``method="detection_limit"``.
        Defaults to the smallest concentration above 0 in
        ``model.ec.concs``.

    Returns
    -------
    pandas.DataFrame
        One row per enzyme that the function acted on, with the columns
        ``uniprot`` and ``new_upper_bound``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if the detection limit is negative, NaN or
        missing (no concentration above 0 and no ``limit``), or if a usage
        reaction of an enzyme to act on is missing.
    NotImplementedError
        If the model is a gecko-light model.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be 'zero' or 'detection_limit', got {method!r}.")

    _, zeroable = get_isozyme_coverage(model)

    if method == "zero":
        new_ub = [0.0] * len(zeroable)
    else:
        if limit is None:
            concs = getattr(model.ec, "concs", None)
            concs = np.asarray([] if concs is None else concs, dtype=float)
            above = concs[concs > 0]
            limit = float(above.min()) if above.size else math.nan
        if (
            isinstance(limit, bool)
            or not isinstance(limit, numbers.Real)
            or math.isnan(limit)
            or limit < 0
        ):
            raise ValueError("The detection limit must be a nonnegative number.")
        new_ub = [float(limit)] * len(zeroable)

    reaction_ids = {r.id for r in model.reactions}
    absent = [e for e in zeroable if f"{USAGE_PREFIX}{e}" not in reaction_ids]
    if absent:
        raise ValueError(f"The model has no usage reaction for {', '.join(absent)}.")

    applied = []
    for enzyme, value in zip(zeroable, new_ub):
        rxn = model.reactions.get_by_id(f"{USAGE_PREFIX}{enzyme}")
        value = min(value, rxn.upper_bound)
        rxn.bounds = (min(rxn.lower_bound, value), value)
        applied.append(value)
    return pd.DataFrame({
        "uniprot": list(zeroable),
        "new_upper_bound": np.array(applied, dtype=float),
    })
