"""Compute per-enzyme usage info from a flux distribution.

Ported from GECKO MATLAB:
src/geckomat/utilities/enzymeUsage.m.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


from ..ec_model.constants import USAGE_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class EnzymeUsageResult:
    """Per-enzyme usage info from a flux distribution.

    Attributes
    ----------
    prot_id
        UniProt IDs (matching ``model.ec.enzymes`` order, modulo
        zero-usage filtering).
    abs_usage
        Absolute enzyme usage in mg/gDCW (= |flux through the
        ``usage_prot_<X>`` reaction|).
    cap_usage
        Capacity usage = ``abs_usage / ub``. ``0`` for enzymes
        whose usage rxn upper bound is 0.
    ub
        Upper bound of each ``usage_prot_<X>`` reaction.
    fluxes
        The original input flux distribution (carried through for
        downstream use, e.g. by ``report_enzyme_usage``).
    """

    prot_id: list[str] = field(default_factory=list)
    abs_usage: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    cap_usage: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    ub: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    fluxes: object = None


def enzyme_usage(
    model: "EcModel",
    fluxes: Union[pd.Series, Mapping[str, float]],
    *,
    include_zero: bool = True,
) -> EnzymeUsageResult:
    """Compute per-enzyme absolute and capacity usage from ``fluxes``.

    For each enzyme in ``model.ec.enzymes``:

    * ``abs_usage = |fluxes[usage_prot_<enzyme>]|``  (mg/gDCW).
    * ``ub`` = the usage reaction's upper bound.
    * ``cap_usage = abs_usage / ub`` (NaN if ``ub == 0``, as
      :attr:`Enzyme.cap_usage` and GECKO MATLAB's ``getEnzymeUsage`` give).

    Ported from GECKO MATLAB:
    src/geckomat/utilities/enzymeUsage.m.

    Parameters
    ----------
    model
        Full (non-light) EcModel with ``usage_prot_*`` reactions
        installed for every enzyme.
    fluxes
        Flux distribution. Typically ``solution.fluxes`` from
        ``model.optimize()``; any dict-like keyed by rxn id works.
    include_zero
        When False, enzymes with zero absolute usage are dropped
        from the result.

    An enzyme without a ``usage_prot_<enzyme>`` reaction in the model is
    left out of the result, with a logged warning.

    Raises
    ------
    NotImplementedError
        If ``model.ec.gecko_light`` is True.
    """
    if model.ec.gecko_light:
        raise NotImplementedError(
            "enzyme_usage does not support gecko-light models."
        )

    cobra_rxn_ids = {r.id for r in model.reactions}
    prot_ids: list[str] = []
    abs_usage_list: list[float] = []
    ub_list: list[float] = []

    left_out: list[str] = []
    for enzyme in model.ec.enzymes:
        rxn_id = f"{USAGE_PREFIX}{enzyme}"
        if rxn_id not in cobra_rxn_ids:
            left_out.append(enzyme)
            continue
        flux = float(_lookup_flux(fluxes, rxn_id))
        rxn_ub = float(model.reactions.get_by_id(rxn_id).upper_bound)
        prot_ids.append(enzyme)
        abs_usage_list.append(abs(flux))
        ub_list.append(rxn_ub)

    if left_out:
        logger.warning(
            "enzyme_usage: the model has no usage reaction for the enzymes "
            "%s; they are left out.", ", ".join(left_out),
        )

    abs_usage_arr = np.array(abs_usage_list, dtype=float)
    ub_arr = np.array(ub_list, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_arr = np.where(
            ub_arr != 0, abs_usage_arr / ub_arr, np.nan,
        )

    if not include_zero:
        keep = abs_usage_arr > 0
        prot_ids = [p for p, k in zip(prot_ids, keep) if k]
        abs_usage_arr = abs_usage_arr[keep]
        cap_arr = cap_arr[keep]
        ub_arr = ub_arr[keep]

    return EnzymeUsageResult(
        prot_id=prot_ids,
        abs_usage=abs_usage_arr,
        cap_usage=cap_arr,
        ub=ub_arr,
        fluxes=fluxes,
    )


def _lookup_flux(
    fluxes: Union[pd.Series, Mapping[str, float]], rxn_id: str,
) -> float:
    """Default-zero lookup that works for both Series and dicts."""
    if isinstance(fluxes, pd.Series):
        if rxn_id in fluxes.index:
            return float(fluxes[rxn_id])
        return 0.0
    return float(fluxes.get(rxn_id, 0.0))
