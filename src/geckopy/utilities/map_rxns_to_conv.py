"""Map a flux vector from ec-rxn space back to conventional-rxn space.

Ported from GECKO MATLAB:
src/change_model/mapRxnsToConv.m.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ..ec_model.constants import (
    POOL_EXCHANGE_ID,
    USAGE_PREFIX,
    canonicalize_rxn_id,
)

if TYPE_CHECKING:
    import cobra

    from ..ec_model.ec_model import EcModel


_POOL_LABEL = "pool"


@dataclass
class MapRxnsResult:
    """Output of mapping ec-rxn fluxes back to conventional rxns.

    Attributes
    ----------
    mapped_flux
        1-D or 2-D array of fluxes in the conventional model's
        ``model.reactions`` order. Same number of columns/scenarios
        as the input (1-D in / 1-D out; 2-D in / 2-D out).
    enz_usage_flux
        Fluxes of the protein-usage reactions
        (``usage_prot_*`` and ``prot_pool_exchange``), in
        ``usage_enz`` order.
    usage_enz
        Labels for ``enz_usage_flux``: protein IDs for
        ``usage_prot_*`` rows, or ``"pool"`` for the exchange row.
    """

    mapped_flux: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    enz_usage_flux: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    usage_enz: list[str] = field(default_factory=list)


def map_rxns_to_conv(
    ec_model: "EcModel",
    model: "cobra.Model",
    flux_vect: Union[np.ndarray, pd.Series],
) -> MapRxnsResult:
    """Translate ec-rxn fluxes back to conventional-rxn fluxes.

    For each ``ec_model.reactions`` entry, the rxn id is canonicalised:

    * ``_REV`` suffix or ``_REV_EXP_`` infix -> the flux is negated
      (reverse direction in geckopy convention) and the suffix
      stripped.
    * ``_EXP_<N>`` suffix -> stripped (isozyme split combined).

    Fluxes belonging to the same canonical rxn id are summed, then
    reordered to match ``model.reactions``. A separate output
    surfaces protein-usage fluxes (``usage_prot_*`` and
    ``prot_pool_exchange``) under their stripped labels.

    Ported from GECKO MATLAB:
    src/change_model/mapRxnsToConv.m.

    Parameters
    ----------
    ec_model
        The EcModel from which ``flux_vect`` was obtained.
    model
        The conventional (non-ec) model that ``ec_model`` was built
        from. ``model.reactions`` defines the output ordering.
    flux_vect
        Either a 1-D ``np.ndarray`` aligned to
        ``ec_model.reactions`` order, a 2-D ``np.ndarray`` with the
        same row order, or a ``pd.Series`` indexed by ec-rxn id
        (missing entries default to 0).

    Returns
    -------
    MapRxnsResult

    Raises
    ------
    ValueError
        If ``flux_vect`` is empty, has the wrong shape, or any
        conventional rxn ID is missing from the canonicalised set.
    """
    n_ec_rxns = len(ec_model.reactions)
    if n_ec_rxns == 0:
        raise ValueError("ec_model has no reactions.")
    ec_rxn_ids = [r.id for r in ec_model.reactions]

    if isinstance(flux_vect, pd.Series):
        flux = np.array(
            [float(flux_vect.get(rid, 0.0)) for rid in ec_rxn_ids],
            dtype=float,
        )
    else:
        flux = np.asarray(flux_vect, dtype=float)

    if flux.size == 0:
        raise ValueError("flux_vect is empty.")

    if flux.ndim == 1:
        if flux.shape[0] != n_ec_rxns:
            raise ValueError(
                f"flux_vect length {flux.shape[0]} does not match "
                f"ec_model.reactions length {n_ec_rxns}."
            )
        flux_2d = flux.reshape(-1, 1)
        was_1d = True
    elif flux.ndim == 2:
        if flux.shape[0] != n_ec_rxns:
            raise ValueError(
                f"flux_vect axis-0 length {flux.shape[0]} does not "
                f"match ec_model.reactions length {n_ec_rxns}."
            )
        flux_2d = flux
        was_1d = False
    else:
        raise ValueError(
            f"flux_vect must be 1-D or 2-D; got {flux.ndim}-D."
        )

    # Negate _REV rxns and canonicalise IDs.
    canonical_ids: list[str] = []
    flux_signed = flux_2d.copy()
    for i, rid in enumerate(ec_rxn_ids):
        canonical, is_rev = canonicalize_rxn_id(rid)
        if is_rev:
            flux_signed[i, :] = -flux_signed[i, :]
        canonical_ids.append(canonical)

    # Group by canonical id and sum.
    summed: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(flux_2d.shape[1], dtype=float)
    )
    for rid, row in zip(canonical_ids, flux_signed):
        summed[rid] += row

    # Reorder to model.reactions order.
    model_rxn_ids = [r.id for r in model.reactions]
    missing = [rid for rid in model_rxn_ids if rid not in summed]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} reaction ID(s) in model.reactions not "
            f"found in canonicalised ec_model rxns "
            f"(examples: {preview}). Is ec_model derived from model?"
        )
    mapped_2d = np.vstack(
        [summed[rid] for rid in model_rxn_ids]
    )

    # Enzyme-usage fluxes (from the original, un-negated input).
    usage_enz: list[str] = []
    usage_indices: list[int] = []
    for i, rid in enumerate(ec_rxn_ids):
        if rid.startswith(USAGE_PREFIX):
            usage_enz.append(rid[len(USAGE_PREFIX):])
            usage_indices.append(i)
        elif rid == POOL_EXCHANGE_ID:
            usage_enz.append(_POOL_LABEL)
            usage_indices.append(i)

    if usage_indices:
        enz_usage_2d = flux_2d[usage_indices, :]
    else:
        enz_usage_2d = np.empty((0, flux_2d.shape[1]), dtype=float)

    if was_1d:
        return MapRxnsResult(
            mapped_flux=mapped_2d.ravel(),
            enz_usage_flux=enz_usage_2d.ravel() if usage_indices else
            np.empty(0, dtype=float),
            usage_enz=usage_enz,
        )
    return MapRxnsResult(
        mapped_flux=mapped_2d,
        enz_usage_flux=enz_usage_2d,
        usage_enz=usage_enz,
    )
