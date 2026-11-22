"""Set model exchange-flux bounds from an experimental dataset.

The constraint logic is
:func:`raven_toolbox.experiments.apply_flux_data_constraints`; this
wrapper takes the biomass reaction and the carbon source from the model
adapter when they are not given.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from raven_toolbox.experiments import apply_flux_data_constraints as _apply

if TYPE_CHECKING:
    from raven_toolbox.experiments import FluxData
    from ..ec_model.ec_model import EcModel


_LooseStrict = Union[Literal["loose"], float]


def apply_flux_data_constraints(
    model: "EcModel",
    flux_data: "FluxData",
    *,
    condition: int | str = 0,
    max_min_growth: Literal["max", "min"] = "max",
    loose_strict_flux: _LooseStrict = "loose",
    bio_rxn: Optional[str] = None,
    c_source: Optional[str] = None,
) -> None:
    """Constrain an EcModel to one measured flux condition.

    Use it to reproduce one measured condition before an analysis, for
    example a flux variability analysis or a proteomics integration. The
    biomass reaction is bounded by the measured growth rate and each
    exchange reaction in ``flux_data.exch_rxn_ids`` by its measured flux in
    the chosen condition (``fluxData.tsv``, see
    :func:`~raven_toolbox.experiments.load_flux_data`). The adapter's
    preferred carbon source is blocked first, so that the data alone
    decide which carbon source is used. The bounds that each setting of
    ``loose_strict_flux`` gives are listed in
    :func:`raven_toolbox.experiments.apply_flux_data_constraints`, which
    does the work; this function only takes the biomass reaction and the
    carbon source from the model adapter.

    Parameters
    ----------
    model
        EcModel with the exchange reactions referenced by
        ``flux_data`` already in place. Mutated in place.
    flux_data
        Flux data, for example from ``load_flux_data``.
    condition
        Either the 0-indexed condition number, or the condition name
        (matched against ``flux_data.conds``).
    max_min_growth
        ``"max"`` sets the ``bio_rxn`` upper bound to the measured
        growth rate (and the lower bound to 0); ``"min"`` sets the lower
        bound (suitable when minimising ``prot_pool_exchange``).
    loose_strict_flux
        ``"loose"`` keeps one bound at 0 and the other at the measured
        value. A number ``p`` in [0, 100] brackets the measured value
        with hard bounds of ``p`` percent in total (``p=10`` allows
        +/-5%).
    bio_rxn
        Biomass reaction ID. Defaults to the adapter's configured
        biomass reaction (``model.adapter.params.bio_rxn``) when
        ``None``.
    c_source
        Reaction ID of the preferred carbon-source uptake reaction;
        its bounds are set to ``(0, 0)`` before applying the flux
        data. Defaults to ``model.adapter.params.c_source`` (or no
        blocking if there is no adapter) when ``None``; pass ``""``
        to skip blocking any reaction. Silently ignored if the
        reaction id is not found in the model.

    Raises
    ------
    ValueError
        If ``bio_rxn`` is ``None`` and ``model.adapter`` is None,
        ``max_min_growth`` / ``loose_strict_flux`` is invalid,
        ``condition`` (as string) is not in ``flux_data.conds``, or any
        ``flux_data.exch_rxn_ids`` is missing from the model.
    IndexError
        If ``condition`` (as int) is out of range.
    """
    from ..adapter import resolve_param

    if c_source is None:
        adapter = getattr(model, "adapter", None)
        c_source = adapter.params.c_source if adapter is not None else ""
    bio_rxn_id = resolve_param(
        model, bio_rxn, "bio_rxn",
        purpose="apply_flux_data_constraints needs the biomass reaction id",
    )
    _apply(
        model, flux_data,
        bio_rxn_id=bio_rxn_id, condition=condition,
        max_min_growth=max_min_growth, loose_strict_flux=loose_strict_flux,
        c_source=c_source,
    )
