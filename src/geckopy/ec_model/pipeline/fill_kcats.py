"""Fill missing kcats by averaging across isozymes.

Ported from GECKO MATLAB:
src/geckomat/change_model/getKcatAcrossIsozymes.m.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

from .apply_kcat import apply_kcat_constraints

if TYPE_CHECKING:
    from ..ec_model import EcModel


logger = logging.getLogger(__name__)

_EXP_SUFFIX_REGEX = re.compile(r"_EXP_\d+$")
_SOURCE_TAG = "isozymes"

_AGGREGATORS = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
}


def fill_kcats_from_isozymes(
    model: "EcModel",
    *,
    apply: bool = False,
    aggregate: Optional[str] = None,
) -> None:
    """Fill in missing kcats by combining known kcats across sibling isozymes.

    Two reactions are siblings if their ``ec.rxns`` IDs are identical
    after stripping any ``_EXP_<n>`` suffix. The ``_REV`` suffix is
    NOT stripped: forward and reverse reactions share chemistry but
    can have different kcats, so they are kept distinct.

    For each reaction with no kcat assigned (``ec.kcat == 0``), find
    its siblings whose kcat is set (``> 0``). If any exist, take their
    mean and assign it. Set ``ec.source`` to ``"isozymes"`` for the
    filled entries. Only a kcat of 0 is filled, as in ``getKcatAcrossIsozymes``:
    a NaN kcat is not a missing kcat and is left as it is. A NaN sibling
    does not contribute to the mean here; in ``getKcatAcrossIsozymes`` it
    does, so there the mean is NaN and the row is not filled.

    Reactions whose siblings all also lack a kcat stay at 0. Reactions
    with a single isozyme (no siblings beyond themselves) also stay at 0.

    ``get_kcat_across_isozymes`` is a deprecated alias for this function.

    Parameters
    ----------
    model
        An EcModel produced by ``make_ec_model``. Mutated in place.
        Must not be a gecko-light model.
    apply
        If True, call ``apply_kcat_constraints`` after updating ec.kcat
        so the new values reflect in the S matrix. Default False, as
        in ``getKcatAcrossIsozymes``, which only changes ``ec.kcat``.
    aggregate
        How to combine sibling kcats: ``"mean"`` (matches MATLAB GECKO),
        ``"median"`` (more robust for log-distributed rate constants), or
        ``"max"``. ``None`` (default) reads
        ``model.adapter.params.kcat_aggregate_isozymes`` if an adapter is
        attached, falling back to ``"mean"`` otherwise.

    Raises
    ------
    NotImplementedError
        If ``model.ec.gecko_light`` is True. Isozyme-averaging does not
        apply to the light formulation since it does not split
        reactions per isozyme.
    """
    if model.ec.gecko_light:
        raise NotImplementedError(
            "fill_kcats_from_isozymes: not applicable to gecko-light models."
        )

    if aggregate is None:
        adapter = getattr(model, "adapter", None)
        aggregate = (
            adapter.params.kcat_aggregate_isozymes
            if adapter is not None else "mean"
        )

    kcat = model.ec.kcat
    known_mask = kcat > 0
    missing_mask = kcat == 0
    if kcat.size == 0 or not known_mask.any():
        logger.warning(
            "fill_kcats_from_isozymes: ec.kcat has no known values to "
            "average from; model unchanged."
        )
        return

    # Strip _EXP_<n> suffix to identify isozyme groups. _REV stays.
    try:
        combine = _AGGREGATORS[aggregate]
    except KeyError:
        raise ValueError(
            f"aggregate must be one of {sorted(_AGGREGATORS)}, got {aggregate!r}"
        ) from None

    base_ids = [_EXP_SUFFIX_REGEX.sub("", r) for r in model.ec.rxns]

    # Group known kcats by base_id.
    known_by_base: dict[str, list[float]] = {}
    for bid, k, known in zip(base_ids, kcat, known_mask):
        if known:
            known_by_base.setdefault(bid, []).append(float(k))

    # For each missing entry, look up its base group and average if available.
    filled_indices: list[int] = []
    for i, (bid, missing) in enumerate(zip(base_ids, missing_mask)):
        if not missing:
            continue
        siblings = known_by_base.get(bid)
        if not siblings:
            continue
        model.ec.kcat[i] = float(combine(siblings))
        model.ec.source[i] = _SOURCE_TAG
        filled_indices.append(i)

    if not filled_indices:
        logger.info(
            "fill_kcats_from_isozymes: no missing kcats had isozymes with "
            "known kcats; nothing filled."
        )
        return

    logger.info(
        "fill_kcats_from_isozymes: filled %d kcat(s) by %s across "
        "isozymes.", len(filled_indices), aggregate,
    )

    if apply:
        filled_rxn_ids = [model.ec.rxns[i] for i in filled_indices]
        apply_kcat_constraints(model, update_rxns=filled_rxn_ids)


def get_kcat_across_isozymes(model: "EcModel", *, apply: bool = False) -> None:
    """Deprecated alias for :func:`fill_kcats_from_isozymes`.

    Kept for backward compatibility with the original MATLAB name.
    Will be removed in a future release; switch to
    ``fill_kcats_from_isozymes``.
    """
    import warnings

    warnings.warn(
        "get_kcat_across_isozymes is deprecated; use "
        "fill_kcats_from_isozymes instead. The old name will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fill_kcats_from_isozymes(model, apply=apply)
