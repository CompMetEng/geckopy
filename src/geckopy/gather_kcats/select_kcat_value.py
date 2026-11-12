"""Pick one kcat per reaction from a kcat list and write to model.ec.kcat.

Ported from GECKO MATLAB:
src/deprecated/selectKcatValue.m.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd

from .kcat_source_format import format_kcat_source

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


_CRITERIA = ("max", "min", "median", "mean")
_OverwriteMode = Union[bool, Literal["if_higher"]]


def _fuzzy_source(source: object, wildcard_level: object, origin: object) -> str:
    """Provenance string of one kcat-list row.

    The string holds the source and, for a fuzzy BRENDA match, the wildcard
    level and origin of the match as the detail pairs ``wc`` and ``origin``.
    """
    detail = {}
    if wildcard_level is not None and pd.notna(wildcard_level):
        detail["wc"] = int(wildcard_level)
        if origin is not None and pd.notna(origin):
            detail["origin"] = int(origin)
    return format_kcat_source(source, detail)


def apply_kcat_list(
    model: "EcModel",
    kcat_list: pd.DataFrame,
    *,
    criteria: Optional[Literal["max", "min", "median", "mean"]] = None,
    overwrite: _OverwriteMode = True,
) -> list[str]:
    """Choose one kcat value per reaction and write it to ``model.ec.kcat``.

    Low-level: ``collect_kcats`` and ``assign_kcats`` run this function; call
    it directly only to change kcats by hand.

    Ported from GECKO MATLAB:
    src/gather_kcats/assignKcatValues.m.

    For each reaction with one or more entries in ``kcat_list``,
    aggregate the candidate kcats per ``criteria`` (``"max"`` is the
    default and most common) and write the result into
    ``model.ec.kcat`` at the matching ``model.ec.rxns`` position.
    The contributing entry's provenance is recorded in
    ``model.ec.source`` via :func:`format_kcat_source`: a lowercase
    string whose first token is the source and whose bracketed detail,
    for a fuzzy BRENDA match, contains the wildcard level and origin
    (e.g. ``"brenda (wc=0, origin=1)"``). If the list has no
    ``wildcard_level`` / ``origin`` columns the source is the bare
    token.

    Rows with ``kcat == 0`` mark "no match" and are dropped before
    aggregation.

    For ``criteria="median"`` or ``"mean"``, the aggregated value does
    not come from a single input row, so the recorded source is that
    of the **first row** in the reaction's group. For ``"max"`` and
    ``"min"`` the source is unambiguous (the row whose kcat won).

    Parameters
    ----------
    model
        EcModel with ``model.ec.rxns``, ``model.ec.kcat``, and
        ``model.ec.source`` already allocated. Mutated in place.
    kcat_list
        DataFrame with at least the columns ``rxn_id``, ``kcat``
        (1/s, ``0`` treated as no-match), and ``source`` (per-row).
        Multiple rows for the same reaction are aggregated per
        ``criteria``.
    criteria
        How to aggregate multiple candidates per reaction:
        ``"max"``, ``"min"``, ``"median"``, or ``"mean"``. ``None``
        (default) reads
        ``model.adapter.params.kcat_aggregate_candidates`` if an adapter
        is attached, falling back to ``"max"`` otherwise.
    overwrite
        ``True`` (default) overwrites unconditionally. ``False``
        only fills entries currently 0 or NaN. ``"if_higher"`` only
        updates when the new kcat exceeds the existing one (with
        0/NaN counted as "infinitely small" so they always update).

    Returns
    -------
    list of str
        Reaction IDs whose kcat was updated.

    Raises
    ------
    ValueError
        If ``criteria`` or ``overwrite`` is invalid, or if any
        ``rxn_id`` in ``kcat_list`` is absent from ``model.ec.rxns``.
    KeyError
        If ``kcat_list`` is missing the ``rxn_id``, ``kcat``, or
        ``source`` column.
    """
    if criteria is None:
        adapter = getattr(model, "adapter", None)
        criteria = (
            adapter.params.kcat_aggregate_candidates
            if adapter is not None else "max"
        )
    if criteria not in _CRITERIA:
        raise ValueError(
            f"criteria must be one of {_CRITERIA}; got {criteria!r}"
        )
    if overwrite not in (True, False, "if_higher"):
        raise ValueError(
            f"overwrite must be True, False, or 'if_higher'; got {overwrite!r}"
        )

    for required in ("rxn_id", "kcat", "source"):
        if required not in kcat_list.columns:
            raise KeyError(
                f"kcat_list missing required column {required!r}"
            )

    nonzero = kcat_list[kcat_list["kcat"] != 0]
    if nonzero.empty:
        return []

    ec_rxn_ids = set(model.ec.rxns)
    unknown = sorted(set(nonzero["rxn_id"].astype(str)) - ec_rxn_ids)
    if unknown:
        preview = unknown[:5]
        raise ValueError(
            f"{len(unknown)} reaction ID(s) in kcat_list are not present "
            f"in model.ec.rxns (examples: {preview})"
        )

    rxn_to_idx = {rid: i for i, rid in enumerate(model.ec.rxns)}
    n_group = len(nonzero)
    wcs_all = (
        nonzero["wildcard_level"].values
        if "wildcard_level" in nonzero.columns
        else [None] * n_group
    )
    origins_all = (
        nonzero["origin"].values
        if "origin" in nonzero.columns
        else [None] * n_group
    )
    nonzero = nonzero.assign(_wc=wcs_all, _origin=origins_all)

    selected: dict[str, tuple[float, str]] = {}
    for rxn_id, group in nonzero.groupby("rxn_id", sort=False):
        kcats = group["kcat"].values.astype(float)
        sources = group["source"].astype(str).values
        wcs = group["_wc"].values
        origins = group["_origin"].values
        if criteria == "max":
            j = int(np.argmax(kcats))
        elif criteria == "min":
            j = int(np.argmin(kcats))
        elif criteria == "median":
            kcat_val, j = float(np.median(kcats)), 0
        else:  # "mean"
            kcat_val, j = float(np.mean(kcats)), 0
        if criteria in ("max", "min"):
            kcat_val = float(kcats[j])
        source_val = _fuzzy_source(sources[j], wcs[j], origins[j])
        selected[str(rxn_id)] = (kcat_val, source_val)

    updated: list[str] = []
    for rxn_id, (kcat_val, source_val) in selected.items():
        i = rxn_to_idx[rxn_id]
        if _should_overwrite(model.ec.kcat[i], kcat_val, overwrite):
            model.ec.kcat[i] = kcat_val
            model.ec.source[i] = source_val
            updated.append(rxn_id)

    return updated


def _should_overwrite(
    current: float, new: float, mode: _OverwriteMode,
) -> bool:
    """Decide whether to write `new` over `current` per the overwrite mode."""
    # 0 and NaN both mean "no kcat assigned" (NaN slips past `== 0`).
    is_unset = current == 0 or bool(np.isnan(current))
    if mode is True:
        return True
    if mode is False:
        return is_unset
    # "if_higher"
    return is_unset or new > current
