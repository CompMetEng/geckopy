"""Set kcat values for individual reactions, with optional auto-apply.

Ported from GECKO MATLAB: src/geckomat/change_model/setKcatForReactions.m.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence, Union


from .apply_kcat import apply_kcat_constraints
from .populate_ec import split_light_rxn_id

if TYPE_CHECKING:
    from ..ec_model import EcModel


_EXP_SUFFIX_PATTERN = re.compile(r"_EXP_\d+$")
_LIGHT_PREFIX_PATTERN = re.compile(r"^\d{3}_")
_SOURCE_TAG = "setKcatForReactions"


def set_kcat_for_reactions(
    model: "EcModel",
    rxn_ids: Sequence[str],
    kcat: Union[float, Sequence[float]],
    *,
    apply: bool = True,
) -> list[str]:
    """Set kcat values for one or more reactions in ec.kcat.

    Each ID in ``rxn_ids`` is interpreted as follows:

    - If the ID ends in ``_EXP_<n>`` (full layout) or starts with
      ``###_`` (gecko-light), only that exact reaction is matched.
    - Otherwise, the ID is treated as a base name and matches every
      ec.rxns entry whose ID, after stripping the layout-specific
      isozyme marker, equals the base name. So in a full model,
      ``"R2"`` matches ``R2``, ``R2_EXP_1``, ``R2_EXP_2``; in a light
      model, ``"R2"`` matches ``001_R2``, ``002_R2``.

    The ``kcat`` argument follows numpy-style broadcasting: a single
    float applies to every matched reaction; a sequence of length
    ``len(rxn_ids)`` gives one value per input ID (broadcast across
    that ID's own matches); a sequence of length equal to the total
    number of matched reactions gives one value per resolved match, in
    ``ec.rxns`` order.

    That last form lets one un-suffixed ``rxn_id`` that expands to
    multiple isozymes take a list of per-isozyme values, matching
    MATLAB ``setKcatForReactions``'s positional semantics exactly:
    values are assigned to matches in whatever order they occur in
    ``ec.rxns``, with no check that this is the order the caller
    intended. This is real but fragile as an API surface -- prefer
    passing the suffixed reaction IDs explicitly when the mapping
    matters, so intent doesn't depend on ``ec.rxns``'s row order.

    ``ec.source`` is set to ``"setKcatForReactions"`` for every
    reaction changed by this call, matching MATLAB's ``ec.source``
    label (its own docstring says ``'from setKcatForReactions'``, but
    the actual assignment writes the bare function name).

    Parameters
    ----------
    model
        An EcModel with ec.rxns populated. Mutated in place.
    rxn_ids
        Reaction identifiers to update.
    kcat
        New kcat value(s) in 1/s. Scalar or sequence (see above).
    apply
        If True (default), call ``apply_kcat_constraints`` after
        updating ec.kcat / ec.source so the change is reflected in the
        S matrix immediately. If False, the caller must invoke
        ``apply_kcat_constraints`` themselves.

    Returns
    -------
    list of str
        IDs of the reactions whose kcat was changed (the post-expansion
        set), in the order in which they were updated.

    Raises
    ------
    ValueError
        If any rxn_id matches zero reactions, or if kcat has a length
        other than 1, ``len(rxn_ids)``, or the total number of resolved
        matches.
    """
    rxn_ids = list(rxn_ids)
    if not rxn_ids:
        return []

    ec_rxns = model.ec.rxns
    if model.ec.gecko_light:
        # Light: strip the leading ``###_`` counter so the cobra reaction
        # id is what the base-name lookup sees.
        base_ec_rxns = [split_light_rxn_id(r)[1] for r in ec_rxns]
        explicit_pattern = _LIGHT_PREFIX_PATTERN
    else:
        base_ec_rxns = [_EXP_SUFFIX_PATTERN.sub("", r) for r in ec_rxns]
        explicit_pattern = _EXP_SUFFIX_PATTERN

    # Resolve every input ID to a list of ec-row indices.
    matches_per_input: list[list[int]] = []
    for rxn_id in rxn_ids:
        if explicit_pattern.search(rxn_id):
            indices = [i for i, r in enumerate(ec_rxns) if r == rxn_id]
        else:
            indices = [
                i for i, r in enumerate(base_ec_rxns) if r == rxn_id
            ]
        if not indices:
            raise ValueError(
                f"rxn_id '{rxn_id}' matched no entries in ec.rxns."
            )
        matches_per_input.append(indices)

    total_matches = sum(len(m) for m in matches_per_input)

    # Resolve kcat to one scalar per ec-row index in resolution order.
    if isinstance(kcat, (int, float)) and not isinstance(kcat, bool):
        kcat_per_index = [float(kcat)] * total_matches
    else:
        kcat_seq = list(kcat)
        # kcat_seq has length len(rxn_ids) (one value per input ID,
        # broadcast across that ID's own matches), or length
        # total_matches (one value per resolved match, in ec.rxns
        # order -- MATLAB setKcatForReactions's positional semantics;
        # see the docstring's fragility note). When every input matches
        # exactly one row the two are the same list, so there's no
        # ambiguity between them.
        if len(kcat_seq) == len(rxn_ids):
            kcat_per_index = []
            for value, indices in zip(kcat_seq, matches_per_input):
                kcat_per_index.extend([float(value)] * len(indices))
        elif len(kcat_seq) == total_matches:
            kcat_per_index = [float(v) for v in kcat_seq]
        else:
            raise ValueError(
                f"kcat has length {len(kcat_seq)}; expected a scalar, "
                f"length {len(rxn_ids)} (one per input rxn_id), or length "
                f"{total_matches} (one per resolved match)."
            )

    # Apply to ec.kcat and ec.source.
    flat_indices: list[int] = []
    for indices in matches_per_input:
        flat_indices.extend(indices)

    for idx, value in zip(flat_indices, kcat_per_index):
        model.ec.kcat[idx] = value
        model.ec.source[idx] = _SOURCE_TAG

    updated_ids = [ec_rxns[i] for i in flat_indices]

    if apply and updated_ids:
        apply_kcat_constraints(model, update_rxns=updated_ids)

    return updated_ids
