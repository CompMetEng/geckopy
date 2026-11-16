"""Copy EC codes from model.ec.eccodes back to per-reaction annotations.

Ported from GECKO MATLAB:
src/deprecated/copyECtoGEM.m.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


def _is_populated(value: object) -> bool:
    """True when an existing ec-code annotation should be preserved
    against an `overwrite=False` copy.

    Treats missing key, empty string, empty list, and None as
    "unpopulated"; anything else (including non-empty list/str) as
    populated.
    """
    if value is None:
        return False
    if isinstance(value, (list, tuple, str)) and len(value) == 0:
        return False
    return True


def copy_ec_to_gem(model: "EcModel", *, overwrite: bool = False) -> None:
    """Copy EC codes from ``model.ec.eccodes`` to per-reaction
    ``annotation['ec-code']`` (the cobrapy-side mirror of MATLAB's
    top-level ``model.eccodes``).

    Ported from GECKO MATLAB:
    src/deprecated/copyECtoGEM.m.

    For each entry of ``model.ec.rxns`` (after stripping a 4-char
    prefix when the model is in gecko-light mode), the corresponding
    cobra.Reaction's ``annotation['ec-code']`` is written as a list
    of EC tokens split from the ``;``-joined ``ec.eccodes[i]``
    string. Empty ``ec.eccodes`` entries are skipped: they neither
    create new annotations nor clobber existing ones, even with
    ``overwrite=True``.

    With ``overwrite=False`` (default) only reactions whose current
    ``ec-code`` annotation is missing or empty (``""``, ``[]``,
    ``None``) are updated. With ``overwrite=True`` any non-empty
    ``ec.eccodes[i]`` replaces the current annotation.

    Reactions in the cobra model that are absent from
    ``model.ec.rxns`` are left untouched. Entries of
    ``model.ec.rxns`` that do not match any cobra reaction are
    silently skipped.

    Parameters
    ----------
    model
        EcModel with populated ``model.ec.rxns`` and
        ``model.ec.eccodes`` (typically by some combination of
        ``fill_eccodes_from_gem``, ``fill_eccodes_from_database``, etc.).
        Mutated in place.
    overwrite
        See above.
    """
    if not model.ec.rxns or not model.ec.eccodes:
        return

    rxn_ids_in_model = {r.id for r in model.reactions}

    for ec_rxn_id, eccode_str in zip(model.ec.rxns, model.ec.eccodes):
        if not eccode_str:
            continue
        rxn_id = ec_rxn_id[4:] if model.ec.gecko_light else ec_rxn_id
        if rxn_id not in rxn_ids_in_model:
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        if not overwrite and _is_populated(rxn.annotation.get("ec-code")):
            continue
        rxn.annotation["ec-code"] = eccode_str.split(";")
