"""SMILES lookup for the metabolites of an ecModel."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from raven_toolbox.annotation import find_met_smiles as _find_met_smiles

if TYPE_CHECKING:
    from pathlib import Path

    import cobra


def find_met_smiles(
    model: "cobra.Model",
    *,
    cache_path: "str | Path",
    skip_prefixes: Iterable[str] = ("prot_",),
) -> None:
    """Write the SMILES of the metabolites of ``model`` to their annotation, from PubChem.

    This is :func:`raven_toolbox.annotation.find_met_smiles` with ``skip_prefixes`` set to
    ``("prot_",)``, so the enzyme metabolites of an ecModel (``prot_<id>``) are not looked up.
    See there for the lookup, the cache file and the other arguments.

    Parameters
    ----------
    model
        Model whose metabolites get a SMILES annotation.
    cache_path
        Cache file (TSV) that is read first and appended to after each lookup.
    skip_prefixes
        Metabolites whose name starts with one of these prefixes are skipped.
    """
    _find_met_smiles(model, cache_path=cache_path, skip_prefixes=skip_prefixes)
