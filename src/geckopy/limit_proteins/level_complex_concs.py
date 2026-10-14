"""One abundance per enzyme complex.

Absolute proteomics recovers the subunits of a complex unevenly, so
after ``fill_enz_concs`` the subunits of one complex can disagree several
fold about how much of it there is. Capped one by one, the complex can
then carry no more flux than its least abundant subunit allows. This
module groups the enzymes of every multi-subunit reaction into complexes
and gives each complex one abundance, scaled to every subunit by its
stoichiometry.

A complex is a set of enzymes that catalyses a reaction together, taken
from ``ec.rxn_enz_mat``. Enzyme sets that share at least half of the
smaller set's subunits are one complex (isozyme variants that swap a
subunit, the same complex on different reactions). A subunit that ends
up in more than one complex, or that also catalyses a reaction on its
own, is *shared*: it is left as measured and does not take part in the
summary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

_STATISTICS = {"median": np.median, "mean": np.mean}


@dataclass
class ComplexFamilies:
    """The complexes of an ecModel and their shared subunits.

    Attributes
    ----------
    complexes
        One dict per complex, ``{enzyme: copies}`` over every subunit
        that appears in any of its enzyme sets.
    shared
        Enzymes that belong to more than one complex, or to a complex
        and also catalyse a reaction alone.
    """

    complexes: list[dict[str, float]] = field(default_factory=list)
    shared: set[str] = field(default_factory=set)

    def members(self, enzyme: str) -> list[str]:
        """The non-shared subunits of the complex ``enzyme`` belongs to, or ``[enzyme]``."""
        if enzyme in self.shared:
            return [enzyme]
        for complex_ in self.complexes:
            if enzyme in complex_:
                return [e for e in complex_ if e not in self.shared]
        return [enzyme]


@dataclass
class LevelComplexResult:
    """What ``level_complex_concs`` changed.

    Attributes
    ----------
    adjusted
        Measured subunits whose concentration was replaced by the complex's.
    added
        Unmeasured subunits given the complex's concentration.
    shared
        Subunits left as measured because they belong to more than one complex.
    skipped
        Complexes (as sorted subunit tuples) without any measured, non-shared subunit.
    """

    adjusted: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    shared: list[str] = field(default_factory=list)
    skipped: list[tuple[str, ...]] = field(default_factory=list)


def complex_families(model: "EcModel", min_overlap: float = 0.5) -> ComplexFamilies:
    """Group the enzymes of multi-subunit reactions into complexes.

    Parameters
    ----------
    model
        An EcModel with ``ec.rxn_enz_mat`` (copies per subunit).
    min_overlap
        Two enzyme sets are one complex when they share at least this
        fraction of the smaller set's subunits.
    """
    matrix = model.ec.rxn_enz_mat.tocsr()
    enzymes = list(model.ec.enzymes)
    sets: dict[frozenset, dict[str, float]] = {}
    alone: set[str] = set()
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        present = {enzymes[j]: float(v) for j, v in zip(matrix.indices[start:end], matrix.data[start:end]) if v}
        if len(present) == 1:
            alone.update(present)
        elif present:
            sets.setdefault(frozenset(present), present)

    keys = list(sets)
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_enzyme: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        for enzyme in key:
            by_enzyme.setdefault(enzyme, []).append(i)
    for indices in by_enzyme.values():
        for x in range(len(indices)):
            for y in range(x + 1, len(indices)):
                i, j = indices[x], indices[y]
                if len(keys[i] & keys[j]) >= min_overlap * min(len(keys[i]), len(keys[j])):
                    parent[find(i)] = find(j)

    grouped: dict[int, dict[str, float]] = {}
    for i, key in enumerate(keys):
        complex_ = grouped.setdefault(find(i), {})
        for enzyme, copies in sets[key].items():
            complex_.setdefault(enzyme, copies)
    complexes = list(grouped.values())

    count: dict[str, int] = {}
    for complex_ in complexes:
        for enzyme in complex_:
            count[enzyme] = count.get(enzyme, 0) + 1
    shared = {e for e, n in count.items() if n > 1} | (alone & set(count))
    return ComplexFamilies(complexes=complexes, shared=shared)


def level_complex_abundances(
    model: "EcModel",
    uniprot_ids: Sequence[str],
    abundances: Sequence[float],
    *,
    statistic: str = "median",
    fill_missing: bool = True,
    families: Optional[ComplexFamilies] = None,
) -> tuple[list[str], np.ndarray, LevelComplexResult]:
    """Level molar abundances to one abundance per complex.

    Parameters
    ----------
    model
        The EcModel whose complexes are used.
    uniprot_ids, abundances
        Measured molar abundances (any molar unit; per subunit, not per complex).
    statistic
        ``"median"`` or ``"mean"`` over the measured, non-shared subunits'
        abundance divided by their copies.
    fill_missing
        Give unmeasured, non-shared subunits the complex's abundance.
    families
        Precomputed ``complex_families(model)``.

    Returns
    -------
    ids, abundances, result
        The levelled data (unmeasured subunits appended when filled) and
        what changed.
    """
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {sorted(_STATISTICS)}, not {statistic!r}")
    summarise = _STATISTICS[statistic]
    families = families or complex_families(model)
    values = {p: float(a) for p, a in zip(uniprot_ids, abundances)}
    order = list(uniprot_ids)
    result = LevelComplexResult(shared=sorted(families.shared & set(values)))

    for complex_ in families.complexes:
        members = {e: c for e, c in complex_.items() if e not in families.shared}
        measured = [values[e] / c for e, c in members.items() if np.isfinite(values.get(e, np.nan)) and values[e] > 0]
        if not measured:
            result.skipped.append(tuple(sorted(complex_)))
            continue
        per_copy = float(summarise(measured))
        for enzyme, copies in members.items():
            level = per_copy * copies
            if enzyme in values and np.isfinite(values[enzyme]) and values[enzyme] > 0:
                if not np.isclose(values[enzyme], level):
                    result.adjusted.append(enzyme)
                values[enzyme] = level
            elif fill_missing:
                if enzyme not in values:
                    order.append(enzyme)
                values[enzyme] = level
                result.added.append(enzyme)
    return order, np.array([values[p] for p in order], dtype=float), result


def level_complex_concs(
    model: "EcModel",
    *,
    statistic: str = "median",
    fill_missing: bool = True,
) -> LevelComplexResult:
    """Give every complex in ``model.ec.concs`` one abundance.

    Run after ``fill_enz_concs`` and before ``constrain_enz_concs``.
    Concentrations (mg/gDCW) are converted to molar amounts with
    ``ec.mw`` for the levelling and back afterwards. Mutates
    ``model.ec.concs``.

    Parameters
    ----------
    model
        EcModel with ``ec.concs`` filled.
    statistic
        ``"median"`` (default) or ``"mean"``.
    fill_missing
        Give unmeasured, non-shared subunits the complex's concentration.
    """
    enzymes = list(model.ec.enzymes)
    mw = np.asarray(model.ec.mw, dtype=float)
    concs = np.asarray(model.ec.concs, dtype=float)
    molar = concs / mw
    measured = [i for i in range(len(enzymes)) if np.isfinite(molar[i])]
    ids, levelled, result = level_complex_abundances(
        model, [enzymes[i] for i in measured], molar[measured],
        statistic=statistic, fill_missing=fill_missing,
    )
    new = concs.copy()
    index = {e: i for i, e in enumerate(enzymes)}
    for enzyme, value in zip(ids, levelled):
        i = index[enzyme]
        new[i] = value * mw[i]
    model.ec.concs = new
    logger.info(
        "level_complex_concs: %d subunits levelled, %d filled, %d shared left as measured",
        len(result.adjusted), len(result.added), len(result.shared),
    )
    return result
