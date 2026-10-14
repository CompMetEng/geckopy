"""One abundance per enzyme complex.

Absolute proteomics recovers the subunits of a complex unevenly, so
after ``fill_enz_concs`` the subunits of one complex can disagree several
fold about how much of it there is. Capped one by one, the complex can
then carry no more flux than its least abundant subunit allows. This
module groups the enzymes of every multi-subunit reaction into complexes
and gives each complex one abundance, scaled to every subunit by its
stoichiometry.

A complex is a set of enzymes that catalyses a reaction together, taken
from ``ec.rxn_enz_mat``. Two enzyme sets are one complex when they share
at least two subunits and at least half of the larger set (isozyme
variants that swap a subunit, the same complex on different reactions).
A single shared protein never merges two sets, so electron carriers and
shared subunits (cytochrome c, thioredoxin, the E3 subunit of the
2-oxo-acid dehydrogenases) do not chain unrelated complexes together.

A subunit in exactly two complexes belongs to the one whose enzyme sets
catalyse more reactions (the core subunits COX1-3 of complex IV also
appear in one assembly reaction). A subunit in three or more complexes,
in two with equal weight, or that also catalyses a reaction on its own
is *shared*: it is left as measured and does not take part in the
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
    owner
        For every subunit that belongs to a complex, the index of that
        complex in ``complexes``.
    shared
        Subunits of a complex that belong to none: carriers and subunits
        shared between complexes.
    """

    complexes: list[dict[str, float]] = field(default_factory=list)
    owner: dict[str, int] = field(default_factory=dict)
    shared: set[str] = field(default_factory=set)

    def owned(self, index: int) -> dict[str, float]:
        """``{enzyme: copies}`` of the subunits complex ``index`` owns."""
        return {e: c for e, c in self.complexes[index].items() if self.owner.get(e) == index}

    def members(self, enzyme: str) -> list[str]:
        """The subunits of the complex ``enzyme`` belongs to, or ``[enzyme]``."""
        index = self.owner.get(enzyme)
        return [enzyme] if index is None else list(self.owned(index))


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


def complex_families(
    model: "EcModel", min_overlap: float = 0.5, min_shared: int = 2,
) -> ComplexFamilies:
    """Group the enzymes of multi-subunit reactions into complexes.

    Parameters
    ----------
    model
        An EcModel with ``ec.rxn_enz_mat`` (copies per subunit).
    min_overlap
        Two enzyme sets are one complex when they share at least this
        fraction of the larger set's subunits...
    min_shared
        ...and at least this many subunits.
    """
    matrix = model.ec.rxn_enz_mat.tocsr()
    enzymes = list(model.ec.enzymes)
    sets: dict[frozenset, dict[str, float]] = {}
    reactions: dict[frozenset, int] = {}
    alone: set[str] = set()
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        present = {enzymes[j]: float(v) for j, v in zip(matrix.indices[start:end], matrix.data[start:end]) if v}
        if len(present) == 1:
            alone.update(present)
        elif present:
            sets.setdefault(frozenset(present), present)
            reactions[frozenset(present)] = reactions.get(frozenset(present), 0) + 1

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
                common = len(keys[i] & keys[j])
                if common >= min_shared and common >= min_overlap * max(len(keys[i]), len(keys[j])):
                    parent[find(i)] = find(j)

    grouped: dict[int, dict[str, float]] = {}
    weight: dict[int, dict[str, int]] = {}
    for i, key in enumerate(keys):
        w = weight.setdefault(find(i), {})
        for enzyme in key:
            w[enzyme] = w.get(enzyme, 0) + reactions[key]
        complex_ = grouped.setdefault(find(i), {})
        for enzyme, copies in sets[key].items():
            if enzyme in complex_ and complex_[enzyme] != copies:
                logger.warning(
                    "complex_families: %s has %g copies in one variant and %g in another; "
                    "the first is used", enzyme, complex_[enzyme], copies,
                )
            complex_.setdefault(enzyme, copies)
    roots = list(grouped)
    complexes = [grouped[r] for r in roots]

    found: dict[str, list[tuple[int, int]]] = {}
    for index, root in enumerate(roots):
        for enzyme, w in weight[root].items():
            found.setdefault(enzyme, []).append((w, index))
    owner: dict[str, int] = {}
    for enzyme, hits in found.items():
        if enzyme in alone or len(hits) > 2:
            continue
        hits.sort(reverse=True)
        if len(hits) == 1 or hits[0][0] > hits[1][0]:
            owner[enzyme] = hits[0][1]
    shared = set(found) - set(owner)
    return ComplexFamilies(complexes=complexes, owner=owner, shared=shared)


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

    for index, complex_ in enumerate(families.complexes):
        members = families.owned(index)
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
