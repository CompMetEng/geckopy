"""Stop sampling distinctions the kcat assignment never made.

A reaction catalysed by several isozymes gets one ``ec.rxns`` row per
isozyme, each with its own kcat. Where the assignment could tell them
apart -- different databases, different sequences, different measured
values -- those are genuinely different parameters. Where it could not,
every copy receives the same number from the same source, and the
tuner is then free to give them different values because no condition
can distinguish which isozyme carries the flux.

It does exactly that. Five copies of ``r_0438`` share a prior of
54.33 1/s and come out of one run at 4.1, 8.1, 16.4 and 427 -- a
hundredfold spread that reports nothing about biology and everything
about an unidentified direction in the objective.

Tying makes the copies one parameter. Copies must share both the prior
value and the source group to be tied: a shared value arrived at by
different routes is a coincidence, not an indistinguishability.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

# Isozyme copies of a reaction have this suffix; the text before it is
# the reaction they share.
ISOZYME_SUFFIX = "_EXP_"


def base_reaction(rxn_id: str) -> str:
    """The reaction an ``ec.rxns`` entry belongs to, suffix removed."""
    return rxn_id.split(ISOZYME_SUFFIX)[0]


def isozyme_tie_map(
    rxn_ids: Sequence[str],
    kcat0: np.ndarray,
    sources: Optional[Sequence[str]] = None,
    *,
    rel_tol: float = 1e-9,
) -> np.ndarray:
    """Index of the parameter each position follows.

    Positions that are their own representative are free; the rest
    follow one. Copies group together when they share a base reaction,
    a prior value (to ``rel_tol``) and -- when ``sources`` is given --
    a source group.

    Returns
    -------
    numpy.ndarray of int, one entry per position. ``tie_map[i] == i``
    for a free parameter, otherwise the index it follows. The
    representative of a group is its lowest index, so the map is
    idempotent: ``tie_map[tie_map] == tie_map``.
    """
    kcat0 = np.asarray(kcat0, dtype=float)
    if len(rxn_ids) != len(kcat0):
        raise ValueError(
            f"rxn_ids has {len(rxn_ids)} entries; kcat0 has {len(kcat0)}."
        )
    tie_map = np.arange(len(kcat0))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, rxn_id in enumerate(rxn_ids):
        # Rounding the prior into the key groups values equal to rel_tol
        # without comparing every pair.
        key = (base_reaction(str(rxn_id)),
               round(float(np.log(kcat0[i])) / rel_tol) if kcat0[i] > 0 else None,
               None if sources is None else str(sources[i]))
        groups[key].append(i)
    for members in groups.values():
        if len(members) > 1:
            tie_map[members] = members[0]
    return tie_map


def apply_ties(particles: np.ndarray, tie_map: np.ndarray) -> np.ndarray:
    """Give every tied position its representative's value.

    ``particles`` is ``(n_params,)`` or ``(n_params, n_particles)``;
    the array is modified in place and returned, so this can sit
    directly in a sampling loop.
    """
    tie_map = np.asarray(tie_map)
    if particles.shape[0] != len(tie_map):
        raise ValueError(
            f"particles has {particles.shape[0]} rows; tie_map has "
            f"{len(tie_map)}."
        )
    particles[...] = particles[tie_map, ...]
    return particles


def n_free(tie_map: np.ndarray) -> int:
    """How many parameters remain free once ties are applied."""
    tie_map = np.asarray(tie_map)
    return int((tie_map == np.arange(len(tie_map))).sum())
