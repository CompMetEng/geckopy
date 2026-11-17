"""Source classification and prior widths for evotune kcat tuning.
"""
from __future__ import annotations

import logging
import warnings
from collections import Counter

import numpy as np

from ...adapter.params import EvotuneParams
from ...gather_kcats.kcat_source_format import parse_kcat_source

logger = logging.getLogger(__name__)

#: Sentinel group name for an ``ec.source`` value matched by no
#: ``source_groups`` entry -- MATLAB's ``noKcatSource``.
UNLABELLED_GROUP = "unlabelled"


def classify_kcat_source(source: str, params: EvotuneParams) -> str:
    """Map one ``ec.source`` string to a trust-tier group name.

    Matching is case-insensitive. An entry of ``sources`` without a trailing
    ``*`` matches the origin of the source (``"brenda"`` for
    ``"brenda (wc=0, origin=1) | tuned=cmaes (prior=2)"``, so the detail and
    the tuning status are ignored) or the whole source string; these exact
    matches are tried over all groups first. Then the entries ending in
    ``*``, prefix patterns on the origin, are tried, the longest prefix first.
    Entries equal in length keep the order of the groups.

    Parameters
    ----------
    source
        One ``ec.source`` entry.
    params
        Supplies ``source_groups``.

    Returns
    -------
    str
        A key of ``params.source_groups``, or :data:`UNLABELLED_GROUP`
        if nothing matches -- callers then fall back to
        ``params.sigma0_log_default``.
    """
    source_lower = source.lower()
    origin = parse_kcat_source(source).origin.lower()
    forms = {source_lower, origin}
    patterns: list[tuple[int, int, str, str]] = []
    for position, (name, rule) in enumerate(params.source_groups.items()):
        for entry in rule.sources:
            entry = entry.lower()
            if entry.endswith("*"):
                patterns.append((-(len(entry) - 1), position, name, entry[:-1]))
            elif entry in forms:
                return name
    for _, _, name, prefix in sorted(patterns):
        if origin.startswith(prefix):
            return name
    return UNLABELLED_GROUP


def classify_kcat_sources(ec_sources: list[str], params: EvotuneParams) -> np.ndarray:
    """Vectorised :func:`classify_kcat_source` over ``ec.source``.

    Returns
    -------
    numpy.ndarray of str, shape ``(len(ec_sources),)``
    """
    return np.array([classify_kcat_source(s, params) for s in ec_sources], dtype=object)


def build_sigma0_log(groups: np.ndarray, params: EvotuneParams) -> np.ndarray:
    """Per-row prior std dev in log-space, from source-group membership.

    Every row starts at ``params.sigma0_log_default``, then rows whose
    group is a real (non-:data:`UNLABELLED_GROUP`) entry get that group's
    ``sigma0_log``, a standard deviation of the natural log of the kcat.

    Parameters
    ----------
    groups
        Per-row group names from :func:`classify_kcat_sources`.
    params
        Supplies ``sigma0_log_default`` and the groups' ``sigma0_log``.

    Returns
    -------
    numpy.ndarray of float, same shape as ``groups``.
    """
    sigma0_log = np.full(len(groups), params.sigma0_log_default, dtype=float)
    for name, rule in params.source_groups.items():
        sigma0_log[groups == name] = rule.sigma0_log
    return sigma0_log


def warn_unlabelled(
    groups: np.ndarray, sources: list[str], params: EvotuneParams, *, max_fraction: float = 0.2,
) -> None:
    """Warn when more than ``max_fraction`` of the tunable kcats are in no group.

    The kcats then take ``params.sigma0_log_default``. The warning names the
    most common origins (up to five) among them, by count descending and then by name.
    """
    unlabelled = [s for s, g in zip(sources, groups) if g == UNLABELLED_GROUP]
    if len(groups) == 0 or len(unlabelled) <= max_fraction * len(groups):
        return
    counts = Counter(parse_kcat_source(s).origin for s in unlabelled)
    origins = sorted(counts, key=lambda o: (-counts[o], o))[:5]
    message = (
        f"{len(unlabelled)} of {len(groups)} tunable kcats match no evotune source group and get "
        f"sigma0_log_default ({params.sigma0_log_default:g}); origins: {', '.join(origins)}"
    )
    logger.warning(message)
    warnings.warn(message, UserWarning, stacklevel=2)
