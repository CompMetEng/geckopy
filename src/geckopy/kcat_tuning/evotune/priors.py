"""Source classification and prior widths for evotune kcat tuning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...adapter.params import EvotuneParams

#: Sentinel group name for an ``ec.source`` value matched by no
#: ``source_groups`` entry -- MATLAB's ``noKcatSource``.
UNLABELLED_GROUP = "unlabelled"


def classify_kcat_source(
    source: str,
    params: EvotuneParams,
    *,
    okp_method: Optional[str] = None,
) -> str:
    """Map one ``ec.source`` string to a trust-tier group name.

    Matching is case-insensitive, mirroring MATLAB's
    ``strcmpi(ecModel.ec.source, kcatSources{i})``, and ignores the
    bracketed fuzzy-match detail geckopy appends to a database source.
    ``ec.source`` reads ``"brenda (wc=0, origin=1)"`` where MATLAB's
    reads ``"brenda"``, so matching the whole string would leave every
    fuzzy-matched kcat unlabelled and silently on the default prior
    width -- which is most of a model's kcats. A ``source_groups`` entry
    listing the full string still matches it.

    Parameters
    ----------
    source
        One ``ec.source`` entry.
    params
        Supplies ``source_groups``.
    okp_method
        The project's configured OpenKineticsPredictor method
        (``OkpParams.method``), if any. A group with ``match_okp=True``
        also matches ``source`` when it equals this (case-insensitive).

    Returns
    -------
    str
        A key of ``params.source_groups``, or :data:`UNLABELLED_GROUP`
        if nothing matches -- callers then fall back to
        ``params.sigma0_log_default`` / ``shrink_thr_default`` /
        ``force_prior_thr_default``.
    """
    source_lower = source.lower()
    token = source_lower.split(" (")[0].strip()
    forms = {source_lower, token}
    for name, rule in params.source_groups.items():
        if any(s.lower() in forms for s in rule.sources):
            return name
        if (
            rule.match_okp
            and okp_method is not None
            and okp_method.lower() in forms
        ):
            return name
    return UNLABELLED_GROUP


def classify_kcat_sources(
    ec_sources: list[str],
    params: EvotuneParams,
    *,
    okp_method: Optional[str] = None,
) -> np.ndarray:
    """Vectorised :func:`classify_kcat_source` over ``ec.source``.

    Returns
    -------
    numpy.ndarray of str, shape ``(len(ec_sources),)``
    """
    return np.array(
        [
            classify_kcat_source(s, params, okp_method=okp_method)
            for s in ec_sources
        ],
        dtype=object,
    )


def build_sigma0_log(groups: np.ndarray, params: EvotuneParams) -> np.ndarray:
    """Per-row prior std dev in log-space, from source-group membership.

    Mirrors MATLAB's ``sigma0log = sigma0logDefault * ones(...);
    sigma0log(uniqKcatParams) = sigma0logSource(kcatSourceIdx(...))``:
    every row starts at ``params.sigma0_log_default``, then rows whose
    group is a real (non-:data:`UNLABELLED_GROUP`) entry get
    overridden by that group's ``sigma0_log_source`` value.

    Parameters
    ----------
    groups
        Per-row group names from :func:`classify_kcat_sources`.
    params
        Supplies ``sigma0_log_default`` and ``sigma0_log_source``.

    Returns
    -------
    numpy.ndarray of float, same shape as ``groups``.
    """
    sigma0_log = np.full(len(groups), params.sigma0_log_default, dtype=float)
    for name, value in params.sigma0_log_source.items():
        sigma0_log[groups == name] = value
    return sigma0_log
