"""Fuzzy-match each ec reaction's EC + substrates against BRENDA kcat data.

Ported from GECKO MATLAB:
src/geckomat/gather_kcats/fuzzyKcatMatching.m.

This subpackage is split into four files by concern:

- ``_escalation.py``   -- EC-token wildcard escalation
- ``_organism.py``     -- organism resolution and per-row filtering
- ``_brenda_query.py`` -- BRENDA-table lookup, kcat selection,
                          and the per-EC iterative match loop
- ``__init__.py``      -- this file: the public ``fuzzy_kcat_matching``
                          orchestrator + small DataFrame helpers

The split is internal: ``from geckopy import fuzzy_kcat_matching``
and ``from geckopy.gather_kcats import fuzzy_kcat_matching`` keep
working unchanged.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd

from ._brenda_query import (
    NO_MATCH_WC,
    SEARCH_RANK,
    build_ec_indices,
    find_ec_rows,
    iterative_match_one_ec,
    match_kcat,
    main_match,
)
from ._escalation import apply_force_wildcards, escalate_wildcard
from ._organism import filter_by_organism, resolve_organism_index

if TYPE_CHECKING:
    from ...adapter import ModelAdapter
    from ...databases import BrendaData, PhylDist
    from ...ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)

# Public surface of the (internally split) subpackage. The helpers are
# re-exported under their plain names; the leading-underscore back-compat
# aliases have been removed.
__all__ = [
    "fuzzy_kcat_matching",
    "NO_MATCH_WC",
    "apply_force_wildcards",
    "build_ec_indices",
    "escalate_wildcard",
    "filter_by_organism",
    "find_ec_rows",
    "iterative_match_one_ec",
    "main_match",
    "match_kcat",
    "resolve_organism_index",
]


# Output DataFrame schema. Used both in the main return and in the
# empty-result helper.
_RESULT_COLUMNS = [
    "rxn_id",
    "source",
    "eccode",
    "substrates",
    "genes",
    "kcat",
    "wildcard_level",
    "origin",
]


def fuzzy_kcat_matching(
    model: "EcModel",
    brenda: "BrendaData",
    phyl_dist: "PhylDist",
    *,
    ec_rxns: Optional[Iterable[str]] = None,
    force_wildcard_level: int = 0,
    aggregate: Optional[str] = None,
    adapter: Optional["ModelAdapter"] = None,
) -> pd.DataFrame:
    """Match each ec reaction to BRENDA kcat data with progressive
    relaxation of EC, substrate, and organism specificity.

    Ported from GECKO MATLAB:
    src/geckomat/gather_kcats/fuzzyKcatMatching.m.

    For each selected reaction, the EC string in
    ``model.ec.eccodes[i]`` is split on ``;`` into one or more EC
    tokens. Each token goes through:

    1. **Wildcard cascade** (per token): try the token as-is. If no
       match, replace its rightmost numeric level with ``-`` and try
       again. Repeat until matched or all four levels are wildcards.
    2. **Six-level matching** (per attempt): try increasingly
       relaxed criteria:

       * org + correct substrate + KCAT (output origin 1)
       * any organism + correct substrate + KCAT (output origin 2)
       * org + any substrate + KCAT (output origin 3)
       * org + specific activity (output origin 5)
       * any organism + any substrate + KCAT (output origin 4)
       * any organism + SA (output origin 6)

       When the EC has any wildcard, substrate-matching levels are
       skipped (the substrate field of an EC carries the catalytic
       specificity that wildcards already discard).

    Across the multiple EC tokens of one reaction, the best match is
    chosen by minimum wildcard count, then the earliest-tried level in
    the order above (so org + SA, origin 5, beats any organism + any
    substrate + KCAT, origin 4), then maximum kcat.

    The model's organism is read from
    ``model.adapter.params.org_name``. When BRENDA has no exact
    organism match, ``phyl_dist`` is used to pick the BRENDA row(s)
    for the phylogenetically closest organism(s) (with genus fallback
    for organisms missing from KEGG).

    The six levels are tried in the fixed order listed above, not in
    order of output rank: org + SA (origin 5) is tried before any
    organism + any substrate + KCAT (origin 4), so when both would
    match, org + SA wins even though origin 4 would otherwise be
    considered the better rank.

    Match statistics (counts per origin and per wildcard level) are
    logged via ``logger.info`` rather than returned.

    Parameters
    ----------
    model
        EcModel with ``model.ec.rxns`` and ``model.ec.eccodes``
        already populated. Reactions whose ``ec.eccodes`` entry is
        empty contribute a row to the output with ``kcat=0`` and
        ``origin=NA``.
    brenda
        BRENDA tables loaded by ``load_brenda_data``.
    phyl_dist
        KEGG phylogenetic distance loaded by ``load_phyl_dist``.
    ec_rxns
        Optional iterable of reaction IDs (must appear in
        ``model.ec.rxns``) restricting which entries are matched.
        ``None`` means match all.
    force_wildcard_level
        Force at least this many wildcards on every EC token before
        the iterative escalation begins. Default 0 (no forcing).
    aggregate
        How to collapse the BRENDA rows matched at a search level to one
        kcat: ``"max"`` (the highest reported turnover) or ``"median"``
        (more robust to assay outliers / engineered mutants). ``None``
        (default) reads the value from
        ``model.adapter.params.kcat_aggregate_brenda``, which itself
        defaults to ``"max"``.

        The choice drives both aggregation layers consistently: it picks
        the matching snapshot view (``brenda.kcat_for(aggregate)`` /
        ``brenda.sa_for(aggregate)``) and applies the same reduction
        across whichever rows survive the EC + organism + substrate
        gates. See ``docs/kcat_aggregation.md`` for the empirical
        rationale.
    adapter
        Model adapter for ``org_name`` and the aggregation default
        (default: ``model.adapter``).

    Returns
    -------
    pandas.DataFrame
        Columns:

        ============ ============================================================
        rxn_id       (str) reaction ID, matching ``model.ec.rxns``
        source       (str) ``"brenda"`` for every row
        eccode       (str) the ``;``-joined EC string used (after
                     ``force_wildcard_level`` but before iterative escalation)
        substrates   (list[str]) per-row substrate names
        genes        (list[str]) always empty for fuzzy matching; kept for
                     schema compatibility with future DLKcat output
        kcat         (float) selected kcat in 1/s, or 0.0 if no match
        wildcard_level (Int64) actual wildcard count at match time, or NA
        origin       (Int64) output origin 1..6, or NA if no match
        ============ ============================================================
    """
    from ...adapter import resolve_adapter
    adapter = resolve_adapter(
        model, adapter,
        purpose="fuzzy_kcat_matching reads params.org_name from the adapter",
    )
    if force_wildcard_level < 0:
        raise ValueError(
            f"force_wildcard_level must be >= 0, got {force_wildcard_level}"
        )

    if aggregate is None:
        aggregate = adapter.params.kcat_aggregate_brenda
    organism_name = (adapter.params.org_name or "").strip()

    n = model.ec.n_rxns
    if n == 0:
        return _empty_result_df()

    if ec_rxns is None:
        positions: list[int] = list(range(n))
    else:
        ec_rxns_list = list(ec_rxns)
        index_by_id = {rid: i for i, rid in enumerate(model.ec.rxns)}
        unknown = [rid for rid in ec_rxns_list if rid not in index_by_id]
        if unknown:
            preview = unknown[:5]
            raise ValueError(
                f"{len(unknown)} reaction ID(s) in ec_rxns are not "
                f"present in model.ec.rxns (examples: {preview})"
            )
        positions = [index_by_id[rid] for rid in ec_rxns_list]

    if not positions:
        return _empty_result_df()

    # The snapshot stores both max and median aggregations per
    # (ec, substrate, organism) triple; pick the view that matches the
    # requested per-query aggregation so both layers are consistent.
    brenda_kcat = brenda.kcat_for(aggregate)
    brenda_sa = brenda.sa_for(aggregate)
    kcat_ec_indices = build_ec_indices(brenda_kcat)
    sa_ec_indices = build_ec_indices(brenda_sa)
    org_index = resolve_organism_index(organism_name, phyl_dist)

    out_rxn_ids: list[str] = []
    out_substrates: list[list[str]] = []
    out_kcats: list[float] = []
    out_eccodes: list[str] = []
    out_wcs: list[object] = []
    out_origins: list[object] = []

    for i in positions:
        rxn_id = model.ec.rxns[i]
        cobra_rxn_id = rxn_id[4:] if model.ec.gecko_light else rxn_id

        try:
            cobra_rxn = model.reactions.get_by_id(cobra_rxn_id)
            substrates = [m.name for m, c in cobra_rxn.metabolites.items() if c < 0]
            substrate_coeffs = [-c for c in cobra_rxn.metabolites.values() if c < 0]
        except KeyError:
            substrates, substrate_coeffs = [], []

        out_rxn_ids.append(rxn_id)
        out_substrates.append(list(substrates))

        ec_string = model.ec.eccodes[i] or ""
        ec_tokens = [
            apply_force_wildcards(t.strip(), force_wildcard_level)
            for t in ec_string.split(";") if t.strip()
        ]

        if not ec_tokens:
            out_kcats.append(0.0)
            out_eccodes.append("")
            out_wcs.append(pd.NA)
            out_origins.append(pd.NA)
            continue

        per_token: list[tuple[float, int, int]] = []
        for token in ec_tokens:
            kcat, origin, wc = iterative_match_one_ec(
                token, substrates, substrate_coeffs,
                brenda_kcat, brenda_sa,
                kcat_ec_indices, sa_ec_indices,
                organism_name, phyl_dist, org_index,
                aggregate=aggregate,
            )
            per_token.append((kcat, origin, wc))

        matched = [r for r in per_token if r[1] > 0]
        out_eccodes.append(";".join(ec_tokens))
        if not matched:
            out_kcats.append(0.0)
            out_wcs.append(pd.NA)
            out_origins.append(pd.NA)
            continue

        min_wc = min(r[2] for r in matched)
        at_min_wc = [r for r in matched if r[2] == min_wc]
        min_rank = min(SEARCH_RANK[r[1]] for r in at_min_wc)
        at_min_rank = [r for r in at_min_wc if SEARCH_RANK[r[1]] == min_rank]
        best = max(at_min_rank, key=lambda r: r[0])

        out_kcats.append(best[0])
        out_wcs.append(best[2])
        out_origins.append(best[1])

    df = pd.DataFrame({
        "rxn_id": out_rxn_ids,
        "source": ["brenda"] * len(out_rxn_ids),
        "eccode": out_eccodes,
        "substrates": out_substrates,
        "genes": [[] for _ in out_rxn_ids],
        "kcat": out_kcats,
        "wildcard_level": pd.array(out_wcs, dtype="Int64"),
        "origin": pd.array(out_origins, dtype="Int64"),
    })

    matched_count = int(df["origin"].notna().sum())
    if matched_count > 0:
        origin_counts = df["origin"].value_counts(dropna=True).to_dict()
        wc_counts = df["wildcard_level"].value_counts(dropna=True).to_dict()
        logger.info(
            "fuzzy_kcat_matching: matched %d of %d reactions. "
            "Origin counts: %s. Wildcard-level counts: %s.",
            matched_count, len(df), dict(origin_counts), dict(wc_counts),
        )
    else:
        logger.info(
            "fuzzy_kcat_matching: no matches for any of %d reactions.",
            len(df),
        )

    return df


# --------------------------------------------------------------------------- #
# Small helpers used only by the orchestrator
# --------------------------------------------------------------------------- #


def _empty_result_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema."""
    return pd.DataFrame({
        "rxn_id": pd.Series([], dtype=str),
        "source": pd.Series([], dtype=str),
        "eccode": pd.Series([], dtype=str),
        "substrates": pd.Series([], dtype=object),
        "genes": pd.Series([], dtype=object),
        "kcat": pd.Series([], dtype=float),
        "wildcard_level": pd.array([], dtype="Int64"),
        "origin": pd.array([], dtype="Int64"),
    })


__all__ = ["fuzzy_kcat_matching"]
