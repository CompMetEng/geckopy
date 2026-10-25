"""CMA-ES kcat tuning: fitting kcats to experimental data.

Three functions, normally called in this order:

- :func:`screen_kcat_leverage` -- report which kcats the data can
  actually speak to, with no optimisation involved. Perturbs each
  tie-group by +-``fold`` and measures the resulting change in RMSE,
  ranked by that leverage weighted by how much the source is trusted.
  Useful on its own for curation, independent of any tuning run.
- :func:`select_tunable_mask` -- turn a screen report into a boolean
  ``tunable_mask``, by keeping the fewest highest-ranked groups whose
  combined leverage reaches ``target_impact_share`` of the total. A
  *relative* cutoff, so the same ``target_impact_share`` selects a
  comparable quality of parameter set on a model with a different
  leverage scale, unlike an absolute threshold or a fixed count.
- :func:`cmaes_kcat_tuning` -- the tuning run. Screens and selects
  automatically when no mask is given.

Parallel scoring (``n_proc``)
------------------------------
Each generation's new candidates are scored independently of each
other, so they parallelise across a process pool -- using
``cobra.util.process_pool.ProcessPool`` (the same primitive cobrapy's
own ``single_gene_deletion``/``single_reaction_deletion``/etc. use for
"repeated FBA on slightly-perturbed model copies", exactly this
module's shape of problem) rather than driving ``multiprocessing``
directly: one persistent ``EcModel`` copy per **worker process**
(deserialised once at pool startup, not once per particle and not one
shared mutable object across processes), which ``ProcessPool`` also
gets right on Windows for free (a temp-file-based initarg handoff that
works around a real performance issue in raw
``multiprocessing.Pool(initializer=...)`` there -- see its docstring).
Each worker then scores every particle it's handed against its own
copy via incremental ``apply_kcat_constraints(update_rxns=...)``
calls, exactly like the serial path -- a per-particle ``EcModel.copy()``
is not used either way (~225x an FBA solve on a real-scale model).

CMA-ES's own ``ask``/``tell`` sequence stays single-threaded in the
main process -- it's cheap relative to FBA and, more importantly,
doing it there keeps the *sequence* of candidates proposed each
generation deterministic given ``seed``, independent of ``n_proc`` or
how work happens to be chunked across workers. Only the scoring of an
already-fixed batch of candidates is parallelised, and scoring is a
pure function of the kcat vector -- each candidate's solves start from
a cold basis (:func:`_reset_solver_basis`), so a worker's result does
not depend on which candidate it scored before. A run with ``n_proc=1``
and the same ``seed`` therefore reproduces bit-for-bit identical
results to one with ``n_proc>1``.

``make_anaerobic``/``change_protein_biomass``, if supplied, must be
importable top-level functions (not lambdas/closures) when running
with ``n_proc>1`` on Windows, where ``multiprocessing`` needs to
pickle them into each worker; POSIX's default start method (typically
``fork``, inherited via copy-on-write with no pickling involved) has
no such restriction.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import cma
import cobra
import numpy as np
import pandas as pd
from cobra.util import ProcessPool

from ...ec_model.pipeline.apply_kcat import apply_kcat_constraints
from ..corrections import Correction, corrections as build_corrections
from .data import EvotuneData, load_evotune_data
from .distance import evotune_distance, compute_excarbon
from .parsimony import fold_change, n_changed
from .priors import build_sigma0_log, classify_kcat_sources
from .simulate import simulate_evotune_dataset
from .tying import isozyme_tie_map

if TYPE_CHECKING:
    from ...adapter import ModelAdapter
    from ...adapter.params import EvotuneParams
    from ...ec_model.ec_model import EcModel

logger = logging.getLogger(__name__)


@dataclass
class EvotuneResult:
    """Outcome of a :func:`cmaes_kcat_tuning` run.

    The model is mutated in place (the best kcat vector found is
    applied to ``model.ec.kcat``), and this result records the run's
    history for inspection.

    Attributes
    ----------
    rxns
        Tunable ``ec.rxns`` IDs, in the column order used throughout
        (``rxns[i]`` corresponds to ``old_kcat[i]``/``new_kcat[i]``/
        ``groups[i]``). Tied isozyme copies are all listed, sharing one
        value in ``new_kcat``.
    old_kcat, new_kcat
        kcat values before/after tuning, parallel to ``rxns``.
    groups
        Source-group name per tunable row (from
        ``priors.classify_kcat_sources``).
    rmse_trace
        Best-so-far plain RMSE, per generation. Always the plain RMSE,
        so runs stay comparable across penalties and against MATLAB
        even when the search used a penalised objective.
    objective_trace
        Best-so-far value of the quantity the search actually
        minimised, per generation. Identical to ``rmse_trace`` when
        ``params.prior_penalty_weight`` is 0.
    n_generations
        Number of generations actually run.
    converged
        True if ``rmse_trace[-1] <= params.rmse_threshold``; False if
        ``max_generations`` was reached first.
    """

    rxns: list[str] = field(default_factory=list)
    old_kcat: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    new_kcat: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    groups: list[str] = field(default_factory=list)
    rmse_trace: list[float] = field(default_factory=list)
    objective_trace: list[float] = field(default_factory=list)
    n_generations: int = 0
    converged: bool = False

    def corrections(
        self,
        *,
        leverage: Optional[np.ndarray] = None,
        names: Optional[Sequence[str]] = None,
        ec_codes: Optional[Sequence[str]] = None,
        rel_tol: float = 0.02,
    ) -> list["Correction"]:
        """The changes this run makes, most consequential first.

        The vector in ``new_kcat`` is the answer; this is the answer in
        a form that can be checked. Pass ``leverage`` from a
        one-at-a-time screen (:func:`screen_kcat_leverage`) to rank by
        how much each change matters rather than by how far it moved,
        and to fill the share columns. ``annotate_from_model`` supplies
        ``names``/``ec_codes``.
        """
        return build_corrections(
            self.new_kcat, self.old_kcat, self.rxns,
            sources=self.groups, names=names, ec_codes=ec_codes,
            leverage=leverage, rel_tol=rel_tol,
        )


def _resolve_context(model, adapter, params, evotune_data, bio_rxn, okp_method):
    """Resolve every optional input against ``model``'s adapter, or
    raise if it can't be and none was given explicitly."""
    if adapter is None:
        adapter = getattr(model, "adapter", None)
    if params is None:
        if adapter is None:
            raise ValueError(
                "params not given and model has no adapter to read "
                "adapter.params.evotune from."
            )
        params = adapter.params.evotune
    if evotune_data is None:
        if adapter is None:
            raise ValueError(
                "evotune_data not given and model has no adapter to load it from."
            )
        evotune_data = load_evotune_data(adapter)
    if evotune_data.flux_data is None and evotune_data.max_grate is None:
        raise ValueError(
            "evotune_data has neither flux_data nor max_grate -- nothing to "
            "tune against."
        )
    if bio_rxn is None:
        if adapter is None:
            raise ValueError("bio_rxn not given and model has no adapter.")
        bio_rxn = adapter.params.bio_rxn
    if okp_method is None and adapter is not None:
        okp_method = adapter.params.okp.method
    return params, evotune_data, bio_rxn, okp_method


def _tunable_context(
    model, params, evotune_data, bio_rxn, okp_method, tunable_mask,
):
    """The tunable subset plus everything derived from it: tie groups,
    trust weights, and the excarbon table. Shared by every function in
    this module so a screen and the run it feeds are always looking at
    exactly the same parameters."""
    is_tunable = model.ec.kcat > 0
    if tunable_mask is not None:
        tunable_mask = np.asarray(tunable_mask, dtype=bool)
        if tunable_mask.shape != is_tunable.shape:
            raise ValueError(
                f"tunable_mask has shape {tunable_mask.shape}; expected "
                f"{is_tunable.shape} to match model.ec.rxns."
            )
        is_tunable = is_tunable & tunable_mask
    tunable_idx = np.flatnonzero(is_tunable)
    if tunable_idx.size == 0:
        raise ValueError(
            "No tunable kcats: model.ec.kcat is all <= 0"
            + (", or tunable_mask excludes every one with a kcat."
               if tunable_mask is not None else ".")
        )
    ec_rxn_ids_tunable = [model.ec.rxns[i] for i in tunable_idx]
    kcat0 = model.ec.kcat[tunable_idx].astype(float).copy()
    sources = [model.ec.source[i] for i in tunable_idx]
    groups = classify_kcat_sources(sources, params, okp_method=okp_method)
    sigma0_log = build_sigma0_log(groups, params)

    excarbon_rxn_ids: set[str] = {bio_rxn}
    for data in (evotune_data.flux_data, evotune_data.max_grate):
        if data is not None:
            excarbon_rxn_ids.update(data.exch_rxn_ids)
    excarbon_rxn_ids.update(evotune_data.zero_flux)
    excarbon = compute_excarbon(model, excarbon_rxn_ids, bio_rxn_id=bio_rxn)

    tie_map = (
        isozyme_tie_map(ec_rxn_ids_tunable, kcat0, list(groups))
        if params.tie_isozymes else np.arange(len(kcat0))
    )
    return (tunable_idx, ec_rxn_ids_tunable, kcat0, groups, sigma0_log,
            excarbon, tie_map)


def screen_kcat_leverage(
    model: "EcModel",
    *,
    adapter: Optional["ModelAdapter"] = None,
    params: Optional["EvotuneParams"] = None,
    evotune_data: Optional[EvotuneData] = None,
    okp_method: Optional[str] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
    tunable_mask: Optional[np.ndarray] = None,
    fold: float = 2.0,
    n_proc: Optional[int] = None,
) -> pd.DataFrame:
    """Report which kcats the data can actually speak to.

    No optimisation happens here -- this only says which parameters
    are worth curating or tuning, and can be read on its own well
    before any tuning run. Each tie-group (isozyme copies sharing a
    prior and a source move together when ``params.tie_isozymes``) is
    perturbed up and down by ``fold``, and its leverage is the largest
    resulting change in plain RMSE -- always the plain fit, never the
    ``prior_penalty_weight``-penalised objective, since leverage asks
    what the data supports, independent of how much a move is charged
    for. Groups are ranked by that leverage weighted by
    ``sigma0_log`` -- :func:`~.parsimony.identifiability_mask`'s
    convention, where a trusted source needs proportionally more
    measured effect to rank alongside an untrusted one. This
    trust-weighting means the reported ``cum_leverage_share`` is a
    different quantity from :func:`~.parsimony.impact_share`, which is
    unweighted.

    Costs one simulation per condition per tie-group probed (two
    probes each, up and down) plus one baseline -- comparable in scale
    to a full tuning run's evaluation budget, not a cheap pre-check.
    FVA-unreachable kcats are not filtered out beforehand: a kcat that
    cannot carry flux under any condition shows zero leverage here by
    construction and simply sorts to the bottom, which is a more
    direct test than a separately computed static reachability mask.

    Returns
    -------
    pandas.DataFrame
        One row per tie-group, sorted by ``rank`` descending:
        ``rxn_id`` (the group's representative), ``n_isozymes``,
        ``source_group``, ``kcat0``, ``leverage``, ``sigma0_log``,
        ``rank`` (``leverage * sigma0_log``), and ``cum_leverage_share``
        (the running total of ``rank``, as a fraction of its sum over
        every row). An internal ``_positions`` column (a tuple of
        indices into ``model.ec.rxns``) is what
        :func:`select_tunable_mask` needs to expand a row back into a
        mask; drop it before showing the table to a person.
    """
    params, evotune_data, bio_rxn, okp_method = _resolve_context(
        model, adapter, params, evotune_data, bio_rxn, okp_method)
    (tunable_idx, ec_rxn_ids_tunable, kcat0, groups, sigma0_log, excarbon,
     tie_map) = _tunable_context(
        model, params, evotune_data, bio_rxn, okp_method, tunable_mask)

    reps = np.flatnonzero(tie_map == np.arange(len(tie_map)))
    members_of = {int(r): np.flatnonzero(tie_map == r) for r in reps}

    if n_proc is None:
        n_proc = cobra.Configuration().processes
    n_proc = max(1, int(n_proc))

    pairs = []
    vectors = [kcat0]
    for rep in reps:
        members = members_of[int(rep)]
        up = kcat0.copy(); up[members] = up[members] * fold
        dn = kcat0.copy(); dn[members] = dn[members] / fold
        pairs.append(int(rep))
        vectors.append(up)
        vectors.append(dn)

    if n_proc == 1:
        rmses = [
            _score_kcat_vector(
                model, tunable_idx, ec_rxn_ids_tunable, evotune_data, excarbon,
                bio_rxn, v, make_anaerobic=make_anaerobic,
                change_protein_biomass=change_protein_biomass,
                max_growth_weight=params.max_growth_weight,
            )[1]
            for v in vectors
        ]
    else:
        with ProcessPool(
            n_proc, initializer=_init_worker,
            initargs=(model, tunable_idx, ec_rxn_ids_tunable, evotune_data,
                     excarbon, bio_rxn, make_anaerobic, change_protein_biomass,
                     params.max_growth_weight, 0.0, None, None),
        ) as pool:
            chunk = max(1, len(vectors) // (n_proc * 4))
            rmses = [r for _, r in pool.map(_score_worker, vectors, chunksize=chunk)]

    # The n_proc==1 path scores directly against `model`, leaving its
    # ec.kcat at whichever probe vector was scored last; a pool worker
    # scores its own copy instead, so this is a no-op there. Either
    # way, `model` must come out holding the prior it went in with.
    model.ec.kcat[tunable_idx] = kcat0
    if ec_rxn_ids_tunable:
        apply_kcat_constraints(model, update_rxns=ec_rxn_ids_tunable)

    base_rmse = rmses[0]
    rows = []
    for i, rep in enumerate(pairs):
        up_rmse, dn_rmse = rmses[1 + 2 * i], rmses[2 + 2 * i]
        leverage = max(abs(up_rmse - base_rmse), abs(dn_rmse - base_rmse))
        members = members_of[rep]
        rows.append({
            "rxn_id": ec_rxn_ids_tunable[rep],
            "n_isozymes": int(len(members)),
            "source_group": str(groups[rep]),
            "kcat0": float(kcat0[rep]),
            "leverage": float(leverage),
            "sigma0_log": float(sigma0_log[rep]),
            "_positions": tuple(int(tunable_idx[m]) for m in members),
        })

    df = pd.DataFrame(rows)
    df["rank"] = df["leverage"] * df["sigma0_log"]
    df = df.sort_values("rank", ascending=False, ignore_index=True)
    total = float(df["rank"].sum())
    df["cum_leverage_share"] = (
        (df["rank"].cumsum() / total) if total > 0 else 0.0
    )
    return df


def select_tunable_mask(
    model: "EcModel",
    screen: pd.DataFrame,
    *,
    target_impact_share: float = 0.9,
) -> np.ndarray:
    """Build a ``tunable_mask`` from a :func:`screen_kcat_leverage` report.

    Keeps the fewest highest-ranked groups whose combined leverage
    reaches ``target_impact_share`` of the total -- a relative cutoff,
    so the same ``target_impact_share`` selects a comparable quality of
    parameter set on any model, rather than an absolute leverage value
    or a fixed count calibrated on a different model's scale.
    ``target_impact_share=0.9`` is a starting point, not a validated
    universal constant; check the resulting mask's size against
    :func:`screen_kcat_leverage`'s own curve before trusting it on a
    new model.

    Pure and cheap -- no simulation, only ``screen`` is read.
    """
    mask = np.zeros(len(model.ec.rxns), dtype=bool)
    if screen.empty:
        return mask
    prev_cum = screen["cum_leverage_share"].shift(1, fill_value=0.0)
    keep = prev_cum < target_impact_share
    for positions in screen.loc[keep, "_positions"]:
        mask[list(positions)] = True
    return mask


def cmaes_kcat_tuning(
    model: "EcModel",
    *,
    adapter: Optional["ModelAdapter"] = None,
    params: Optional["EvotuneParams"] = None,
    evotune_data: Optional[EvotuneData] = None,
    okp_method: Optional[str] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
    tunable_mask: Optional[np.ndarray] = None,
    screen: Optional[pd.DataFrame] = None,
    target_impact_share: float = 0.9,
    popsize: Optional[int] = None,
    n_proc: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> EvotuneResult:
    """Tune kcats against experimental data with CMA-ES.

    Configuration comes entirely from ``EvotuneParams``: trust tiers,
    ``max_growth_weight``, ``prior_penalty_weight`` and ``tie_isozymes``
    control the search, and ``max_generations``/``rmse_threshold`` are
    its stopping conditions.

    Parameters
    ----------
    model
        EcModel with a populated ``ec.kcat``. Mutated in place: on
        return, tunable rows carry the best kcat vector CMA-ES found,
        with kcat constraints already applied.
    tunable_mask, screen, target_impact_share
        Control which kcats are searched over, in order of precedence.
        Pass ``tunable_mask`` to fix the set yourself. Otherwise, a
        mask is built by :func:`select_tunable_mask` at
        ``target_impact_share``, from ``screen`` if given (so a
        previously computed :func:`screen_kcat_leverage` report need
        not be recomputed) or from a fresh screen otherwise.
    popsize
        CMA-ES population size. Defaults to ``cma``'s own
        dimension-scaled default (``4 + floor(3 ln n)``), so it adapts
        to however many free parameters the mask/tying produced rather
        than a value calibrated on one particular model.
    n_proc
        Number of worker processes for scoring each generation's
        candidates. Defaults to ``cobra.Configuration().processes``.
        ``1`` runs the original serial path (no ``Pool`` at all). See
        the module docstring's "Parallel scoring" section.
    seed
        If given, seeds the search, for reproducible runs.
    verbose
        Whether per-generation progress is logged at INFO.
    make_anaerobic, change_protein_biomass
        Forwarded to ``simulate.simulate_evotune_dataset`` -- see its
        docstring; geckopy has no generic organism-agnostic
        implementation of these yet. See the module docstring's
        "Parallel scoring" section for a picklability caveat when
        ``n_proc>1``.

    Returns
    -------
    EvotuneResult
        ``rxns``/``old_kcat``/``new_kcat``/``groups`` cover the
        selected tunable set, tied groups included (their members share
        one value in ``new_kcat``). ``rmse_trace``/``objective_trace``
        record the best-so-far plain RMSE and optimised objective per
        generation.

    Raises
    ------
    ValueError
        If there are no tunable kcats, or fewer than two free
        parameters remain after masking and tying -- too little for
        CMA-ES to search over.
    """
    params, evotune_data, bio_rxn, okp_method = _resolve_context(
        model, adapter, params, evotune_data, bio_rxn, okp_method)

    if tunable_mask is None:
        if screen is None:
            screen = screen_kcat_leverage(
                model, adapter=adapter, params=params, evotune_data=evotune_data,
                okp_method=okp_method, bio_rxn=bio_rxn,
                make_anaerobic=make_anaerobic,
                change_protein_biomass=change_protein_biomass,
                n_proc=n_proc,
            )
        tunable_mask = select_tunable_mask(
            model, screen, target_impact_share=target_impact_share)

    (tunable_idx, ec_rxn_ids_tunable, kcat0, groups, sigma0_log, excarbon,
     tie_map) = _tunable_context(
        model, params, evotune_data, bio_rxn, okp_method, tunable_mask)

    reps = np.flatnonzero(tie_map == np.arange(len(tie_map)))
    if len(reps) < 2:
        raise ValueError(
            f"Only {len(reps)} free parameter(s) after masking and tying; "
            "too few for CMA-ES. Widen tunable_mask/target_impact_share."
        )
    assign = np.searchsorted(reps, tie_map)
    x0 = np.log(kcat0[reps])
    sig = sigma0_log[reps]
    lo, hi = kcat_bounds(kcat0)
    bounds = ([float(np.log(v)) for v in lo[reps]],
              [float(np.log(v)) for v in hi[reps]])

    def expand(xm: np.ndarray) -> np.ndarray:
        """(n_free, n_particles) log-space matrix -> (n_tunable,
        n_particles) kcat matrix, tied rows sharing their rep's value."""
        return np.exp(xm[assign, :])

    if n_proc is None:
        n_proc = cobra.Configuration().processes
    n_proc = max(1, int(n_proc))

    options = {
        "verbose": -9, "maxiter": params.max_generations,
        "CMA_diagonal": True, "bounds": bounds,
        "scaling_of_variables": (sig / np.median(sig)).tolist(),
    }
    if popsize is not None:
        options["popsize"] = int(popsize)
    if seed is not None:
        # cma treats seed=0 as falsy and falls back to an unseeded,
        # time-based draw instead of raising -- verified directly
        # against this cma version, not assumed. +1 keeps every
        # caller-facing seed (0, 1, 2, ...) off that value.
        options["seed"] = int(seed) + 1
    es = cma.CMAEvolutionStrategy(x0, float(np.median(sig)), options)

    pool_cm = (
        nullcontext(None) if n_proc == 1 else
        ProcessPool(
            n_proc, initializer=_init_worker,
            initargs=(model, tunable_idx, ec_rxn_ids_tunable, evotune_data,
                     excarbon, bio_rxn, make_anaerobic, change_protein_biomass,
                     params.max_growth_weight, params.prior_penalty_weight,
                     kcat0, sigma0_log),
        )
    )

    def _score_batch(kcat_matrix, pool):
        if pool is None:
            return np.array([
                _score_kcat_vector(
                    model, tunable_idx, ec_rxn_ids_tunable, evotune_data,
                    excarbon, bio_rxn, kcat_matrix[:, j],
                    make_anaerobic=make_anaerobic,
                    change_protein_biomass=change_protein_biomass,
                    max_growth_weight=params.max_growth_weight,
                    prior_penalty_weight=params.prior_penalty_weight,
                    kcat0=kcat0, sigma0_log=sigma0_log,
                )
                for j in range(kcat_matrix.shape[1])
            ])
        cols = [kcat_matrix[:, j] for j in range(kcat_matrix.shape[1])]
        chunk = max(1, len(cols) // (n_proc * 4))
        return np.array(pool.map(_score_worker, cols, chunksize=chunk))

    with pool_cm as pool:
        obj0, rmse0 = _score_batch(kcat0.reshape(-1, 1), pool)[0]
        best_obj, best_rmse, best_vec = float(obj0), float(rmse0), kcat0.copy()

        rmse_trace: list[float] = []
        objective_trace: list[float] = []
        generation = 0
        while not es.stop():
            asks = es.ask()
            xm = np.array(asks).T
            scores = _score_batch(expand(xm), pool)
            es.tell(asks, list(scores[:, 0]))
            generation += 1

            j = int(np.argmin(scores[:, 0]))
            if scores[j, 0] < best_obj:
                best_obj = float(scores[j, 0])
                best_rmse = float(scores[j, 1])
                best_vec = expand(xm[:, [j]])[:, 0].copy()
            rmse_trace.append(best_rmse)
            objective_trace.append(best_obj)
            if verbose:
                logger.info(
                    "generation %d: objective %.4f, rmse %.4f",
                    generation, best_obj, best_rmse,
                )
            if best_rmse <= params.rmse_threshold:
                break

    model.ec.kcat[tunable_idx] = best_vec
    if ec_rxn_ids_tunable:
        apply_kcat_constraints(model, update_rxns=ec_rxn_ids_tunable)

    return EvotuneResult(
        rxns=ec_rxn_ids_tunable, old_kcat=kcat0.copy(), new_kcat=best_vec,
        groups=list(groups), rmse_trace=rmse_trace,
        objective_trace=objective_trace, n_generations=generation,
        converged=best_rmse <= params.rmse_threshold,
    )


def tune_prior_penalty_weight(
    model: "EcModel",
    *,
    adapter: Optional["ModelAdapter"] = None,
    params: Optional["EvotuneParams"] = None,
    evotune_data: Optional[EvotuneData] = None,
    okp_method: Optional[str] = None,
    bio_rxn: Optional[str] = None,
    make_anaerobic: Optional[Callable[["EcModel"], None]] = None,
    change_protein_biomass: Optional[Callable[["EcModel", float], None]] = None,
    tunable_mask: Optional[np.ndarray] = None,
    screen: Optional[pd.DataFrame] = None,
    target_impact_share: float = 0.9,
    candidates: Sequence[float] = (0.0, 0.01, 0.03, 0.1),
    seeds: Sequence[int] = (0, 1),
    fold_threshold: float = 2.0,
    popsize: Optional[int] = None,
    n_proc: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep ``prior_penalty_weight`` and report the fit/reproducibility
    trade-off, so the value doesn't have to be picked blind.

    ``prior_penalty_weight`` charges for moving a kcat away from its
    prior, and mainly exists to keep large corrections reproducible
    across seeds rather than landing on an arbitrary point along a flat
    direction. How strong a charge is enough is model- and
    data-specific: too little and the search's largest corrections
    disagree across seeds; too much and the search stops finding real
    corrections at all, giving up fit for nothing. This runs
    :func:`cmaes_kcat_tuning` at every value in ``candidates``, each at
    every seed in ``seeds``, all against the *same* tunable set (built
    once, the way :func:`cmaes_kcat_tuning` would on its own if
    ``tunable_mask``/``screen`` aren't given), and reports both fit and
    cross-seed reproducibility per value.

    This is exactly as expensive as it sounds: ``len(candidates) *
    len(seeds)`` full :func:`cmaes_kcat_tuning` runs, none of them
    reusable across candidates. Reuse a ``screen`` you already have to
    at least avoid recomputing that part.

    ``model`` is restored to its original kcats before every run and
    left that way when this returns -- unlike :func:`cmaes_kcat_tuning`,
    which mutates ``model`` with its result, this function only
    reports, since there is no single "best" candidate for it to leave
    the model in. Fewer than two ``seeds`` makes every reproducibility
    column ``NaN``, since there is nothing to compare.

    Parameters
    ----------
    candidates
        ``prior_penalty_weight`` values to try.
    seeds
        Seeds to run each candidate at. Reproducibility columns pool
        every pair of seeds, so more than two sharpens them but costs
        proportionally more.
    fold_threshold
        A kcat counts as "moved" past this fold change from its prior.
        Matches :func:`~.parsimony.fold_change`'s convention.
    tunable_mask, screen, target_impact_share, popsize, n_proc,
    make_anaerobic, change_protein_biomass, okp_method, bio_rxn
        As in :func:`cmaes_kcat_tuning`.

    Returns
    -------
    pandas.DataFrame
        One row per candidate: ``prior_penalty_weight``, ``n_seeds``,
        ``distance_mean`` and ``distance_spread`` (max minus min across
        seeds), ``fit_cost`` (``distance_mean`` relative to the best
        candidate's, as a fraction), ``n_changed_mean``, ``n_movers``
        (kcats past ``fold_threshold`` in *either* seed of a pair, summed
        over every pair), ``n_both_moved`` (past it in *both* -- the
        denominator for direction agreement, since a kcat only one seed
        moved has no direction to agree about), ``pct_direction_agree``,
        ``median_fold_spread`` and ``max_fold_spread`` (among movers,
        the largest fold-change between what the two seeds landed on).
    """
    params, evotune_data, bio_rxn, okp_method = _resolve_context(
        model, adapter, params, evotune_data, bio_rxn, okp_method)

    if tunable_mask is None:
        if screen is None:
            screen = screen_kcat_leverage(
                model, adapter=adapter, params=params, evotune_data=evotune_data,
                okp_method=okp_method, bio_rxn=bio_rxn,
                make_anaerobic=make_anaerobic,
                change_protein_biomass=change_protein_biomass,
                n_proc=n_proc,
            )
        tunable_mask = select_tunable_mask(
            model, screen, target_impact_share=target_impact_share)

    prior_kcat = model.ec.kcat.copy()

    def _reset():
        model.ec.kcat[:] = prior_kcat
        apply_kcat_constraints(model, update_rxns=model.ec.rxns)

    try:
        rows = []
        for lam in candidates:
            lam_params = params.model_copy(
                update={"prior_penalty_weight": float(lam)})
            vectors, distances, changed = [], [], []
            old_kcat = None
            for seed in seeds:
                _reset()
                result = cmaes_kcat_tuning(
                    model, adapter=adapter, params=lam_params, evotune_data=evotune_data,
                    okp_method=okp_method, bio_rxn=bio_rxn,
                    make_anaerobic=make_anaerobic,
                    change_protein_biomass=change_protein_biomass,
                    tunable_mask=tunable_mask, popsize=popsize, n_proc=n_proc,
                    seed=seed, verbose=verbose,
                )
                vectors.append(result.new_kcat)
                distances.append(
                    result.rmse_trace[-1] if result.rmse_trace else float("nan"))
                changed.append(n_changed(result.new_kcat, result.old_kcat))
                old_kcat = result.old_kcat
                if verbose:
                    logger.info(
                        "prior_penalty_weight=%g seed=%d: distance %.4f, "
                        "%d changed", lam, seed, distances[-1], changed[-1],
                    )

            n_movers = n_both = n_agree = 0
            spreads: list[float] = []
            for i, j in combinations(range(len(vectors)), 2):
                fa = fold_change(vectors[i], old_kcat)
                fb = fold_change(vectors[j], old_kcat)
                movers = (fa > fold_threshold) | (fb > fold_threshold)
                both = (fa > fold_threshold) & (fb > fold_threshold)
                agree = np.sign(np.log(vectors[i][both] / old_kcat[both])) == \
                    np.sign(np.log(vectors[j][both] / old_kcat[both]))
                n_movers += int(movers.sum())
                n_both += int(both.sum())
                n_agree += int(agree.sum())
                if movers.any():
                    ratio = vectors[i][movers] / vectors[j][movers]
                    spreads.extend(np.maximum(ratio, 1.0 / ratio).tolist())

            rows.append({
                "prior_penalty_weight": float(lam),
                "n_seeds": len(seeds),
                "distance_mean": float(np.mean(distances)),
                "distance_spread": float(np.ptp(distances)) if len(distances) > 1 else 0.0,
                "n_changed_mean": float(np.mean(changed)),
                "n_movers": n_movers,
                "n_both_moved": n_both,
                "pct_direction_agree": (n_agree / n_both) if n_both > 0 else float("nan"),
                "median_fold_spread": float(np.median(spreads)) if spreads else float("nan"),
                "max_fold_spread": float(np.max(spreads)) if spreads else float("nan"),
            })
    finally:
        _reset()

    df = pd.DataFrame(rows)
    best = df["distance_mean"].min()
    df["fit_cost"] = df["distance_mean"] / best - 1.0
    return df


# --------------------------------------------------------------------------- #
# Scoring core, shared by the screen and the search above.
# --------------------------------------------------------------------------- #

def kcat_bounds(kcat0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Biologically plausible bounds for proposed kcats, in 1/s.

    Proposals outside these bounds are not hypotheses worth spending an
    FBA solve on. The window is 1e-2 to 1e4 for an ordinary kcat, and
    always widened far enough to contain the prior with a hundred-fold
    margin either side, so an unusually slow or fast enzyme keeps a
    window around itself rather than being clipped towards the generic
    range.

    Parameters
    ----------
    kcat0
        Prior kcat per tunable row, 1/s.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper bound per row.
    """
    kcat0 = np.asarray(kcat0, dtype=float)
    lo = np.minimum(1e-2, kcat0 / 100.0)
    hi = np.maximum(1e4, kcat0 * 100.0)
    return lo, hi


def _reset_solver_basis(model: "EcModel") -> None:
    """Discard the solver's incumbent basis so the solves below start cold.

    The distance must be a function of the kcat vector alone. Solvers
    warm-start from the previous solve's basis, and these LPs have
    alternate optima: resuming from a basis that is already optimal
    for the new problem returns that vertex in zero iterations, so the
    reported exchange fluxes -- and hence the RMSE -- would otherwise
    depend on which candidate this model scored before. That would
    make the result depend on ``n_proc`` and on how the pool happened
    to schedule candidates across workers.

    A no-op for solver interfaces whose problem object exposes no
    ``reset`` (only the model's own state then determines the solve,
    which is the property this guarantees for the rest).
    """
    problem = getattr(model.solver, "problem", None)
    reset = getattr(problem, "reset", None)
    if callable(reset):
        reset()


def _score_kcat_vector(
    model: "EcModel",
    tunable_idx: np.ndarray,
    ec_rxn_ids_tunable: list[str],
    evotune_data: EvotuneData,
    excarbon: dict[str, float],
    bio_rxn_id: str,
    kcat_vec: np.ndarray,
    *,
    make_anaerobic,
    change_protein_biomass,
    max_growth_weight: float = 1.0,
    prior_penalty_weight: float = 0.0,
    kcat0: Optional[np.ndarray] = None,
    sigma0_log: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Score one kcat vector against ``model`` (mutated in place).

    Writes the candidate kcat vector into ``model.ec.kcat`` and
    rewrites just the tunable rows' stoichiometry via
    ``apply_kcat_constraints(update_rxns=...)`` -- no per-candidate
    ``EcModel.copy()`` (prohibitively expensive: ~225x an FBA solve on
    a real-scale model). Called directly (looped in-process) for the
    serial path, or via
    :func:`_score_worker` from inside a pool worker for the parallel
    path -- either way, ``model`` is one persistent object reused
    across every candidate it's asked to score.
    """
    model.ec.kcat[tunable_idx] = kcat_vec
    if ec_rxn_ids_tunable:
        apply_kcat_constraints(model, update_rxns=ec_rxn_ids_tunable)
    _reset_solver_basis(model)

    flux_sims = None
    if evotune_data.flux_data is not None:
        flux_sims = simulate_evotune_dataset(
            model, evotune_data.flux_data,
            constrain=True, zero_flux_rxns=evotune_data.zero_flux,
            bio_rxn_id=bio_rxn_id,
            make_anaerobic=make_anaerobic,
            change_protein_biomass=change_protein_biomass,
        )
    max_grate_sims = None
    if evotune_data.max_grate is not None:
        max_grate_sims = simulate_evotune_dataset(
            model, evotune_data.max_grate,
            constrain=False, zero_flux_rxns=[],
            bio_rxn_id=bio_rxn_id,
            make_anaerobic=make_anaerobic,
            change_protein_biomass=change_protein_biomass,
        )
    rmse, _ = evotune_distance(
        evotune_data, flux_sims=flux_sims, max_grate_sims=max_grate_sims,
        excarbon=excarbon, bio_rxn_id=bio_rxn_id,
        max_growth_weight=max_growth_weight,
    )
    # The search may run on a penalised objective, but the plain RMSE
    # is always carried alongside it so a run stays comparable to
    # MATLAB's and to runs at other penalties.
    objective = rmse
    if prior_penalty_weight and kcat0 is not None and sigma0_log is not None:
        dev = np.log(np.asarray(kcat_vec, dtype=float) / kcat0) / sigma0_log
        objective = rmse + prior_penalty_weight * float(np.mean(dev ** 2))
    return objective, rmse


# --------------------------------------------------------------------------- #
# Parallel scoring (n_proc > 1): worker-process globals, populated once
# by _init_worker and reused for every candidate that worker scores.
# --------------------------------------------------------------------------- #

_WORKER_MODEL: Optional["EcModel"] = None
_WORKER_TUNABLE_IDX: Optional[np.ndarray] = None
_WORKER_EC_RXN_IDS_TUNABLE: Optional[list[str]] = None
_WORKER_EVOTUNE_DATA: Optional[EvotuneData] = None
_WORKER_EXCARBON: Optional[dict[str, float]] = None
_WORKER_BIO_RXN: Optional[str] = None
_WORKER_MAKE_ANAEROBIC = None
_WORKER_CHANGE_PROTEIN_BIOMASS = None
_WORKER_MAX_GROWTH_WEIGHT: float = 1.0
_WORKER_PRIOR_PENALTY_WEIGHT: float = 0.0
_WORKER_KCAT0: Optional[np.ndarray] = None
_WORKER_SIGMA0_LOG: Optional[np.ndarray] = None


def _init_worker(
    model: "EcModel",
    tunable_idx: np.ndarray,
    ec_rxn_ids_tunable: list[str],
    evotune_data: EvotuneData,
    excarbon: dict[str, float],
    bio_rxn_id: str,
    make_anaerobic,
    change_protein_biomass,
    max_growth_weight: float = 1.0,
    prior_penalty_weight: float = 0.0,
    kcat0: Optional[np.ndarray] = None,
    sigma0_log: Optional[np.ndarray] = None,
) -> None:
    """Pool initializer: stash this worker process's own EcModel copy
    (deserialised by ``ProcessPool``, not by us) and everything else
    needed to score a candidate."""
    global _WORKER_MODEL, _WORKER_TUNABLE_IDX, _WORKER_EC_RXN_IDS_TUNABLE
    global _WORKER_EVOTUNE_DATA, _WORKER_EXCARBON, _WORKER_BIO_RXN
    global _WORKER_MAKE_ANAEROBIC, _WORKER_CHANGE_PROTEIN_BIOMASS
    global _WORKER_MAX_GROWTH_WEIGHT, _WORKER_PRIOR_PENALTY_WEIGHT
    global _WORKER_KCAT0, _WORKER_SIGMA0_LOG
    _WORKER_MODEL = model
    _WORKER_TUNABLE_IDX = tunable_idx
    _WORKER_EC_RXN_IDS_TUNABLE = ec_rxn_ids_tunable
    _WORKER_EVOTUNE_DATA = evotune_data
    _WORKER_EXCARBON = excarbon
    _WORKER_BIO_RXN = bio_rxn_id
    _WORKER_MAKE_ANAEROBIC = make_anaerobic
    _WORKER_CHANGE_PROTEIN_BIOMASS = change_protein_biomass
    _WORKER_MAX_GROWTH_WEIGHT = max_growth_weight
    _WORKER_PRIOR_PENALTY_WEIGHT = prior_penalty_weight
    _WORKER_KCAT0 = kcat0
    _WORKER_SIGMA0_LOG = sigma0_log


def _score_worker(kcat_vec: np.ndarray) -> tuple[float, float]:
    assert _WORKER_MODEL is not None, "_score_worker called before _init_worker"
    return _score_kcat_vector(
        _WORKER_MODEL, _WORKER_TUNABLE_IDX, _WORKER_EC_RXN_IDS_TUNABLE,
        _WORKER_EVOTUNE_DATA, _WORKER_EXCARBON, _WORKER_BIO_RXN, kcat_vec,
        make_anaerobic=_WORKER_MAKE_ANAEROBIC,
        change_protein_biomass=_WORKER_CHANGE_PROTEIN_BIOMASS,
        max_growth_weight=_WORKER_MAX_GROWTH_WEIGHT,
        prior_penalty_weight=_WORKER_PRIOR_PENALTY_WEIGHT,
        kcat0=_WORKER_KCAT0, sigma0_log=_WORKER_SIGMA0_LOG,
    )
