"""Evotune: CMA-ES kcat tuning against experimental data.

See ``docs/evotune_kcat_tuning.md`` for the walkthrough:
:func:`~.tuning.screen_kcat_leverage`, then
:func:`~.tuning.select_tunable_mask`, then
:func:`~.tuning.cmaes_kcat_tuning`.

Modules in this package that need ``cma`` import it unconditionally --
it's a hard requirement here, gated one level up instead (see
``kcat_tuning/__init__.py``) so importing plain
``geckopy.kcat_tuning`` never requires the optional
``evotune`` extra. Install it with
``pip install geckopy[evotune]``.
"""
from .data import EvotuneData, load_evotune_data
from .distance import (
    BIOMASS_CARBON_EQUIV,
    INFEASIBLE_PENALTY,
    evotune_distance,
    compute_excarbon,
    dataset_rmse,
)
from .mask_size import (
    MaskSizePoint,
    recommend_target_impact_share,
    sweep_tunable_mask_size,
)
from .parsimony import best_parsimonious, n_changed, parsimony_frontier
from .priors import (
    UNLABELLED_GROUP,
    build_sigma0_log,
    classify_kcat_source,
    classify_kcat_sources,
)
from .simulate import ConditionSimResult, simulate_evotune_dataset
from .tying import isozyme_tie_map
from .tuning import (
    EvotuneResult,
    cmaes_kcat_tuning,
    screen_kcat_leverage,
    select_tunable_mask,
    tune_prior_penalty_weight,
)

__all__ = [
    "BIOMASS_CARBON_EQUIV",
    "INFEASIBLE_PENALTY",
    "UNLABELLED_GROUP",
    "EvotuneData",
    "EvotuneResult",
    "ConditionSimResult",
    "MaskSizePoint",
    "evotune_distance",
    "best_parsimonious",
    "build_sigma0_log",
    "classify_kcat_source",
    "classify_kcat_sources",
    "cmaes_kcat_tuning",
    "compute_excarbon",
    "dataset_rmse",
    "isozyme_tie_map",
    "load_evotune_data",
    "n_changed",
    "parsimony_frontier",
    "recommend_target_impact_share",
    "screen_kcat_leverage",
    "select_tunable_mask",
    "simulate_evotune_dataset",
    "sweep_tunable_mask_size",
    "tune_prior_penalty_weight",
]
