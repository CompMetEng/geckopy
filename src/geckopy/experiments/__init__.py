"""Experimental data on growth and fluxes, and how a model is scored against it.

Several functions compare a model with measured conditions: the kcat tuning
of :mod:`geckopy.kcat_tuning` and the flux constraints of
:mod:`geckopy.limit_proteins`. They share the files that hold the
measurements (:func:`load_experiment_data`), the setup of one condition in
the model and its simulation (:func:`simulate_conditions`) and the distance
between simulated and measured values (:func:`experiment_distance`).
"""
from .conditions import ConditionSimResult, simulate_conditions
from .data import ExperimentData, load_experiment_data
from .distance import (
    BIOMASS_CARBON_EQUIV,
    INFEASIBLE_PENALTY,
    compute_excarbon,
    dataset_rmse,
    experiment_distance,
)

__all__ = [
    "BIOMASS_CARBON_EQUIV",
    "INFEASIBLE_PENALTY",
    "ConditionSimResult",
    "ExperimentData",
    "compute_excarbon",
    "dataset_rmse",
    "experiment_distance",
    "load_experiment_data",
    "simulate_conditions",
]
