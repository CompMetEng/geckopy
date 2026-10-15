"""Apply enzyme-concentration constraints to ec models.

Mirrors GECKO MATLAB's `src/geckomat/limit_proteins/` directory.
"""
from .calculate_f_factor import calculate_f_factor
from .constrain_enz_concs import constrain_enz_concs
from .constrain_flux_data import (
    apply_flux_data_constraints,
    constrain_flux_data,
)
from .fill_enz_concs import fill_enz_concs
from .flexibilize_enz_concs import FlexEnzResult, flexibilize_enz_concs
from .get_conc_control_coeffs import get_conc_control_coeffs
from .level_complex_concs import (
    ComplexFamilies,
    LevelComplexResult,
    complex_families,
    level_complex_abundances,
    level_complex_concs,
)
from .relax_enz_concs import relax_enz_concs
from .relax_proteomics_greedy import (
    GreedyRelaxResult,
    RelaxationStep,
    relax_proteomics_greedy,
)

__all__ = [
    "ComplexFamilies",
    "FlexEnzResult",
    "LevelComplexResult",
    "GreedyRelaxResult",
    "RelaxationStep",
    "apply_flux_data_constraints",
    "calculate_f_factor",
    "complex_families",
    "constrain_enz_concs",
    "constrain_flux_data",  # deprecated; alias of apply_flux_data_constraints
    "fill_enz_concs",
    "flexibilize_enz_concs",
    "get_conc_control_coeffs",
    "level_complex_abundances",
    "level_complex_concs",
    "relax_enz_concs",
    "relax_proteomics_greedy",
]
