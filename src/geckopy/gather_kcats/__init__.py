"""Gather kcat values for ec model reactions from various sources.

Mirrors GECKO MATLAB's `src/geckomat/gather_kcats/` directory.
"""
from .fuzzy_kcat_matching import fuzzy_kcat_matching
from .kcat_assignment import assign_kcats, collect_kcats, update_kcat_sources
from .homologs import kcats_from_homologs
from .kcat_sources import ec_rxn_keys, load_kcat_source, read_kcat_source, save_kcat_source
from .kcats_from import kcats_from_brenda, kcats_from_dlkcat, kcats_from_ec_tree, kcats_from_okp
from .get_standard_kcat import assign_standard_kcat, get_standard_kcat
from .merge_dlkcat_and_fuzzy_kcats import merge_dlkcat_and_fuzzy_kcats
from .merge_kcats import merge_kcats, normalize_source
from .open_kinetics_predictor import (
    OKPClient,
    OKPError,
    fetch_open_kinetics_predictor,
    submit_open_kinetics_predictor,
)
from .read_dlkcat_output import read_dlkcat_output
from .remove_standard_kcat import remove_standard_kcat
from .review import EcStats, ec_stats_from_brenda, review_assignment
from .run_dlkcat import run_dlkcat
from .select_kcat_value import (
    apply_kcat_list,
    format_kcat_source,
    select_kcat_value,
)
from .write_dlkcat_input import extract_enzyme_substrate_pairs, write_dlkcat_input

__all__ = [
    "EcStats",
    "OKPClient",
    "OKPError",
    "apply_kcat_list",
    "assign_kcats",
    "assign_standard_kcat",
    "collect_kcats",
    "ec_rxn_keys",
    "ec_stats_from_brenda",
    "extract_enzyme_substrate_pairs",
    "fetch_open_kinetics_predictor",
    "format_kcat_source",
    "fuzzy_kcat_matching",
    "get_standard_kcat",  # deprecated; alias of assign_standard_kcat
    "kcats_from_brenda",
    "kcats_from_dlkcat",
    "kcats_from_ec_tree",
    "kcats_from_homologs",
    "kcats_from_okp",
    "load_kcat_source",
    "merge_dlkcat_and_fuzzy_kcats",  # deprecated; use merge_kcats
    "merge_kcats",
    "normalize_source",
    "read_kcat_source",
    "read_dlkcat_output",
    "remove_standard_kcat",
    "review_assignment",
    "run_dlkcat",
    "save_kcat_source",
    "select_kcat_value",  # deprecated; alias of apply_kcat_list
    "submit_open_kinetics_predictor",
    "update_kcat_sources",
    "write_dlkcat_input",
]
