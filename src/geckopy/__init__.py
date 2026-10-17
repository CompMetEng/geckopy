"""geckopy: Enzyme-constrained genome-scale metabolic modeling in Python.

Everything you need to build, edit, analyse, save, and load ecModels
is reachable from the top-level namespace -- ``from geckopy import X``
works for every public name. The subpackages (``geckopy.ec_model``,
``geckopy.gather_kcats``, ``geckopy.limit_proteins``, ...) still
exist for code organisation and can be imported directly if you
prefer, but the flat top-level surface is the canonical user API.

The most-used names, grouped by stage of a typical workflow:

- **Project setup**: ``ModelAdapter``
- **Build**: ``load_conventional_gem``, ``make_ec_model``
- **Enrich kcats**: ``fuzzy_kcat_matching``, ``read_dlkcat_output``,
  ``merge_dlkcat_and_fuzzy_kcats``, ``apply_kcat_list``,
  ``apply_custom_kcats``, ``fill_kcats_from_isozymes``,
  ``assign_standard_kcat``, ``apply_kcat_constraints``
- **Set the protein budget**: ``calculate_f_factor``,
  ``set_prot_pool_size``, ``fit_sigma``
- **Curate**: ``set_kcat_for_reactions``, ``sensitivity_tuning``
- **Integrate proteomics**: ``load_prot_data``, ``fill_enz_concs``,
  ``constrain_enz_concs``, ``flexibilize_enz_concs``,
  ``relax_proteomics_greedy``
- **Apply experimental flux data**: ``load_flux_data``,
  ``apply_flux_data_constraints``
- **Analyse**: ``ec_fva``, ``ec_fseof``, ``enzyme_usage``,
  ``report_enzyme_usage``, ``get_enzyme_bottlenecks``,
  ``pfba_enzymes``, ``map_rxns_to_conv``
- **I/O**: ``save_ec_model``, ``load_ec_model``  (YAML only — SBML
  ecModel I/O was removed; the YAML schema and the (de)serialisation
  live in raven-toolbox, see ``raven_toolbox.io.ec_data``)
- **Per-enzyme proxy**: ``Enzyme`` (lazy view via ``model.enzymes``)
"""
# Adapter
from geckopy.adapter import ModelAdapter, ModelParameters

# Databases (loaders + dataclasses)
from geckopy.databases import (
    BrendaData,
    ComplexPortalEntry,
    DLKcatIgnoreLists,
    FluxData,
    PhylDist,
    ProtData,
    UniprotDB,
    calculate_mw,
    find_met_smiles,
    get_complex_data,
    load_brenda_data,
    load_complex_portal_json,
    load_dlkcat_ignore_lists,
    load_flux_data,
    load_pax_db,
    load_phyl_dist,
    load_prot_data,
    load_uniprot_tsv,
)

# Core data + builder + per-enzyme accessor
from geckopy.ec_model import EcData, EcModel, Enzyme, make_ec_model

# Pipeline stages (these are public utilities, not just internal)
from geckopy.ec_model.pipeline import (
    apply_complex_data,
    apply_custom_kcats,
    apply_kcat_constraints,
    fill_kcats_from_isozymes,
    get_kcat_across_isozymes,  # deprecated; alias of fill_kcats_from_isozymes
    get_reactions_from_enzyme,
    set_kcat_for_reactions,
    set_prot_pool_size,
)

# Gather-kcats
from geckopy.gather_kcats import (
    apply_kcat_list,
    assign_kcats,
    assign_standard_kcat,
    collect_kcats,
    fetch_open_kinetics_predictor,
    fuzzy_kcat_matching,
    get_standard_kcat,  # deprecated; alias of assign_standard_kcat
    kcats_from_brenda,
    kcats_from_dlkcat,
    kcats_from_okp,
    load_kcat_source,
    merge_dlkcat_and_fuzzy_kcats,  # deprecated; alias of merge_kcats
    merge_kcats,
    read_dlkcat_output,
    read_kcat_source,
    remove_standard_kcat,
    run_dlkcat,
    save_kcat_source,
    select_kcat_value,  # deprecated; alias of apply_kcat_list
    submit_open_kinetics_predictor,
    update_kcat_sources,
    write_dlkcat_input,
)

# get_enzyme_data
from geckopy.get_enzyme_data import (
    copy_ec_to_gem,
    fill_eccodes_from_database,
    fill_eccodes_from_gem,
    find_ec_in_db,
    get_ec_from_database,  # deprecated; alias of fill_eccodes_from_database
    get_ec_from_gem,  # deprecated; alias of fill_eccodes_from_gem
)

# kcat_tuning
from geckopy.kcat_tuning import (
    SigmaFitterResult,
    TunedKcatsResult,
    find_max_value,
    fit_sigma,
    sensitivity_tuning,
    sigma_fitter,  # deprecated; alias of fit_sigma
    truncate_values,
)

# limit_proteins
from geckopy.limit_proteins import (
    FlexEnzResult,
    GreedyRelaxResult,
    RelaxationStep,
    apply_flux_data_constraints,
    calculate_f_factor,
    constrain_enz_concs,
    constrain_flux_data,  # deprecated; alias of apply_flux_data_constraints
    fill_enz_concs,
    flexibilize_enz_concs,
    get_conc_control_coeffs,
    relax_proteomics_greedy,
)

# utilities
from geckopy.utilities import (
    AddNewRxnsResult,
    EnzymeUsageReport,
    EnzymeUsageResult,
    MapRxnsResult,
    NewEnzyme,
    add_new_rxns_to_ec,
    ec_fseof,
    ec_fva,
    enzyme_usage,
    get_enzyme_bottlenecks,
    get_subset_ec_model,
    load_conventional_gem,
    load_ec_model,
    map_rxns_to_conv,
    pfba_enzymes,
    report_enzyme_usage,
    save_ec_model,
)

__all__ = [
    # Adapter
    "ModelAdapter",
    "ModelParameters",
    # Core data + builder + per-enzyme accessor
    "EcData",
    "EcModel",
    "Enzyme",
    "make_ec_model",
    # Databases (loaders + dataclasses)
    "BrendaData",
    "ComplexPortalEntry",
    "DLKcatIgnoreLists",
    "FluxData",
    "PhylDist",
    "ProtData",
    "UniprotDB",
    "calculate_mw",
    "find_met_smiles",
    "get_complex_data",
    "load_brenda_data",
    "load_complex_portal_json",
    "load_dlkcat_ignore_lists",
    "load_flux_data",
    "load_pax_db",
    "load_phyl_dist",
    "load_prot_data",
    "load_uniprot_tsv",
    # Pipeline / model edits
    "apply_complex_data",
    "apply_custom_kcats",
    "apply_kcat_constraints",
    "fill_kcats_from_isozymes",
    "get_reactions_from_enzyme",
    "set_kcat_for_reactions",
    "set_prot_pool_size",
    # Gather-kcats
    "apply_kcat_list",
    "assign_kcats",
    "assign_standard_kcat",
    "collect_kcats",
    "fetch_open_kinetics_predictor",
    "fuzzy_kcat_matching",
    "kcats_from_brenda",
    "kcats_from_dlkcat",
    "kcats_from_okp",
    "load_kcat_source",
    "merge_dlkcat_and_fuzzy_kcats",  # deprecated; alias of merge_kcats
    "merge_kcats",
    "read_dlkcat_output",
    "read_kcat_source",
    "remove_standard_kcat",
    "run_dlkcat",
    "save_kcat_source",
    "submit_open_kinetics_predictor",
    "update_kcat_sources",
    "write_dlkcat_input",
    # get_enzyme_data
    "copy_ec_to_gem",
    "fill_eccodes_from_database",
    "fill_eccodes_from_gem",
    "find_ec_in_db",
    # kcat sensitivity analysis
    "SigmaFitterResult",
    "TunedKcatsResult",
    "find_max_value",
    "fit_sigma",
    "sensitivity_tuning",
    "truncate_values",
    # limit_proteins
    "FlexEnzResult",
    "GreedyRelaxResult",
    "RelaxationStep",
    "apply_flux_data_constraints",
    "calculate_f_factor",
    "constrain_enz_concs",
    "fill_enz_concs",
    "flexibilize_enz_concs",
    "get_conc_control_coeffs",
    "relax_proteomics_greedy",
    # utilities
    "AddNewRxnsResult",
    "EnzymeUsageReport",
    "EnzymeUsageResult",
    "MapRxnsResult",
    "NewEnzyme",
    "add_new_rxns_to_ec",
    "ec_fseof",
    "ec_fva",
    "enzyme_usage",
    "get_enzyme_bottlenecks",
    "get_subset_ec_model",
    "load_conventional_gem",
    "load_ec_model",
    "map_rxns_to_conv",
    "pfba_enzymes",
    "report_enzyme_usage",
    "save_ec_model",
    # Deprecated aliases (kept reachable so old `from geckopy import X` still works)
    "constrain_flux_data",  # -> apply_flux_data_constraints
    "get_ec_from_database",  # -> fill_eccodes_from_database
    "get_ec_from_gem",  # -> fill_eccodes_from_gem
    "get_kcat_across_isozymes",  # -> fill_kcats_from_isozymes
    "get_standard_kcat",  # -> assign_standard_kcat
    "select_kcat_value",  # -> apply_kcat_list
    "sigma_fitter",  # -> fit_sigma
]
