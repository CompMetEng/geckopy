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
- **Enrich kcats**: ``update_kcat_sources`` (``kcats_from_brenda``,
  ``kcats_from_dlkcat``, ``kcats_from_okp``, ...), ``collect_kcats``,
  ``assign_kcats``
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
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geckopy")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

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
    apply_kcat_constraints,
    fill_kcats_from_isozymes,
    get_reactions_from_enzyme,
    set_kcat_for_reactions,
    set_prot_pool_size,
)

# Gather-kcats
from geckopy.gather_kcats import (
    EcStats,
    KcatSource,
    apply_kcat_list,
    assign_kcats,
    assign_standard_kcat,
    collect_kcats,
    ec_stats_from_brenda,
    format_kcat_source,
    fuzzy_kcat_matching,
    kcats_from_brenda,
    kcats_from_dlkcat,
    kcats_from_ec_tree,
    kcats_from_homologs,
    kcats_from_okp,
    load_kcat_source,
    merge_kcats,
    parse_kcat_source,
    read_kcat_source,
    remove_standard_kcat,
    review_assignment,
    save_kcat_source,
    update_kcat_sources,
)

# get_enzyme_data
from geckopy.get_enzyme_data import (
    copy_ec_to_gem,
    fill_eccodes_from_database,
    fill_eccodes_from_gem,
    find_ec_in_db,
)

# kcat_tuning
from geckopy.kcat_tuning import (
    SigmaFitterResult,
    TunedKcatsResult,
    fit_sigma,
    sensitivity_tuning,
)

# limit_proteins
from geckopy.limit_proteins import (
    ComplexFamilies,
    FlexEnzResult,
    GreedyRelaxResult,
    LevelComplexResult,
    RelaxationStep,
    apply_flux_data_constraints,
    calculate_f_factor,
    complex_families,
    constrain_enz_concs,
    fill_enz_concs,
    flexibilize_enz_concs,
    get_conc_control_coeffs,
    get_isozyme_coverage,
    level_complex_abundances,
    level_complex_concs,
    relax_enz_concs,
    relax_proteomics_greedy,
    zero_unmeasured_isozymes,
)

# utilities
from geckopy.utilities import (
    AddNewRxnsResult,
    EcFSEOFResult,
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

from geckopy.upgrade import convert_custom_kcats, upgrade_ec_model, upgrade_project

__all__ = [
    # Upgrades
    "convert_custom_kcats",
    "upgrade_ec_model",
    "upgrade_project",
    # Adapter
    "ModelAdapter",
    "ModelParameters",
    # Core data + builder + per-enzyme accessor
    "EcData",
    "EcModel",
    "EcStats",
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
    "apply_kcat_constraints",
    "fill_kcats_from_isozymes",
    "get_reactions_from_enzyme",
    "set_kcat_for_reactions",
    "set_prot_pool_size",
    # Gather-kcats
    "KcatSource",
    "apply_kcat_list",
    "assign_kcats",
    "assign_standard_kcat",
    "collect_kcats",
    "ec_stats_from_brenda",
    "format_kcat_source",
    "fuzzy_kcat_matching",
    "kcats_from_brenda",
    "kcats_from_dlkcat",
    "kcats_from_ec_tree",
    "kcats_from_homologs",
    "kcats_from_okp",
    "load_kcat_source",
    "merge_kcats",
    "parse_kcat_source",
    "read_kcat_source",
    "remove_standard_kcat",
    "review_assignment",
    "save_kcat_source",
    "update_kcat_sources",
    # get_enzyme_data
    "copy_ec_to_gem",
    "fill_eccodes_from_database",
    "fill_eccodes_from_gem",
    "find_ec_in_db",
    # kcat sensitivity analysis
    "SigmaFitterResult",
    "TunedKcatsResult",
    "fit_sigma",
    "sensitivity_tuning",
    # limit_proteins
    "ComplexFamilies",
    "FlexEnzResult",
    "GreedyRelaxResult",
    "LevelComplexResult",
    "RelaxationStep",
    "apply_flux_data_constraints",
    "calculate_f_factor",
    "complex_families",
    "constrain_enz_concs",
    "fill_enz_concs",
    "flexibilize_enz_concs",
    "get_conc_control_coeffs",
    "get_isozyme_coverage",
    "level_complex_abundances",
    "level_complex_concs",
    "relax_enz_concs",
    "relax_proteomics_greedy",
    "zero_unmeasured_isozymes",
    # utilities
    "AddNewRxnsResult",
    "EcFSEOFResult",
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
]
