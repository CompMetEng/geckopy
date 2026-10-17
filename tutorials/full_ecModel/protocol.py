# %% [markdown]
# # full_ecModel tutorial
#
# Python port of the MATLAB tutorial that accompanies the GECKO 3
# Nature Protocols paper (https://doi.org/10.1038/s41596-023-00931-7).
# Demonstrates the reconstruction and analysis of a *full* ecModel
# using yeast-GEM as the starting point. STEP numbers match the
# MATLAB tutorial and the Nature Protocols paper.
#
# Covers Stages 0-5: model construction, kcat integration, tuning,
# proteomics integration, and simulation/analysis.
#
# **Do not use the ecModel produced here outside this tutorial.**
# A maintained ecYeastGEM is distributed via
# https://github.com/SysBioChalmers/yeast-GEM (release 9.2.0+).

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import cobra

from geckopy import (
    ModelAdapter,
    apply_complex_data,
    assign_kcats,
    fill_eccodes_from_database,
    find_met_smiles,
    kcats_from_brenda,
    kcats_from_dlkcat,
    load_brenda_data,
    load_conventional_gem,
    load_dlkcat_ignore_lists,
    load_ec_model,
    load_phyl_dist,
    load_uniprot_tsv,
    make_ec_model,
    save_ec_model,
    set_kcat_for_reactions,
    set_prot_pool_size,
    write_dlkcat_input,
)
from geckopy.databases import download_phyl_dist, load_kegg_tsv

# %% [markdown]
# ## STAGE 0: Preparation
#
# **STEP 1-7** Project structure and adapter parameters are already
# in place: see `model_adapter.toml` for the organism-specific
# parameters and the `data/` and `models/` subfolders.

# %% [markdown]
# ## STAGE 1: Expansion from a starting GEM to an ecModel structure
#
# **STEP 8** Set the model adapter. geckopy has no global default
# adapter; the loaded adapter must be passed explicitly to functions
# that need it, or attached to `model.adapter`.

# %%
adapter = ModelAdapter.from_folder(Path(__file__).parent)
params = adapter.params
print(f"Organism: {params.org_name}")
print(f"Biomass reaction: {params.bio_rxn}")

# %% [markdown]
# **STEP 9** Load conventional yeast-GEM. `load_conventional_gem`
# reads the file at `adapter.params.conv_gem`.

# %%
model = load_conventional_gem(adapter)
print(f"Conventional GEM: {len(model.reactions)} reactions, "
      f"{len(model.metabolites)} metabolites, {len(model.genes)} genes")

# %% [markdown]
# **STEP 10-11** Build the ecModel. We make a *full* GECKO ecModel
# (`gecko_light=False`); see `tutorials/light_ecModel` for the light
# variant. UniProt data is loaded from `data/uniprot.tsv`.

# %%
uniprot_db = load_uniprot_tsv(params.path / "data" / "uniprot.tsv")
ec_model = make_ec_model(model, adapter, uniprot_db=uniprot_db)
print(f"ecModel: {len(ec_model.reactions)} reactions, "
      f"{len(ec_model.ec.enzymes)} enzymes")

# %% [markdown]
# **STEP 12-13** Annotate with complex data. The Complex Portal
# JSON is shipped at `data/ComplexPortal.json`; the downloader
# `geckopy.databases.get_complex_data` would refresh it.

# %%
apply_complex_data(ec_model, path=params.path / "data" / "ComplexPortal.json")

# %% [markdown]
# **STEP 14** Save Stage 1 ecModel.

# %%
save_ec_model(ec_model, "ecYeastGEM_stage1.yml", adapter=adapter)

# %% [markdown]
# ## STAGE 2: Integration of kcat into the ecModel structure
#
# To resume from disk, uncomment:
# ```python
# ec_model = load_ec_model("ecYeastGEM_stage1.yml", adapter=adapter)
# ```

# %% [markdown]
# **STEP 16-17** Gather EC numbers via `fill_eccodes_from_database`,
# which overwrites `ec.eccodes` with database-derived values for
# every reaction, since yeast-GEM's own EC annotations are not
# thoroughly curated. UniProt is queried first; KEGG (`data/kegg.tsv`)
# fills reactions where UniProt has no EC number or only an
# incomplete one (ending in `-`).

# %%
kegg_db = load_kegg_tsv(params.path / "data" / "kegg.tsv")
fill_eccodes_from_database(ec_model, uniprot_db, kegg_db=kegg_db)

# %% [markdown]
# **STEP 18-19** Gather kcat values from BRENDA via fuzzy matching.
# Each kcat source is kept as a file in `data/kcats` (`brenda.tsv`,
# `dlkcat.tsv`, `okp_<method>.tsv`, `custom.tsv`), which `assign_kcats`
# combines in STEP 26-31; `[kcat_assignment]` in model_adapter.toml sets
# how. `update_kcat_sources(ec_model)` gathers all sources that the
# configuration needs in one call. The BRENDA data ships with geckopy. When BRENDA has no value for
# S. cerevisiae itself, the value of the phylogenetically closest
# organism is used, based on the KEGG phylogenetic distances in
# `data/PhylDist.mat`. That file (~5 MB) is downloaded from RAVEN on
# the first run.

# %%
brenda = load_brenda_data(adapter.get_brenda_db_folder())
phyl_dist_path = adapter.get_phyl_dist_path()
if not phyl_dist_path.is_file():
    download_phyl_dist(phyl_dist_path)
phyl_dist = load_phyl_dist(phyl_dist_path)
kcat_list_fuzzy = kcats_from_brenda(ec_model, brenda=brenda, phyl_dist=phyl_dist)
print(f"Fuzzy BRENDA matches: {len(kcat_list_fuzzy)} rows, kept as data/kcats/brenda.tsv")

# %% [markdown]
# **STEP 20-22** Gather metabolite SMILES. `find_met_smiles` queries
# PubChem; with a populated cache (`data/smilesDB.tsv`) no network
# is needed.

# %%
find_met_smiles(ec_model, cache_path=params.path / "data" / "smilesDB.tsv")

# %% [markdown]
# **STEP 23** Prepare DLKcat input file. The `data/DLKcat.tsv`
# shipped here already has predicted kcat values; uncomment to
# regenerate (this discards existing predictions).
#
# ```python
# ignore_lists = load_dlkcat_ignore_lists(params.path / "data")
# write_dlkcat_input(
#     ec_model, params.path / "data" / "DLKcat.tsv",
#     ignore_lists, overwrite=True,
# )
# ```

# %% [markdown]
# **STEP 24** Run DLKcat. (External tool; not invoked here. The
# shipped `data/DLKcat.tsv` already has output from a previous run.)

# %% [markdown]
# **STEP 25** Load DLKcat output from `data/DLKcat.tsv`, kept as
# `data/kcats/dlkcat.tsv`.

# %%
kcat_list_dlkcat = kcats_from_dlkcat(ec_model)
print(f"DLKcat predictions: {len(kcat_list_dlkcat)} rows")

# %% [markdown]
# **STEP 26-31** Assign kcat values. `assign_kcats` combines the kcat
# source files and applies the result to `ec_model`, in the stages that
# model_adapter.toml sets:
#
# - **STEP 26** For each reaction, the first source in the priority list
#   that has a kcat is used: custom kcats, then BRENDA matches without EC
#   wildcard and with origin up to 4 (`database_top`), then DLKcat, then
#   any remaining BRENDA match (`database_bottom`). See
#   `help(merge_kcats)` for the tier definitions. (The GECKO tutorial
#   uses the `gecko_classic` preset, which takes origin up to 6.)
# - **STEP 27** The kcats are written to `ec_model.ec.kcat`.
# - **STEP 28** Manually curated kcat values from `data/customKcats.tsv`,
#   or from `data/kcats/custom.tsv` when that exists.
# - **STEP 29** Reactions without a kcat take one from their isozymes.
# - **STEP 30** Get standard kcat. Assigns a protein cost to reactions
#   without a gene association, except for exchange / transport /
#   pseudoreactions and any reactions listed in `data/pseudoRxns.tsv`.
# - **STEP 31** Apply kcat constraints to the stoichiometric matrix:
#   `ec.kcat`, `ec.mw` and `ec.rxn_enz_mat` become the protein
#   pseudo-substrate stoichiometry in `model.S`. Re-run
#   `apply_kcat_constraints` any time `ec.kcat`, `ec.rxn_enz_mat`, or
#   `ec.mw` changes.
#
# The decision per reaction is in `output/<model id>_kcatDecisions.tsv`:
# which stage and source gave its kcat, and how many values there were.
# `collect_kcats` gives the same decisions without changing the model.

# %%
kcat_decisions, kcat_report = assign_kcats(ec_model, uniprot_db=uniprot_db)
print(kcat_decisions["stage"].value_counts().to_string())

# %% [markdown]
# **STEP 32** Set the upper bound of the protein pool exchange.
# `Ptot * f * sigma` from the adapter; can be overridden per call.

# %%
set_prot_pool_size(
    ec_model,
    p_tot=params.p_tot, f=params.f, sigma=params.sigma,
)

# %%
save_ec_model(ec_model, "ecYeastGEM_stage2.yml", adapter=adapter)

# %% [markdown]
# ## STAGE 3: Model tuning
#
# **STEP 33-38** Test the maximum growth rate with an unconstrained
# glucose uptake, setting the bound directly on the reaction and
# solving with `model.optimize()`.

# %%
ec_model.reactions.get_by_id(params.c_source).lower_bound = -1000
ec_model.objective = params.bio_rxn
sol = ec_model.optimize()
growth_rate = sol.fluxes[params.bio_rxn]
print(f"Growth rate: {growth_rate:.4f} /hour")
print(f"(Below the 0.41 /hour reference for S. cerevisiae)")

# %% [markdown]
# **STEP 43-44** Sensitivity tuning. Iteratively bumps the most
# limiting kcat by `fold_change` until the model can reach
# `desired_growth_rate` (defaults to `adapter.params.gr_exp`).
# Returns a result with the tuned kcats.

# %%
from geckopy import sensitivity_tuning

tuning_result = sensitivity_tuning(ec_model)
final_growth = ec_model.optimize().fluxes[params.bio_rxn]
print(f"Tuned {len(tuning_result.rxns)} kcats; "
      f"final growth rate: {final_growth:.4f}")

# %% [markdown]
# Besides step-wise sensitivity tuning, kcats can be fitted to
# experimental growth rates and exchange fluxes with CMA-ES
# (`geckopy.kcat_tuning.evotune`, installed with the `evotune` extra;
# see docs/evotune_kcat_tuning.md):
#
# ```python
# from geckopy.kcat_tuning.evotune import (
#     cmaes_kcat_tuning, screen_kcat_leverage, select_tunable_mask,
# )
# screen = screen_kcat_leverage(ec_model)
# mask = select_tunable_mask(ec_model, screen)
# result = cmaes_kcat_tuning(ec_model, tunable_mask=mask)
# ```
#
# Its hyperparameters are set in the `[evotune]` section of
# `model_adapter.toml`, but each model may need its own tuning of
# them. As in the MATLAB tutorial, it is not run here; the maintained
# ecYeastGEM in the yeast-GEM repository is tuned this way.

# %% [markdown]
# **STEP 45-51** Curate kcat values based on the tuning result.
# As an example the MATLAB tutorial increases the kcat of r_0079
# (5'-phosphoribosylformyl glycinamidine synthetase) from 0.05 to a
# computed 5.34 /sec, derived from a paper-reported specific
# activity. We replicate that here.

# %%
import numpy as np

enz_idx = ec_model.ec.enzymes.index("P38972")
enz_mw = ec_model.ec.mw[enz_idx]
sa = 2.15  # umol/min/mg protein
kcat_per_sec = sa / 1000 / 60 * enz_mw
print(f"Computed kcat for r_0079: {kcat_per_sec:.4f} /sec")
set_kcat_for_reactions(ec_model, ["r_0079"], kcat_per_sec)

# %%
save_ec_model(ec_model, "ecYeastGEM_stage3.yml", adapter=adapter)

# %% [markdown]
# **STEP 52** Save the curated ecModel without proteomics
# integration (the canonical "ecYeastGEM" file).

# %%
save_ec_model(ec_model, "ecYeastGEM.yml", adapter=adapter)

# %% [markdown]
# ## STAGE 4: Integration of proteomics data into the ecModel
#
# To resume from disk, uncomment:
# ```python
# ec_model = load_ec_model("ecYeastGEM_stage3.yml", adapter=adapter)
# ```

# %% [markdown]
# **STEP 53-57** Load proteomics data and constrain the ecModel.
# `data/proteomics.tsv` has 3 replicates of a single condition,
# expressed as `[3]` in geckopy (a list with one element per
# condition).

# %%
from geckopy import (
    apply_flux_data_constraints,
    calculate_f_factor,
    constrain_enz_concs,
    fill_enz_concs,
    flexibilize_enz_concs,
    load_flux_data,
    load_prot_data,
)

prot_data = load_prot_data(
    params.path / "data" / "proteomics.tsv", repl_per_cond=[3],
)
fill_enz_concs(ec_model, prot_data)
constrain_enz_concs(ec_model)
n_constrained = int((~np.isnan(ec_model.ec.concs)).sum())
print(f"Constrained {n_constrained} of {len(ec_model.ec.enzymes)} enzymes "
      f"with measured concentrations")

# %% [markdown]
# **STEP 58** Update protein pool. The f-factor can be
# recomputed from the proteomics data, then `set_prot_pool_size`
# is re-applied using the condition-specific total protein content
# from `fluxData.tsv`.

# %%
f_pdata = calculate_f_factor(ec_model, prot_data)
flux_data = load_flux_data(params.path / "data" / "fluxData.tsv")
set_prot_pool_size(
    ec_model, p_tot=float(flux_data.p_tot[0]), f=f_pdata,
)
print(f"Recomputed f-factor: {f_pdata:.3f} "
      f"(adapter default: {params.f:.3f}); "
      f"Ptot from flux data: {float(flux_data.p_tot[0]):.3f}")

# %% [markdown]
# **STEP 59-63** Constrain exchange fluxes from the flux data.

# %%
apply_flux_data_constraints(
    ec_model, flux_data,
    condition=0, max_min_growth="max", loose_strict_flux="loose",
)
sol = ec_model.optimize()
print(f"Growth rate after flux constraints: {sol.objective_value:.4f} /hour "
      f"(target: {float(flux_data.gr_rate[0]):.4f})")

# %% [markdown]
# **STEP 64-65** Flexibilize enzyme concentrations until the
# experimental growth rate is reached.

# %%
flex_result = flexibilize_enz_concs(
    ec_model, exp_growth=float(flux_data.gr_rate[0]), fold_change=10.0,
)
sol = ec_model.optimize()
print(f"Growth rate after flexibilizing: {sol.objective_value:.4f} /hour")
print(f"Flexibilized {len(flex_result.uniprot_ids)} enzymes "
      f"(showing top 5 by fold-increase):")
for i in np.argsort(flex_result.ratio_incr)[::-1][:5]:
    print(f"  {flex_result.uniprot_ids[i]}: "
          f"{flex_result.old_concs[i]:.3g} -> "
          f"{flex_result.flex_concs[i]:.3g} "
          f"(x{flex_result.ratio_incr[i]:.1f})")

# %% [markdown]
# As a sanity check, confirm the starting (non-ec) GEM also can't
# reach the experimental growth rate when given the same exchange
# constraints. If it can't, the measurement set itself is
# overconstraining; the ecModel will only be able to match what
# the metabolic network allows.

# %%
conv_model = load_conventional_gem(adapter)
# apply_flux_data_constraints reads bio_rxn / c_source from `model.adapter`;
# attach the same adapter so it works on the plain cobra.Model too.
conv_model.adapter = adapter
apply_flux_data_constraints(
    conv_model, flux_data,
    condition=0, max_min_growth="max", loose_strict_flux="loose",
)
sol_conv = conv_model.optimize()
print(f"Starting GEM growth rate: {sol_conv.objective_value:.4f} /hour")

# %%
save_ec_model(ec_model, "ecYeastGEM_stage4.yml", adapter=adapter)

# %% [markdown]
# ## STAGE 5: Simulation and analysis
#
# **STEP 66** Reload the ecModel WITHOUT proteomics integration
# (the canonical `ecYeastGEM.yml` from STEP 52). The proteomics-
# constrained model from Stage 4 is kept separately as
# `ecModelProt` for the FVA comparison below.

# %%
ecModelProt = ec_model
ec_model = load_ec_model("ecYeastGEM.yml", adapter=adapter)

# %% [markdown]
# **STEP 67-68** Simulate Crabtree effect. The plot helpers live
# in `tutorials/full_ecModel/code/`; we add that directory to
# `sys.path` once so the imports below work.

# %%
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend; plots are saved to PDF

sys.path.insert(0, str(params.path / "code"))
from plot_crabtree import plot_crabtree   # noqa: E402
from plot_ec_fva import plot_ec_fva       # noqa: E402

output_dir = params.path / "output"
output_dir.mkdir(exist_ok=True)

# %%
plot_crabtree(
    ec_model,
    data_path=params.path / "data" / "vanHoek1998.tsv",
    save_path=output_dir / "crabtree.pdf",
)

# %% [markdown]
# For comparison, run the same simulation with an unconstrained
# protein pool (mimicking a conventional GEM). Crabtree effect
# should disappear.

# %%
import math

ec_model_inf = load_ec_model("ecYeastGEM.yml", adapter=adapter)
set_prot_pool_size(ec_model_inf, p_tot=math.inf)
plot_crabtree(
    ec_model_inf,
    data_path=params.path / "data" / "vanHoek1998.tsv",
    save_path=output_dir / "crabtree_infProt.pdf",
)

# %% [markdown]
# And on the pre-tuning ecModel from Stage 2 (before sensitivity
# tuning relaxed the kcat values). The model should get
# constrained at too low growth rates, with no feasible solutions
# at high growth.

# %%
ec_model_preTuning = load_ec_model("ecYeastGEM_stage2.yml", adapter=adapter)
ec_model_preTuning.reactions.get_by_id(params.c_source).lower_bound = -1000
plot_crabtree(
    ec_model_preTuning,
    data_path=params.path / "data" / "vanHoek1998.tsv",
    save_path=output_dir / "crabtree_preTuning.pdf",
)

# %% [markdown]
# **STEP 69-70** Selecting objective functions. Solve for max
# growth, then constrain growth to 99% of that and minimise
# protein-pool usage.

# %%
ec_model.objective = params.bio_rxn
sol = ec_model.optimize()
max_growth = sol.fluxes[params.bio_rxn]
print(f"Max growth rate: {max_growth:.4f} /hour")

ec_model.reactions.get_by_id(params.bio_rxn).lower_bound = 0.99 * max_growth
ec_model.objective = {
    ec_model.reactions.get_by_id("prot_pool_exchange"): -1.0,
}
sol = ec_model.optimize()
print(f"Minimum protein pool usage at 99% of max growth: "
      f"{abs(sol.fluxes['prot_pool_exchange']):.2f} mg/gDCW")

# %% [markdown]
# **STEP 71** Inspect enzyme usage at a non-trivial growth rate.

# %%
from geckopy import enzyme_usage, report_enzyme_usage

ec_model.objective = "r_1714"  # max glucose uptake
ec_model.reactions.get_by_id(params.bio_rxn).lower_bound = 0.25
sol = ec_model.optimize()
usage = enzyme_usage(ec_model, sol.fluxes)
report = report_enzyme_usage(ec_model, usage)
print("Top 10 enzymes by absolute usage:")
print(report.top_abs_usage.head(10))

# %% [markdown]
# **STEP 72** Compare ecModel fluxes to a conventional GEM by
# mapping ec-rxn fluxes back via `map_rxns_to_conv`.

# %%
from geckopy import map_rxns_to_conv

sol = ec_model.optimize()
mapped = map_rxns_to_conv(ec_model, conv_model, sol.fluxes)
print(f"Mapped fluxes shape: {mapped.mapped_flux.shape}; "
      f"conv model reactions: {len(conv_model.reactions)}")

# %% [markdown]
# **STEP 73-75** Perform (ec)FVA on the starting GEM, the ecModel,
# and the ecModel with proteomics integration, all under the same
# exchange flux constraints. `ec_fva` maps each ecModel's variability
# back onto the starting GEM's reactions, so the three results line
# up row by row. This is two LPs per reaction per model; with Gurobi
# or CPLEX it takes a few minutes, with GLPK well over an hour, so it
# is skipped on GLPK (set `cobra.Configuration().solver = "gurobi"`,
# or another commercial solver, to run it).

# %%
import pandas as pd

from geckopy.utilities import ec_fva

fva_solver = cobra.Configuration().solver.__name__.rsplit(".", 1)[-1]
if fva_solver.startswith("glpk"):
    print("STEP 73-75 skipped: ec_fva is too slow with GLPK.")
else:
    model_fva = load_conventional_gem(adapter)
    model_fva.adapter = adapter
    ec_model_fva = load_ec_model("ecYeastGEM.yml", adapter=adapter)

    # The proteomics-constrained model reaches at most 0.088 /hour, so
    # use that as target growth rate for all three models.
    flux_data.gr_rate[0] = 0.088
    for m in (model_fva, ec_model_fva, ecModelProt):
        apply_flux_data_constraints(
            m, flux_data,
            condition=0, max_min_growth="max", loose_strict_flux="loose",
        )

    fva = {
        "GEM": ec_fva(model_fva, model_fva),
        "ecModel": ec_fva(ec_model_fva, model_fva),
        "ecModel + proteomics": ec_fva(ecModelProt, model_fva),
    }

    fva_all = pd.DataFrame({
        "rxn_id": [r.id for r in model_fva.reactions],
        "rxn_name": [r.name for r in model_fva.reactions],
    })
    for label, col in zip(fva, ("", "ec-", "ecP-")):
        fva_all[f"{col}minFlux"] = fva[label]["min_flux"].values
        fva_all[f"{col}maxFlux"] = fva[label]["max_flux"].values
    fva_all.to_csv(output_dir / "ecFVA.tsv", sep="\t", index=False)

    plot_ec_fva(
        np.column_stack([r["min_flux"].values for r in fva.values()]),
        np.column_stack([r["max_flux"].values for r in fva.values()]),
        labels=list(fva),
        save_path=output_dir / "ecFVA.pdf",
    )

# %% [markdown]
# **STEP 76-77** Compare light and full ecModels. For a fair
# comparison, `plot_light_vs_full` (in `code/`) builds a light and a
# full ecModel from yeast-GEM with only BRENDA kcats and no tuning,
# and compares their flux distributions at maximum growth rate.

# %%
from plot_light_vs_full import plot_light_vs_full  # noqa: E402

flux_light, flux_full = plot_light_vs_full(
    load_conventional_gem(adapter), adapter,
    uniprot_db=uniprot_db, brenda=brenda, phyl_dist=phyl_dist,
    save_path=output_dir / "lightVSfull.pdf",
)

# %% [markdown]
# The ratio between the two flux distributions shows which reactions
# carry a flux that deviates more than 0.1% between the light and
# full ecModel.

# %%
with np.errstate(divide="ignore", invalid="ignore"):
    flux_ratio = flux_full / flux_light
changed_flux = np.abs(flux_ratio - 1) > 0.001
print(f"{int(changed_flux.sum())} reactions carry a different flux:")
for rxn, changed in zip(conv_model.reactions, changed_flux):
    if changed:
        print(f"  {rxn.id}: {rxn.name}")

# %% [markdown]
# **STEP 78-79 skipped:** these refer to the
# `tutorials/light_ecModel` protocol which is out of scope for
# this tutorial.

print("Stage 5 complete. Outputs written to "
      f"{output_dir.relative_to(Path.cwd()) if output_dir.is_relative_to(Path.cwd()) else output_dir}/")
