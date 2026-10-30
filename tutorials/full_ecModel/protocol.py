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
    apply_custom_kcats,
    apply_kcat_constraints,
    apply_kcat_list,
    assign_standard_kcat,
    fill_eccodes_from_database,
    fill_kcats_from_isozymes,
    find_met_smiles,
    fuzzy_kcat_matching,
    load_brenda_data,
    load_conventional_gem,
    load_dlkcat_ignore_lists,
    load_ec_model,
    load_phyl_dist,
    load_uniprot_tsv,
    make_ec_model,
    merge_dlkcat_and_fuzzy_kcats,
    read_dlkcat_output,
    save_ec_model,
    set_kcat_for_reactions,
    set_prot_pool_size,
    write_dlkcat_input,
)

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
# which overwrites `ec.eccodes` with UniProt-derived values for
# every reaction, since yeast-GEM's own EC annotations are not
# thoroughly curated.

# %%
fill_eccodes_from_database(ec_model, uniprot_db)

# %% [markdown]
# **STEP 18-19** Gather kcat values from BRENDA via fuzzy matching.
# The BRENDA dumps live at `data/max_KCAT.txt` etc.; PhylDist.mat is
# the KEGG phylogenetic-distance file.
#
# Note: the PhylDist.mat shipped here is a stub. For
# better-than-tutorial results, build a real PhylDist via KEGG (a
# `get_phyl_dist` function is on the porting roadmap).

# %%
brenda = load_brenda_data(adapter.get_brenda_db_folder())
phyl_dist = load_phyl_dist(params.path / "data" / "PhylDist.mat")
kcat_list_fuzzy = fuzzy_kcat_matching(ec_model, brenda, phyl_dist)
print(f"Fuzzy BRENDA matches: {len(kcat_list_fuzzy)} rows")

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
# **STEP 25** Load DLKcat output.

# %%
kcat_list_dlkcat = read_dlkcat_output(
    ec_model, params.path / "data" / "DLKcat.tsv",
)
print(f"DLKcat predictions: {len(kcat_list_dlkcat)} rows")

# %% [markdown]
# **STEP 26** Combine kcat from BRENDA and DLKcat.

# %%
kcat_list_merged = merge_dlkcat_and_fuzzy_kcats(
    kcat_list_dlkcat, kcat_list_fuzzy,
)

# %% [markdown]
# **STEP 27** Populate `ec_model.ec.kcat` from the merged list.

# %%
apply_kcat_list(ec_model, kcat_list_merged)

# %% [markdown]
# **STEP 28** Apply manually-curated kcat values from
# `data/customKcats.tsv`.

# %%
apply_custom_kcats(ec_model, path=params.path / "data" / "customKcats.tsv")

# %% [markdown]
# **STEP 29** Propagate kcat values across isozymes (sibling
# enzymes that catalyse the same reaction).

# %%
fill_kcats_from_isozymes(ec_model)

# %% [markdown]
# **STEP 30** Get standard kcat. Assigns a protein cost to
# reactions without a gene association, except for exchange /
# transport / pseudoreactions and any reactions listed in
# `data/pseudoRxns.tsv`.

# %%
assign_standard_kcat(ec_model, uniprot_db)

# %% [markdown]
# **STEP 31** Apply kcat constraints to the stoichiometric matrix.
# `apply_kcat_constraints` translates `ec.kcat`, `ec.mw`, and
# `ec.rxn_enz_mat` into the protein pseudo-substrate stoichiometry
# in `model.S`. Re-run any time `ec.kcat`, `ec.rxn_enz_mat`, or
# `ec.mw` changes.

# %%
apply_kcat_constraints(ec_model)

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
# Note: fitting kcats to experimental data with CMA-ES (see
# docs/evotune_kcat_tuning.md) is not part of this protocol walkthrough.

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
# **STEP 73-75 (deferred)** Three-way FVA across bare GEM,
# full ecModel, and ecModel + proteomics integration is the
# original MATLAB step. With the default open-source solver (GLPK)
# this takes well over an hour on yeast-GEM-sized models: ~4000
# canonical reactions x 3 models x 2 LPs each. The code below is
# kept commented out; rerun once a faster solver (Gurobi or CPLEX)
# is configured via `ec_model.solver = "gurobi"`. The `plot_ec_fva`
# helper in `code/plot_ec_fva.py` is already in place to render
# the CDF comparison once FVA results are available.
#
# ```python
# from geckopy.utilities import ec_fva
#
# model_bare = load_conventional_gem(adapter)
# model_bare.adapter = adapter
# ec_model_bare = load_ec_model("ecYeastGEM.yml", adapter=adapter)
#
# # Cap target growth at what the proteomics-constrained model can
# # reach, and apply the same exchange constraints to all three.
# flux_data.gr_rate[0] = 0.088
# for m in (model_bare, ec_model_bare, ecModelProt):
#     apply_flux_data_constraints(
#         m, flux_data,
#         condition=0, max_min_growth="max", loose_strict_flux="loose",
#     )
#
# fva_bare = ec_fva(ec_model_bare, model_bare)
# fva_full = ec_fva(ec_model_bare, model_bare)
# fva_prot = ec_fva(ecModelProt, model_bare)
#
# import pandas as pd
# fva_all = pd.DataFrame({
#     "rxn_id": fva_bare.index,
#     "minFlux_bare": fva_bare["min_flux"].values,
#     "maxFlux_bare": fva_bare["max_flux"].values,
#     "minFlux_ec": fva_full["min_flux"].values,
#     "maxFlux_ec": fva_full["max_flux"].values,
#     "minFlux_ecProt": fva_prot["min_flux"].values,
#     "maxFlux_ecProt": fva_prot["max_flux"].values,
# })
# fva_all.to_csv(output_dir / "ecFVA.tsv", sep="\t", index=False)
#
# import numpy as np
# min_flux_mat = np.column_stack([
#     fva_bare["min_flux"].values,
#     fva_full["min_flux"].values,
#     fva_prot["min_flux"].values,
# ])
# max_flux_mat = np.column_stack([
#     fva_bare["max_flux"].values,
#     fva_full["max_flux"].values,
#     fva_prot["max_flux"].values,
# ])
# plot_ec_fva(
#     min_flux_mat, max_flux_mat,
#     labels=["bare GEM", "ecModel", "ecModel + proteomics"],
#     save_path=output_dir / "ecFVA.pdf",
# )
# ```

# %% [markdown]
# **STEP 76-77 skipped:** light vs full ecModel comparison
# requires the light variant of `make_ec_model`, which is not yet
# ported (the existing code path raises `NotImplementedError` for
# `gecko_light=True`).
#
# **STEP 78-79 skipped:** these refer to the
# `tutorials/light_ecModel` protocol which is out of scope for
# this tutorial.

print("Stage 5 complete. Outputs written to "
      f"{output_dir.relative_to(Path.cwd()) if output_dir.is_relative_to(Path.cwd()) else output_dir}/")
