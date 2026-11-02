# %% [markdown]
# # light_ecModel tutorial
#
# Python port of the MATLAB tutorial that accompanies the GECKO 3
# Nature Protocols paper (https://doi.org/10.1038/s41596-023-00931-7),
# `tutorials/light_ecModel/protocol.m`. Demonstrates how to build and
# analyse a *light* ecModel.
#
# **Why light?** Light ecModels skip the isozyme split (cobra reactions
# stay singular), omit per-enzyme ``prot_<id>`` pseudometabolites and
# ``usage_prot_<id>`` reactions, and constrain enzyme usage solely
# through the shared ``prot_pool``. The LP is much smaller — important
# for GEMs the size of Human-GEM (~13k reactions) or Recon3D, where
# the full layout's per-enzyme bookkeeping becomes impractical.
#
# **Why a tiny demo model?** The MATLAB tutorial uses Human-GEM as the
# starting GEM. We mirror it with ecTestGEM (5 genes, 5 reactions) so
# the tutorial runs in seconds without external data. For a real-scale
# build that mirrors the MATLAB tutorial more faithfully, see
# ``tests/test_light_humangem_smoke.py`` — that smoke test builds an
# ecModel from a full Human-GEM YAML when one is available.
#
# **Do not use the ecModel produced here outside this tutorial.**

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import cobra

from geckopy import (
    ModelAdapter,
    apply_kcat_constraints,
    load_uniprot_tsv,
    make_ec_model,
    set_kcat_for_reactions,
    set_prot_pool_size,
)

# %% [markdown]
# ## STAGE 1: Expansion from a starting GEM to a light ecModel
#
# **STEP 8** Load the adapter. ecTestGEM ships in ``examples/`` so the
# tutorial is self-contained; no local ``model_adapter.toml`` is needed.

# %%
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "ecTestGEM"
adapter = ModelAdapter.from_folder(EXAMPLES)
params = adapter.params
print(f"Organism: {params.org_name}")
print(f"Biomass reaction: {params.bio_rxn}")

# %% [markdown]
# **STEP 9** Load the starting GEM.

# %%
model = cobra.io.read_sbml_model(str(params.conv_gem))
print(f"Conventional GEM: {len(model.reactions)} reactions, "
      f"{len(model.metabolites)} metabolites, {len(model.genes)} genes")

# %% [markdown]
# **STEP 10-11** Build the *light* ecModel. The key difference from
# the full-model tutorial is the ``gecko_light=True`` flag. With it,
# ``make_ec_model``:
#
# - skips the isozyme split (cobra reactions stay singular);
# - emits one row per isozyme in ``ec.rxns`` with a ``###_`` counter
#   prefix (``001_R2``, ``002_R2``, ...);
# - does not add per-enzyme ``prot_<id>`` pseudometabolites or
#   ``usage_prot_<id>`` reactions — only the shared ``prot_pool`` and
#   its exchange reaction.

# %%
uniprot_db = load_uniprot_tsv(params.path / "data" / "uniprot.tsv")
ec_model = make_ec_model(model, adapter, gecko_light=True, uniprot_db=uniprot_db)
print(
    f"Light ecModel: {len(ec_model.reactions)} reactions, "
    f"{ec_model.ec.n_enzymes} enzymes, {ec_model.ec.n_rxns} ec rows"
)

# %% [markdown]
# **Inspecting the light shape.** ec.rxns uses the ``###_<cobra_id>``
# convention. R2 (which has two isozymes) appears twice in both
# directions; R3 / R5 (one isozyme each) appear once.

# %%
print("ec.rxns:")
for rid in ec_model.ec.rxns:
    print(f"  {rid}")

print(f"\nMetabolites containing 'prot': "
      f"{[m.id for m in ec_model.metabolites if 'prot' in m.id]}")
print(f"Reactions containing 'prot': "
      f"{[r.id for r in ec_model.reactions if 'prot' in r.id]}")

# %% [markdown]
# ## STAGE 2: Integrating kcat values
#
# **STEP 27 (simplified)** In a real workflow you'd populate kcat
# values from BRENDA + DLKcat (see the full_ecModel tutorial for the
# pipeline). Here we set a couple of kcats by hand to demonstrate the
# light dispatch.
#
# In a light model the ``ec.rxns`` IDs carry a ``###_`` prefix, which is
# part of the ID for ``set_kcat_for_reactions``, as in GECKO's
# ``setKcatForReactions``: ``"001_R2"`` targets exactly that isozyme row,
# so you can set different kcats per isozyme, and the bare cobra ID
# ``"R2"`` matches nothing. The call changes ``ec.kcat`` and ``ec.source``
# only; ``apply_kcat_constraints`` writes the S matrix afterwards.

# %%
set_kcat_for_reactions(ec_model, ["001_R2", "002_R2"], 5.0)
set_kcat_for_reactions(ec_model, ["001_R3"], 10.0)
set_kcat_for_reactions(ec_model, ["001_R5"], 20.0)
print("ec.kcat after manual edits:")
for rid in ec_model.ec.rxns:
    idx = ec_model.ec.rxns.index(rid)
    print(f"  {rid:15s} kcat = {ec_model.ec.kcat[idx]:6.2f}  "
          f"source = {ec_model.ec.source[idx]!r}")

# %% [markdown]
# **STEP 31** Push the kcats into the S-matrix.
#
# Light ``apply_kcat_constraints`` differs from the full version: for
# each cobra reaction it picks the *cheapest* isozyme (smallest
# ``MW_sum / kcat``) across all its ``###_`` rows, then writes a single
# ``-MW_sum / (kcat * 3600)`` coefficient at ``S[prot_pool, rxn]``. No
# per-enzyme constraints are written.

# %%
apply_kcat_constraints(ec_model)

for rid in ("R2", "R3", "R5"):
    rxn = ec_model.reactions.get_by_id(rid)
    coef = next(
        (c for m, c in rxn.metabolites.items() if m.id == "prot_pool"),
        0.0,
    )
    print(f"  {rid}: S[prot_pool, {rid}] = {coef:.4e}")

# %% [markdown]
# **STEP 32** Constrain the total protein pool and solve.

# %%
set_prot_pool_size(ec_model)
print(f"prot_pool_exchange upper bound: "
      f"{ec_model.reactions.prot_pool_exchange.upper_bound:.4f} mg/gDCW")

# %% [markdown]
# ## What's not available on light models
#
# - **Per-enzyme analyses.** Anything that reads or writes individual
#   ``prot_<id>`` mass balances (per-enzyme shadow prices, proteomics
#   integration via ``constrain_enz_concs``, ``flexibilize_enz_concs``)
#   raises ``NotImplementedError``. The shared ``prot_pool`` is the
#   only enzyme constraint, so per-enzyme distinctions don't exist in
#   the LP.
# - **Isozyme-spanning kcat helpers.** ``fill_kcats_from_isozymes`` /
#   ``get_kcat_across_isozymes`` are full-model only — the light layout
#   already represents isozymes natively as separate ec rows.
#
# **What still works.** ``Enzyme`` proxy reads (``mw``, ``gene``,
# ``sequence``, ``kcats``, ``reactions``), most kcat-source helpers
# (BRENDA fuzzy matching, DLKcat I/O, custom-kcat overrides), and
# anything that operates on the cobra LP without inspecting individual
# enzyme metabolites.
