# What belongs in `ravenpy` vs `geckopy`

## Premise

Proposed three-layer stack:

```
cobrapy   — core GEM object model + FBA/FVA/IO          (no GECKO, no RAVEN)
  ▲
ravenpy   — generic GEM reconstruction/annotation/analysis utilities
            useful for ANY model, including ecGEMs        (depends on cobrapy)
  ▲
geckopy   — everything specific to enzyme-constrained GEMs (depends on ravenpy)
```

The test applied to every function below: **"Would a researcher working on a plain
GEM (no enzyme constraints) ever want this?"** If yes → `ravenpy`. If it only makes
sense once a model carries kcats / a protein pool / enzyme usage → `geckopy`.

A second, sharper test for the borderline cases: **does the function actually read or
write the `ec` substructure, or does it just operate on a `cobra.Model`?** Many
functions currently take an `EcModel` purely out of convenience but never touch
`.ec` — those are generic and the `EcModel` type hint is the only thing tying them
to geckopy.

Note: this is a *destination* analysis, not a "rip it out now" plan. Several items
are generic in intent but currently coupled to `model.ec` or to the ecModel adapter;
for those the recommendation is "the kernel belongs in ravenpy, with a thin ec-aware
wrapper left in geckopy." Those are called out explicitly.

---

## Tier A — move as-is (clearly generic, little/no coupling)

These operate on a `cobra.Model` (or raw data) and have obvious value for non-ec GEMs.
Several already carry docstrings naming their RAVEN MATLAB origin.

### Structural model surgery — `ec_model/pipeline/preprocess.py`, `expand.py`
| function | what it does | why generic |
|---|---|---|
| `convert_to_irreversible` | split reversible non-exchange reactions into fwd/`_REV` | docstring already says *"Wraps RAVEN: convertToIrrev.m. RAVENpy candidate."* Irreversible models are a generic preprocessing form (sampling, MILP, FSEOF). |
| `invert_backwards_only_reactions` | rewrite `lb<0, ub==0` reactions in canonical forward form | pure bound/stoichiometry canonicalization; nothing enzyme-related |
| `remove_pseudoreaction_gprs` | clear GPRs on biomass/pseudo reactions | GPR hygiene any reconstruction wants; takes only a model + a TSV path |
| `expand_model` + `_gpr_to_dnf` / `_node_to_dnf` | split isozyme (OR-GPR) reactions, one per clause | docstring: *"Equivalent to RAVEN: expandModel.m"*. The GPR→DNF helper is a generally useful GPR utility (cobrapy has no DNF expander). |

These four touch only `cobra` objects — no `.ec` access at all. They are the single
strongest, lowest-risk batch to lift into `ravenpy`.

### Sequence / annotation databases — `databases/`
| module | what it does | why generic |
|---|---|---|
| `uniprot_download.py` + `uniprot_loader.py` (`UniprotDB`) | fetch & parse UniProt entries (sequence, MW, gene names, EC) by taxon/proteome | genome annotation; RAVEN's `getUniprot`. Useful for any reconstruction. |
| `kegg_download.py` + `kegg_loader.py` | fetch & parse KEGG gene entries (sequence, EC, pathway, MW) | RAVEN's `getKEGGModelForOrganism`/KEGG helpers. Generic. |
| `phyl_dist.py` (`PhylDist`) | KEGG phylogenetic distance matrix + genus lookup | RAVEN's `getPhylDist`. A generic comparative-genomics resource. |
| `mw.py` (`calculate_mw`) | molecular weight from an amino-acid sequence | a generic protein property; nothing ec about it |
| `pubchem.py` (`find_met_smiles`) | look up metabolite SMILES by name | generic metabolite annotation (cheminformatics, met matching, thermo) |
| `flux_data.py` (`FluxData`, `load_flux_data`) | parse experimental exchange/flux measurements with `(rxn)` headers | generic experimental-data ingestion |
| `pax_db_loader.py`, `prot_data_loader.py` (`ProtData`) | load & filter proteomics abundances | generic omics integration (E-Flux, GIMME, etc. all consume proteomics) |

Their result dataclasses (`UniprotDB`, `PhylDist`, `ProtData`, `FluxData`) move with them.

### Generic constraint application — `limit_proteins/constrain_flux_data.py`
`apply_flux_data_constraints` (aliased `constrain_flux_data`) applies measured exchange
and growth fluxes as bounds. **Despite living in `limit_proteins`, it has nothing to do
with proteins** — it constrains any GEM to experimental flux data. Clear `ravenpy`
function. (It currently reaches into the adapter for `c_source`/`bio_rxn`; see the
adapter note below.)

### Model loading — `utilities/load_conventional_gem.py`
A thin reader that dispatches YAML vs SBML. Generic ("import a GEM"); RAVEN's
`importModel`. The only coupling is that it takes the adapter to find the path —
trivial to also expose a plain `load_gem(path)`. (Also a good place to add the missing
`.json`/`.mat` dispatch noted in the code review.)

---

## Tier B — generic kernel, currently ec-coupled (move the kernel, keep a thin wrapper)

These accomplish something generic but are wired to `model.ec` today. The
recommendation is to factor the generic operation into `ravenpy` operating on a
`cobra.Model`, and leave a small geckopy wrapper that also maintains the `ec` arrays.

### EC-code assignment — `get_enzyme_data/`
EC numbers are standard metabolic annotation, valuable for **any** GEM (pathway
mapping, model comparison, gap-filling), not just ecGEMs. The logic here is generic;
only the *storage target* (`model.ec.eccodes`) is ec-specific.

| module | generic core | ec coupling to strip |
|---|---|---|
| `ec_from_gem.py` (`fill_eccodes_from_gem`) | read & validate `ec-code` annotations off reactions | writes to `model.ec.eccodes` |
| `ec_from_database.py` (`fill_eccodes_from_database`) | look up EC numbers from UniProt/KEGG by gene | writes to `model.ec`; uses adapter |
| `find_ec_in_db.py` | EC-code matching incl. wildcard subsumption (`1.2.-.-`) | the wildcard/subsumption logic is fully generic; the MW return is enzyme-flavoured |
| `copy_ec_to_gem.py` | write EC strings back into reaction annotations | reads `model.ec.eccodes` |

Suggested shape: ravenpy gets `annotate_ec_from_database(model, ...)` writing to
`reaction.annotation['ec-code']`; geckopy's version additionally mirrors into
`ec.eccodes`. The EC-wildcard matcher in `find_ec_in_db.py` is independently useful and
should be a public ravenpy helper.

### Strain-design scanning — `utilities/ec_fseof.py`
FSEOF (Flux Scanning with Enforced Objective Function, Choi et al. 2010) is a **generic
strain-design method that predates ecModels**. The flux-scan-and-rank core works on any
GEM. The geckopy version adds enzyme/usage-reaction targets and the `usage_prot_*`
exclusion. Split: a `ravenpy.fseof(model, target, ...)` returning reaction/gene targets,
and a geckopy `ec_fseof` that layers enzyme targets on top.

### Flux aggregation over split reactions — `utilities/map_rxns_to_conv.py`
Maps `_EXP_`/`_REV` split reactions back to their conventional parent and sums fluxes.
This is the natural companion to `convert_to_irreversible` + `expand_model` (both Tier
A) — **any** irreversible/expanded RAVEN-style model needs to fold fluxes back. The
flux-folding logic is generic; only the `enz_usage_2d` extraction is ec-specific. Move
the generic reverse-mapping to ravenpy.

### Model subsetting — `utilities/get_subset_ec_model.py`
Extracting a sub-model by genes/reactions and pruning orphans is generic (RAVEN's
`getModelSubset`/`removeReactions`/`removeGenes`). Here it additionally trims the `ec`
arrays. Generic subsetting → ravenpy; the ec-array trimming stays as a geckopy override.

### Reaction addition with isozyme expansion — `utilities/add_new_rxns_to_ec.py`
Adding reactions and expanding their GPRs is generic; the `ec` array bookkeeping is
specific. The expand-on-insert behavior reuses the Tier-A `expand_model` logic.

### Complex stoichiometry data — `databases/complex_portal_download.py` + `complex_portal_loader.py`
Downloading/parsing Complex Portal protein-complex stoichiometry is generic
protein-annotation data (interactome studies, complex-aware analyses). The *consumer*
(`apply_complex_data`, which writes subunit counts into `rxn_enz_mat`) is ec-specific.
Move the acquisition/parsing (`ComplexPortalEntry`, `get_complex_data`,
`load_complex_portal_json`) to ravenpy; keep `apply_complex_data` in geckopy.

---

## Tier C — the adapter / project scaffolding (split the schema)

`adapter/` is generic *project* infrastructure (a folder convention, a TOML config, a
template generator, organism metadata) carrying an ec-specific *payload*.

- **Generic → ravenpy**: the `ModelAdapter` machinery (`from_folder`, `resolve.py`,
  `template.py`), and the generic fields of `ModelParameters`: `path`, `conv_gem`,
  `org_name`, `c_source`, `bio_rxn`, and the data-source sub-schemas `KeggParams`,
  `UniprotParams`, `ComplexParams`.
- **ec-specific → geckopy**: `sigma`, `p_tot`, `f`, `gr_exp`, `enzyme_comp`, and the
  `EvotuneParams` / `OkpParams` (kcat tuning / prediction) sub-schemas.

Cleanest design: a `ravenpy` base adapter/params, subclassed in geckopy to add the
enzyme parameters. This also fixes a coupling smell flagged in the code review — many
generic functions currently demand a *full* ecModel adapter (`resolve_adapter`) when
they only need `path`, `org_name`, or one reaction ID.

---

## Stays in geckopy (genuinely enzyme-constrained)

For completeness, these are correctly ec-specific and should **not** move:

- **Core ec object model**: `ec_model/ec_model.py` (`EcModel`), `ec_data.py` (`EcData`),
  `enzyme.py` (`Enzyme`/`EnzymeView`), `make_ec_model.py`, `constants.py`.
- **Building the enzyme layer**: `pipeline/populate_ec.py`, `apply_kcat.py`,
  `protein_pool.py`, `apply_complex_data.py` (the *apply*, not the download),
  `apply_custom_kcats.py`, `fill_kcats.py`, `set_kcat.py`, `query.py`.
- **kcat acquisition**: all of `gather_kcats/` (DLKcat, fuzzy BRENDA matching, OKP,
  merge/select), plus `databases/brenda*` and `databases/dlkcat_ignore_lists.py`.
- **Protein-budget constraints**: `limit_proteins/` except `constrain_flux_data.py` —
  i.e. `calculate_f_factor`, `constrain_enz_concs`, `fill_enz_concs`,
  `flexibilize_enz_concs`, `get_conc_control_coeffs`, `relax_proteomics_greedy`.
- **kcat tuning**: all of `kcat_sensitivity_analysis/`.
- **Enzyme analytics**: `utilities/enzyme_usage.py`, `report_enzyme_usage.py`,
  `bottlenecks.py`.
- **ec model I/O**: `io/sbml.py`, `utilities/save_ec_model.py`, `load_ec_model.py` (the
  GECKO SBML/YAML extension format).
- **ec FVA**: `utilities/ec_fva.py` — though see the note below; the parallel-FVA
  harness inside it is generic even if the conventional-reaction grouping is not.

### Two near-misses worth noting
- **`databases/brenda*`** is the one "database" cluster that stays: BRENDA is queried
  here purely for turnover numbers (kcat / specific activity), which is intrinsically
  enzyme-kinetic. If ravenpy ever wanted generic enzyme-kinetic annotation (Km, etc.),
  the BRENDA *download/parse* infrastructure could be promoted, but as used today it is
  ec-specific.
- **`databases/dlkcat_ignore_lists.py`** is DLKcat-specific, but the *concept* of a
  "currency metabolite" blocklist is generic (network-distance, transport detection in
  FSEOF all need it). Consider a generic currency-metabolite list in ravenpy that the
  DLKcat list extends.

---

## Cross-cutting consequences of the split

1. **Dependency direction & types.** ravenpy functions must type-hint `cobra.Model`,
   not `EcModel`. Today many generic functions hint `EcModel` only for convenience —
   that hint is what would otherwise force a circular `ravenpy → geckopy` dependency.
   Audit every Tier-A/B function and widen the type to `cobra.Model`.

2. **Drop the `ec_`/`*_ec_*` naming in ravenpy.** `ec_fseof`→`fseof`,
   `load_conventional_gem`→`load_gem`, `fill_eccodes_from_gem`→`annotate_ec_from_gem`,
   `get_subset_ec_model`→`get_model_subset`, `add_new_rxns_to_ec`→`add_reactions`. The
   geckopy wrappers keep the ec-flavoured names.

3. **The adapter must be decoupled** (Tier C) before the database/flux functions can
   move cleanly, since most of them currently pull organism/path/taxon parameters out
   of the ecModel adapter.

4. **`model.ec` writes are the dividing line.** A useful mechanical first pass: grep
   each candidate for `.ec` access. Zero `.ec` references ⇒ Tier A (move outright).
   `.ec` only on output ⇒ Tier B (factor the kernel, keep an ec-writing wrapper).

---

## Suggested migration order

1. **Tier A structural surgery** (`preprocess.py`, `expand.py`) — zero `.ec` coupling,
   already RAVEN-annotated, immediate win.
2. **Tier A databases + `flux_data`/proteomics + `apply_flux_data_constraints`** — but
   first do the **adapter split** (Tier C) so they don't drag the ec adapter along.
3. **Tier B EC-code assignment and FSEOF** — refactor to operate on `cobra.Model`,
   leave ec wrappers in geckopy.
4. **Tier B subsetting / reaction-add / map-to-conv / complex download** — last, since
   they interleave generic and ec bookkeeping most tightly.

The biggest payoff is conceptual: after the split, `geckopy` shrinks to *only* the
enzyme-constraint mechanics (kcats, protein pool, enzyme usage, kcat tuning), and a
large, independently useful reconstruction/annotation/analysis toolkit (`ravenpy`)
becomes available to the whole cobrapy ecosystem.
