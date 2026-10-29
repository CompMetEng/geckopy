# Changelog

## Unreleased

* A scipy `EcModel` is copied and pickled into the sparse problem. optlang pickles a model as JSON and restores it by adding the variables and constraints to a dense problem, so every worker process of `ec_fva` (and `cobra.Model.copy`, `copy.deepcopy`) spent tens of minutes rebuilding the matrix of a genome-scale ecModel; ecYeastGEM now restores in 1.6 s and copies in 2.2 s. The solver of a repaired or converted scipy model is a `geckopy.solver_support._SparseModel`.
* `load_ec_model` reads the file on glpk and converts the ecModel to scipy when scipy is cobra's default solver, as `load_conventional_gem` does, and setting `model.solver = "scipy"` on an `EcModel` that is not on scipy builds the sparse problem of `geckopy.solver_support.switch_to_scipy`. optlang's own clone grows the dense coefficient matrix row by row: loading ecYeastGEM (8001 reactions) on scipy did not finish in 25 minutes and is now a matter of seconds. The scenarios that load a genome-scale ecModel on scipy spent most of their hours there. `geckopy.solver_support.is_scipy_interface` tells whether a solver name or interface is scipy.
* `get_subset_ec_model` removes the metabolites and genes that no reaction of the subset uses, as the `removeGenes` and `removeReactions` calls of GECKO's `getSubsetEcModel` do, also those that were unused in the big ecModel. The HT29-GEM subset of the light Human-GEM ecModel had the pseudo-gene `standard` and six metabolites (`MAM00579m`, `MAM01009c`, `MAM01099c`, `MAM01420m`, `MAM02549c`, `MAM02579m`) more than MATLAB's. A context model equal to the big model gives a copy without its unused metabolites and genes.
* `load_ec_model` loads the two forms of ecModel file that GECKO's `loadEcModel` reads and the YAML parser of raven-toolbox rejects, with the values MATLAB loads. `- enzymes: !!omap` with no entries, as GECKO writes it for an ec reaction without enzymes (the `standard` rows), raised `AttributeError: 'ScalarNode' object has no attribute 'flow_style'`; the row has no enzymes. A value in double quotes that holds unescaped double quotes, such as the notes `""text""` of the published eciYali5-GEM (RAVEN 2's `writeYAMLmodel` wraps a value in quotes without escaping the quotes in it, and GECKO 3 read the notes column of `customKcats.tsv` with `%s`, which keeps the quotes of the file), raised a parser error; the note is `"text"`, the outer pair of quotes removed and the inner ones kept. A file that the parser rejects is read again after these two forms are rewritten as YAML.
* `load_uniprot_tsv` and `ModelAdapter.get_uniprot_ids_from_table` read the cells as MATLAB's `textscan` with `%q` does, which the docstring of `load_uniprot_tsv` already claimed: a cell in double quotes is given without the quotes (`"1.1.1.25; 2.5.1.19"` gives `1.1.1.25; 2.5.1.19`), `""` inside a quoted cell is one quote, tabs and line breaks inside quotes belong to the cell, and leading spaces of a cell are dropped. A line with fewer cells is padded with empty cells and the cells of a line beyond the last column start a new row, as `textscan` does. The conversion table maps a gene that is listed twice to its first row, as `ismember` does, and a gene on a line without a second cell to an empty UniProt ID. `geckopy.databases.uniprot_loader.read_quoted_tsv` does the splitting.
* `assign_standard_kcat` compares the subsystem names of the reactions without regard to letter case, as MATLAB's `strcmpi` does: `Pentose phosphate pathway` and `Pentose Phosphate Pathway` are one subsystem, so their reactions count together toward the `threshold` and give one median. A reaction with an empty subsystem has none and gets the model-wide standard kcat.
* `get_subset_ec_model` removes the genes of the big ecModel that the context model lacks with `cobra.manipulation.remove_genes`, so a reaction that survives the removal (an isozyme remains, as with the rule `G1 and G2 or G3` without `G3`) loses the gene from its rule. Before, the gene left `model.genes` while the rule still named it, and removing that reaction later because the context model lacks it raised `ValueError: G3 not found`, as it did for the HT29-GEM subset of the light Human-GEM ecModel. In a gecko-light model the `001_` prefix of the ec reaction ids is no longer part of the id the ec rows are matched against: the subset used to drop every row of `ec.rxns`, `kcat`, `source`, `notes`, `eccodes` and `rxn_enz_mat`. A context model equal to the big model gives a copy, as before.
* The scipy solver (optlang's HiGHS interface) gives the results of glpk and gurobi. optlang renumbers the variable columns wrongly after a variable is removed (it stops at `len(constraints)` instead of `len(variables)`), so cobra read fluxes from the wrong columns after `make_ec_model`, `Reaction.delete` or any other removal: `ec_fva` returned 0 for every reaction of ecTestGEM although the LP optimum was 3272.7. An `EcModel` now repairs the problem whenever `model.solver` is read; `geckopy.solver_support.repair_scipy_problem(model)` does the same for a plain `cobra.Model`. The repaired problem stores the coefficients sparsely, because optlang copies a dense matrix on every added or removed variable or constraint, and it keeps the solution when `cobra.Model.optimize` sets the objective direction again, so `Reaction.flux` and `Enzyme.flux` work after `model.optimize()`. After an infeasible solve it returns NaN values and objective, so `model.optimize(raise_error=False)` returns a solution with status `infeasible` instead of raising a `TypeError`.
* With scipy as cobra's default solver, `load_conventional_gem` reads the model on glpk and `make_ec_model` builds the ecModel on glpk, and both convert the result to scipy (`geckopy.solver_support.switch_to_scipy`). Building the yeast-GEM ecModel on the scipy interface took more than 40 minutes in optlang's dense matrix operations without finishing.
* `Enzyme.shadow_price`, `get_enzyme_bottlenecks` and `relax_proteomics_greedy` work on the scipy solver, which has no constraint duals in optlang: `geckopy.solver_support.constraint_shadow_prices` reads the HiGHS row marginals of the last solve and uses them only if they are an optimum of the problem as it is now. `relax_proteomics_greedy` computes the shadow prices of all candidates once per iteration and takes the lowest ID among enzymes with equal absolute shadow price.
* `flexibilize_enz_concs` and `sensitivity_tuning` decide that the starting LP is infeasible from the solver status, not from the objective value, so they behave the same on every solver. Before, an infeasible start raised on Gurobi (no objective value) but not on GLPK (a finite objective value). `flexibilize_enz_concs` now returns an empty result for it on all solvers, as MATLAB's `flexibilizeEnzConcs` does, instead of raising on Gurobi. `sensitivity_tuning` raises on all solvers and stops with a warning when a later solve is not optimal.
* `apply_kcat_constraints` treats an empty `update_rxns` as all reactions, as MATLAB's
  `applyKcatConstraints` does with an empty `updateRxns`; it used to do nothing. The callers
  that build the list from a filter (`set_kcat_for_reactions`, the evotune scoring and reset)
  skip the call when the list is empty.
* `apply_custom_kcats` reads a reaction list in double quotes (`"R3,R5"`) without the quotes,
  as MATLAB's `%q` does, so the quoted list matches R3 and R5 instead of nothing.
* `make_ec_model` refuses what GECKO's `makeEcModel` refuses, with a `ValueError`: a reaction ID
  that starts with `usage_prot_` or `prot_pool` or ends in `_REV` or `_EXP_<digits>`, a metabolite
  ID that starts with `prot_`, an `enzyme_comp` that is not one of the compartment names of the
  model (a compartment ID that differs from its name no longer passes), and a model without
  genes. Only a reaction with `lb < 0 < ub` that is not an exchange reaction is split into a
  `_REV` pair: a reaction with both bounds negative stays as it is instead of raising, and a
  reaction that only consumes or only produces metabolites, such as `A + B <=>`, is an exchange
  reaction, as in RAVEN's `getExchangeRxns`.
* `set_kcat_for_reactions` follows `setKcatForReactions`. `apply` defaults to `False`, so only
  `ec.kcat` and `ec.source` change; pass `apply=True` to write the S matrix. IDs match without
  regard to letter case, an ID that matches nothing is ignored, and in a gecko-light model the
  `###_` prefix is part of the ID, so `"R2"` no longer expands to `001_R2` and `002_R2`.
  A kcat list with one value per matched row takes its values by position even when another
  ID matches nothing, as in MATLAB: `["R2", "nonsense"]` with `[1.0, 2.0]` sets the two `R2` rows to 1.0 and 2.0.
* `flexibilize_enz_concs` with a levelled complex scores a unit by its probes when the solver
  returns no usable reduced costs (the scipy solver), where a NaN score could pick the first unit.
* `get_conc_control_coeffs` keeps a reduced cost when raising the bound by 0.01 percent raises
  the optimum by at least half of the rise the reduced cost predicts, and by more than 1e-12
  times max(1, |optimum|). A small coefficient on a small bound is kept; the test used to need a
  rise above 1e-10, which dropped any coefficient with a reduced cost times bound below 1e-6.
* `fit_sigma(method="bisect")` gives the first grid sigma, `1 / n_sigma_steps`, for a model that
  cannot grow even at sigma 1, as the grid does; it used to give about 1e-6.
* `truncate_values` rounds a value exactly halfway between two candidates away from
  zero, as MATLAB's `round(x, n)` does: 100000.5 gives 100001 and 1234565 gives
  1234570, where they gave 100000 and 1234560.
* `save_kcat_source` writes the free text of the `detail` column as
  `saveKcatSource` does: a tab or line break becomes a space, so a row can no longer
  be split, and an empty part leaves no leading, trailing or repeated `; `.
* `fill_kcats_from_isozymes` and `get_kcat_across_isozymes` default to `apply=False`,
  as `getKcatAcrossIsozymes` only changes `ec.kcat`; pass `apply=True` to rewrite the
  stoichiometry. Only a kcat of 0 is filled: a NaN kcat is left as it is, and a NaN
  sibling does not enter the mean. `assign_kcats` passes `apply=True`, as before.
* `fit_sigma` scans the 0.01 grid of GECKO's `sigmaFitter` by default (`method="bisect"`
  stays available and still gives a continuous sigma). A target that growth cannot
  reach gives the first sigma of the growth plateau, where it used to give 1.0. The
  bounds of the objective reaction are set to `[0, 1000]` before the scan and stay so on
  the fitted model, and a solve that is not optimal counts as growth 0.
* `sensitivity_tuning` follows `sensitivityTuning.m`. The result lists the tuned reactions
  in model order, and reports the kcat and source of the input model, so a reaction an
  earlier run had tuned no longer reports NaN. `ec.notes` prints the pre-tune kcat as
  `num2str` does (`preTuneKcat=10`). A `prot_to_ignore` id is matched without regard to
  case and raises `ValueError` when no usage reaction has it. A tuned kcat has no upper
  limit. A gecko-light model is tuned, in place of raising `NotImplementedError`.
* `get_conc_control_coeffs` also takes the one-solve path with the scipy solver. optlang does not expose
  variable duals there, so the reduced cost is read from the HiGHS bound marginal of the last `linprog`
  result, after checking that the stored result is the solution `model.optimize` returned. On the tutorial
  yeast ecModel this takes seconds where the finite-difference loop took several minutes. A scipy result
  without bound marginals still falls back to the finite-difference loop.
* `get_conc_control_coeffs` reads the reduced cost of each usage reaction's forward
  variable and keeps it where a 0.01 percent rise of the bound raises the optimum,
  in place of the `prot_<id>` shadow price. An enzyme whose own bound does not
  bind now gets 0 while the protein pool binds, where every used enzyme used to
  get the pool's price, and a degenerate vertex no longer gives a spurious
  coefficient. Every nonzero reduced cost is checked, however small, and a usage
  reaction held at its lower bound gets 0 without a check. Where the solver exposes
  no variable duals, or returns NaN, the finite-difference loop is used.
* New: `save_ec_model` records the ecModel format (`gecko_format: "4"`) and
  the release that wrote the file (`generator: geckopy <version>`) in the
  `metaData`, the keys GECKO's `saveEcModel` writes, in place of
  `geckopy_version`. The model's own metaData, its date included, is kept.
  `load_ec_model` refuses a `gecko_format` above 4 and warns about a GECKO 3
  ecModel with a message that names `upgrade_project`; it turns the same
  protein reactions as GECKO's `loadEcModel`, including a usage reaction fixed
  to zero. `upgrade_project` reports an ecModel it cannot read as failed
  and goes on with the other steps.
* New: `level_complex_concs` and `level_complex_abundances` give every enzyme
  complex one abundance, the median per copy over its measured subunits, so a
  single badly quantified subunit no longer sets a complex's capacity.
  `complex_families` groups isozyme variants into complexes and leaves shared
  subunits and electron carriers as measured. See `docs/relaxation_methods.md`.
* New: `relax_enz_concs` relaxes enzyme caps (and, if need be, the protein
  pool) by the least total fold increase that reaches the target growth, in one
  LP, so the result does not depend on the order in which limiting enzymes are
  found.
* `flexibilize_enz_concs` gained `per_complex=True`: complexes are ranked and
  raised as one unit, keeping levelled subunits levelled. Off by default.

* Fix: `write_dlkcat_input` and the OpenKineticsPredictor input builder
  raised `TypeError: cannot use 'list' as a set element` on any model that
  had been saved and reloaded. cobra returns an annotation value as a list
  for a model read from a file, so `annotation['smiles']` is `"CCO"` on a
  model annotated in memory by `find_met_smiles` and `["CCO"]` on the same
  model after a round trip, and the ignore-list check hashed it. The new
  `databases.met_smiles` reads either shape; a reloaded model now writes
  the same DLKcat input as a freshly annotated one.
* Fix: `parse_okp_output` raised `TypeError: cannot use 'list' as a dict
  key` on a reloaded model for the same reason, when indexing metabolites
  by their SMILES annotation to map each prediction back to an `ec.rxn`.
* Fix: evotune's trust tiers never applied to a fuzzy-matched kcat.
  `ec.source` reads `"brenda (wc=0, origin=1)"` where MATLAB's reads
  `"brenda"`, and `classify_kcat_source` compared the whole string against
  the `source_groups` entries, so every fuzzy database match fell through
  to `unlabelled` and took `sigma0_log_default` instead of its tier's
  width -- on the yeast tutorial, 3 497 of 4 671 kcats. The bracketed
  detail is now ignored when matching; a group that lists the full string
  still matches it. Tuning runs on models with fuzzy-matched kcats will
  move by this.
* `fuzzy_kcat_matching`: when a reaction has several EC numbers, the EC
  matched at the earliest-tried level now wins, using the same search order
  as within one EC (a correct-organism specific activity, origin 5, before
  any-organism any-substrate kcat, origin 4), matching `fuzzyKcatMatching.m`.
  Previously the lower origin number won. This changes the kcat of
  multi-EC reactions; on the yeast-GEM tutorial the pre-tuning growth rate
  goes from 0.0013 to 0.12 /hour, in line with MATLAB.
* `load_conventional_gem` reads YAML with raven-toolbox's `read_yaml_model`
  instead of `cobra.io.load_yaml_model`, so RAVEN fields such as
  per-reaction `eccodes` are kept (`fill_eccodes_from_gem` found no EC
  numbers on RAVEN-format GEMs).
* New: `geckopy.databases.download_phyl_dist` fetches the KEGG
  phylogenetic-distance file distributed with RAVEN.
* full_ecModel tutorial: downloads the full `PhylDist.mat` instead of
  shipping a two-organism stub, uses KEGG as fallback for EC numbers, uses
  `merge_kcats`, runs ecFVA (STEP 73-75, skipped on GLPK) and the light
  versus full comparison (STEP 76-77, new `code/plot_light_vs_full.py`).
* Docs: `max_growth_weight` is weighted as in MATLAB's `maxGrowthWeight`
  (the previous description said otherwise).

## 4.0.0b1 — 2026-09-08

Version realignment: geckopy numbering will be aligned with GECKO (MATLAB),
justifying a jump to 4.x. This is the first release published to PyPI.

* New: `kcat_tuning.evotune` fits `ec.kcat` to measured growth rates and
  exchange fluxes with CMA-ES (`screen_kcat_leverage`, `select_tunable_mask`,
  `cmaes_kcat_tuning`, `tune_prior_penalty_weight`), replacing the earlier
  ABC-SMC approach. `EvotuneResult.corrections()` reports a tuning run as a
  ranked table of what changed and why, rather than a raw kcat vector. Gated
  behind the new `evotune` extra.
* New: `gather_kcats.review.review_assignment` flags kcat-assignment
  problems (repeated values, EC-maximum fallback, implausible magnitude)
  worth curating before a tuning run.
* `kcat_sensitivity_analysis` renamed to `kcat_tuning`, with the CMA-ES
  subpackage named `evotune` rather than `bayesian` — the shipped method is
  CMA-ES, not Bayesian inference.
* `flexibilize_enz_concs`: the post-loop refinement now uses MATLAB's soft
  ±0.25% growth band instead of pinning to `exp_growth` exactly, matching
  `flexibilizeEnzConcs.m`.
* `make_ec_model` no longer splits exchange reactions in the irreversibility
  stage, matching `makeEcModel.m`'s `nonExchRxns` exclusion.
* `get_reactions_from_enzyme` now matches `protein_id` case-insensitively,
  matching MATLAB's `strcmpi`.
* `read_dlkcat_output` gained an opt-in `strict=True` fail-fast on
  unrecognized substrates, matching MATLAB, and now reports drop reasons
  separately instead of lumping them together.
* `assign_standard_kcat`: a NaN-kcat reaction no longer counts toward a
  subsystem's threshold eligibility or gets folded into the subsystem
  median; unset kcats now get filled for NaN too, not just literal zeros.
* `parse_okp_output` now tags provenance with the `OKP-` prefix MATLAB
  applies; `merge_kcats` strips it again before source-priority matching so
  tagged rows still rank correctly.
* `set_kcat_for_reactions` now accepts a per-isozyme kcat list against an
  un-suffixed base name (assigned positionally, matching MATLAB), and writes
  `ec.source = "setKcatForReactions"` (MATLAB's actual value) instead of
  `"manual"`.
* `ec_fseof`: scan floor now anchors to the production target's flux at
  biomass-optimum instead of a fixed fraction of the theoretical maximum,
  and candidates are restricted to gene-associated, non-standard reactions
  before classification — both matching `ecFSEOF.m`.
* `ec_fva`'s parallel path now uses `cobra.util.process_pool.ProcessPool`
  instead of a hand-rolled `multiprocessing.Pool` — fixes a slow Windows
  initializer handoff and gives graceful pool teardown. Public signature
  unchanged.
* `raven-toolbox` dependency switched from a `git+...@develop` pin to the
  PyPI `>=3.0.0b1` release, which carries the `convert_to_irreversible(...,
  rxns=...)` addition `make_ec_model` needs.
* Packaging: the sdist was bundling the entire local `.venv-win`
  virtualenv (~60MB, including a compiled scipy wheel) because `.gitignore`
  listed `.venv/` but not `.venv-win/`, and hatchling's default sdist
  selection only reads the top-level `.gitignore`.
* `release.yml`: PyPI publish job enabled (trusted publishing).
* README: disambiguation note simplified and repointed at the specific old
  `geckopy` release (`2.0.2` and earlier), now that this project has taken
  over the PyPI project name itself.

## 0.3.0 — 2026-08-30

Parity pass against the MATLAB GECKO Toolbox: geckopy's core functions now
reproduce every value GECKO's own unit-test suite pins down.

* New: `tests/test_gecko_matlab_parity.py`, a 1:1 port of GECKO's
  `test/unit_tests/geckoCoreFunctionTests.m` (47 tests, MATLAB test cases
  `tc0001`–`tc0013`), run against the same `ecTestGEM` fixture MATLAB uses.
  Every expected value is copied verbatim from the MATLAB sources rather than
  re-derived from geckopy's output, so the file is the executable definition
  of "geckopy builds the same ecModel as GECKO". Covers `makeEcModel` (full
  and light), `applyComplexData`, `setProtPoolSize`, `getECfromGEM`,
  `getECfromDatabase`, save/load round-trip, `fuzzyKcatMatching`,
  `writeDLKcatInput`, `mergeDLKcatAndFuzzyKcats`, `selectKcatValue`,
  `applyKcatConstraints`, `getKcatAcrossIsozymes`, `applyCustomKcats`,
  `findMetSmiles` and the proteomics-integration chain.
* `make_ec_model` now sorts identifiers for full models, as MATLAB
  `makeEcModel` does (`sortIdentifiers`, full models only, before the protein
  pseudoreactions are appended) — geckopy kept the input GEM's reaction
  order, so a full ecModel came out with the same reactions in a different
  order from MATLAB's.
* `write_dlkcat_input` now emits rows in reaction order, matching MATLAB's
  column-major `find(clearedRedS < 0)` — numpy's `where` is row-major, so
  geckopy grouped the DLKcat input by substrate instead of by reaction.
* `EcModel.copy()` now clones the `ec` substructure. `cobra.Model.copy`
  rebuilds the network but carried every other attribute over by reference,
  so `model.copy().ec` was the same `EcData` object as `model.ec` — writing
  a kcat, eccode, source, conc or coupling coefficient on the copy silently
  changed the model it was copied from. `adapter` stays a shared reference
  (project configuration, not model state); `copy.deepcopy(model)` was
  already correct and is unchanged.
* Reverse (`_REV`) reactions now keep their EC code, via a fix in
  raven-toolbox's `convert_to_irreversible` (it now copies annotations,
  subsystem and notes onto the reverse reaction, as MATLAB's
  `convertToIrrev` copies `eccodes`/`rxnMiriams`/`subSystems`/`rxnNotes`).
  Requires raven-toolbox at or after that fix.
* `copy.deepcopy(ec_model)` (and of any `cobra.Model`) no longer
  `RecursionError`s on Python 3.14. The cause was in cobra:
  `Reaction.__copy__`/`__deepcopy__` delegated to `super()` to avoid
  infinitely recursing into themselves, relying on `copy` memoizing an
  object before recursing into its state — Python 3.14's `copy` module
  stopped doing so for that code path. Fixed upstream in cobra 0.31.0 by
  deleting the overrides; requires cobra at or after 0.31.1.
* `examples/yeast-GEM`'s adapter config used a rejected pydantic field:
  `model_adapter.toml` set `uniprot.tax_id`, but `ModelParameters`'s schema
  has no such field (it's `id`, already defaulting to the taxonomy id), so
  `ModelAdapter.from_folder` raised a validation error unconditionally. A
  parametrized regression test now runs over every `examples/` subfolder to
  catch schema drift like this for any future adapter config too.
* Docstrings across geckopy's source and tests reviewed for clarity and
  accuracy: `MATLAB-COMPAT` comparison paragraphs replaced with plain
  statements of current behavior, other historical/regression narration
  dropped, stale docstrings corrected. Comment/docstring-only, no functional
  changes.
* `protein_pool.py`'s module docstring no longer claims `prot_pool_exchange`'s
  direction diverges from MATLAB pending a "future I/O layer" — GECKO
  MATLAB's `makeEcModel.m` already writes the same forward (`lb=0`,
  `ub=1000`) convention as geckopy; the reverse convention only applies to
  legacy pre-GECKO-4 files, which `load_ec_model.py` already normalizes on
  load.
* `mw.py`'s stale `MATLAB-COMPAT` comments removed now that `calculate_mw`
  matches GECKO MATLAB's current water-mass constant (18.01528) and mean
  X-residue mass (118.885), closed upstream as GECKO#459. No behavioral
  change.
* Removed the dead `_run_preprocess` stub from `test_populate_ec`: an empty
  helper that was never called.
* `merge_kcats` is n-ary and concatenates surviving rows list by list, so
  the equivalent of MATLAB's `mergeDLKcatAndFuzzyKcats(dlkcat, fuzzy)` —
  which always emits fuzzy rows first regardless of argument order — is
  `merge_kcats(fuzzy, dlkcat, ...)`. The deprecated
  `merge_dlkcat_and_fuzzy_kcats` alias is `merge_kcats`, so it follows the
  geckopy convention, not MATLAB's signature. Row order has no numerical
  effect: `apply_kcat_list` aggregates per reaction.

## 0.2.1 — 2026-07-16

Bugfix release. `make_ec_model` no longer mutates the GEM it is given.

* `make_ec_model` now leaves the input GEM unchanged. Stages 1–5
  (`remove_pseudoreaction_gprs`, `invert_backwards_only_reactions`,
  `convert_to_irreversible`, `expand_model`) ran in place on the caller's
  model, leaving it with `_REV`/`_EXP_` reaction splits, `prot_`
  pseudometabolites, and stripped pseudoreaction GPRs. The pipeline now
  works on an internal `model.copy()`, matching MATLAB GECKO's
  `makeEcModel` value semantics.
* `test_loads_sbml_gem` on Windows: the test wrote a backslash path into the
  adapter TOML, which TOML reads as an escape sequence.
* README: disambiguation note stating that this geckopy is a from-scratch
  port of the MATLAB GECKO Toolbox, unrelated to and sharing no code with
  the separate PyPI `geckopy` package described in Carrasco Muriel et al.
  (2023).
* CI: leaner test matrix — full Python range (3.11–3.13) on Linux plus a
  single Python (3.12) on macOS and Windows, instead of the full 3×3
  OS × Python grid. Same OS coverage at roughly half the runner cost.
* Removed unused test imports flagged by ruff (F401).

## 0.2.0 — 2026-06-14

Dependency rename + CI hardening. geckopy now targets raven-toolbox (the
renamed raven-python) and runs its test matrix on every pull request.

* Depend on `raven-toolbox` (renamed from `raven-python`): all imports are
  now `raven_toolbox.*`, and the `pyproject.toml` git-URL dependency and
  project URL point at `SysBioChalmers/raven-toolbox`.
* Use the public `raven_toolbox.manipulation.expand.gpr_to_dnf` instead of
  the former private `_gpr_to_dnf`.
* CI: run the test matrix on every pull request, not only PRs into
  `main`/`master`.
* CI: bump GitHub Actions to Node 24 runtimes — `checkout@v5`,
  `setup-python@v6`, `upload-artifact@v7`, `download-artifact@v8`.
* Rewrote `migrating_from_gecko_matlab.md` as a complete reference; moved
  internal planning notes into `docs/internal/`.

## 0.1.0a3 — 2026-05-30

Build-config hotfix. v0.1.0a2 was tagged but its release.yml build failed:
hatchling refuses PEP 508 direct-URL dependencies (`raven-toolbox @
git+https://...`) by default. The v0.1.0a2 tag and release were deleted;
v0.1.0a3 is the published artifact carrying the 0.1.0a2 content described
below. No code changes vs. v0.1.0a2.

* `pyproject.toml`: `[tool.hatch.metadata] allow-direct-references = true`
  so hatchling accepts the `raven-toolbox` git URL dependency. PyPI publish
  stays disabled (geckopy's name is taken, raven-toolbox isn't on PyPI yet);
  the direct reference is acceptable for the git-install path the alpha
  series uses.

## 0.1.0a2 — 2026-05-30

Second public alpha. Layering pass: the bits of geckopy that aren't
GECKO-specific moved into raven-toolbox, and `gecko-light` ecModels are now
buildable end-to-end. Repo housekeeping for a discoverable first GitHub
release (LICENSE, planning docs out of the user-facing tree).

* gecko-light end-to-end: `make_ec_model(gecko_light=True)` now produces a
  working light ecModel — cobra reactions stay singular; per-isozyme
  coupling lives in `ec.rxns` as duplicate rows with a `###_` counter
  prefix; only the shared `prot_pool` constrains enzyme usage.
  `apply_kcat_constraints` picks the lowest-cost (smallest `MW_sum/kcat`)
  isozyme per cobra reaction and writes one `prot_pool` stoichiometric
  coefficient. `set_kcat_for_reactions` recognises both base names and the
  `###_` prefix; the `Enzyme` proxy's read paths work on light models. New
  `tutorials/light_ecModel/protocol.py` mirrors MATLAB GECKO's light
  tutorial; opt-in Human-GEM smoke test (`pytest -m smoke`) builds an
  ecModel from the unmodified Human-GEM YAML in ~2.5 min. See
  [`docs/gecko_light_status.md`](docs/gecko_light_status.md).
* YAML I/O delegated to raven-toolbox: `save_ec_model`/`load_ec_model` are
  now ~80-LOC wrappers around `raven_toolbox.io.read_yaml_model`/
  `write_yaml_model`. raven-toolbox owns the typed `EcData` (now
  re-exported as `geckopy.EcData`), the `ec-rxns`/`ec-enzymes`/`gecko_light`
  YAML schema, and the three legacy GECKO normalisations (top-level
  `smiles` → `annotation`, reverse-direction `usage_prot_*` flip, bare-`-`
  document root). Geckopy keeps file-extension dispatch, adapter-aware path
  resolution, and provenance/diagnostics. SBML ecModel I/O removed
  (`geckopy.io.sbml` deleted; `cobra.io.read_sbml_model` is still used for
  loading the conventional starting GEM). See
  [`docs/raven_integration.md`](docs/raven_integration.md).
* `ec_fseof` re-aligned with `raven_toolbox.analysis.fseof`: thin wrapper
  over raven's regression-based FSEOF, dropping `usage_prot_*` from the
  scan + targets, resolving `bio_rxn` from the adapter, and emitting an
  optional carbon-source consistency warning. Selection moves from MATLAB
  GECKO's strict-monotonicity + top-25%-by-slope to raven's
  `|correlation| ≥ threshold` with pFBA per step. Result type is raven's
  `FSEOFResult` (replaces `EcFseofResult`); action labels are
  `amplify`/`knockdown`/`knockout` (replaces `OE`/`KD`/`KO`); the per-gene
  `essentiality` column drops (use
  `cobra.flux_analysis.single_gene_deletion`).
* `raven-toolbox` pin moved from `@main` to `@develop`. raven-toolbox's
  `main` is empty in the current iteration; all releaseable work lives on
  `develop`.
* `get_conc_control_coeffs` solver-state guard: both `_shadow_price_coeffs`
  and `_finite_difference_coeffs` previously accepted any solve whose
  `objective_value` was non-`None` and non-NaN, including infeasible
  glpk-via-optlang solves (which return `status="infeasible"` but a
  non-NaN objective from the binding-constraint rhs). New
  `_solution_is_optimal` helper requires `sol.status == "optimal"` at every
  solve point.
* Removed `geckopy.io.sbml.{read,write}_sbml_ec_model` and the `geckopy.io`
  package (SBML ecModel I/O dropped; YAML is the supported on-disk format).
* Removed `geckopy.EcFseofResult` (use
  `raven_toolbox.analysis.fseof.FSEOFResult`).
* Removed the top-level `requirements.txt` (a `pip freeze` snapshot that
  duplicated `pyproject.toml`'s declared dependencies); pyproject is the
  source of truth.
* Internal planning docs (`brenda_refresh_plan`, `openkineticspredictor_plan`,
  `porting_plan`, `raven_inventory`, `code_review`) moved from `docs/` to
  `docs/internal/` so the user-facing docs tree only carries reference
  documentation.
* `LICENSE` file added (MIT was already declared in `pyproject.toml` and
  classifier; the text file was missing).

## 0.1.0a1 — 2026-05-30

First public alpha. Every MATLAB GECKO 3.2.5 function used in the standard
ecModel build is ported; the yeast-GEM tutorial runs end-to-end.

* Project skeleton: `EcModel`/`EcData`/`Enzyme` core classes built on
  cobrapy; `ModelParameters`/`ModelAdapter` scaffolding (TOML-driven config,
  template generator, `geckopy init` CLI).
* `make_ec_model` pipeline: 9-stage MATLAB-port pipeline that builds an
  ecModel from a conventional GEM + adapter. Fixes a unit bug where MW
  values were treated as kDa instead of Da.
* Kcat constraints + protein-complex/MW: `apply_kcat_constraints`,
  `set_kcat_for_reactions`, `apply_custom_kcats`, isozyme averaging, Complex
  Portal loader, `find_met_smiles`, `calculate_mw`.
* BRENDA + DLKcat pipelines: EC-code resolution helpers,
  `fuzzy_kcat_matching` MATLAB-port against BRENDA, DLKcat wrappers
  (`write_dlkcat_input`, `run_dlkcat`, `read_dlkcat_output`), standard-kcat
  fallback.
* Wide BRENDA snapshot: `kcat.tsv`/`sa.tsv`/`mw.tsv` ship both `max` and
  `median` per `(ec, substrate, organism)` triple (plus `n` = measurement
  count). `BrendaData.kcat_for("max" | "median")` picks the view; three
  adapter fields (`kcat_aggregate_brenda`/`kcat_aggregate_candidates`/
  `kcat_aggregate_isozymes`) let projects flip aggregation defaults
  globally. See [`docs/kcat_aggregation.md`](docs/kcat_aggregation.md).
* Generalised kcat merge: `merge_kcats` records per-row provenance and
  accepts any combination of BRENDA/DLKcat/OpenKineticsPredictor/manual
  sources.
* Data-source downloaders: `geckopy brenda-refresh` CLI (BRENDA bulk JSON),
  KEGG as an alternative protein/EC source (with a warning when fallback
  uses a bare KEGG gene id), OpenKineticsPredictor REST-API submit/fetch
  wrappers, YAML I/O aligned with cobrapy + GECKO keys.
* Protein concentrations + sigma + sensitivity: `calculate_f_factor` +
  `load_pax_db`, `fill_enz_concs`, `constrain_enz_concs`,
  `get_conc_control_coeffs`, `flexibilize_enz_concs`, `constrain_flux_data`;
  sigma fitting and `sensitivity_tuning`.
* Analysis utilities: `load_flux_data`, `load_prot_data`, `enzyme_usage` +
  `report_enzyme_usage`, `map_rxns_to_conv`, `ec_fva`, `ec_fseof`,
  `add_new_rxns_to_ec`, `get_subset_ec_model`.
* Persistence: `load_ec_model` + `save_ec_model` (YAML + SBML round-trip
  with `ec_*` metadata), `merge_dlkcat_and_fuzzy_kcats`,
  `load_conventional_gem`.
* Enzyme accessor + analysis utilities + SBML I/O: `Enzyme`/`EnzymeView`
  accessor, `pfba_enzymes`, `get_enzyme_bottlenecks`, parallel `ec_fva` via
  multiprocessing, `relax_proteomics_greedy`, SBML reader/writer with
  MW_KCAT encoding, flatter top-level imports.
* Dedup, renames, splits: de-duped ec-layer constants across 12 files,
  renamed 7 functions for verb-style consistency (with deprecated aliases),
  flattened top-level API, `resolve_adapter` helper, `fuzzy_kcat_matching`
  split into a 4-file subpackage, SBML name aliases, auto-flip of legacy
  reverse-direction protein reactions, `ec.kcat` sentinel switched from
  `NaN` to `0`.
* Bisect sigma, shadow-price control coeffs, diagonal ecFVA, median kcat:
  `fit_sigma(method="bisect")` exploiting monotonicity; `calculate_mw`
  returns NaN on empty/all-skipped sequence; `get_conc_control_coeffs`
  switches to LP shadow-price with scipy auto-fallback; `ec_fva` reports
  the exact per-reaction range (diagonal); `median` option for kcat
  aggregation.
* Code review hardening — silent-data-loss guards: `genes`-list length
  validation, numpy-scalar coercion, kcat/relaxation loop non-termination
  guards, enzyme-provenance round-trip across SBML, UniProt-accession
  dedup, Complex Portal sub-complex classification, downloader hardening,
  NaN-aware kcat overwrite, non-fatal DLKcat unknown-substrate handling,
  `ec_fseof`/`pfba_enzymes` feasibility checks, OKP GET retries, current
  PubChem SMILES property, perf hoists, BayesianParams length validation,
  robust BRENDA protein lookup.
* Code review hardening — adapter decoupling: adapter-free read/inspect of
  ecModels; per-parameter overrides decouple analysis functions from the
  adapter; `bio_rxn` overrides + loader robustness; writable BRENDA data
  kept out of the install tree.
* Code review hardening — cleanup: sanity guards + DLKcat ignore-list
  header tolerance; dedup helpers + standardised infeasibility check;
  dead-code removal; tuned-kcat capping + negative-concentration guards;
  dropped private back-compat aliases in `fuzzy_kcat_matching`.
* Tutorial + packaging: full_ecModel notebook tutorial walking through
  Stages 0–5 (build → proteomics → simulation), `ec_fva` tqdm progress bar,
  ruff baseline, `.gitattributes` LF enforcement, PyPI-ready packaging
  metadata, GitHub Actions for tests/lint/release.
* Docs sweep: README rewrite for non-specialist readability, friendlier
  docstrings on the public-API modules, Carrasco et al. (2023) citation,
  third-party library gotchas (SBML/multiprocessing/ruamel), RAVEN function
  inventory + ravenpy scope, gecko-light status.
* raven-toolbox is now a hard dependency: geckopy delegates `expand_model`
  and `convert_to_irreversible` to raven-toolbox (the functions originated
  in geckopy and were adopted upstream as their canonical home).
  raven-toolbox is not yet on PyPI; install both via the git URLs in the
  README. See [`docs/raven_integration.md`](docs/raven_integration.md) for
  the current delegation and the planned future migrations.
