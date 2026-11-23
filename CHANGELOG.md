# Changelog

## 4.0.0b3 (2026-11-23)

Third beta of geckopy 4. The main changes are the move of the experiment functions to raven-toolbox, stricter checks of `model_adapter.toml`, source groups for evotune and the GPL-3.0 licence. Changes below are relative to 4.0.0b2, and geckopy 4.0.0b3 requires raven-toolbox 3.0.0b3.

To update a `model_adapter.toml` written for 4.0.0b2, remove the keys `[kcat_sources.okp_<x>] method`, `[evotune.sigma0_log_source]`, `match_okp`, `kcat_aggregate_brenda`, `kcat_aggregate_candidates` and `kcat_aggregate_isozymes`, which are errors. Set `gr_exp`, `uniprot.id` and `kegg.id`, which have no default. Run `geckopy adapter show` to list the values the adapter resolves to.

### Breaking changes

* `load_experiment_data`, `ExperimentData`, `FluxData`, `load_flux_data`, `simulate_conditions`, `experiment_distance` and `compute_carbon_weights` are in `raven_toolbox.experiments`; geckopy imports them and `geckopy.databases` still exports `FluxData` and `load_flux_data`.
* `load_experiment_data` takes the data folder (was: the adapter).
* The evotune functions take `experiment_data` (was: `evotune_data`) and `max_growth` (was: `max_grate`).
* `FluxData.weight` is the per-condition weight (was: `evotune_rmse_weight`), and the file column is `weight` (was: `evotuneRMSEweight`).
* Loading `fluxData.tsv` or `maxGrowth.tsv` for the tuning raises an error that names `weight` when the file has the column `evotuneRMSEweight`.
* The experiment data files are `fluxData.tsv`, `maxGrowth.tsv` and `zeroFlux.tsv` (were: `evotuneFluxData.tsv`, `evotuneMaxGrowth.tsv` and `evotuneZeroExch.tsv`).
* `experiment_distance` and `dataset_rmse` take no biomass reaction.
* `geckopy.solver_support` is removed; its functions are in `raven_toolbox.utils.solver` (`optimize_model`, `repair_scipy_problem`, `switch_to_scipy`, `edit_on_fast_solver`, `fast_solver_while_building`, `is_scipy_interface`, `constraint_shadow_prices`).
* `calculate_mw` and `met_smiles` are in raven-toolbox (`raven_toolbox.utils.protein` and `raven_toolbox.annotation`), and `geckopy.databases` exports them from there.
* `find_met_smiles` calls `raven_toolbox.annotation.find_met_smiles` with `skip_prefixes=("prot_",)`, so the enzyme metabolites are not looked up.
* `find_met_smiles` needs `cache_path` and does not default to `data/smilesDB.tsv` of the adapter.
* `load_uniprot_tsv` always splits a gene cell (the `split_gene_cells` option is removed).
* The deprecated aliases with GECKO 3 names are removed from `geckopy`: `run_dlkcat`, `read_dlkcat_output`, `write_dlkcat_input`, `apply_custom_kcats`, `get_kcat_across_isozymes`, `get_standard_kcat`, `select_kcat_value`, `merge_dlkcat_and_fuzzy_kcats`, `get_ec_from_database`, `get_ec_from_gem`, `sigma_fitter` and `constrain_flux_data`.
* `find_max_value`, which ports `findMaxValue` that GECKO 4 removed, is removed from `geckopy`.
* `extract_enzyme_substrate_pairs` stays public.
* `submit_open_kinetics_predictor`, `fetch_open_kinetics_predictor`, `build_okp_input_csv` and `parse_okp_output` are private (`_submit_open_kinetics_predictor` and so on) and not exported from `geckopy`, `geckopy.gather_kcats` or `geckopy.gather_kcats.open_kinetics_predictor`.
* `OKPClient` and `OKPError` stay public. `kcats_from_okp` and `kcats_from_dlkcat(via="okp")` submit and fetch OpenKineticsPredictor jobs. GECKO moves its OKP engine functions into `kcatsFromOKP` in the same way.
* `truncate_values` is removed from `geckopy.kcat_tuning` and `geckopy`; nothing in geckopy called it, and GECKO has removed `truncateValues`.
* `update_prot_pool` is not ported; `set_prot_pool_size` sets the protein pool for a condition.
* `cma` is a required dependency of geckopy and the extra `evotune` is removed.
* `uniprot.id` and `kegg.id` are empty by default (was: `"559292"` and `"sce"`, the identifiers of Saccharomyces cerevisiae), as in GECKO; `adapterParameters.tsv` is the same as GECKO's.
* `make_ec_model` and `calculate_f_factor` raise `FileNotFoundError` naming `[uniprot] id` when `data/uniprot.tsv` is missing and the identifier is empty, as GECKO's `loadDatabases` raises `loadDatabases:noUniprotID`.
* An empty `kegg.id` leaves KEGG out with a warning.
* `gr_exp` has no default (was: `0.41`, the yeast growth rate).
* `sensitivity_tuning`, `fit_sigma`, `flexibilize_enz_concs` and `relax_enz_concs` called without a growth rate raise `ValueError: gr_exp is not set in model_adapter.toml` when the file does not set it.
* An unknown key in model_adapter.toml is an error, whatever its place: `ModelAdapter.from_folder` raises `ValueError` with `<file>: unknown key <dotted.key> in model_adapter.toml`.
* The unknown keys include keys under `evotune.source_groups.<group>`, the `kcat_assignment.stage` entries and `kcat_sources.<name>`. The removed keys are unknown keys, and no alias or warning remains for them.
* `kcat_aggregate_brenda` is removed from model_adapter.toml (and `ModelParameters`); `[kcat_sources.brenda] aggregation` is the only BRENDA aggregation setting.
* `kcats_from_brenda` and `fuzzy_kcat_matching` without an explicit aggregation use the aggregation the kcat assignment config resolves: median for the default preset `brenda_ec_tree` and the other median presets, max for `gecko_classic`, and max when neither the preset nor the file sets it (was: `kcat_aggregate_brenda`, which defaulted to `"max"`, so a call without an aggregation under the default preset took the maximum).
* `kcat_aggregate_candidates` and `kcat_aggregate_isozymes` are removed from model_adapter.toml (and `ModelParameters`).
* `apply_kcat_list` takes `criteria="max"` and `fill_kcats_from_isozymes` takes `aggregate="mean"` as their defaults, and neither reads an adapter value.
* `[kcat_sources.okp] method` is the only setting that names the OpenKineticsPredictor method (default `"KinForm-H"`); `[okp]` has no `method` key. `kcats_from_okp` without `method` uses it.
* `[kcat_sources.okp_<x>] method` is removed. The method of the file `okp_<x>` is the known method whose file name is `okp_<x>` (`okp_` and the method in lower case, each run of other characters than letters and digits as `_`).
* `update_kcat_sources` reports any `okp_<x>` that is not the file of a known OKP method as failed (`<name> is not the file of a known OKP method; the files are ...`).
* `CatPred` and `TurNup` are not accepted as `[kcat_sources.okp] method`: `kcats_from_okp` rejects them, so the configuration fails early with `Unknown OKP method`.
* The `okp` source group of the presets is the OpenKineticsPredictor method KinForm-H by default, and `update_kcat_sources` gathers `okp_kinform_h.tsv` only. GECKO uses the same default.
* Evotune source groups are one table each, `[evotune.source_groups.<group>]` with `sources` and `sigma0_log`, both required.
* `[evotune.sigma0_log_source]` and `SourceGroupRule.match_okp` are removed, and so is the `okp_method` argument of `classify_kcat_source`, `classify_kcat_sources`, `screen_kcat_leverage`, `cmaes_kcat_tuning`, `sweep_tunable_mask_size` and the other tuning functions.
* Without `[evotune.source_groups]` the groups are `custom` 0.1, `brenda` 0.2, `ec_tree` 0.3 and `combined` 0.4 (was: `dlkcat` 0.4, `brenda` 0.2 and `custom` 0.1); groups in the file replace the defaults entirely.
* `collect_kcats` and `assign_kcats` do not apply `data/customKcats.tsv`: when a stage lists custom, `data/kcats/custom.tsv` does not exist and `data/customKcats.tsv` does, they raise `ValueError` naming `convert_custom_kcats`, before they change the model. GECKO's `collectKcats` does the same.
* The decisions have no stage `custom`.
* `download_uniprot` and `load_uniprot_tsv` download reviewed and unreviewed entries by default (`reviewed=False`), as the adapter default `uniprot.reviewed = false` (was: reviewed entries only).
* `geckopy uniprot-download` keeps reviewed and unreviewed entries by default, as the adapter default `uniprot.reviewed = false` and GECKO's `loadDatabases` do; `--reviewed-only` keeps the reviewed (Swiss-Prot) entries only. The option `--include-unreviewed` is removed (was: the command kept only reviewed entries by default).
* `ec_fseof` follows `ecFSEOF.m` in its labels, default and gene targets. `target_type` is `OE`, `KD` or `KO` (was: `amplify`, `knockdown` or `knockout`), and `n_steps` defaults to 16 (was: 10).
* `ec_fseof` returns an `EcFSEOFResult` (exported from `geckopy` and `geckopy.utilities`), raven-toolbox's `FSEOFResult`.
* The `gene_targets` of `ec_fseof` has one row per gene with the columns `gene`, `name`, `slope` (the mean absolute slope of its target reactions), `action`, `essentiality` and `reactions`, ordered by slope (was: the columns `gene`, `target_type` (the actions joined by `;`), `reactions` and `max_abs_slope`).
* A gene with several actions is `OE` when one of them is `OE`, otherwise `KO` when it is not essential and `KD` when it is. Essential means that growth is infeasible or below 1e-8 with the usage of its enzymes blocked and the production fixed at the last enforced level.
* `format_kcat_source(origin, detail=None, tuned=(), prior=None)` replaces `format_kcat_source(source, wildcard_level, origin)` and writes the same string for the same non-empty source, wildcard level and origin; an empty source with a detail is written as `unknown`.
* `sensitivity_tuning` does not write `sensitivityTuning` to `ec.source` or the kcat before tuning to `ec.notes` (was: it wrote both).
* `set_kcat_for_reactions` writes the source `manual` (was: `setKcatForReactions`) and drops any tuning status.

### Experiments

* `simulate_conditions` raises `ValueError` when the biomass reaction, an exchange reaction of the data or a reaction of the zero-flux list is not in the model.
* `apply_flux_data_constraints` takes the constraint logic from `raven_toolbox.experiments.apply_flux_data_constraints` and keeps the adapter defaults for `bio_rxn` and `c_source`.
* In the tutorial, `fluxData.tsv` holds the condition of the protein step as its first row, with weight 0.

### Adapter

* `geckopy adapter show [FOLDER]` (default: the current folder) and `show_adapter_params(adapter)` print one line per item in the order of `adapterParameters.tsv`, without a header: `<key> = <value>  [<origin>]` with the origin `file`, `default`, `preset <name>` or `not set` (`<key> =  [not set]`). GECKO has `showAdapterParams`.
* `geckopy adapter show` lists the kcat assignment stages (`kcat_assignment.stage[<i>]`), the `kcat_sources.<name>.<key>` settings and the evotune source groups with their `sigma0_log` as resolved.
* `geckopy adapter show` prints whole-valued numbers without decimals and lists the `kcat_sources` settings by source (brenda, ec_tree, homologs, okp, then others alphabetically) and key.
* `geckopy init` writes `uniprot.id` and `kegg.id` uncommented as `""`, with examples in the comments.
* `geckopy init` writes `gr_exp` commented out, with a note that the functions that tune kcats or fit sigma require it.
* The keys of `[kcat_sources.<name>]` are `aggregation` and `force_wildcard_level` for `brenda`, `k` for `ec_tree`, `k`, `min_identity` and `coverage` for `homologs`, `method` for `okp`, and none for `custom`, `dlkcat` and `okp_<file of a known OKP method>`; another source name is an unknown key.
* `ModelAdapter.from_folder` raises `kcat_sources.okp.method must be a string` and `Unknown OKP method <name> in [kcat_sources.okp] method; supported: <known>` when `[kcat_sources.okp] method` is not a string or does not name a known OKP method. The message lists the supported methods comma-separated.
* `ModelAdapter.from_folder` raises `<file>: kcat_sources.<name> must be a table` when a `[kcat_sources]` entry is not a table.
* `ModelAdapter.from_folder` requires `[kcat_sources.brenda] aggregation` to be `max` or `median`, an evotune group's `sources` to be a non-empty list of strings, and `sigma0_log` to be finite and positive.
* A malformed evotune group raises `<file>: evotune.source_groups.<g> must be a table with sources and sigma0_log`, `... needs both sources and sigma0_log`, `... .sources must be a list of strings` or `... .sigma0_log must be a positive number`.
* A function given another BRENDA aggregation raises `aggregation must be max or median`.

### Evotune

* The scoring of the evotune functions (`screen_kcat_leverage`, `cmaes_kcat_tuning`, `tune_prior_penalty_weight`) maximises the biomass reaction `bio_rxn` of the adapter, whatever objective the model has, and leaves the objective of the model as it was. GECKO's `evotuneScore` does the same.
* The model objective does not need to be growth, and `simulate_conditions` issues no objective warning during tuning.
* `sigma0_log` of an evotune source group is a standard deviation of the natural log of the kcat.
* A `sources` entry ending in `*` is a prefix pattern on the origin (case-insensitive). Exact matches over all groups are tried first, then patterns, the longest prefix first, and groups of equal length in file order.
* A group covers OpenKineticsPredictor kcats with a pattern such as `okp_*`.
* Tuning warns when more than 20 % of the tunable kcats are in no group: `<n> of <m> tunable kcats match no evotune source group and get sigma0_log_default (<value>); origins: <up to 5, by count descending and then by name>`.
* The unlabelled-kcat warning of evotune tuning is also issued as a `UserWarning`.

### Kcat sources

* `kcat_evidence` shows, for selected reactions, the kcat of every source file next to the kcat of the model and, for a tuned kcat, the kcat before tuning. GECKO has `kcatEvidence`.
* `kcat_evidence` selects the reactions by name (`rxns`), by enzyme (`enzymes`, for example the enzymes that `flexibilize_enz_concs` relaxed) or by the result of `sensitivity_tuning` (`tuned`).
* `kcat_evidence` with `compute=["ec_tree"]` adds the EC-tree estimate without a source file.
* `kcats_for_reactions` gets the kcats of every source for a few reactions and computes the EC-tree estimate. With `okp`, it predicts them with one, several or all OpenKineticsPredictor methods. GECKO has `kcatsForReactions`.
* `kcats_from_okp` has `job_tag`, which keeps a job for a few reactions apart from the job of all reactions, with its own job, input and result files. GECKO has the `jobTag` option of `kcatsFromOKP`.
* `kcats_from_ec_tree` marks a reverse direction (`_REV`) in `detail` with `direction=reverse`. GECKO's `kcatsFromECTree` does the same.
* `kcats_from_ec_tree` gives the reverse direction the value of the forward direction, as the estimate depends on the EC number only and BRENDA measurements have no direction.
* `convert_custom_kcats` (or `geckopy upgrade`) turns `data/customKcats.tsv` into `data/kcats/custom.tsv`.
* `kcats_from_dlkcat` takes `via`: `"okp"` predicts with the DLKcat method of OpenKineticsPredictor (as `kcats_from_okp`), `"file"` reads `data/DLKcat.tsv` and `"docker"` runs DLKcat locally. The default is `"file"` when `data/DLKcat.tsv` exists and `"okp"` otherwise; `run=True` is the same as `via="docker"`.
* `kcats_from_dlkcat` returns `(kcat_list, done)` as `kcats_from_okp` does, with `kcat_list` `None` while an OKP job runs, and takes `wait`. GECKO's `kcatsFromDLKcat` has the same option.
* `update_kcat_sources` gathers `dlkcat` through OKP when `data/DLKcat.tsv` is missing, needs an OKP API key for it and reports `pending` while the job runs (was: it needed Docker).

### Provenance

* `format_kcat_source` and `parse_kcat_source` write and read the provenance strings of `model.ec.source`, `<origin>[ (<detail>)][ | tuned=<method>[+<method>...] (prior=<value>)]`, and `KcatSource` holds the parts of one. GECKO's `formatKcatSource` and `parseKcatSource` do the same.
* `sensitivity_tuning` and `cmaes_kcat_tuning` keep the origin and detail of the source of a kcat they change and add the method (`sensitivity`, `cmaes`) to its tuning status, with the kcat before the first tuning as `prior`.
* `tune_prior_penalty_weight` restores `ec.source` with `ec.kcat`.
* `assign_standard_kcat`, `remove_standard_kcat`, `review_assignment`, `convert_custom_kcats` and the evotune trust tiers read the origin of `ec.source`, so a source with a detail or a tuning status is recognised.
* `remove_standard_kcat` keeps a standard kcat with a tuning status, as it was changed on purpose.

### Databases

* `load_uniprot_tsv` splits a gene cell of uniprot.tsv with several gene names, separated by semicolons and/or whitespace (`YPR080W; YBR118W`), into one row per gene name. The other columns are repeated on every row and an empty cell stays one row. A model gene that is any of the names finds the entry. GECKO's `loadDatabases` does the same.
* `make_ec_model` without `uniprot_db` or `kegg_db` downloads a missing `data/uniprot.tsv` or `data/kegg.tsv` with the `[uniprot]` and `[kegg]` settings of the adapter, as GECKO's `makeEcModel` does through `loadDatabases` (was: `FileNotFoundError` for a missing `uniprot.tsv`, and no KEGG data without `kegg.tsv`).
* `make_ec_model` leaves KEGG out with a warning when `kegg.id` is empty. A project without `kegg.tsv` and with a `kegg.id` gets the KEGG fallback for genes UniProt does not know.
* `load_kegg_tsv` with an empty `kegg_id` and no `kegg.tsv` logs a warning and returns None, as GECKO's `loadDatabases` does without a `kegg.ID` (was: it called `download_kegg` with an empty organism code, which raised).
* `calculate_f_factor` takes no `prot_data`: it then reads `data/paxDB.tsv` of the adapter's project with `load_pax_db`, and without that file logs a warning and returns the default f of 0.5, as GECKO's `calculateFfactor` does (was: `prot_data` was required).

### Enzyme concentrations

* `level_complex_concs` takes `families`, the complexes of the model as `complex_families` returns them (default: calculated from the model), as GECKO's `levelComplexConcs` does.
* `level_complex_concs` and `level_complex_abundances` keep a subunit with a concentration of 0 at 0 instead of filling it with the level of its complex.
* `flexibilize_enz_concs` warns about enzymes with a concentration of 0, which it cannot raise, and does not report them as flexibilized.
* `flexibilize_enz_concs` with `per_complex=True` records the count of a complex on its first subunit with a concentration above 0.
* `load_prot_data` takes `zeros_as_measured`, as GECKO's `loadProtData` option `zerosAsMeasured`. By default an abundance of 0 is read as not measured (NaN) before any filter, also with `filter_data=False`, so the mean of a protein with a zero replicate leaves the zeros out.
* `load_prot_data` with `zeros_as_measured=True` counts a 0 as a measured replicate, keeps it in the mean and exempts it from `min_val` and `cut_lowest`.
* `get_isozyme_coverage` reports the isozyme groups of `model.ec.rxns` that have both a measured and an unmeasured entry, and the enzymes that `zero_unmeasured_isozymes` can limit, as GECKO's `getIsozymeCoverage`.
* `zero_unmeasured_isozymes` lowers the usage upper bound of the unmeasured isozymes of a reaction that has a measured isozyme, to 0 or to a detection limit (`method="detection_limit"`), as GECKO's `zeroUnmeasuredIsozymes`. It lowers the bounds in place and returns a DataFrame, and a lower bound above the new upper bound is lowered with it.
* `enzyme_usage` reports `measured_protein`, `demanded_protein` and `demand_ratio`, the usage of the enzymes with a measured concentration divided by their measured concentrations, and the same restricted to enzymes with a usage above 1e-7 mg/gDCW (`measured_protein_active`, `demand_ratio_active`).
* `report_enzyme_usage` reports `demand_ratio` and `demand_ratio_active`. GECKO's `getEnzymeUsage` and `reportEnzymeUsage` give the same values.

### Packaging and documentation

* geckopy is licensed under the GNU General Public License v3 (GPL-3.0), as RAVEN and raven-toolbox. `LICENSE`, the `license` field and the classifier of `pyproject.toml`, and the README state it. Releases up to 4.0.0b2 were published under the MIT licence.
* `geckopy.__version__` is the installed version of geckopy.
* `import geckopy` imports `cma`.
* `upgrade_project` (`geckopy upgrade`) reports the calls of removed or replaced functions in the project's scripts with their replacement, for example `submit_open_kinetics_predictor`, `fetch_open_kinetics_predictor`, `build_okp_input_csv` and `truncate_values`.
* `upgrade_project` (`geckopy upgrade`) reports each removed alias in a project's scripts with its replacement, which `geckopy.upgrade.RENAMED` lists.
* The user guides (evotune, YAML format, relaxation methods, migration from GECKO MATLAB, kcat sources) moved to [gecko-docs](https://gecko-docs.readthedocs.io/en/latest/).
* The `docs/` folder is gone; the developer notes are in `dev/`.

### Fixes

* `sensitivity_tuning` selects the first enzyme or reaction of the model when two have equal flux (within a relative 1e-9), as `sensitivityTuning` does (was: the first reaction of an unordered set, so a tie could give a different kcat than MATLAB).
* `get_conc_control_coeffs` raises `ValueError` when the solve of the model as given is not optimal (for example infeasible), as GECKO's `getConcControlCoeffs` raises `getConcControlCoeffs:infeasible` (was: no analysed proteins and zero coefficients).
* `ec_fseof` raises `ValueError` when the top of the scan is not above the flux of the production target at maximum growth by more than `1e-9 * max(1, |top|)`, also for a target that cannot carry flux; GECKO's `ecFSEOF` raises `ecFSEOF:noRange` in the same case (was: the scan ran on levels that do not rise and returned no targets).
* A `[kcat_sources.<name>]` table of model_adapter.toml overrides only the keys it sets of the preset's settings for that source, as in GECKO (was: it replaced them, so `[kcat_sources.brenda]` without `aggregation` under a median preset gathered BRENDA with the default aggregation, the maximum).
* `apply_complex_data` reports up to two proposed complexes for a reaction, the closest superset and the best partial match, as GECKO's `applyComplexData` does (was: the partial match only when there was no superset).
* `collect_kcats` returns the model with the decided kcats before `apply_kcat_constraints`, also for the kcats filled from isozymes, as GECKO's `collectKcats` does (was: the constraints of the isozyme-filled reactions were applied to the returned copy). `assign_kcats` gives the same model as in 4.0.0b2.
* `fit_sigma` raises `ValueError` for a target growth rate that is not above 0 (0, negative or NaN), before it changes the model, as GECKO's `sigmaFitter` raises `sigmaFitter:growthRate` (was: for 0 it fitted sigma to the smallest growth).
* `relax_proteomics_greedy` checks the growth after the last relaxation it is allowed: it returns a converged result when that relaxation reaches `minimal_growth`, and a result with `converged=False` and a warning when no candidate is left. GECKO's `relaxProteomicsGreedy` does the same.
* `relax_proteomics_greedy` raises `RuntimeError` only when `max_iterations` relaxations leave growth below the target and candidates remain (was: it raised after `max_iterations` relaxations without checking the last one).
* `get_complex_data` without a taxonomic ID (`taxonomic_id=None`) logs a warning and returns no complexes, without a query or a file written, as GECKO's `getComplexData` does (was: `ValueError`).
* `geckopy init` writes a `.gitignore` with `data/okpApiKey.txt` in the new project, and the geckopy repository ignores `okpApiKey.txt`, so the OpenKineticsPredictor API key file stays out of git (was: the help said the file was git-ignored, which was not the case).

## 4.0.0b2 (2026-11-02)

Second beta of geckopy 4. The main changes are a repaired and faster scipy solver, the ecModel format keys in the `metaData`, complex levelling with `relax_enz_concs`, and closer agreement with GECKO in many functions. Changes below are relative to 4.0.0b1.

### Breaking changes

* `get_enzyme_bottlenecks` ranks the enzymes by `objective_gain`, the change of the objective after the upper bound of one enzyme's usage reaction is raised by `fold_change` (default 2.0) and the model is solved again (was: the absolute shadow price of the `prot_<id>` constraint). At a degenerate optimum several capacity constraints are active and the solver assigns the dual to one of them, so the dual depends on the basis.
* `get_enzyme_bottlenecks` solves again only the enzymes at their bound (`upper_bound > 0` and `|flux| / upper_bound >= 1 - 1e-6`); every other enzyme has `objective_gain` 0. Ties are broken by the position in `model.enzymes`, so the order does not depend on the solver.
* The columns of `get_enzyme_bottlenecks` are `gene, objective_gain, shadow_price, flux, cap_usage, upper_bound, multiple_active`; `shadow_price` is a reference column and moved from position 2 to 3, so callers that read columns by position must use the column names.
* `get_enzyme_bottlenecks` sets `multiple_active` to True when another capacity constraint (an enzyme at its bound, or `prot_pool_exchange` at its upper bound) is active at the same optimum. `df.attrs["n_resolved"]` and `df.attrs["active_constraints"]` hold the number of solves and of active constraints.
* `get_enzyme_bottlenecks` warns when no enzyme is at its usage bound, as on ecYeastGEM where the protein pool is the active constraint, and the rows are in model order.
* The `targets` of `ec_fseof` lists the reaction targets first and then the transport targets (MATLAB's `rxnTargets` and `transportTargets`), each group sorted by score, and has a new boolean column `transport`. A reaction is a transport target when a metabolite name occurs more than once among its metabolites, not counting `prot_*`.
* The order of `targets` of `ec_fseof` changes the row order of `amplification`, `knockout` and `gene_targets`, which are derived from it, for callers that take the top N: transport targets come after all reaction targets.
* `set_kcat_for_reactions` follows `setKcatForReactions`. `apply` defaults to `False`, so only `ec.kcat` and `ec.source` change; pass `apply=True` to write the S matrix.
* `set_kcat_for_reactions` matches IDs without regard to letter case and ignores an ID that matches nothing. In a gecko-light model the `###_` prefix is part of the ID, so `"R2"` does not expand to `001_R2` and `002_R2`.
* `set_kcat_for_reactions` takes a kcat list with one value per matched row by position even when another ID matches nothing, as in MATLAB: `["R2", "nonsense"]` with `[1.0, 2.0]` sets the two `R2` rows to 1.0 and 2.0.
* `fill_kcats_from_isozymes` and `get_kcat_across_isozymes` default to `apply=False`, as `getKcatAcrossIsozymes` only changes `ec.kcat`; pass `apply=True` to rewrite the stoichiometry. `assign_kcats` passes `apply=True`.
* `fill_kcats_from_isozymes` fills only a kcat of 0: a NaN kcat is left as it is, and a NaN sibling does not enter the mean.
* `get_conc_control_coeffs` takes the control coefficient by finite difference by default, as GECKO's `getConcControlCoeffs` does: it raises one usage bound by `fold_change` (2), re-solves and divides the rise of the optimum by the rise of the bound.
* The reduced-cost path of `get_conc_control_coeffs`, which reads the dual of the usage reaction and is unreliable at a degenerate vertex, is `method="shadow_price"`.
* The finite-difference path of `get_conc_control_coeffs` needs one solve per protein, so each iteration of `flexibilize_enz_concs`, which calls it, takes minutes on a genome-scale ecModel.
* `fit_sigma` scans the 0.01 grid of GECKO's `sigmaFitter` by default (`method="bisect"` stays available and gives a continuous sigma).
* `save_ec_model` records the ecModel format (`gecko_format: "4"`) and the release that wrote the file (`generator: geckopy <version>`) in the `metaData`, the keys GECKO's `saveEcModel` writes (was: `geckopy_version`). The model's own metaData, its date included, is kept.
* `make_ec_model` refuses what GECKO's `makeEcModel` refuses, with a `ValueError`: a reaction ID that starts with `usage_prot_` or `prot_pool` or ends in `_REV` or `_EXP_<digits>`, a metabolite ID that starts with `prot_`, an `enzyme_comp` that is not one of the compartment names of the model (was: a compartment ID that differs from its name passed), and a model without genes.
* `make_ec_model` splits into a `_REV` pair only a reaction with `lb < 0 < ub` that is not an exchange reaction. A reaction with both bounds negative stays as it is (was: it raised).
* `make_ec_model` treats a reaction that only consumes or only produces metabolites, such as `A + B <=>`, as an exchange reaction, as in RAVEN's `getExchangeRxns`.

### ecModel files

* New: `load_ec_model` refuses a `gecko_format` above 4 and warns about a GECKO 3 ecModel with a message that names `upgrade_project`.
* `load_ec_model` turns the same protein reactions as GECKO's `loadEcModel`, including a usage reaction fixed to zero.
* `upgrade_project` reports an ecModel it cannot read as failed and goes on with the other steps.
* `save_ec_model` keeps the description of a model that has one (`metaData` of the file it was loaded from) and writes `Enzyme-constrained model of <id>` only when there is none (was: it overwrote the description). GECKO's `saveEcModel` does the same.
* `save_ec_model` writes an ec reaction without enzymes with no `enzymes` entry, as GECKO's `saveEcModel` does (was: `enzymes: !!omap []`).
* `load_ec_model` reads a missing `enzymes` entry as no enzymes, and still reads `enzymes: !!omap` with no entries and `enzymes: !!omap []` from files written by earlier versions of GECKO, RAVEN and geckopy.
* `load_ec_model` reads `- enzymes: !!omap` with no entries, as GECKO writes it for an ec reaction without enzymes (the `standard` rows); the row has no enzymes (was: `AttributeError: 'ScalarNode' object has no attribute 'flow_style'`).
* `load_ec_model` reads a value in double quotes that holds unescaped double quotes, such as the notes `""text""` of the published eciYali5-GEM, as `"text"`, with the outer pair of quotes removed and the inner ones kept (was: a parser error). RAVEN 2's `writeYAMLmodel` wraps a value in quotes without escaping the quotes in it, and GECKO 3 read the notes column of `customKcats.tsv` with `%s`, which keeps the quotes of the file.
* `load_ec_model` reads a file that the parser rejects again after these two forms are rewritten as YAML. It loads both forms with the values MATLAB loads.
* `load_conventional_gem` reads YAML with raven-toolbox's `read_yaml_model` (was: `cobra.io.load_yaml_model`), so RAVEN fields such as per-reaction `eccodes` are kept (was: `fill_eccodes_from_gem` found no EC numbers on RAVEN-format GEMs).
* `save_kcat_source` writes the free text of the `detail` column as `saveKcatSource` does: a tab or line break becomes a space, so a row cannot be split, and an empty part leaves no leading, trailing or repeated `; `.

### Solvers

* The scipy solver (optlang's HiGHS interface) gives the results of glpk and gurobi. optlang renumbers the variable columns wrongly after a variable is removed (it stops at `len(constraints)` instead of `len(variables)`), so cobra read fluxes from the wrong columns after `make_ec_model`, `Reaction.delete` or any other removal; `ec_fva` returned 0 for every reaction of ecTestGEM although the LP optimum was 3272.7.
* An `EcModel` repairs the scipy problem whenever `model.solver` is read; `geckopy.solver_support.repair_scipy_problem(model)` does the same for a plain `cobra.Model`.
* The repaired scipy problem stores the coefficients sparsely, because optlang copies a dense matrix on every added or removed variable or constraint.
* The repaired scipy problem keeps the solution when `cobra.Model.optimize` sets the objective direction again, so `Reaction.flux` and `Enzyme.flux` work after `model.optimize()`.
* After an infeasible solve, the scipy interface returns NaN values and objective, so `model.optimize(raise_error=False)` returns a solution with status `infeasible` (was: a `TypeError`).
* With scipy as cobra's default solver, `load_conventional_gem` reads the model on glpk and `make_ec_model` builds the ecModel on glpk, and both convert the result to scipy (`geckopy.solver_support.switch_to_scipy`) (was: building the yeast-GEM ecModel on the scipy interface took more than 40 minutes in optlang's dense matrix operations without finishing).
* `switch_to_scipy`, and so `make_ec_model`, `load_conventional_gem` and `load_ec_model` on scipy, create each constraint with its whole expression (was: setting the coefficients one at a time, which made the scipy interface remove and add the rows of a constraint for every call; the light Human-GEM ecModel took 2 h 17 min to build on scipy against 1.5 min on Gurobi).
* `apply_kcat_constraints` edits the model on glpk when it updates 100 or more ec reactions of a scipy model, for the same reason.
* `load_ec_model` reads the file on glpk and converts the ecModel to scipy when scipy is cobra's default solver, as `load_conventional_gem` does. Setting `model.solver = "scipy"` on an `EcModel` that is not on scipy builds the sparse problem of `geckopy.solver_support.switch_to_scipy`.
* Loading ecYeastGEM (8001 reactions) on scipy takes seconds (was: it did not finish in 25 minutes, as optlang's own clone grows the dense coefficient matrix row by row). The scenarios that load a genome-scale ecModel on scipy spent most of their hours there.
* `geckopy.solver_support.is_scipy_interface` tells whether a solver name or interface is scipy.
* A scipy `EcModel` is copied and pickled into the sparse problem (was: optlang pickled a model as JSON and restored it by adding the variables and constraints to a dense problem, so every worker process of `ec_fva`, `cobra.Model.copy` and `copy.deepcopy` spent tens of minutes rebuilding the matrix of a genome-scale ecModel).
* ecYeastGEM restores in 1.6 s and copies in 2.2 s. The solver of a repaired or converted scipy model is a `geckopy.solver_support._SparseModel`.
* `model.tolerance` reaches HiGHS on the scipy solver (was: optlang's scipy interface ignored every tolerance, so a model set to `1e-9`, as the Gurobi and GLPK runs of the parity scenarios are, was solved at HiGHS's `1e-7`, and the flux ranges of `ec_fva` on ecYeastGEM differed from Gurobi's by up to 0.2 percent).
* The feasibility tolerance becomes `primal_feasibility_tolerance` and the optimality tolerance `dual_feasibility_tolerance`; both stay at HiGHS's default until set, and survive a copy or a pickle of the model.
* A solve that HiGHS ends with numerical difficulties (status 4) at `1e-9` is solved again at its default, and a solve that still ends so has the status `numeric` (was: `KeyError`).
* `Enzyme.shadow_price`, `get_enzyme_bottlenecks` and `relax_proteomics_greedy` work on the scipy solver, which has no constraint duals in optlang: `geckopy.solver_support.constraint_shadow_prices` reads the HiGHS row marginals of the last solve and uses them only if they are an optimum of the problem as it is at that point.
* `relax_proteomics_greedy` computes the shadow prices of all candidates once per iteration and takes the lowest ID among enzymes with equal absolute shadow price.
* `get_conc_control_coeffs` takes the one-solve path with the scipy solver as well. optlang does not expose variable duals there, so the reduced cost is read from the HiGHS bound marginal of the last `linprog` result, after checking that the stored result is the solution `model.optimize` returned.
* On the tutorial yeast ecModel the one-solve path of `get_conc_control_coeffs` on scipy takes seconds (was: several minutes for the finite-difference loop). A scipy result without bound marginals falls back to the finite-difference loop.
* `geckopy.solver_support.optimize_model` returns the infeasible solution (status `infeasible`, NaN objective and fluxes) in the solves of the limit-protein, kcat-tuning and analysis functions that follow the starting-LP check.
* `get_subset_ec_model` works on the scipy solver (was: `LookupError: Constraint prot_P1 ... not in solver` at the next `optimize` when it had removed two or more constraints).
* `get_subset_ec_model` removes genes, reactions and metabolites on glpk and converts the subset back to scipy, because optlang rewrites every constraint expression for each removed variable (65 minutes for the HT29-GEM subset of the light Human-GEM ecModel). The sparse scipy model removes constraints correctly. optlang's scipy interface calls the generic removal inside its per-constraint loop, with the whole list each time, so the second pass finds the constraints gone.
* `geckopy.solver_support.edit_on_fast_solver` is the context manager that does this for any structural edit of a scipy model.

### Kcat sources

* `read_kcat_source` and `load_kcat_source` skip a row whose fields are all empty quoted fields (`""<TAB>""`), as they skip a row of tabs (was: the row was reported as invalid).
* `read_kcat_source` and `load_kcat_source` raise `ValueError` for a field that opens a double quote and does not close it (was: the row was read with a kcat that is not a number).
* `read_dlkcat_output` drops a row whose kcat is infinite (`inf`), as it drops `NA` and `None`, so the kcat list holds finite numbers only.
* `write_dlkcat_input` and `extract_enzyme_substrate_pairs` write the rows in the order of `model.ec.rxns` whatever the order of `ec_rxns`, and a reaction named twice once, as GECKO's `writeDLKcatInput` does with its logical vector.
* `ECTree` raises `ValueError` for `k = NaN`, as GECKO's `ecTreeEstimate` does (was: NaN estimates).
* `kcats_from_ec_tree` with an empty `ec_rxns` returns no kcats, as an empty selection means no reaction (was: it raised).
* `kcats_from_ec_tree` counts a reaction named twice once in the `ec_rxns` parameter of the source file.
* `assign_standard_kcat` leaves out every reaction without a substrate or without a product (RAVEN's `getExchangeRxns`), such as `A + B =>` (was: single-metabolite boundary reactions only).
* `assign_standard_kcat` compares the subsystem names of the reactions without regard to letter case, as MATLAB's `strcmpi` does: `Pentose phosphate pathway` and `Pentose Phosphate Pathway` are one subsystem, so their reactions count together toward the `threshold` and give one median.
* `assign_standard_kcat` gives a reaction with an empty subsystem the model-wide standard kcat.
* `review_assignment` recognises the source labels `custom` and `standard` without regard to letter case, as MATLAB's `reviewKcatAssignment` does: a source `Custom` is checked for a duplicate of a database value and `STANDARD` is the fallback.
* `review_assignment`, `EcStats` and `ec_stats_from_brenda` are exported from `geckopy.gather_kcats` and `geckopy`, and listed in `__all__`.
* `apply_custom_kcats` reads a reaction list in double quotes (`"R3,R5"`) without the quotes, as MATLAB's `%q` does, so the quoted list matches R3 and R5 (was: nothing).
* `find_met_smiles` skips a metabolite without a name with a logged warning and gives it no SMILES (was: it looked the metabolite up by its id).
* `find_met_smiles` uses the last row when the cache file holds a name more than once, as the docstring states.
* `fuzzy_kcat_matching` with several EC numbers for a reaction takes the EC matched at the earliest-tried level, using the same search order as within one EC (a correct-organism specific activity, origin 5, before any-organism any-substrate kcat, origin 4), matching `fuzzyKcatMatching.m` (was: the lower origin number won).
* The change in `fuzzy_kcat_matching` changes the kcat of multi-EC reactions; on the yeast-GEM tutorial the pre-tuning growth rate goes from 0.0013 to 0.12 /hour, in line with MATLAB.
* New: `geckopy.databases.download_phyl_dist` fetches the KEGG phylogenetic-distance file distributed with RAVEN.
* New: `databases.met_smiles` reads the SMILES annotation of a metabolite whether it is a string or a list.

### Kcat tuning

* `flexibilize_enz_concs` and `sensitivity_tuning` decide that the starting LP is infeasible from the solver status, not from the objective value, so they behave the same on every solver (was: an infeasible start raised on Gurobi, which has no objective value, but not on GLPK, which has a finite one).
* `flexibilize_enz_concs` returns an empty result for an infeasible start on all solvers, as MATLAB's `flexibilizeEnzConcs` does (was: it raised on Gurobi).
* `sensitivity_tuning` raises on an infeasible start on all solvers and stops with a warning when a later solve is not optimal.
* The starting-LP check of `flexibilize_enz_concs` and `sensitivity_tuning` reads the solver status after `slim_optimize`. Gurobi holds no primal values for an infeasible LP that it presolved and the scipy interface returns `None` for them, so `model.optimize()` raised `GurobiError: Unable to retrieve attribute 'X'` or `TypeError` there, and the checks that an LP is infeasible failed in a few percent of full test runs.
* `flexibilize_enz_concs` with a levelled complex scores a unit by its probes when the solver returns no usable reduced costs (the scipy solver) (was: a NaN score could pick the first unit).
* `get_conc_control_coeffs` keeps a reduced cost when raising the bound by 0.01 percent raises the optimum by at least half of the rise the reduced cost predicts, and by more than 1e-12 times max(1, |optimum|). A small coefficient on a small bound is kept (was: the test needed a rise above 1e-10, which dropped any coefficient with a reduced cost times bound below 1e-6).
* `get_conc_control_coeffs` reads the reduced cost of the forward variable of each usage reaction and keeps it where a 0.01 percent rise of the bound raises the optimum (was: the `prot_<id>` shadow price).
* An enzyme whose own bound does not bind gets 0 from `get_conc_control_coeffs` while the protein pool binds (was: every used enzyme got the pool's price), and a degenerate vertex gives no spurious coefficient.
* `get_conc_control_coeffs` checks every nonzero reduced cost, however small, and gives a usage reaction held at its lower bound 0 without a check.
* `get_conc_control_coeffs` uses the finite-difference loop where the solver exposes no variable duals, or returns NaN.
* `fit_sigma(method="bisect")` gives the first grid sigma, `1 / n_sigma_steps`, for a model that cannot grow even at sigma 1, as the grid does (was: about 1e-6).
* `fit_sigma` gives a target that growth cannot reach the first sigma of the growth plateau (was: 1.0).
* `fit_sigma` sets the bounds of the objective reaction to `[0, 1000]` before the scan, keeps them on the fitted model, and counts a solve that is not optimal as growth 0.
* `sensitivity_tuning` follows `sensitivityTuning.m`. The result lists the tuned reactions in model order.
* `sensitivity_tuning` reports the kcat and source of the input model (was: a reaction an earlier run had tuned reported NaN).
* `sensitivity_tuning` prints the pre-tune kcat in `ec.notes` as `num2str` does (`preTuneKcat=10`).
* `sensitivity_tuning` matches a `prot_to_ignore` id without regard to case and raises `ValueError` when no usage reaction has it. A tuned kcat has no upper limit.
* `sensitivity_tuning` tunes a gecko-light model (was: `NotImplementedError`).
* `truncate_values` rounds a value exactly halfway between two candidates away from zero, as MATLAB's `round(x, n)` does: 100000.5 gives 100001 and 1234565 gives 1234570 (was: 100000 and 1234560).

### Evotune

* `geckopy.kcat_tuning.evotune` exports `best_parsimonious`, `n_changed`, `parsimony_frontier` and `isozyme_tie_map`, the helpers behind the parsimony sweep and the tying of isozyme copies, which GECKO has as `bestParsimonious`, `nChanged`, `parsimonyFrontier` and `isozymeTieMap`.
* `best_parsimonious` skips points with a NaN `rmse`, as MATLAB's `bestParsimonious` does with `min(rmses)`. It still raises when every `rmse` is NaN (was: a NaN at the first point of the frontier made every comparison false and raised `ValueError`).
* `screen_kcat_leverage` keeps rows of equal rank in the order of the tie groups (a stable sort), as MATLAB's `screenKcatLeverage` does.
* `compute_excarbon` warns when no metabolite of the model has a formula (every exchange then gets the carbon weight 1).
* `cmaes_kcat_tuning` runs `max_generations` generations on a flat objective (was: it stopped after the first, as the flat-fitness criteria of `cma` do not end the search), as MATLAB's `cmaesKcatTuning` does.
* `classify_kcat_source` ignores the bracketed detail of `ec.source` when matching; a group that lists the full string still matches it.
* Evotune's trust tiers apply to a fuzzy-matched kcat (was: `ec.source` reads `"brenda (wc=0, origin=1)"` where MATLAB's reads `"brenda"`, and `classify_kcat_source` compared the whole string against the `source_groups` entries, so every fuzzy database match fell through to `unlabelled` and took `sigma0_log_default` instead of its tier's width, 3 497 of 4 671 kcats on the yeast tutorial). Tuning runs on models with fuzzy-matched kcats move by this.
* The docs describe `max_growth_weight` as MATLAB's `maxGrowthWeight` weights it (was: another description).

### Enzyme concentrations and usage

* `pfba_enzymes` with `fraction_of_optimum` below 1 lets the objective fall by that fraction of the magnitude of its optimum, for a negative optimum and for a minimised objective as for a maximised positive one, as MATLAB's `getPfbaEnzymes` does (was: `Infeasible` for these objectives).
* `pfba_enzymes` fixes the objective at 0.999999 of its optimum by default, as MATLAB's `getPfbaEnzymes` does through RAVEN's `solveLP`; a given `fraction_of_optimum` is multiplied by 0.999999 as well, so `fraction_of_optimum=0.9` fixes the objective at 0.8999991 of its optimum and the default 1.0 at 0.999999.
* `apply_kcat_constraints` takes cobra reaction ids as well as ec reaction ids (with the `###_` prefix) in `update_rxns` on a gecko-light model, as GECKO's `applyKcatConstraints` does for `model.rxns` ids; naming the cobra reaction updates it as a whole. An id that is both kinds or neither raises a `ValueError`.
* `apply_kcat_constraints` treats an empty `update_rxns` as all reactions, as MATLAB's `applyKcatConstraints` does with an empty `updateRxns` (was: it did nothing). The callers that build the list from a filter (`set_kcat_for_reactions`, the evotune scoring and reset) skip the call when the list is empty.
* `constrain_enz_concs` resets each processed usage reaction to a lower bound of 0 and a `prot_pool` coefficient of -1 before it sets the upper bound, as GECKO's `constrainEnzConcs` does (was: a prior lower bound above the new upper bound raised).
* `fill_enz_concs` raises a `ValueError` for a negative abundance of a protein that is an enzyme of the model, as GECKO's `fillEnzConcs` does with the error `fillEnzConcs:negativeAbundance` (was: it skipped the value with a warning).
* `constrain_enz_concs` raises a `ValueError` for a negative concentration in `model.ec.concs` before it changes any bound, as GECKO's `constrainEnzConcs` does with the error `constrainEnzConcs:negativeConcentration` (was: it skipped the enzyme with a warning). `remove_constraints=True` does not read the concentrations.
* `enzyme_usage` logs a warning that lists the enzymes without a `usage_prot_<enzyme>` reaction, which it leaves out of the result, as GECKO's `getEnzymeUsage` does with the warning `getEnzymeUsage:missingUsageReaction`.
* `report_enzyme_usage` raises a `ValueError` for a negative `top_abs_usage` (was: it listed all enzymes except the last ones, counted from the end of the ranking).
* `load_flux_data` raises a `ValueError` for a row with a different number of cells than the header, as GECKO's `loadFluxData` does with the error `loadFluxData:raggedRow` (was: it padded the missing cells with NaN).
* `load_flux_data` reads a header `X (EX_a(e))` as the metabolite `X` and the reaction id `EX_a(e)`: the reaction id is the text between the parenthesis that matches the final one and that one (was: a header with nested parentheses gave the whole header for both).

### ec_fseof

* `ec_fseof` ranks the targets by the score `|slope| * |correlation|` rounded to 9 significant digits, with scores that are equal after rounding in model order, as GECKO's `ecFSEOF` does (was: scores that differ by solver noise decided the order).
* `ec_fseof` follows `ecFSEOF.m`. The target flux is fixed at each level of the scan (`equality=True` of the raven-toolbox `fseof`, which `ecFSEOF.m` sets with `setParam(model, 'eq', ...)`) (was: lower-bounded).

### Subsets

* `get_subset_ec_model` removes the metabolites and genes that no reaction of the subset uses, as the `removeGenes` and `removeReactions` calls of GECKO's `getSubsetEcModel` do, also those that were unused in the big ecModel.
* The HT29-GEM subset of the light Human-GEM ecModel had the pseudo-gene `standard` and six metabolites (`MAM00579m`, `MAM01009c`, `MAM01099c`, `MAM01420m`, `MAM02549c`, `MAM02579m`) more than MATLAB's. A context model equal to the big model gives a copy without its unused metabolites and genes.
* `get_subset_ec_model` removes the genes of the big ecModel that the context model lacks with `cobra.manipulation.remove_genes`, so a reaction that survives the removal (an isozyme remains, as with the rule `G1 and G2 or G3` without `G3`) loses the gene from its rule.
* Removing a reaction later because the context model lacks it raised `ValueError: G3 not found` for the HT29-GEM subset of the light Human-GEM ecModel (was: the gene left `model.genes` while the rule still named it).
* In a gecko-light model, `get_subset_ec_model` matches the ec rows against the ec reaction id without the `001_` prefix (was: the subset dropped every row of `ec.rxns`, `kcat`, `source`, `notes`, `eccodes` and `rxn_enz_mat`). A context model equal to the big model gives a copy.

### Enzyme complexes

* New: `level_complex_concs` and `level_complex_abundances` give every enzyme complex one abundance, the median per copy over its measured subunits, so a single badly quantified subunit does not set a complex's capacity. See `docs/relaxation_methods.md`.
* `complex_families` groups isozyme variants into complexes and leaves shared subunits and electron carriers as measured.
* New: `relax_enz_concs` relaxes enzyme caps (and, if need be, the protein pool) by the least total fold increase that reaches the target growth, in one LP, so the result does not depend on the order in which limiting enzymes are found.
* `flexibilize_enz_concs` takes `per_complex=True`: complexes are ranked and raised as one unit, keeping levelled subunits levelled. It is off by default.

### Fixes

* `write_dlkcat_input` and the OpenKineticsPredictor input builder work on a model that has been saved and reloaded (was: `TypeError: cannot use 'list' as a set element`). cobra returns an annotation value as a list for a model read from a file, so `annotation['smiles']` is `"CCO"` on a model annotated in memory by `find_met_smiles` and `["CCO"]` on the same model after a round trip, and the ignore-list check hashed it.
* A reloaded model writes the same DLKcat input as a freshly annotated one.
* `parse_okp_output` works on a reloaded model, when indexing metabolites by their SMILES annotation to map each prediction back to an `ec.rxn` (was: `TypeError: cannot use 'list' as a dict key`).
* `load_uniprot_tsv` and `ModelAdapter.get_uniprot_ids_from_table` read the cells as MATLAB's `textscan` with `%q` does, which the docstring of `load_uniprot_tsv` already claimed: a cell in double quotes is given without the quotes (`"1.1.1.25; 2.5.1.19"` gives `1.1.1.25; 2.5.1.19`), `""` inside a quoted cell is one quote, tabs and line breaks inside quotes belong to the cell, and leading spaces of a cell are dropped.
* A line with fewer cells is padded with empty cells and the cells of a line beyond the last column start a new row, as `textscan` does.
* The conversion table maps a gene that is listed twice to its first row, as `ismember` does, and a gene on a line without a second cell to an empty UniProt ID. `geckopy.databases.uniprot_loader.read_quoted_tsv` does the splitting.

### Tutorials

* The full_ecModel tutorial downloads the full `PhylDist.mat` (was: a two-organism stub), uses KEGG as fallback for EC numbers, uses `merge_kcats`, runs ecFVA (STEP 73-75, skipped on GLPK) and the light versus full comparison (STEP 76-77, new `code/plot_light_vs_full.py`).

## 4.0.0b1 (2026-09-08)

First release published to PyPI. The numbering of geckopy is aligned with GECKO (MATLAB), which justifies the jump to 4.x. The main changes are the new evotune tuning, `review_assignment` and closer agreement with GECKO.

### Breaking changes

* `kcat_sensitivity_analysis` is renamed to `kcat_tuning`.
* `make_ec_model` does not split exchange reactions in the irreversibility stage, matching the `nonExchRxns` exclusion of `makeEcModel.m`.
* `set_kcat_for_reactions` writes `ec.source = "setKcatForReactions"`, MATLAB's actual value (was: `"manual"`).
* The `raven-toolbox` dependency is the PyPI release `>=3.0.0b1` (was: a `git+...@develop` pin). It carries the `convert_to_irreversible(..., rxns=...)` addition `make_ec_model` needs.

### Evotune and assignment review

* New: `kcat_tuning.evotune` fits `ec.kcat` to measured growth rates and exchange fluxes with CMA-ES (`screen_kcat_leverage`, `select_tunable_mask`, `cmaes_kcat_tuning`, `tune_prior_penalty_weight`). It is gated behind the new `evotune` extra.
* `EvotuneResult.corrections()` reports a tuning run as a ranked table of what changed and why (was: a raw kcat vector).
* New: `gather_kcats.review.review_assignment` flags kcat-assignment problems (repeated values, EC-maximum fallback, implausible magnitude) that need curating before a tuning run.

### Kcat sources

* `set_kcat_for_reactions` accepts a per-isozyme kcat list against an un-suffixed base name, assigned positionally, matching MATLAB.
* `assign_standard_kcat` does not count a NaN-kcat reaction toward a subsystem's threshold eligibility and does not fold it into the subsystem median. Unset kcats are filled for NaN too, not just literal zeros.
* `read_dlkcat_output` takes an opt-in `strict=True` that fails fast on unrecognized substrates, matching MATLAB, and reports drop reasons separately (was: lumped together).
* `parse_okp_output` tags provenance with the `OKP-` prefix MATLAB applies; `merge_kcats` strips it again before source-priority matching, so tagged rows rank correctly.

### Fixes

* `flexibilize_enz_concs` uses MATLAB's soft ±0.25% growth band in the post-loop refinement, matching `flexibilizeEnzConcs.m` (was: pinning to `exp_growth` exactly).
* `get_reactions_from_enzyme` matches `protein_id` case-insensitively, matching MATLAB's `strcmpi`.
* `ec_fseof` anchors the scan floor to the flux of the production target at the biomass optimum (was: a fixed fraction of the theoretical maximum), matching `ecFSEOF.m`.
* `ec_fseof` restricts the candidates to gene-associated, non-standard reactions before classification, matching `ecFSEOF.m`.
* `ec_fva` uses `cobra.util.process_pool.ProcessPool` in its parallel path (was: a hand-rolled `multiprocessing.Pool`). This fixes a slow Windows initializer handoff and gives graceful pool teardown. The public signature is unchanged.

### Packaging

* The sdist excludes the local `.venv-win` virtualenv (was: it bundled about 60MB, including a compiled scipy wheel, because `.gitignore` listed `.venv/` but not `.venv-win/`, and the default sdist selection of hatchling only reads the top-level `.gitignore`).
* `release.yml` enables the PyPI publish job (trusted publishing).
* The README disambiguation note is simplified and points at the specific old `geckopy` release (`2.0.2` and earlier), as this project has taken over the PyPI project name itself.

## 0.3.0 (2026-08-30)

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

## 0.2.1 (2026-07-16)

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

## 0.2.0 (2026-06-14)

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

## 0.1.0a3 (2026-05-30)

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

## 0.1.0a2 (2026-05-30)

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

## 0.1.0a1 (2026-05-30)

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
  PubChem SMILES property, perf hoists, tuning-parameter length validation,
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
