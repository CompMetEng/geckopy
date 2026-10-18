# Migrating from GECKO 4 (MATLAB) to geckopy

geckopy is the Python port of the GECKO toolbox. It builds the same
enzyme-constrained models, follows the same algorithms, and writes the
same on-disk format as **GECKO 4** (MATLAB). This guide is for someone
who knows GECKO 4 in MATLAB and wants to do the same work in Python.

The port's goal is **behavioural and on-disk fidelity**: same reactions,
metabolites, coefficients, kcats, YAML output. Differences come from
being idiomatic Python and from using cobrapy instead of RAVEN. None of
them require you to change your model files.

> **Coming from GECKO 3?** GECKO 4 introduced a handful of changes
> (forward-direction protein reactions, several bug fixes, a reformatted
> BRENDA snapshot, the REST-based OpenKineticsPredictor flow) that
> geckopy also follows — see
> [GECKO 3 to GECKO 4](https://gecko-docs.readthedocs.io/en/latest/gecko3-to-gecko4.html)
> for the full list. Throughout this guide, individual functions that
> are affected by those 3→4 changes carry a **`[3→4]`** marker; that
> marker is geckopy's signal that the function's behaviour or schema
> differs from what GECKO 3 did, not a Python-vs-MATLAB difference.

---

## 1. At a glance

|                                 | GECKO 4 (MATLAB)                       | geckopy                                                 |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| Model toolbox                   | RAVEN + COBRA Toolbox                  | cobrapy + raven-toolbox                                  |
| Model object                    | RAVEN struct + `ec` field              | `EcModel(cobra.Model)` with an `.ec` dataclass          |
| Per-entity classes              | none — parallel cell arrays            | `cobra.{Metabolite,Reaction,Gene}` instances            |
| Per-enzyme accessor             | indexing into `model.ec.enzymes{i}`    | `Enzyme` proxy: `model.enzymes.get_by_id("P00350")`     |
| Adapter                         | `ModelAdapter` classdef + `ModelAdapterManager` | `ModelAdapter` + `model_adapter.toml` (pydantic-validated) |
| Default adapter                 | global `ModelAdapterManager.getDefault()` | none; pass `adapter=` or set `model.adapter`         |
| Return style                    | `[a,b,c] = f(...)`                     | single return; model mutated in place                   |
| Naming                          | `camelCase`                            | `snake_case`                                            |
| Indexing                        | 1-based                                | 0-based                                                 |
| Missing data                    | often a silent default                 | usually a raised exception                              |
| Tabular output                  | cell arrays / structs                  | `pandas.DataFrame`                                      |
| HTTP (BRENDA/KEGG/UniProt/OKP)  | `webread`/`webwrite`                   | `requests`                                              |
| On-disk format                  | RAVEN/cobrapy YAML (+ SBML)            | RAVEN/cobrapy YAML (SBML ecModel I/O dropped in `0.1.0a2`) |

The two toolboxes exchange ecModels directly: a model saved by either
side loads in the other (see [§7](#7-file-exchange)).

---

## 2. The model object

GECKO MATLAB uses a RAVEN model struct with parallel cell arrays plus a
`model.ec` struct and side cells (`model.metSmiles`, `model.eccodes`).
The struct itself isn't a class — everything is an array, indexed by
position.

geckopy's `EcModel` **subclasses** `cobra.Model`. cobrapy is
class-based: `cobra.Metabolite`, `cobra.Reaction`, `cobra.Gene` are
real classes, so `model.metabolites[17]` returns a `Metabolite`
instance (with `.id`, `.name`, `.formula`, `.annotation`, …), not a
string.

For the GECKO-specific data, geckopy keeps the **parallel-arrays
layout** rather than a cobrapy-style per-enzyme class, because the
algorithms (`apply_kcat_constraints`, sensitivity tuning) are bulk
vector operations on `kcat` / `mw` and the sparse coupling matrix —
those are several orders of magnitude faster as numpy arrays than as a
loop over N enzyme objects. The layout is in a dataclass on
`model.ec`:

```python
model.ec.rxns          # list[str]                  parallel to ec.kcat etc.
model.ec.kcat          # np.ndarray (per-rxn)       — 0 means "unset"
model.ec.source        # list[str]                  per-rxn provenance tag
model.ec.notes         # list[str]                  per-rxn free-text
model.ec.eccodes       # list[str]                  ';' -joined per-rxn
model.ec.enzymes       # list[str]                  per-enzyme accessions
model.ec.genes         # list[str]                  per-enzyme gene name
model.ec.mw            # np.ndarray                 per-enzyme; NaN == "unknown"
model.ec.sequence      # list[str]                  per-enzyme
model.ec.concs         # np.ndarray                 per-enzyme proteomics; NaN == "not measured"
model.ec.rxn_enz_mat   # scipy.sparse.csr_matrix    shape (n_rxns, n_enzymes), subunit counts
model.ec.gecko_light   # bool
```

For ergonomic per-enzyme access, geckopy adds a separate **proxy
class** `Enzyme` reached via `model.enzymes.get_by_id("P00350")`. The
proxy holds no state of its own; it forwards reads/writes into the
`model.ec.*` arrays. This gives you cobrapy-style code
(`enz.kcats["R_FOO"] = 30.0`) without paying for N real objects. See
the per-table explanation in §5 for which entry-points read it.

The `EcData` dataclass itself is owned by **raven-toolbox**
(`raven_toolbox.io.EcData`); geckopy re-exports it as `geckopy.EcData`
so existing imports keep working. Mirrors MATLAB RAVEN's split:
`model.ec` is RAVEN-owned; GECKO is just a consumer.

Other side-cells move into cobrapy's `annotation` dict:

| MATLAB                                                            | geckopy                                                         |
| ----------------------------------------------------------------- | --------------------------------------------------------------- |
| `model.metSmiles{i}`                                              | `metabolite.annotation['smiles']`                               |
| `model.eccodes{i}`                                                | `reaction.annotation['ec-code']` (`list[str]`)                  |
| `model.ec.rxnEnzMat`                                              | `model.ec.rxn_enz_mat` (scipy sparse)                           |
| `model.ec.kcat`, `.mw`, `.sequence`, `.source`, `.notes`, `.concs` | `model.ec.kcat`, `.mw`, `.sequence`, `.source`, `.notes`, `.concs` |
| `model.grRules` (strings)                                         | cobrapy `GPR` objects (parsed AST)                              |

Loaders return dataclasses / DataFrames rather than structs: e.g.
`BrendaData`, `ProtData`, `FluxData`, `UniprotDB`, `PhylDist`. Helpers
that produce tables (`fuzzyKcatMatching`, `ecFVA`,
`getReactionsFromEnzyme`, `enzymeUsage`, `reportEnzymeUsage`,
`mapRxnsToConv`) return `pandas.DataFrame`.

---

## 3. Adapters and configuration

In MATLAB you write an adapter `classdef` and register a default with
`ModelAdapterManager.setDefault`. In geckopy you create an adapter
folder with a `model_adapter.toml`, loaded via
`ModelAdapter.from_folder(path)`. The TOML is parsed into pydantic
models (`ModelParameters` with nested `kegg`, `uniprot`, `okp`,
`evotune`, …) and validated strictly, so config typos are
caught early.

There is **no global default adapter**. Functions read `model.adapter`
or take an explicit `adapter=` argument. Where a MATLAB function takes
a `modelAdapter` only to derive a path, the geckopy version takes the
path (or pre-loaded data) directly. The pattern is:

```python
adapter = ModelAdapter.from_folder("my_project")    # has params.path, params.bio_rxn, ...
model   = make_ec_model(conv_gem, adapter)          # adapter stored as model.adapter
# downstream functions read model.adapter implicitly:
apply_complex_data(model)
# ...or take an explicit override:
apply_complex_data(model, path=Path("/custom/complex.json"))
```

**Secrets** (e.g. an OpenKineticsPredictor API key) are never stored
in the adapter. Provide them via a function argument, an environment
variable (`OKP_API_KEY`), or a git-ignored file in the project
`data/` folder (`data/okpApiKey.txt`).

---

## 4. Calling conventions

### Mutate in place, return one thing

MATLAB functions return several outputs including a copy of the model.
geckopy mutates the model in place and returns at most one value;
diagnostics that MATLAB returned as extra outputs go to the logger and
to `rxn.notes['geckopy_warning']`.

| MATLAB                                                     | geckopy                                                                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `[model, noUniprot] = makeEcModel(...)`                    | `make_ec_model(...)` mutates; logs unmatched genes, annotates `rxn.notes['geckopy_warning']` |
| `[model, rxnUpdated, notMatch] = applyCustomKcats(...)`    | `apply_custom_kcats(...)` returns `None`; logs the rows                                  |
| `[model, ...] = getStandardKcat(...)`                      | `assign_standard_kcat(...)` mutates; diagnostics via logging                             |
| `[minFlux, maxFlux] = ecFVA(...)`                          | `ec_fva(...)` returns a `DataFrame` indexed by reaction id                               |
| `[rxns, kcat, name, gpr, idx] = getReactionsFromEnzyme(...)` | `get_reactions_from_enzyme(...)` returns one `DataFrame`                               |
| `[model, kcatList] = sensitivityTuning(...)`               | `sensitivity_tuning(...)` mutates; returns a `TunedKcatsResult` dataclass               |
| `[mappedFlux, enzUsageFlux, usageEnz] = mapRxnsToConv(...)` | `map_rxns_to_conv(...)` returns one `MapRxnsResult` dataclass                          |

### Errors instead of silent defaults

Where MATLAB silently substitutes a default, geckopy tends to raise so
problems surface early — e.g. missing proteome data raises
`FileNotFoundError` instead of returning an f-factor of `0.5`; an
unknown protein id raises `ValueError` instead of returning empty.

### Names and indexing

`camelCase` → `snake_case` throughout. **Most renames are mechanical**
(`makeEcModel → make_ec_model`); a handful are intentional re-naming
for clarity (kept reachable as deprecated aliases — `from geckopy
import old_name` still works for one minor cycle):

| MATLAB                  | geckopy (canonical)            | Deprecated alias              |
| ----------------------- | ------------------------------ | ----------------------------- |
| `selectKcatValue`       | `apply_kcat_list`              | `select_kcat_value`           |
| `getKcatAcrossIsozymes` | `fill_kcats_from_isozymes`     | `get_kcat_across_isozymes`    |
| `getStandardKcat`       | `assign_standard_kcat`         | `get_standard_kcat`           |
| `sigmaFitter`           | `fit_sigma`                    | `sigma_fitter`                |
| `getECfromDatabase`     | `fill_eccodes_from_database`   | `get_ec_from_database`        |
| `getECfromGEM`          | `fill_eccodes_from_gem`        | `get_ec_from_gem`             |
| `constrainFluxData`     | `apply_flux_data_constraints`  | `constrain_flux_data`         |
| `mergeDLKcatAndFuzzyKcats` | `merge_kcats` (generalised) | `merge_dlkcat_and_fuzzy_kcats` |

MATLAB is 1-indexed; geckopy is 0-indexed, which shows up in arguments
like `data_col` (default `0`) and `condition` (in `apply_flux_data_constraints`).

### gecko-light

Implemented end-to-end. `make_ec_model(model, adapter, gecko_light=True)`
produces a working light ecModel; `apply_kcat_constraints`,
`set_kcat_for_reactions`, and the `Enzyme` proxy's read paths all
handle the light layout. The per-enzyme prot-pool analyses (proteomics
integration via `constrain_enz_concs`, `flexibilize_enz_concs`,
`pfba_enzymes`) and the isozyme-aggregating helpers
(`fill_kcats_from_isozymes`) raise `NotImplementedError` on a light
model with a clear message — they have no meaning in the light layout
by design. See [gecko_light_status.md](gecko_light_status.md) for the
full availability table and a worked example.

---

## 5. Function correspondence

Each row maps a MATLAB function to its geckopy equivalent. The
**Notes** column flags

- **`[3→4]`** — function affected by GECKO 3→4 changes (direction
  flip, bug fix, schema change). Behaviour matches GECKO 4, not
  GECKO 3.
- **`[Py]`** — geckopy-specific divergence from MATLAB GECKO 4 (a
  port choice, not something flowing back to MATLAB).
- **(new)** — geckopy entry-point with no direct MATLAB counterpart.

### 5.1 Build / edit the ecModel (`src/geckomat/change_model`)

| MATLAB                       | geckopy                       | Notes                                                                                                                                                                          |
| ---------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `makeEcModel`                | `make_ec_model`               | **`[3→4]`** ec.genes/enzymes/mw/sequence are sorted alphabetically for stable order; KEGG fallback for unmatched genes is opt-in via `kegg_db=`. **`[Py]`** Single return; unmatched genes are logged and annotated on `rxn.notes['geckopy_warning']` instead of being returned. |
| `applyComplexData`           | `apply_complex_data`          | —                                                                                                                                                                              |
| `applyCustomKcats`           | `apply_custom_kcats`          | **`[Py]`** Mutates; diagnostics via logging. Takes a pre-loaded DataFrame, or reads `<adapter.path>/data/customKcats.tsv` by default.                                          |
| `applyKcatConstraints`       | `apply_kcat_constraints`      | **`[3→4]`** Writes coefficients with the forward-direction `usage_prot_*` / `prot_pool_exchange` convention.                                                                  |
| `findMetSmiles`              | `find_met_smiles`             | **`[Py]`** Asynchronous PubChem cache via `requests`; uses a TSV cache (`data/smilesDB.tsv`) so re-runs are offline.                                                          |
| `getComplexData`             | `get_complex_data`            | —                                                                                                                                                                              |
| `getKcatAcrossIsozymes`      | `fill_kcats_from_isozymes`    | **`[Py]`** Renamed; `get_kcat_across_isozymes` kept as deprecated alias. Aggregation strategy (`max` / `median` / `mean`) is adapter-configurable via `kcat_aggregate_isozymes`. |
| `getReactionsFromEnzyme`     | `get_reactions_from_enzyme`   | **`[Py]`** Returns one `DataFrame` instead of five parallel cells.                                                                                                            |
| `setKcatForReactions`        | `set_kcat_for_reactions`      | **`[Py]`** Recognises both `_EXP_<N>` (full) and `###_` (light) suffixes; understands base names that broadcast across isozymes.                                              |
| `setProtPoolSize`            | `set_prot_pool_size`          | **`[3→4]`** Sets the **upper** bound of `prot_pool_exchange` (forward direction).                                                                                              |

### 5.2 Gather kcats (`src/geckomat/gather_kcats`)

| MATLAB                              | geckopy                              | Notes                                                                                                                                                                                                                            |
| ----------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fuzzyKcatMatching`                 | `fuzzy_kcat_matching`                | **`[3→4]`** Wildcard escalation no longer escalates past the fully-wildcarded form. **`[Py]`** Returns a `DataFrame` (rows: ec / substrate / organism / kcat / source / wildcard\_level …) instead of a struct; takes pre-loaded `BrendaData` + `PhylDist`. |
| `getStandardKcat`                   | `assign_standard_kcat`               | **`[3→4]`** Applies per-subsystem mean kcat (GECKO 3 fell back to the global standard kcat unless every subsystem matched). **`[Py]`** Renamed; `get_standard_kcat` kept as deprecated alias.                                  |
| `mergeDLKcatAndFuzzyKcats`          | `merge_kcats`                        | **`[Py]`** Generalised to accept any combination of BRENDA / DLKcat / OpenKineticsPredictor / manual rows; records per-row provenance. `merge_dlkcat_and_fuzzy_kcats` kept as deprecated alias.                                |
| `readDLKcatOutput`                  | `read_dlkcat_output`                 | **`[3→4]`** Case-insensitive substrate-name matching.                                                                                                                                                                          |
| `removeStandardKcat`                | `remove_standard_kcat`               | —                                                                                                                                                                                                                                |
| `runDLKcat`                         | `run_dlkcat`                         | —                                                                                                                                                                                                                                |
| `selectKcatValue`                   | `apply_kcat_list`                    | **`[Py]`** Renamed; `select_kcat_value` kept as deprecated alias.                                                                                                                                                              |
| `writeDLKcatInput`                  | `write_dlkcat_input`                 | **`[3→4]`** Bounds-safe when `ec_rxns` is a subset.                                                                                                                                                                            |
| `submitOpenKineticsPredictor`       | `submit_open_kinetics_predictor`     | **`[3→4]`** New REST-based submit (replaced `writeOpenKineticsPredictorInput`). API key via argument / `OKP_API_KEY` env / `data/okpApiKey.txt`.                                                                                |
| `fetchOpenKineticsPredictor`        | `fetch_open_kinetics_predictor`      | **`[3→4]`** New REST-based fetch (replaced `readOpenKineticsPredictorOutput`); per-row `kcatSource` reports actual provenance (CataPro / BRENDA / Sabio-RK).                                                                  |
| `writeOpenKineticsPredictorInput`   | *(removed)*                          | **`[3→4]`** Replaced by `submit_open_kinetics_predictor`.                                                                                                                                                                      |
| `readOpenKineticsPredictorOutput`   | *(removed)*                          | **`[3→4]`** Replaced by `fetch_open_kinetics_predictor`.                                                                                                                                                                       |
| `kcatsFromBrenda`                   | `kcats_from_brenda`                  | Fuzzy BRENDA matching, kept as `data/kcats/brenda.tsv`. Loads the BRENDA data and KEGG distances from the adapter unless passed in. |
| `kcatsFromDLKcat`                   | `kcats_from_dlkcat`                  | DLKcat (run in Docker with `run=True`, or an earlier output), kept as `data/kcats/dlkcat.tsv`. |
| `kcatsFromOKP`                      | `kcats_from_okp`                     | One OpenKineticsPredictor method, kept as `data/kcats/okp_<method>.tsv`; submits a job when there is none, writes the file once it is done. Returns `(kcat_list, done)`. |
| `saveKcatSource`, `loadKcatSource` | `save_kcat_source`, `load_kcat_source`, `read_kcat_source` | Kcat source files: one per source, keyed by conventional reaction and enzyme set, the same file format in both tools. `load_kcat_source` maps a file to the current model and reports rows that fit no reaction. **`[Py]`** `read_kcat_source` reads a file without a model; GECKO gives the same in `loadKcatSource`'s `report.header` and `report.records`. |
| `getEcRxnKeys`                      | `ec_rxn_keys`                        | Conventional reaction and enzyme set per `ec.rxns` entry. |
| `kcatsFromECTree`                   | `kcats_from_ec_tree`                 | EC median of the BRENDA measurement table, shrunk toward the EC's 3- and 2-digit classes (weight n / (n + k)), kept as `data/kcats/ec_tree.tsv`. |
| `kcatsFromHomologs`                 | `kcats_from_homologs`                | Identity-weighted mean of up to 10 measured homologs (DIAMOND, ≥ 40% identity, 80% coverage) on the reaction's substrate, kept as `data/kcats/homologs.tsv`. |
| `extractEnzymeSubstratePairs`       | `extract_enzyme_substrate_pairs`     | The enzyme–substrate pairs DLKcat and the homolog search use; geckopy also returns `met_id`. |
| `ecTreeEstimate`                    | `ECTree.estimate` (`geckopy.gather_kcats.ec_tree`) | The EC-tree value for one or more EC numbers. |
| `loadBrendaMeasurements`            | `load_brenda_measurements`           | The BRENDA measurement table (`measurements.tsv`): wild-type kcats with an EC number, after the outlier screen. |
| `fetchBrendaAsset`                  | `fetch_brenda_asset`                 | Downloads (once) the BRENDA files of the toolbox-assets release `brenda-2026.1-1` and returns their folder; used when a project has no `data/brenda`. |
| `convertCustomKcats`                | `convert_custom_kcats`               | `data/customKcats.tsv` to the curated source file `data/kcats/custom.tsv`, with the kcats `apply_custom_kcats` gives the ecModel. |
| `upgradeEcModel`                    | `upgrade_ec_model`                   | Turns GECKO 3 protein usage and pool reactions forward; `load_ec_model` already does this for the files it reads. |
| `upgradeGECKOproject`               | `upgrade_project` (`geckopy upgrade`)| Lists, or carries out, the steps from a GECKO 3 project to the GECKO 4 layout. The adapter-class step is MATLAB only. |
| `collectKcats`                      | `collect_kcats`                      | Decides one kcat per reaction from the source files, as `[kcat_assignment]` in model_adapter.toml sets (preset `brenda_ec_tree`, the default, `gecko_median` or `gecko_classic`, or stages), without changing the model. Writes `output/<model id>_kcatDecisions.tsv`. |
| `assignKcats`                       | `assign_kcats`                       | `collect_kcats`, then sets `ec.kcat` and `ec.source` and applies the kcat constraints; replaces `merge_kcats` → `apply_kcat_list` → `apply_custom_kcats` → `fill_kcats_from_isozymes` → `assign_standard_kcat` → `apply_kcat_constraints`. In place. |
| `updateKcatSources`                 | `update_kcat_sources`                | Gathers the source files the configuration needs that are missing, or outdated against `[kcat_sources.<name>]`; `dry_run` only reports. |

### 5.3 Get enzyme data (`src/geckomat/get_enzyme_data`)

| MATLAB                | geckopy                                                                | Notes                                                                                                                                                                                                                                      |
| --------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `calculateMW`         | `calculate_mw`                                                         | **`[Py]`** Stand-alone function (takes a sequence string, returns the MW); no model argument needed.                                                                                                                                       |
| `copyECtoGEM`         | `copy_ec_to_gem`                                                       | —                                                                                                                                                                                                                                          |
| `downloadKEGG`        | `geckopy.databases.kegg_download.download_kegg`                        | **`[3→4]`** Standalone helper (was split out in 4). **`[Py]`** Per-gene REST cache; `geckopy brenda-refresh` and friends drive it lazily.                                                                                                  |
| `downloadUniProt`     | `geckopy.databases.uniprot_download.download_uniprot`                  | **`[3→4]`** Standalone helper (was split out in 4).                                                                                                                                                                                        |
| `findECInDB`          | `find_ec_in_db`                                                        | **`[3→4]`** No duplicate EC codes emitted.                                                                                                                                                                                                |
| `getECfromDatabase`   | `fill_eccodes_from_database`                                           | **`[3→4]`** Can consult KEGG when UniProt's EC field is empty or ends with `-` (opt-in via `kegg_db=`). **`[Py]`** Renamed; `get_ec_from_database` kept as deprecated alias.                                                              |
| `getECfromGEM`        | `fill_eccodes_from_gem`                                                | **`[3→4]`** Actually assigns EC codes (the GECKO 3 validation regex silently discarded every EC string). **`[Py]`** Renamed; `get_ec_from_gem` kept as deprecated alias.                                                                  |
| `getECstring`         | *(internal helper inside `fill_eccodes_*`)*                            | Not exposed as a public entry-point — folded into the EC-fill helpers.                                                                                                                                                                     |
| `loadBRENDAdata`      | `load_brenda_data`                                                     | **`[3→4]`** Reads the new TSV schema (`kcat.tsv` / `sa.tsv` / `mw.tsv`, bare EC codes, plain organism names, `references` column). **`[Py]`** Ships **both** `max` and `median` per (ec, substrate, organism) triple — see [kcat_aggregation.md](kcat_aggregation.md). Returns a `BrendaData` dataclass. |
| `loadDatabases`       | *(implicit)*                                                           | **`[Py]`** No combined-load entry-point; users call `load_brenda_data`, `load_uniprot_tsv`, etc. directly.                                                                                                                                |

### 5.4 Kcat sensitivity analysis and tuning (`src/kcat_tuning`)

| MATLAB                                | geckopy                                                            | Notes                                                                                                                                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sensitivityTuning`                   | `sensitivity_tuning`                                               | **`[Py]`** Returns a `TunedKcatsResult` dataclass (kcat changes table + final flux).                                                                                                                                              |
| `sigmaFitter`                         | `fit_sigma`                                                        | **`[3→4]`** Returns the model fitted to the **optimal** sigma (GECKO 3 returned the last trial value, sigma = 1.0). **`[Py]`** Renamed; `sigma_fitter` kept as deprecated alias. Returns a `SigmaFitterResult`.                  |
| `findMaxValue`                        | `find_max_value`                                                   | **`[3→4]`** Wildcard EC branch matches real codes.                                                                                                                                                                                |
| `truncateValues`                      | `truncate_values`                                                  | —                                                                                                                                                                                                                                |
| *(none)*                              | `kcat_tuning.evotune.cmaes_kcat_tuning` + helpers      | **`[Py]`** Fits kcats to experimental flux/growth data with CMA-ES rather than MATLAB's ABC-SMC `bayesianSensitivityTuning`; see [evotune_kcat_tuning.md](evotune_kcat_tuning.md) and [cmaes_vs_abc_smc.md](cmaes_vs_abc_smc.md). |
| `bayesianSensitivityTuning` + helpers | *(not ported)*                                                     | MATLAB's ABC-SMC sampler itself is not ported; geckopy uses CMA-ES instead (see above).                                                                                                                                          |

### 5.5 Limit proteins (`src/geckomat/limit_proteins`)

| MATLAB                | geckopy                              | Notes                                                                                                                                                                                                                                                                                                          |
| --------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `calculateFfactor`    | `calculate_f_factor`                 | **`[Py]`** Raises `FileNotFoundError` when no proteome data is provided (MATLAB silently returns `0.5`).                                                                                                                                                                                                       |
| `constrainEnzConcs`   | `constrain_enz_concs`                | **`[3→4]`** Sets the **upper** bound of `usage_prot_*` (forward direction). **`[Py]`** Raises `NotImplementedError` on gecko-light (no per-enzyme constraint to set).                                                                                                                                          |
| `constrainFluxData`   | `apply_flux_data_constraints`        | **`[Py]`** Renamed; `constrain_flux_data` kept as deprecated alias. Takes pre-loaded `FluxData`. 0-indexed `condition` (default `0`, MATLAB `1`).                                                                                                                                                              |
| `fillEnzConcs`        | `fill_enz_concs`                     | —                                                                                                                                                                                                                                                                                                              |
| `flexibilizeEnzConcs` | `flexibilize_enz_concs`              | **`[Py]`** Returns a `FlexEnzResult` dataclass. Raises `NotImplementedError` on gecko-light.                                                                                                                                                                                                                  |
| `getConcControlCoeffs` | `get_conc_control_coeffs`            | **`[3→4]`** Probe direction reads the forward `usage_prot_*` upper bound. **`[Py]`** Default path uses the prot-met shadow price (one LP solve, *n* times fewer LPs than the finite-difference probe); falls back to finite-difference when the solver doesn't expose LP duals. Gates on `sol.status == "optimal"`. |
| `updateProtPool`      | *(not ported)*                       | Obsolete since GECKO 3.2.0. Use `set_prot_pool_size` instead.                                                                                                                                                                                                                                                  |
| —                     | `relax_proteomics_greedy` *(new)*    | **`[Py]`** Alternative to `flexibilize_enz_concs`: relaxes a fixed top-*k* of measured concentrations in a greedy loop until a target growth is reached. Returns a `GreedyRelaxResult`.                                                                                                                       |

### 5.6 Utilities (`src/geckomat/utilities`)

| MATLAB                | geckopy                                | Notes                                                                                                                                                                                                                                                                                                                                                          |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `addNewRxnsToEC`      | `add_new_rxns_to_ec`                   | **`[Py]`** Returns an `AddNewRxnsResult` dataclass.                                                                                                                                                                                                                                                                                                            |
| `ecFSEOF`             | `ec_fseof`                             | **`[Py]`** Thin wrapper over `raven_toolbox.analysis.fseof`. Selection method is regression-based (`|correlation| ≥ threshold` + `|slope| ≥ eps`) with pFBA per step, replacing MATLAB's strict-monotonicity + top-25 %-by-slope filter. Result type is raven's `FSEOFResult` (replaces `EcFseofResult`). Action labels are `amplify` / `knockdown` / `knockout` (replaces `OE` / `KD` / `KO`). The per-gene `essentiality` column is dropped — use `cobra.flux_analysis.single_gene_deletion`. |
| `ecFVA`               | `ec_fva`                               | **`[Py]`** Returns a `DataFrame` indexed by reaction id with `minimum` / `maximum` columns; can run in parallel (`n_jobs=`).                                                                                                                                                                                                                                  |
| `enzymeUsage`         | `enzyme_usage`                         | **`[3→4]`** Usage flux read from positive `usage_prot_*` flux. **`[Py]`** Returns an `EnzymeUsageResult` dataclass (`per_enzyme` DataFrame + summary scalars).                                                                                                                                                                                                |
| `findGECKOroot`       | *(not needed)*                         | Python: `pathlib.Path(__file__)` works.                                                                                                                                                                                                                                                                                                                        |
| `getSubsetEcModel`    | `get_subset_ec_model`                  | **`[Py]`** Returns a new `EcModel`; the input is not mutated.                                                                                                                                                                                                                                                                                                  |
| `loadConventionalGEM` | `load_conventional_gem`                | —                                                                                                                                                                                                                                                                                                                                                              |
| `loadEcModel`         | `load_ec_model`                        | **`[3→4]`** Auto-flips legacy reverse-direction `usage_prot_*` / `prot_pool_exchange` to the forward convention (warns once). Reads the new cobrapy-style YAML. **`[Py]`** Thin wrapper over `raven_toolbox.io.read_yaml_model` (which owns the YAML schema + typed `EcData`); SBML ecModel I/O dropped in `0.1.0a2`. |
| `loadFluxData`        | `load_flux_data`                       | **`[Py]`** Returns a `FluxData` dataclass.                                                                                                                                                                                                                                                                                                                    |
| `loadProtData`        | `load_prot_data`                       | **`[Py]`** Returns a `ProtData` dataclass.                                                                                                                                                                                                                                                                                                                    |
| `mapRxnsToConv`       | `map_rxns_to_conv`                     | **`[Py]`** Returns a `MapRxnsResult` dataclass (was three parallel outputs).                                                                                                                                                                                                                                                                                  |
| `plotEcFVA`           | *(not ported)*                         | Use matplotlib / seaborn against the `ec_fva` DataFrame directly.                                                                                                                                                                                                                                                                                              |
| `reportEnzymeUsage`   | `report_enzyme_usage`                  | **`[3→4]`** Reads positive `usage_prot_*` flux. **`[Py]`** Returns an `EnzymeUsageReport` dataclass.                                                                                                                                                                                                                                                          |
| `saveEcModel`         | `save_ec_model`                        | **`[3→4]`** Emits the new cobrapy-style YAML; `usage_prot_*` written in forward convention. **`[Py]`** Thin wrapper over `raven_toolbox.io.write_yaml_model`; YAML only (`.yml` / `.yaml`), SBML ecModel I/O dropped in `0.1.0a2`. Injects a `metaData` provenance block. |
| `startGECKOproject`   | `geckopy init` *(CLI)*                 | **`[Py]`** Project skeleton generator runs from the `geckopy` CLI (`geckopy init my_project`).                                                                                                                                                                                                                                                              |
| `updateGECKOdoc`      | *(not applicable)*                     | Python docs are docstring-driven; no MATLAB Toolbox metadata to regenerate.                                                                                                                                                                                                                                                                                    |
| —                     | `pfba_enzymes` *(new)*                 | **`[Py]`** pFBA variant that minimises total `usage_prot_*` flux instead of all fluxes. Raises `NotImplementedError` on gecko-light.                                                                                                                                                                                                                          |
| —                     | `get_enzyme_bottlenecks` *(new)*       | **`[Py]`** Per-enzyme dual-based bottleneck ranker; returns a DataFrame.                                                                                                                                                                                                                                                                                       |

### 5.7 Model adapter (`src/geckomat/model_adapter`)

| MATLAB                  | geckopy                                                    | Notes                                                                                                                                                          |
| ----------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModelAdapter` classdef | `ModelAdapter` class + `model_adapter.toml`                | **`[Py]`** Config is in TOML, validated against a pydantic `ModelParameters` schema. Override via subclass when you need custom logic (rarely).               |
| `ModelAdapterManager`   | *(no global default)*                                      | **`[Py]`** Removed by design — pass `adapter=` or set `model.adapter` so adapter use is explicit.                                                              |
| `adapterTemplate.m`     | `geckopy init` *(CLI)* / `templates/model_adapter.toml`   | **`[Py]`** `geckopy init <name>` scaffolds a project folder with a starter TOML, `data/`, and `models/`.                                                       |

---

## 6. A minimal workflow comparison

**MATLAB** (sketch):

```matlab
adapter   = ModelAdapterManager.setDefault('MyAdapter.m');
model     = makeEcModel(loadConventionalGEM(), false, adapter);
kcatList  = fuzzyKcatMatching(model);
model     = selectKcatValue(model, kcatList);
model     = applyKcatConstraints(model);
model     = setProtPoolSize(model);
saveEcModel(model, 'ecModel.yml');
```

**geckopy**:

```python
from geckopy import (
    ModelAdapter, make_ec_model, load_conventional_gem,
    load_brenda_data, load_phyl_dist,
    fuzzy_kcat_matching, apply_kcat_list, apply_kcat_constraints,
    set_prot_pool_size, save_ec_model,
)
from geckopy.databases import download_phyl_dist

adapter   = ModelAdapter.from_folder("my_model")
model     = make_ec_model(load_conventional_gem(adapter), adapter)
brenda    = load_brenda_data(adapter.get_brenda_db_folder())
if not adapter.get_phyl_dist_path().is_file():
    download_phyl_dist(adapter.get_phyl_dist_path())       # RAVEN's keggPhylDist.mat
phyl_dist = load_phyl_dist(adapter.get_phyl_dist_path())
kcats     = fuzzy_kcat_matching(model, brenda, phyl_dist)   # returns a DataFrame
apply_kcat_list(model, kcats)                               # mutates in place
apply_kcat_constraints(model)
set_prot_pool_size(model)
save_ec_model(model, "ecModel.yml", adapter=adapter)
```

Three patterns to notice:

1. **Loaders are explicit.** Where MATLAB's `fuzzyKcatMatching`
   re-loads BRENDA + phyl-dist internally on every call, geckopy
   passes the pre-loaded `BrendaData` + `PhylDist` so you control
   when (and from where) they load. MATLAB reads the phylogenetic
   distances from RAVEN's `keggPhylDist.mat`; `download_phyl_dist`
   fetches that same file into the project's `data/` folder.
2. **`apply_kcat_list` mutates.** No reassignment to `model`.
3. **`save_ec_model` is YAML-only.** SBML ecModel I/O was dropped in
   `0.1.0a2`; `load_conventional_gem` still reads starting GEMs from
   SBML (`cobra.io.read_sbml_model`) as well as from RAVEN/cobra YAML
   (raven-toolbox's `read_yaml_model`).

The full tutorial (yeast-GEM, end-to-end, ~600 lines including kcat
curation + proteomics + sensitivity tuning) lives at
[`tutorials/full_ecModel/protocol.py`](../tutorials/full_ecModel/protocol.py).
A scale-down for the light layout (also a working end-to-end script,
~170 lines) is at [`tutorials/light_ecModel/protocol.py`](../tutorials/light_ecModel/protocol.py).

---

## 7. File exchange

ecModels move between the two tools without conversion:

- **Writing.** Both tools emit the same cobrapy-style YAML
  (cobra-shaped portion as `!!omap`; the GECKO ec sections —
  `ec-rxns`, `ec-enzymes`, `gecko_light` — as top-level keys cobrapy
  silently ignores). A geckopy `.yml` loads in GECKO 4 and vice versa.
- **Reading.** Backward-compatible: geckopy reads GECKO 3 / older
  RAVEN ecModels too, and auto-flips legacy reverse-direction protein
  reactions on load (warns once). Save the loaded model and it comes
  out as a current model.

The YAML schema and the legacy normalisations are owned by raven-toolbox
(`raven_toolbox.io.ec_data.EcData`,
`raven_toolbox.io.yaml.{read,write}_yaml_model`); geckopy's
`save_ec_model` / `load_ec_model` are thin wrappers that add geckopy's
application-level concerns (adapter-aware path resolution,
empty-ecModel guard, provenance injection). See
[raven_integration.md](raven_integration.md) for the split, and
[yaml_format.md](yaml_format.md) for the schema reference.

**BRENDA data files.** **`[3→4]`** GECKO 4 / geckopy ship
`kcat.tsv` / `sa.tsv` / `mw.tsv` (refreshed from the BRENDA bulk
JSON; bare EC numbers, plain organism names, `#` release header,
`references` column). GECKO 3's `max_KCAT.txt` etc. are no longer
read. **`[Py]`** geckopy additionally ships both **max** and
**median** rows per (ec, substrate, organism) triple in one file,
distinguished by an `aggregation` column. The default
`kcat_aggregate_brenda = "max"` preserves MATLAB-GECKO behaviour.
See [kcat_aggregation.md](kcat_aggregation.md).

---

## 8. What's available where

### Same algorithms, aligned results

The port found and fixed several GECKO 3 bugs (EC-code assignment,
sigma fitting, standard-kcat subsystem means, duplicate EC codes,
…); **`[3→4]`** GECKO 4 carries the same fixes, so the two toolboxes
agree. The full per-function divergence list lives on the
[GECKO vs. geckopy](https://gecko-docs.readthedocs.io/en/latest/gecko-to-geckopy.html) page
of gecko-docs.readthedocs.io, plus [future_improvements.md](future_improvements.md)
for the MATLAB-side items the port surfaced.

### Available in both (added during the port)

- **KEGG as a fallback protein/EC source** (`make_ec_model(kegg_db=…)`,
  `fill_eccodes_from_database`).
- **OpenKineticsPredictor over its REST API** —
  `submit_open_kinetics_predictor` / `fetch_open_kinetics_predictor`.
- **Forward-direction protein reactions** (`usage_prot_*` /
  `prot_pool_exchange`).
- **Refreshed BRENDA TSV schema** (geckopy adds the max+median
  variant; MATLAB ships only max so far).

### geckopy-only (no MATLAB counterpart)

- `pfba_enzymes` — enzyme-pool-minimising pFBA.
- `get_enzyme_bottlenecks` — dual-based bottleneck ranking.
- `relax_proteomics_greedy` — greedy alternative to
  `flexibilize_enz_concs`.
- `merge_kcats` — generalised n-source kcat merge (replaces the
  DLKcat-only MATLAB helper).
- `geckopy brenda-refresh` CLI — rebuilds the BRENDA TSV from a
  bulk-JSON snapshot.
- `get_subset_ec_model` — *(also exists in MATLAB)* but the Python
  version returns a fresh model rather than mutating in place.
- The `Enzyme` proxy (`model.enzymes.get_by_id(...)`).
- `geckopy init` — project skeleton CLI.

### Not yet in geckopy

- **ABC-SMC kcat tuning** (`bayesianSensitivityTuning` + helpers).
  geckopy fits kcats to experimental data with CMA-ES instead; see
  [evotune_kcat_tuning.md](evotune_kcat_tuning.md) and
  [cmaes_vs_abc_smc.md](cmaes_vs_abc_smc.md).
- **`plotEcFVA`** — Python users plot from the `ec_fva` DataFrame
  with matplotlib / seaborn directly.

---

## 9. Where to look when something differs

- Each ported function's docstring opens with
  `Ported from GECKO MATLAB: <path>.` so you can find the exact
  MATLAB original.
- For current MATLAB ↔ Python behavioural differences, see the
  [GECKO vs. geckopy](https://gecko-docs.readthedocs.io/en/latest/gecko-to-geckopy.html) page
  on [gecko-docs.readthedocs.io](https://gecko-docs.readthedocs.io/), the
  shared documentation site for both toolboxes.
- MATLAB-side bugs and rough edges the port found are tracked in
  [future_improvements.md](future_improvements.md).
- For the GECKO 3→4 changes as they apply on the MATLAB side, see
  [GECKO 3 to GECKO 4](https://gecko-docs.readthedocs.io/en/latest/gecko3-to-gecko4.html)
  in the GECKO repository.
- For the YAML schema (shared with MATLAB GECKO 4 / RAVEN), see
  [yaml_format.md](yaml_format.md).
- For the gecko-light availability table, see
  [gecko_light_status.md](gecko_light_status.md).
- For the raven-toolbox / geckopy responsibility split, see
  [raven_integration.md](raven_integration.md).
