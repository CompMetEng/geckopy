# WILDkCAT integration plan (Strategy B — side-by-side, no deprecation)

## Goal

Add [WILDkCAT](https://github.com/sysbiolux/WILDkCAT) (PyPI: `wildkcat`)
as an **alternative** kcat-gathering pathway in geckopy. Keep the
existing fuzzy BRENDA matcher and ML wrappers
([gather_kcats/fuzzy_kcat_matching/](../../src/geckopy/gather_kcats/fuzzy_kcat_matching/),
[run_dlkcat.py](../../src/geckopy/gather_kcats/run_dlkcat.py),
[read_dlkcat_output.py](../../src/geckopy/gather_kcats/read_dlkcat_output.py))
fully functional. Both pathways must terminate in the same
downstream API: a dataframe that
[apply_kcat_list](../../src/geckopy/gather_kcats/select_kcat_value.py)
accepts.

Users opt into WILDkCAT explicitly. A later PR — outside this plan —
can promote WILDkCAT to default and deprecate the in-tree path once
benchmarks back the switch.

## Why Strategy B

- Lowest risk: zero behaviour change for existing users.
- WILDkCAT is a heavy optional dep (BRENDA login, Entrez email,
  CataPro weights) and we should not force it on the default install.
- Lets us run a head-to-head benchmark on the Crabtree tutorial
  before retiring ~1.5k LOC of fuzzy + ML scaffolding.

## Integration boundary

Both pipelines must produce a `pandas.DataFrame` with at least
these columns (current contract of
[fuzzy_kcat_matching](../../src/geckopy/gather_kcats/fuzzy_kcat_matching/__init__.py#L67-L78)
and consumed by
[apply_kcat_list](../../src/geckopy/gather_kcats/select_kcat_value.py)):

| column   | type  | meaning                                                |
|----------|-------|--------------------------------------------------------|
| rxn_id   | str   | matches `model.ec.rxns`                                |
| source   | str   | `"brenda"` / `"dlkcat"` / **new:** `"wildkcat"`        |
| kcat     | float | s^-1; `0.0` means "no match"                           |

Optional columns already used downstream (origin/wildcard) stay
opt-in. Anything WILDkCAT-specific (penalty_score, db_source,
SMILES) is exposed as **extra columns** on the same dataframe so
the user can inspect them; `apply_kcat_list` ignores them.

## Files to add

```
src/geckopy/gather_kcats/wildkcat/
    __init__.py        # public surface: gather_kcats_wildkcat
    config.py          # .env / BRENDA / Entrez resolution + validation
    runner.py          # build WILDkCAT input from model, invoke pipeline
    adapter.py         # WILDkCAT output -> geckopy kcat dataframe
tests/
    test_wildkcat_adapter.py     # pure mapping tests, no network
    test_wildkcat_runner.py      # runner with mocked wildkcat module
    fixtures/
        wildkcat_output_sample.tsv
```

## Files to modify

- `pyproject.toml` — new optional-deps group:
  ```toml
  [project.optional-dependencies]
  wildkcat = ["wildkcat>=<min-ver>", "python-dotenv>=1"]
  ```
  Select `<min-ver>` from the first PyPI release that includes the
  programmatic entry point we depend on (see step 1 below).
- [src/geckopy/gather_kcats/__init__.py](../../src/geckopy/gather_kcats/__init__.py)
  — re-export `gather_kcats_wildkcat` **inside a lazy import**
  so users without the optional dep are not penalised on
  `import geckopy`.
- [src/geckopy/__init__.py](../../src/geckopy/__init__.py) — do **not**
  promote to top-level for now; users call
  `from geckopy.gather_kcats.wildkcat import gather_kcats_wildkcat`.
- [docs/internal/porting_plan.md](porting_plan.md) — add a note that WILDkCAT
  is now a parallel pathway.
- [README.md](../../README.md) — one sentence under "Installation"
  pointing to the new optional install + the doc page.

## New doc

- `docs/guides/wildkcat.md` — user-facing how-to: install,
  `.env` setup, minimal call, output schema, when to prefer it
  over the in-tree pipeline.

## Implementation steps

### 1. Pin the WILDkCAT API surface
- `pip install wildkcat` in a scratch venv.
- Identify the **programmatic entry** (function, not just CLI).
  WILDkCAT's README documents a CLI; check
  `wildkcat.__init__`, `wildkcat/main.py` or equivalent for a
  Python callable. If only a CLI exists, fall back to driving
  it via `subprocess` and parsing the TSV — this is acceptable
  for Strategy B but should be flagged.
- Record the **input schema** WILDkCAT expects: at minimum
  `(rxn_id, ec_number, substrate_name | substrate_inchi |
  substrate_smiles, organism)`.
- Record the **output schema** WILDkCAT emits and which columns
  contain kcat (s^-1 vs min^-1; confirm and unit-convert in
  the adapter if needed).

### 2. `config.py`
- Resolve credentials in this order:
  1. arguments passed to `gather_kcats_wildkcat(...)`
  2. environment variables (`BRENDA_EMAIL`, `BRENDA_PASSWORD`,
     `ENTREZ_EMAIL`, plus any CataPro paths)
  3. `.env` in the working directory (via `python-dotenv`,
     loaded only on demand)
- Raise `ValueError` with an actionable message — naming the
  missing variable — if nothing is found.
- **Do not** fall through silently to an anonymous BRENDA call;
  WILDkCAT will fail and produce a confusing stack trace.

### 3. `runner.py`
- `gather_kcats_wildkcat(model, *, organism=None, config=None,
  workdir=None, **wildkcat_kwargs) -> pd.DataFrame`
- Mirror the entry-point shape of
  [fuzzy_kcat_matching](../../src/geckopy/gather_kcats/fuzzy_kcat_matching/__init__.py#L78-L99):
  same first positional arg, same return type, no surprises.
- Build the WILDkCAT input table by walking `model.ec.rxns`:
  - `rxn_id` <- `model.ec.rxns[i]` (preserve the `ec_` prefix
    for light models so the adapter can read the id back)
  - `ec_number` <- `model.ec.eccodes[i]`
  - `substrate` <- substrate name(s) from the EC reaction's
    metabolites; reuse the substrate-extraction logic that
    used to live in `_extract_substrates` and now lives inline
    in [fuzzy_kcat_matching.__init__](../../src/geckopy/gather_kcats/fuzzy_kcat_matching/__init__.py)
    (refactor: lift it back into
    `gather_kcats/_substrates.py` so both pathways share it
    — single source of truth, ~30 LOC)
  - `organism` <- `organism` arg, falling back to
    `model.adapter.params.org_name`
- Reactions with empty EC or no substrates contribute a row
  with `kcat=0.0` and `source="wildkcat"` — matches the
  fuzzy-matcher convention.
- Persist the WILDkCAT input TSV to `workdir` (default a
  temp dir, but accept user path for reproducibility), invoke
  WILDkCAT, read back the output TSV.
- Wrap WILDkCAT failures in a `RuntimeError` that surfaces the
  log path; do not swallow the original exception.

### 4. `adapter.py`
- `wildkcat_tsv_to_kcat_df(path_or_df) -> pd.DataFrame`
- Pure function, no I/O beyond reading the supplied path.
- Column-by-column mapping with a documented map at the top of
  the module. Unit-convert if WILDkCAT emits min^-1.
- Drop unknown rows (rxn_id not in the original input) with a
  `logger.warning` listing them — symmetrical to how
  [apply_kcat_constraints](../../src/geckopy/ec_model/pipeline/apply_kcat.py)
  warns on unmatched rxns.
- Preserve WILDkCAT-specific columns (`penalty_score`,
  `db_source`, etc.) so the caller can inspect them.

### 5. Re-export with a lazy guard
In [gather_kcats/__init__.py](../../src/geckopy/gather_kcats/__init__.py):

```python
def __getattr__(name):
    if name == "gather_kcats_wildkcat":
        try:
            from .wildkcat import gather_kcats_wildkcat
        except ImportError as exc:
            raise ImportError(
                "WILDkCAT support requires the optional dependency: "
                "pip install 'geckopy[wildkcat]'"
            ) from exc
        return gather_kcats_wildkcat
    raise AttributeError(name)
```

`__all__` stays as-is — we do not advertise the symbol in
tab-completion until the optional dep is present.

### 6. Tests

- `test_wildkcat_adapter.py` — feed a 5-row fixture TSV into
  `wildkcat_tsv_to_kcat_df`, assert column names, types, unit
  conversion, source label, warning on unknown rxn_id.
- `test_wildkcat_runner.py` — monkeypatch the WILDkCAT entry
  point to a stub that writes a known TSV; assert the runner:
  (a) builds the correct input table, (b) returns the adapted
  df, (c) raises `ValueError` on missing credentials,
  (d) returns `kcat=0.0, source="wildkcat"` rows for ec
  reactions with no substrate.
- Mark **no** network test as required in CI. Add an opt-in
  `@pytest.mark.wildkcat_live` marker, off by default, gated
  on real credentials in `.env`. Document in
  [README.md](../../README.md) how to enable it.
- Target test count after this PR: 172 (current) + ~12 new.

### 7. Benchmark helper (optional, can be included in the same PR)

`src/geckopy/gather_kcats/benchmark.py`:

```python
def benchmark_kcat_pipelines(model, organism, *, brenda_path,
                              wildkcat_config=None) -> pd.DataFrame
```

Returns one row per ec reaction with columns
`rxn_id, fuzzy_kcat, fuzzy_source, wildkcat_kcat,
wildkcat_source, agree_within_2x`. Intended to be run once
on the [full_ecModel tutorial](../../tutorials/) and pasted into
the eventual default-switch PR.

This helper is the data that will justify deprecating the
fuzzy + ML path in a future PR. Without it, the deprecation
decision has nothing to stand on.

### 8. Docs

`docs/guides/wildkcat.md`:
- One-paragraph context (what WILDkCAT is, why offer it)
- Install: `pip install 'geckopy[wildkcat]'`
- `.env` template (no real credentials in the file)
- Minimal example: model load -> `gather_kcats_wildkcat` ->
  `apply_kcat_list` -> `apply_kcat_constraints` -> solve
- Output-column reference
- Caveats: first run downloads CataPro weights, requires
  Python >= 3.11 (matches geckopy)

## Out of scope (explicit non-goals)

- Removing or deprecating [fuzzy_kcat_matching](../../src/geckopy/gather_kcats/fuzzy_kcat_matching/)
  or DLKcat wrappers. Tracked for a follow-up after benchmarks.
- Wrapping CataPro standalone (Strategy D). The full WILDkCAT
  pipeline already runs CataPro.
- Promoting `gather_kcats_wildkcat` to top-level `geckopy.*`.
- Changing the MATLAB GECKO side.

## Verification

- `cd /mnt/c/Work/GitHub/geckopy && source .venv/bin/activate &&
  pytest` — expect baseline + ~12 new tests, all passing on a
  machine **without** the `wildkcat` extra installed
  (the lazy `__getattr__` keeps it green).
- `pip install '.[wildkcat]' && pytest` — same, plus the
  WILDkCAT-specific tests are now collected and pass with the
  mocked entry point.
- Manual: run `gather_kcats_wildkcat` on the Crabtree tutorial
  ecYeastGEM, save the output, diff against the in-tree
  pipeline using the new benchmark helper, attach the table
  to the PR description.

## Rollout

1. Land this PR. No user-visible default change.
2. Run the benchmark on ecYeastGEM + at least one other
   organism (E. coli or human) — attach to the PR or to a
   follow-up issue.
3. If WILDkCAT coverage and accuracy meet or beat the in-tree
   pipeline, open a second PR that:
   - flips the default in the tutorial,
   - emits `DeprecationWarning` from
     `fuzzy_kcat_matching` and `run_dlkcat`,
   - schedules removal for the next minor.
4. Removal PR after a release cycle.
