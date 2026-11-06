# BRENDA database refresh plan

> **Historical planning doc.** The implementation diverged in May 2026:
> the snapshot now includes **both** max and median per (ec, substrate,
> organism) triple in long-format `kcat.tsv` / `sa.tsv` (not just max
> in `max_kcat.tsv` etc.). See [kcat_aggregation.md](kcat_aggregation.md)
> for the current schema and rationale.

## Goal

Replace the 2018-vintage Python 2 SOAP-scrape pipeline that produced
`max_KCAT.txt` / `max_SA.txt` / `max_MW.txt` with a modern Python 3
build step that:

1. Downloads the **BRENDA bulk JSON** (Release 2026.1, March 2026,
   ~80 MB, CC BY 4.0).
2. Walks the JSON and emits three deterministically-sorted, diffable
   5-column TSVs into the geckopy repo.
3. Drops dead columns (KEGG-code, taxonomy) and adds a `references`
   column carrying PMIDs.

Phylogenetic distance is **out of scope**: `keggPhylDist.mat` was
refreshed in RAVEN on 2026-04-19 (commit `9ff834be`) and will be
owned by ravenpy in the future.

## Background

### What exists today

- [GECKO/src/geckopy/brenda_parser/](../../GECKO/src/geckopy/brenda_parser/) —
  three Python 2 scripts (`retrieveBRENDA.py`, `createECfiles.py`,
  `findMaxKvalues_AllOrgs.py`) last touched 2018-04-10. They depend
  on the abandoned `SOAPpy` library and hard-coded paths. Do not
  run on any modern interpreter.
- [GECKO/databases/max_KCAT.txt](../../GECKO/databases/max_KCAT.txt) (30,162 rows),
  `max_SA.txt` (15,777 rows), `max_MW.txt` (25,050 rows).
  Schema: `ECNNN \t substrate \t organism//taxonomy//kegg_code \t value \t pathways`.
- Consumers: [loadBRENDAdata.m](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m)
  on the MATLAB side, [brenda_loader.py](../src/geckopy/databases/brenda_loader.py)
  on the geckopy side.

### What both consumers actually read

- Column 1: EC number (loader strips the `EC` prefix).
- Column 2: substrate name (or `*` for SA/MW).
- Column 3: **only the substring before the first `//`** — both
  loaders explicitly discard taxonomy and KEGG code:
  - [loadBRENDAdata.m:79](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m#L79):
    `regexprep(data_cell{3},'\/\/.*','')`
  - [brenda_loader.py:164](../src/geckopy/databases/brenda_loader.py#L164):
    `organism_blob.split("//", 1)[0]`
- Column 4: numeric value.
- Column 5: **not used by any code path** (pathways column is dead).

This means the taxonomy and KEGG-code columns are vestigial overhead
from the 2018 pipeline. Dropping them changes nothing at runtime.

### Why the 2018 pipeline cannot be incrementally fixed

It is a per-EC SOAP scrape rate-limited to 1 request/s. ~15,000 EC
numbers × 5 fields ≈ 4-6 h per refresh with constant babysitting,
on a stack (Python 2 + `SOAPpy`) that is decade-dead.

The same data is now available as a **single 80 MB HTTP download**
([BRENDA download page](https://www.brenda-enzymes.org/download.php)),
refreshed roughly quarterly. Replace the pipeline, not patch it.

## API choice: bulk JSON, nothing else

[BRENDA schema 2.0.0](https://www.brenda-enzymes.org/schemas/2.0.0/brenda.schema.json)
is JSON-Schema-Draft-2020-12 compliant. Top-level structure:

```
{
  release:  "2026.1",
  version:  "2.0.0",
  data: {
    "1.1.1.1": {                          // one entry per EC
      id, recommended_name, ...,
      turnover_number:   [ numeric_dataset, ... ],   // kcat in 1/s
      specific_activity: [ numeric_dataset, ... ],   // umol/min/mg
      molecular_weight:  [ numeric_dataset, ... ],   // g/mol (holoenzyme)
      protein:   { "<pid>": { id, organism, accessions, source, ... } },
      reference: { "<rid>": { title, authors, journal, year, pmid } }
    },
    ...
  }
}
```

`numeric_dataset` shape:
```
{
  value:      "23.5 {ethanol}",   // numeric prefix; substrate in {...}
  comment:    "...",              // free text; mutant info lives here
  proteins:   ["1", "5", ...],   // keys into the .protein map
  references: ["3", "7", ...]    // keys into the .reference map
}
```

No SOAP fallback. No NCBI fetch. No KEGG fetch.

## Output schema

5 columns, tab-separated, UTF-8 NFC normalised, `\n` line endings,
deterministically sorted by `(ec_number, substrate, organism)`:

```
# BRENDA release 2026.1 generated 2026-05-18 - CC BY 4.0
ec_number   substrate   organism                       value   references
1.1.1.1     ethanol     escherichia coli               12.1    PMID:12345;PMID:67890
1.1.1.1     ethanol     saccharomyces cerevisiae       23.5    PMID:11111
1.1.1.1     methanol    saccharomyces cerevisiae       0.42    PMID:22222
...
```

| col | meaning | notes |
|-----|---------|-------|
| 1 | EC number | no `EC` prefix |
| 2 | substrate | lowercased; `*` for SA/MW rows |
| 3 | organism | lowercased; **no `//taxonomy//kegg`** — bare organism string |
| 4 | value | kcat in 1/s, SA in umol/min/mg, MW in g/mol |
| 5 | references | semicolon-joined `PMID:NNN` list (or `*` if none) |

### Compatibility with existing loaders

The legacy 5-column format expected `organism//tax//kegg` in column 3.
With the new bare-organism layout, [loadBRENDAdata.m:79](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m#L79)
and [brenda_loader.py:164](../src/geckopy/databases/brenda_loader.py#L164)
still work as-is: the `//` strip becomes a no-op. The legacy
column-5 `pathways` slot is replaced by `references` and neither
loader reads it, so this is a free rename.

### The `#` header line

Verified in MATLAB R2024b (test cases T1-T5):
- Without `'CommentStyle','#'`, [loadBRENDAdata.m](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m)
  parses the `#` line as a data row and corrupts the table.
- With `'CommentStyle','#'`, the header is skipped cleanly.

**Required loader changes** (one line each):
- MATLAB: [loadBRENDAdata.m:74](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m#L74):
  add `,'CommentStyle','#'` to the `textscan` call.
- Python: [brenda_loader.py](../src/geckopy/databases/brenda_loader.py):
  skip lines starting with `#` in the per-line iteration.

## Files

### New

```
src/geckopy/databases/brenda/
    __init__.py        # re-exports: refresh_brenda, parse_brenda_json
    download.py        # bulk JSON fetch + local cache + sha256 check
    parse.py           # JSON -> (ec, substrate, organism, value, refs) rows
    aggregate.py       # max per (ec, substrate, organism); deterministic sort; TSV writer
    cli.py             # geckopy brenda-refresh entrypoint glue

src/geckopy/data/brenda/
    max_kcat.tsv       # committed; regenerated by the CLI
    max_sa.tsv         # committed; regenerated by the CLI
    max_mw.tsv         # committed; regenerated by the CLI

tests/
    test_brenda_download.py    # cache hit/miss + integrity check (mocked HTTP)
    test_brenda_parse.py       # synthetic JSON fragments -> expected row tuples
    test_brenda_aggregate.py   # filters, max, sort determinism, TSV format
    fixtures/
        brenda_minimal.json    # 3-EC handcrafted sample exercising every branch
```

### Modified

- [pyproject.toml](../pyproject.toml) — `requests` is already a dep;
  add nothing.
- [src/geckopy/cli.py](../src/geckopy/cli.py) — register
  `brenda-refresh` subcommand.
- [src/geckopy/databases/__init__.py](../src/geckopy/databases/__init__.py)
  — re-export `refresh_brenda` for `from geckopy.databases import ...`.
- [src/geckopy/databases/brenda_loader.py](../src/geckopy/databases/brenda_loader.py)
  — skip `#`-prefixed lines in the per-line iteration.
- [.gitignore](../.gitignore) — add `src/geckopy/data/brenda/_cache/`.

### Cross-repo

- [GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m:74](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m#L74)
  — add `'CommentStyle','#'` to the `textscan` call.
- [GECKO/databases/max_KCAT.txt](../../GECKO/databases/max_KCAT.txt)
  + `max_SA.txt` + `max_MW.txt` — replace with refreshed copies (or
  point GECKO at the geckopy-owned files via a path tweak).

### Deletable after this lands

- [GECKO/src/geckopy/brenda_parser/](../../GECKO/src/geckopy/brenda_parser/)
  — entire folder, all four 2018 Python 2 scripts.

## Implementation steps

### 1. `download.py`

- `download_brenda_json(*, cache_dir, release=None, force=False) -> Path`
- Default cache dir: `src/geckopy/data/brenda/_cache/`
  (gitignored).
- Fetch `https://www.brenda-enzymes.org/download/brenda_<release>.json.tar.gz`
  (URL pattern to be verified — the download page does not require
  registration for the bulk file but uses a click-through licence
  acceptance; if a direct URL is not available, accept that the user
  manually downloads to the cache dir and the CLI just validates
  presence + sha256).
- Cache hit if file exists and `force=False`.
- After download, untar the JSON file (one file inside the tarball).
- Log release version and file size.

### 2. `parse.py`

- `parse_brenda_json(path) -> Iterator[Row]` where `Row = (kind, ec, substrate, organism, value, references)`.
- `kind` is one of `"kcat"`, `"sa"`, `"mw"`.
- Stream the JSON with `json.load` (80 MB fits in memory; ijson only if
  profiling shows we need it).
- For each `ec`:
  - Build `pid -> organism` from `data[ec].protein`.
  - Build `rid -> "PMID:<n>"` from `data[ec].reference` (skip refs
    without a `pmid` field; if none survive emit `*`).
  - For each `numeric_dataset` row in `turnover_number`:
    - Parse `value` string: split on `{`, take the numeric prefix
      and the substrate token between `{...}`.
    - Skip rows where comment matches `/mutant|mutated/i`.
    - Skip `value == -999`.
    - Skip `kcat > 1e7` (Bar-Even et al. 2011 physical limit).
    - For ranges (`"0.1-2.5 {NADH}"`): take the **upper bound**.
    - Fan out one row per `pid` in `proteins`; emit
      `(ec, substrate, organism, kcat, references)`.
  - Same for `specific_activity` (substrate = `*`).
  - Same for `molecular_weight` (substrate = `*`).
- Skip `data["spontaneous"]`.
- Skip rows with empty organism string.

### 3. `aggregate.py`

- `aggregate_and_write(rows, out_dir) -> None`
- Group by `(kind, ec, substrate, organism)`, keep **max value**;
  for `references`, take the **union** of PMIDs across the merged
  rows (sorted, deduped, semicolon-joined).
- Sort lexicographically by `(ec, substrate, organism)`.
- Write three TSVs to `out_dir/`:
  - `max_kcat.tsv` (kcat in 1/s)
  - `max_sa.tsv` (umol/min/mg)
  - `max_mw.tsv` (g/mol)
- First line of each: `# BRENDA release <X.Y> generated <YYYY-MM-DD> - CC BY 4.0`.

### 4. `cli.py` integration

- `geckopy brenda-refresh [--cache-dir DIR] [--out-dir DIR] [--force]`
- Default `out-dir`: `src/geckopy/data/brenda/`.
- Sequence: download -> parse -> aggregate -> write -> print summary
  (release, row counts per file, sha256 of each output).
- Non-zero exit on download failure or empty output.

### 5. Loader updates

- [brenda_loader.py](../src/geckopy/databases/brenda_loader.py) per-line
  iteration: prepend `if line.startswith("#"): continue`.
  Add a unit test in [tests/test_brenda_loader.py](../tests/test_brenda_loader.py)
  with a `#`-headed fixture.
- [loadBRENDAdata.m](../../GECKO/src/geckomat/get_enzyme_data/loadBRENDAdata.m)
  `textscan` call: add `'CommentStyle','#'`. Verified with the T1-T5
  test suite already run against MATLAB R2024b.

### 6. Tests

- `test_brenda_parse.py`: feed `brenda_minimal.json` (3 hand-crafted
  EC entries covering: normal kcat, value range, mutant comment,
  `-999`, kcat > 1e7, missing protein, missing references, `spontaneous`),
  assert exact emitted row tuples.
- `test_brenda_aggregate.py`: synthetic input rows, assert max-per-key,
  references-union, sort determinism, header line, NFC normalisation,
  `\n` line endings, byte-identical output across two runs.
- `test_brenda_download.py`: mock `requests.get` with a tiny tarball,
  assert cache reuse, sha256 check, force flag behaviour.
- `test_brenda_loader.py`: add a fixture with a `#` header line,
  assert it is skipped.
- Total ~25 new tests; target 172 (current) + 25 = 197 passing.

### 7. Re-run + commit refreshed TSVs

- One-off: `geckopy brenda-refresh` -> three new TSVs in
  `src/geckopy/data/brenda/`.
- Spot-check against the 2018 files: select 5 well-studied ECs
  (`1.1.1.1`, `2.7.1.40`, `3.4.21.1`, `4.2.1.1`, `1.11.1.6`),
  diff max-kcat per organism, sanity-check the order of magnitude.
- Copy the three files into [GECKO/databases/](../../GECKO/databases/)
  (or update `loadBRENDAdata.m` to read from a path inside geckopy
  — design call: not in this PR).
- Commit the TSVs to the geckopy repo. The first commit will be a
  large diff; subsequent refreshes will diff cleanly because of the
  deterministic sort.

## Filters and edge cases (forward-ported from 2018)

| filter | source | action |
|--------|--------|--------|
| `value == -999` | BRENDA's missing-value marker | skip row |
| comment matches `/mutant\|mutated/i` | mutant enzyme | skip row |
| `kcat > 1e7` | Bar-Even et al. 2011 physical limit | skip row |
| value is a range `0.1-2.5` | BRENDA convention | take upper bound |
| `data["spontaneous"]` | non-enzymatic | skip entire EC |
| empty organism string | no protein assignment | skip row |
| EC has `B` prefix in last component (`1.1.1.B7`) | BRENDA-supplementary EC | keep, treat as normal |

## Out of scope

- Phylogenetic distance source (RAVEN owns it; ravenpy will inherit).
- KEGG REST or FTP integration. The 2018 pipeline only joined KEGG
  to fill the taxonomy column, which is dead data on both loaders.
- BRENDA SOAP API. Bulk JSON is the only ingest path.
- BRENDA text file (.txt) format. JSON has a stable schema and is
  easier to parse.
- Migrating away from .mat for PhylDist. Separate concern; ravenpy
  will revisit.
- Changing the GECKO MATLAB `loadBRENDAdata.m` beyond the one-line
  `CommentStyle` addition.

## Verification

- `cd /mnt/c/Work/GitHub/geckopy && source .venv/bin/activate && pytest`
  -> baseline + 25 new tests passing.
- `geckopy brenda-refresh --force` end-to-end on a developer machine
  with the BRENDA bulk JSON in the cache, completes in under 5 min,
  emits three TSVs of expected order-of-magnitude size:
  - `max_kcat.tsv`: ~30-50k rows
  - `max_sa.tsv`: ~15-25k rows
  - `max_mw.tsv`: ~25-40k rows
- MATLAB-side: load the new TSV in GECKO with the patched
  `loadBRENDAdata.m`, run the full_ecModel Crabtree tutorial,
  confirm it solves without the loader complaining.
- Diff `max_kcat.tsv` against the 2018 `max_KCAT.txt` for EC `1.1.1.1`,
  spot-check 10 high-value rows survive in both files within an
  order of magnitude.

## Attribution

The TSV header contains `CC BY 4.0`; add a one-paragraph
attribution to [README.md](../README.md) crediting BRENDA
(citation: Chang et al., Nucleic Acids Res., latest annual update)
and linking to <https://www.brenda-enzymes.org>.

## Rollout

1. Land this PR. Refresh once on commit, include the new TSVs in the
   geckopy repo, submit the one-line `loadBRENDAdata.m` patch upstream
   to GECKO.
2. Future refreshes: re-run `geckopy brenda-refresh` when BRENDA
   publishes a new release (announcement on
   <https://www.brenda-enzymes.org/news.php>); review the diff;
   commit the new TSVs. No automation, no schedule.
3. Once the dust settles, delete [GECKO/src/geckopy/brenda_parser/](../../GECKO/src/geckopy/brenda_parser/)
   in a follow-up GECKO PR (its only role was generating these files,
   and that role has moved here).
