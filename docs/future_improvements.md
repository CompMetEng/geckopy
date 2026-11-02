# Future improvements

A running list of ideas the port surfaced that are worth doing but
weren't blocking. Three audiences:

- **geckopy users / contributors** — sections "Conventions / data
  model", "Third-party library gotchas", and "Deferred migrations".
- **MATLAB GECKO maintainers** — section "MATLAB GECKO changes".
  Each item is a bug or rough edge in MATLAB GECKO that the port
  spotted while comparing the two implementations side by side. For
  the current, maintained list of MATLAB ↔ Python behavioural
  differences (broader than just what's tracked here), see the
  [GECKO vs. geckopy](https://gecko-docs.readthedocs.io/en/latest/gecko-to-geckopy.html) page
  on [gecko-docs.readthedocs.io](https://gecko-docs.readthedocs.io/).
- **anyone hitting a strange dependency bug** — section
  "Third-party library gotchas".

## Conventions / data model

One cross-cutting decision where geckopy and MATLAB GECKO are
still misaligned. Not blocking, but worth doing eventually.

- **Custom subunit stoichiometry file format.** The `stoicho`
  column in `customKcats.tsv` was meant to let users override
  subunit counts manually. The MATLAB code never actually reads
  it; geckopy also ignores it. Two options if we want to support
  it properly: a separate `customComplexes.tsv` file, or a new
  "user_overrides" section in the ComplexPortal JSON. Deferred
  until someone needs it.

## Third-party library gotchas

A scratchpad of brittle behaviours we ran into in geckopy's hard
dependencies. Each is worked around in source today; the entries
exist so the next person hitting the same wall can find the fix
fast, and so we have a list of things to retest when we upgrade.

- **`multiprocessing.Pool(context="spawn")` deadlocks on some WSL
  kernels.** Even a trivial 2-worker pool with a `lambda x: x*2`
  task hangs forever — no error, no timeout. The original plan for
  parallel `ec_fva` called for `spawn` because it's the only context
  available on Windows. Workaround: pick the context based on the
  platform — `fork` on POSIX (which is also faster, no module
  re-import), `spawn` on Windows. See `geckopy.utilities.ec_fva.ec_fva`.

- **`ruamel.yaml`'s `ScalarInt` / `ScalarFloat` leak out of cobra
  YAML loads.** After loading a YAML model with cobra, attributes
  like `Metabolite.charge` are ruamel scalar types (subclasses of
  int/float that carry YAML formatting metadata). The ruamel
  safe-dumper has no representer for them, so writing the model
  back out fails with `RepresenterError: cannot represent an
  object: 0`. Workaround: walk the assembled document and coerce
  everything to plain Python primitives just before writing. See
  `geckopy.utilities.save_ec_model._to_native`. This bites any
  YAML-to-YAML round-trip of a cobra-loaded model.

## Deferred migrations

Pieces of geckopy that have a clear destination (upstream package
or a different in-tree shape) but whose migration was deferred to
keep an in-flight release on schedule.

- **`brenda/parse.py` range collapse.** A BRENDA value reported as
  a range (e.g. `"0.1-2.5"`) is collapsed to its upper bound
  unconditionally in `parse_brenda_json`. After the kcat-aggregation
  refactor moved the rest of the pipeline to be configurable
  (`kcat_aggregate_brenda = "max" | "median"`), this is the one
  residual max-leaning step. Honouring the adapter setting here
  would make every layer consistent. Small effect (only rows
  reported as ranges) — deferred.

## MATLAB GECKO changes

Aimed at MATLAB GECKO maintainers. Each item is something the
geckopy port found while comparing the two implementations: bugs,
dead code, unit-comment slip-ups, or unclear behaviour.

Notable items:

- Implement gene-cell splitting in the UniProt loader.
- Apply the `stoicho` column from `customKcats.tsv`, or drop it from
  the schema.
- Rename `getKcatAcrossIsozymes` to `fillKcatsFromIsozymes` — the
  verb `get` is misleading, the function mutates `ec.kcat` in place.
  geckopy renamed its Python counterpart; the MATLAB side is the
  remaining half of the coordinated rename.
- Clean up the `applyCustomKcats` Mode B docstring (proteins only,
  no rxns) — its docstring and code disagree on whether the
  full-match gate fires. Either clarify the docstring or drop the
  mode.
- Delete `getECstring` from MATLAB GECKO. After the `findECInDB`
  refactor (geckopy works with raw `;`-joined EC tokens throughout)
  the function has no remaining callers. It also has three latent
  bugs not worth fixing in place: an accumulator footgun (callers
  must add a trailing space to `EC_set` before calling, otherwise
  tokens collide), an empty-input quirk (returns `"EC"` instead of
  `""` because `strsplit("", " ")` returns `{''}`), and no
  validation (would happily emit `ECEC1.1.1.1` for already-prefixed
  input or `ECnotanec` for junk).
- Fix the validation regex in `getECfromGEM`. The current pattern
  `(\d\.(\w|-)+\.(\w|-)+\.(\w|-)+)(;\w+\.(\w|-)+\.(\w|-)+\.(\w|-)+)*(.*)`
  substituted with `$3` returns the level-3 character of the first EC
  for any valid input (e.g. `"1.2.3.4"` -> `"3"`), and the subsequent
  `~cellfun(@isempty, ...)` then flags every non-empty result as
  invalid. As written, every non-empty EC string is silently
  discarded. Replace with a straightforward
  `^TOKEN(;TOKEN)*$` validation where `TOKEN` is the canonical
  four-level dotted EC pattern with `-` allowed in any level.
- Dedupe the `intersection` helper output in `findECInDB`. The
  helper currently produces duplicates when multiple subunits share
  the same EC (e.g. `"1.1.1.1;1.1.1.1"` for a two-subunit complex
  whose subunits both map to EC 1.1.1.1). Apply a final
  `compare_wild` pass on the intersection result, mirroring what
  geckopy does.
- Simplify `loadBRENDAdata` signature: drop the `modelAdapter` arg
  and `ModelAdapterManager.getDefault()` fallback; take the folder
  path directly. The adapter resolution belongs at the call site.
- Resolve the `'add'` action in `getECfromDatabase`: either implement
  the commented-out `addMultipleMatches` branch (and drop the
  apologetic `% I don't understand the purpose of this, let's skip
  it for now` comment) or remove the option from the documented API.
  geckopy currently only supports `'display'` and `'ignore'`.
- Subset by `ecRxns` upfront in `getECfromDatabase` rather than
  iterating every reaction and discarding the unwanted results at
  the end. The function's own inline comment already calls this out
  ("Probably faster to subset with ecRxns in the beginning of the
  script, but this was at the moment simpler to implement").
  geckopy already does this.
- Fix the search-order/output-ranking inconsistency in
  `fuzzyKcatMatching`. The search order inside `mainMatch` tries
  org-SA (output rank 5) BEFORE any-organism-no-substrate-kcat
  (output rank 4). When both are available for a reaction, MATLAB
  returns the org-SA result and reports its origin as 5, even though
  the docstring's origin ranking implies origin 4 should win. Either
  swap the search order so any-no-subs-kcat is tried before org-SA,
  or update the docstring to match the search order. geckopy
  replicates the current MATLAB behavior, both within one EC number
  and when selecting among the EC numbers of one reaction; see
  `SEARCH_LEVELS` and `SEARCH_RANK` in
  `fuzzy_kcat_matching/_brenda_query.py`.
- Delete `updateProtPool` from MATLAB GECKO. The function has been
  obsolete since GECKO 3.2.0 (all enzymes, measured and unmeasured,
  draw from the protein pool); its sole runtime behaviour on a
  3.2.0+ model is to raise an error pointing users to
  `setProtPoolSize`. geckopy does not port it; the recommended
  replacement is `set_prot_pool_size` in `protein_pool.py`.
- Make `calculateFfactor` raise (or return NaN with a warning)
  when no proteome data is provided. Silently returning 0.5 hides
  the absence of real data; downstream calculations using this f
  factor look fine until they don't. geckopy's split design
  (`load_pax_db` raises FileNotFoundError, `calculate_f_factor`
  requires pre-loaded data) makes the missing-data case explicit.
