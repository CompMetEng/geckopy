# geckopy ecModel YAML format

This document defines the YAML format that geckopy uses to save and
load ecModels.

## It *is* the cobrapy YAML format, plus a few GECKO keys

There is no bespoke geckopy format. An ecModel is saved as **exactly
the YAML that cobrapy writes** (`cobra.io.save_yaml_model`), with a
handful of GECKO-specific top-level keys added alongside for the data
cobra doesn't model: kcat values, the enzyme list (MW, sequence,
concentration), and the reaction-enzyme coupling matrix.

geckopy delegates the I/O to
[raven-toolbox](https://github.com/CompMetEng/raven-toolbox), which
calls cobrapy for the cobra-shaped portion and parses the GECKO
`ec-rxns` / `ec-enzymes` / `gecko_light` sections into a typed
`EcData` substructure attached as `model.ec`. That means:

- any cobrapy-aware tool (Escher, Memote, plain cobra) loads the file
  and simply ignores the extra `ec-*` keys;
- MATLAB GECKO / RAVEN write and read the same format, so ecModels
  exchange between the two toolboxes with no translator.

## Design goals

1. **Identical to cobrapy.** The cobra-shaped portion is exactly what
   `cobra.io.model_to_dict` / `model_from_dict` round-trips; no custom
   parser, no custom schema.
2. **Round-trippable with MATLAB GECKO.** RAVEN's `writeYAMLmodel`
   and `readYAMLmodel` emit and accept this same format.
3. **Backward compatible.** Both geckopy and RAVEN also load the
   legacy MATLAB / RAVEN layout (see the last section). Empty strings
   and NaN values can be omitted; the reader fills in the defaults.

## Top-level structure

The whole file is one YAML mapping (a top-level dictionary). It
contains the following keys:

| Key | Required? | Owner | What it holds |
|---|---|---|---|
| `id` | yes | cobra | Model id |
| `name` | optional | cobra | Human-readable name |
| `version` | optional | cobra | cobra schema version (omit if unused) |
| `compartments` | optional | cobra | Mapping of compartment id -> name |
| `metabolites` | yes | cobra | List of metabolite entries (see below) |
| `reactions` | yes | cobra | List of reaction entries (see below) |
| `genes` | yes | cobra | List of gene entries (see below) |
| `ec-rxns` | yes (for ecModel) | raven-toolbox | Per-reaction ec data (see below) |
| `ec-enzymes` | yes (for ecModel) | raven-toolbox | Per-enzyme ec data (see below) |
| `gecko_light` | optional | raven-toolbox | Boolean flag; defaults to `false` |
| `metaData` | optional | raven-toolbox | Free-form provenance: version, date, author, taxonomy, note |

The "Owner" column says which library defines and parses each field.
Tools that only understand the cobra side (cobrapy, escher, memote,
...) silently ignore the GECKO-specific keys (`ec-rxns`,
`ec-enzymes`, `gecko_light`, `metaData`), so the same file loads
cleanly as a plain cobra model in those tools — you just lose the ec
layer.

## Cobra-shaped section

This is exactly the schema produced by
`cobra.io.dict.model_to_dict`. A few conventions worth noting:

- `compartments` is a flat mapping (`{c: cytosol, e: extracellular}`),
  not a list.
- `metabolites`, `reactions`, `genes` are lists of plain mappings
  (one per metabolite/reaction/gene).
- A reaction's `metabolites` field is a flat mapping
  `met_id -> stoichiometric_coefficient`.
- `annotation` is a mapping `key -> [list, of, strings]`. RAVEN's
  per-metabolite `smiles` field lives at
  `annotation: {smiles: ["..."]}`.

Example metabolite:

```yaml
- id: s_0001
  name: "(1->3)-beta-D-glucan"
  compartment: ce
  formula: C6H10O5
  charge: 0
  annotation:
    bigg.metabolite: ["13BDglcn"]
    chebi: ["CHEBI:37671"]
    kegg.compound: ["C00965"]
    metanetx.chemical: ["MNXM6492"]
    sbo: ["SBO:0000247"]
    smiles: ["C(C1C(C(C(C(O1)O)O)O)O)O"]
```

Example reaction:

```yaml
- id: r_0001
  name: "(R)-lactate:ferricytochrome-c 2-oxidoreductase"
  metabolites:
    s_0027: -1.0
    s_0556: 1.0
  lower_bound: 0
  upper_bound: 1000
  gene_reaction_rule: "(YDL174C and YML054C) or (YEL024W and YML054C)"
  subsystem: "Oxidative phosphorylation"
  annotation:
    ec-code: ["1.1.2.4"]
    kegg.reaction: ["R00196"]
```

## `ec-rxns`: per-reaction ec data

A list, one entry per catalysed reaction. When the build pipeline
splits a reaction across isozymes (`_EXP_<N>` suffix) or into
forward/reverse pairs (`_REV` suffix), each variant gets its own
entry here. Fields:

| Field | Type | Required? | Notes |
|---|---|---|---|
| `id` | string | yes | Matches a model reaction id; may carry `_EXP_<N>` and/or `_REV` suffixes |
| `kcat` | number | yes | Turnover number in s^-1; write NaN as `.nan` |
| `source` | string | optional | Where the kcat came from (`"brenda"`, `"dlkcat"`, `"manual"`, ...). Default `""` |
| `notes` | string | optional | Free-form note. Default `""` |
| `eccodes` | string \| list | optional | One EC code as a string (`"1.1.2.4"`), or a list when several apply. Default `""` |
| `enzymes` | mapping `enzyme_id -> stoich` | optional | Which enzymes catalyse this reaction, and the subunit count of each. Every key must appear in `ec-enzymes`. A reaction without enzymes (such as a `standard` kcat row) has no `enzymes` entry; the loader also reads the `enzymes: !!omap` and `enzymes: !!omap []` forms that earlier versions of GECKO, RAVEN and raven-toolbox wrote for it |

Example:

```yaml
- id: r_0001_EXP_1
  kcat: 1500.0
  source: "brenda"
  eccodes: ["1.1.2.4", "1.1.99.40"]
  enzymes:
    P00045: 1
    P32891: 1
```

## `ec-enzymes`: per-enzyme ec data

A list, one entry per unique enzyme. Fields:

| Field | Type | Required? | Notes |
|---|---|---|---|
| `genes` | string | yes | Gene id; must match a `genes[].id` from the cobra section |
| `enzymes` | string | yes | Enzyme id (usually a UniProt accession) |
| `mw` | number | optional | Molecular weight in Da. Write NaN as `.nan`. Default NaN |
| `sequence` | string | optional | Amino acid sequence. Default `""` |
| `concs` | number | optional | Measured concentration in mmol/gDW (from proteomics). Write NaN as `.nan`. Default NaN |

The `standard` pseudo-gene (added by `get_standard_kcat` to cover
reactions without a real gene assignment) appears as both `genes:
"standard"` and `enzymes: "standard"`.

Example:

```yaml
- genes: "Q0045"
  enzymes: "P00401"
  mw: 58798.0
  sequence: "MVQRWLYSTNAKDIAVLY..."
```

## Sparse defaults

To keep files compact, writers can omit any field that takes the
documented default (empty string, NaN). Readers fill the defaults
back in when loading. The exception is numeric NaN that you want
to keep explicit (e.g. "this kcat is deliberately unknown"): write
it as `.nan` so the reader sees it.

## Legacy MATLAB / RAVEN format

Older RAVEN/GECKO ecModels used a slightly different schema. **Both
geckopy and RAVEN load it** — geckopy's loader normalises it on read
(`_normalize_legacy_layout`), and RAVEN's `readYAMLmodel` auto-detects
it. The schema differences are:

| Aspect | Legacy MATLAB / RAVEN | Current (cobrapy + GECKO keys) |
|---|---|---|
| Top-level `id` / `name` / `version` | nested under `metaData` | top level (where cobra reads them) |
| `smiles` per metabolite | top-level metabolite key | under `annotation` as `{smiles: ["..."]}` |
| Annotation values | scalar strings | list of strings |
| `metaData.geckoLight` | inside `metaData`, string `"true"`/`"false"` | top-level `gecko_light: <bool>` |
| Outer `---` document marker | present | absent (cobra omits it) |
| `metaData` provenance fields | inside `metaData` | inside `metaData` (unchanged) |

The normalisation lifts `id`/`name`/`version` to the top level and
moves per-metabolite `smiles` into `annotation`; cobra tolerates the
scalar-vs-list annotation difference on read. A legacy file written as
a bare `---` sequence of single-key maps (no `!!omap`) is merged into a
mapping by geckopy's reader. In short: the two layouts hold the same
information, and the tooling reads both.
