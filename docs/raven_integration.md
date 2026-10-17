# raven-toolbox integration

geckopy depends on [raven-toolbox](https://github.com/CompMetEng/raven-toolbox),
the Python port of the RAVEN Toolbox. raven-toolbox owns the generic
model-manipulation primitives and the YAML I/O (including the GECKO
ec-model substructure); geckopy keeps the enzyme-constraint *algorithms*
that operate on top.

## What geckopy delegates to raven-toolbox today

| geckopy module / API | re-exports from raven-toolbox |
|---|---|
| `geckopy.EcData` | `raven_toolbox.io.EcData` |
| `geckopy.ec_model.pipeline.expand.expand_model` | `raven_toolbox.manipulation.expand.expand_model` |
| `geckopy.ec_model.pipeline.preprocess.convert_to_irreversible` | `raven_toolbox.manipulation.irreversible.convert_to_irreversible` |
| `geckopy.utilities.add_new_rxns_to_ec` (internal `_gpr_to_dnf`) | `raven_toolbox.manipulation.expand._gpr_to_dnf` |
| `geckopy.utilities.save_ec_model` | `raven_toolbox.io.write_yaml_model` |
| `geckopy.utilities.load_ec_model` | `raven_toolbox.io.read_yaml_model` |
| `geckopy.utilities.ec_fseof` | `raven_toolbox.analysis.fseof` (ec-specific post-filter: drops `usage_prot_*` from the scan + targets; adds adapter-aware `bio_rxn` resolution and optional carbon-source consistency check) |

This mirrors MATLAB GECKO + RAVEN: `model.ec` is a RAVEN-owned struct,
and `readYAMLmodel.m` / `writeYAMLmodel.m` populate / serialise it
without any GECKO involvement. Downstream consumers (geckopy / GECKO)
operate on the populated struct.

## YAML I/O split

raven-toolbox's `read_yaml_model` and `write_yaml_model`:

- handle the cobra-shaped portion (metabolites / reactions / genes / compartments / annotations);
- on read, parse the `ec-rxns` / `ec-enzymes` / `gecko_light` top-level sections into a typed `EcData` (`raven_toolbox.io.ec_data`) and attach it as `model.ec`;
- on write, serialise `model.ec` back to the same top-level sections when present;
- normalise three legacy MATLAB GECKO quirks transparently: top-level `smiles` → `annotation['smiles']`, reverse-direction `usage_prot_*` / `prot_pool_exchange` flipped to forward, bare-`-` document root merged into one mapping;
- preserve other unknown top-level keys opaquely via `model.notes['_yaml_sections']` for round-trip.

geckopy's `save_ec_model` / `load_ec_model` are thin wrappers that
add three application-level concerns on top:

1. **File-extension validation** — only `.yml` / `.yaml` is accepted (SBML ecModel I/O was dropped; the message points users at the new contract).
2. **Adapter-aware path resolution** — relative filenames resolve under `<adapter.params.path>/models/<filename>`.
3. **Provenance + diagnostics** — `save_ec_model` injects a `metaData` block (date + geckopy version + description); both wrappers fast-fail with clear messages when the YAML isn't an ecModel or the in-memory model has an empty `ec`.

On-disk format is the same RAVEN/cobrapy YAML in both directions: files
written by geckopy load in raven-toolbox (as a `cobra.Model` with
`model.ec` populated as a typed `EcData`), and files written by
raven-toolbox / MATLAB RAVEN / MATLAB GECKO load in geckopy as full
`EcModel`s.

## Planned future migrations

No outstanding migrations at the moment; the surface gap was closed by
the YAML / `EcData` / FSEOF delegations above. Candidates for the next
round (BRENDA, KEGG REST, phyl-dist, the protein-pool machinery) are
unlikely to land in raven-toolbox — they're GECKO-specific. Genuine new
overlaps will be added back to this table as raven-toolbox grows.

## What geckopy keeps in-tree (not raven-toolbox territory)

- The `EcModel` class (subclass of `cobra.Model` with `model.ec` +
  `model.adapter` + `model.enzymes`) and the `make_ec_model` 9-stage
  pipeline (GECKO-specific).
- The `Enzyme` proxy (`model.enzymes.get_by_id(...)`).
- BRENDA loaders + fuzzy kcat matching + DLKcat wrappers
  (`geckopy.gather_kcats`, `geckopy.databases.brenda`).
- Per-gene KEGG REST-API downloader (`geckopy.databases.kegg_download`).
  raven-toolbox has a different KEGG concern: bulk-file dumps for de-novo
  reconstruction (`raven_toolbox.reconstruction.kegg.download_kegg_dump`).
- Phylogenetic-distance matrix loading (`geckopy.databases.phyl_dist`).
  raven-toolbox has a different concern: KEGG taxonomy parsing
  (`raven_toolbox.reconstruction.kegg.taxonomy`).
- Everything under `geckopy.limit_proteins`,
  `geckopy.kcat_tuning`,
  `geckopy.ec_model.pipeline.protein_pool`, etc. — pure
  enzyme-constraint mechanics, no general-GEM analog in RAVEN.
