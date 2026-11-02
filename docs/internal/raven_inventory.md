# RAVEN function inventory

GECKO MATLAB is built on top of
[RAVEN](https://github.com/CompMetEng/RAVEN), a MATLAB toolkit
for genome-scale metabolic modelling. RAVEN handles all the
low-level work: parsing SBML files, editing reactions and
metabolites, solving FBAs, manipulating gene-protein-reaction
rules, and so on.

As the geckopy port progressed, every RAVEN call in the MATLAB
source had to land somewhere on the Python side. There were three
possibilities:

1. **cobra-py already does it.** Most cases — cobra-py is RAVEN's
   Python-side equivalent and covers the bread-and-butter
   operations.
2. **geckopy implements it internally.** A few RAVEN functions
   were specific enough to GECKO's needs that re-implementing them
   in geckopy itself made more sense than pulling in a separate
   dependency.
3. **No Python equivalent exists yet.** These are the
   functions worth eventually porting into a separate **ravenpy**
   package. cobra-py either doesn't do them at all or does them
   differently enough that callers can't drop one in for the
   other.

This document catalogues all three groups, plus a fourth (RAVEN
helpers that don't translate to Python at all — MATLAB-specific
glue like file dialogs).

The list is based on a scan of `src/geckomat/**/*.m` in GECKO
MATLAB and the existing pipeline modules in
`src/geckopy/ec_model/pipeline/`. It only covers RAVEN functions
that GECKO actually uses; the full RAVEN library is much larger.

## Why this document is useful

Two reasons:

- **For porting more MATLAB code.** If you're translating a piece
  of MATLAB GECKO (or any RAVEN-based code) to Python, look here
  first to find the cobra-py / geckopy substitute.
- **For scoping ravenpy.** Group 3 is the proposed v0.1 of a
  ravenpy package — a Python port of the bits of RAVEN that
  cobra-py doesn't cover.

## 1. Replaced by cobra-py directly

These are the easy ones. RAVEN's API was MATLAB-flavoured (verb +
model + string id); cobra-py's is object-oriented (model.<thing>.
get_by_id(...).<attr>), but they do the same thing.

| RAVEN | cobra-py equivalent |
|---|---|
| `setParam(model, 'lb'\|'ub'\|'eq', rxn, val)` | `model.reactions.get_by_id(rxn).lower_bound = val` (or `.upper_bound`; for `obj`, `model.objective = rxn`) |
| `solveLP(model)` | `model.optimize()` |
| `importModel(filename)` | `cobra.io.read_sbml_model(filename)` |
| `exportModel(model, filename)` | `cobra.io.write_sbml_model(model, filename)` |
| `getIndexes(model, ids, 'rxns'\|'mets'\|'genes')` | `model.reactions.get_by_id(id)` (cobra-py is keyed by id, no numeric indices needed) |
| `addRxns(model, rxnsToAdd)` | `model.add_reactions([cobra.Reaction(...)])` |
| `addMets(model, metsToAdd)` | `model.add_metabolites([cobra.Metabolite(...)])` |
| `removeReactions(model, rxnList)` | `model.remove_reactions(rxnList)` |
| `removeMets(model, metList)` | `model.remove_metabolites(metList)` |
| `removeGenes(model, geneList)` | `model.genes.remove(model.genes.get_by_id(g))` per id |
| `addGenes(model, geneList)` | not needed — cobra-py adds genes automatically when a reaction's GPR mentions them |

## 2. Replaced by geckopy internals

A few RAVEN functions are tied closely enough to GECKO's needs
(isozyme expansion, reversibility splitting) that geckopy
re-implements them in the build pipeline rather than depending
on an external package.

| RAVEN | geckopy module |
|---|---|
| `expandModel` | `src/geckopy/ec_model/pipeline/expand.py::expand_model`. Uses cobra-py's GPR abstract syntax tree instead of RAVEN's string-manipulation approach (more robust against unusual GPR formatting). |
| `convertToIrrev` | `src/geckopy/ec_model/pipeline/preprocess.py::convert_to_irreversible`. Splits each reversible reaction into a forward and a reverse half, the reverse getting a `_REV` suffix. |
| pseudoreaction handling / backward-only inversion / bound validation | Also in `preprocess.py` — stages 1-4 of the build pipeline. |
| `readYAMLmodel`, `writeYAMLmodel` (cobra portion only) | `src/geckopy/utilities/load_ec_model.py`, `save_ec_model.py`. But geckopy uses its own canonical YAML schema — see [yaml_format.md](yaml_format.md) for the migration story. |

## 3. RAVEN-only — ravenpy candidates

These have no drop-in cobra-py equivalent. Each one provides a
useful, biochemically-informed structural or optimization utility
that's orthogonal to cobra-py's core FBA engine. They're the
suggested scope for a future **ravenpy** package.

| Function | What it does | Why cobra-py can't do it natively |
|---|---|---|
| `fillGaps` | Repairs an infeasible model by adding reactions from a reference model (gap-filling) | cobra-py has `cobra.flux_analysis.gapfill` but it's a different algorithm with a different signature; calling it as a `fillGaps` drop-in doesn't work |
| `simplifyModel` | Removes dead-end metabolites and orphan reactions by iteratively propagating constraints | cobra-py has no built-in equivalent; everyone re-rolls it |
| `contractModel` | Reduces a model by removing reactions that flux analysis shows are redundant | not in cobra-py |
| `haveFlux(model, rxnList)` | Asks per reaction "could this carry non-zero flux?" with numerical-tolerance handling | cobra-py has `find_blocked_reactions` but the tolerance semantics differ |
| `getAllowedBounds(model)` | Tightens each reaction's bounds via FVA under the current constraints | cobra-py has `flux_variability_analysis` but it returns a DataFrame, not a tightened model |
| `gprConsolidate(model, grRules)` | Simplifies GPRs using Boolean algebra (`(A or A) and B` -> `A and B`) | cobra-py's `GPR` class parses and evaluates, but doesn't simplify |
| `standardizeGrRules(model)` | Normalises GPR syntax (consistent parenthesisation, whitespace) | cobra-py's parser is liberal in what it accepts, but doesn't normalise on output |
| `changeGeneAssociation(model, rxn, newGpr)` | Replaces a reaction's GPR and keeps the gene list consistent | `rxn.gene_reaction_rule = ...` works in cobra-py, but doesn't garbage-collect orphan genes |
| `getGenesFromGrRules(grRules)` | Extracts gene IDs from a GPR rule string | cobra-py's `GPR.genes` works on the parsed AST, not raw strings |
| `mergeCompartments` | Consolidates two compartments into one | not in cobra-py |
| `copyToCompartments(model, mets, src, tgt)` | Replicates metabolites and their reactions into another compartment | not in cobra-py |
| `removeBadRxns` | Flags reactions with structural issues (impossible stoichiometry, missing genes, ...) for review | not in cobra-py |
| `readYAMLmodel`, `writeYAMLmodel` (RAVEN dialect) | Reads / writes the RAVEN-flavoured YAML, which differs schematically from cobra-py's | needed only for reading legacy `ecYeastGEM.yml`-style files; geckopy uses a different YAML format (see [yaml_format.md](yaml_format.md)) |

## 4. MATLAB-specific helpers — not needed in Python

RAVEN bits that don't translate at all, either because they're
MATLAB-specific (struct manipulation, file dialogs) or because
their job is already handled inside cobra-py / optlang.

| Function | Why it's not ported |
|---|---|
| `solveLP(..., minFlux, hsSol)` hot-start | MATLAB-specific solver state; optlang / cobra-py handle warm-start internally |
| `setRavenSolver(...)` | MATLAB CPLEX / Gurobi setup; cobra-py uses `model.solver = "gurobi"` |
| `importExcelModel`, `checkFileExistence`, dialog UI prompts | File I/O glue, not algorithmic |
| `convertCharArray`, `dispEM` | MATLAB cell-array and struct utilities |

## Suggested ravenpy scope

If someone starts a ravenpy package, the highest-value v0.1
covers the GPR utilities (`gprConsolidate`,
`standardizeGrRules`, `getGenesFromGrRules`) plus `fillGaps` and
`simplifyModel`. These are the most-called RAVEN gaps in MATLAB
GECKO and have the clearest "RAVEN does this, cobra-py doesn't"
signature.

The compartment-manipulation functions
(`mergeCompartments`, `copyToCompartments`) and `removeBadRxns`
are next on the list but get called in fewer places.

The YAML reader/writer is its own concern. geckopy already picked
a canonical cobra-py-compatible schema (see
[yaml_format.md](yaml_format.md)), so the RAVEN-dialect reader is
really only useful for backward compatibility with legacy files.
