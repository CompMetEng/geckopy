# GECKO to geckopy porting plan

How the MATLAB-to-Python port was scoped, what each piece maps to,
and which external libraries do what.

## Target version

Pinned to **GECKO v3.2.5** so the port has a stable reference and
the in-source `Ported from GECKO MATLAB: ...` comments resolve to
exact line numbers.

## Scope

Full port of all 82 MATLAB functions in `src/geckomat/` plus the
two reference tutorials (`full_ecModel`, `light_ecModel`). The
full ecModel build is complete; the light variant is documented
but not yet implemented — see [gecko_light_status.md](gecko_light_status.md).

## Fidelity goal

Model structure (which reactions, which metabolites, which
coefficients) and on-disk I/O must match the MATLAB output. Small
numerical differences are acceptable when they come from solver or
library choices (LP solvers do not always give the same alternate
optimum; floating-point math differs between Python and MATLAB).

## Data model

`EcModel` subclasses `cobra.Model` and has an extra `.ec`
attribute. `.ec` is an `EcData` dataclass that holds the kcat
array, enzyme list, MW array, the sparse reaction-enzyme coupling
matrix, and a few other vectors. See
[src/geckopy/ec_model/ec_data.py](../src/geckopy/ec_model/ec_data.py)
for the exact field set.

## Dependency mapping

| MATLAB-side dependency | Python-side replacement |
|---|---|
| **RAVEN** (model manipulation, FBA, I/O) | Mostly cobra-py; a handful of functions are re-implemented in geckopy itself. The remaining gap is catalogued in [raven_inventory.md](raven_inventory.md) as the suggested scope for a future ravenpy package. |
| **COBRA Toolbox** | cobra-py |
| **Gurobi / GLPK / CPLEX solvers** | The same solvers, accessed via cobra-py and its optlang layer |
| **BRENDA / UniProt / KEGG HTTP APIs** | `requests` |
| **MATLAB `.mat` reference files** | `scipy.io.loadmat` |

## Function inventory

Per-subsystem porting status. (The table below is a placeholder
from early planning; it has not been kept up to date as functions
landed. The authoritative status is the existence of the Python
module and its passing tests — see the source tree under
`src/geckopy/` and `tests/`.)

| Subsystem | MATLAB function | Python target | Deps (RAVEN / other) | Status |
|-----------|-----------------|---------------|----------------------|--------|
| model_adapter | ... | ... | ... | todo |
