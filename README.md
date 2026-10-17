# geckopy

[![tests](https://github.com/CompMetEng/geckopy/actions/workflows/test.yml/badge.svg)](https://github.com/CompMetEng/geckopy/actions/workflows/test.yml)

Enzyme-constrained genome-scale metabolic modelling in Python.

> **Full documentation for both MATLAB GECKO and geckopy — the six-stage
> protocol, installation, troubleshooting, and complete API reference —
> lives at [gecko-docs.readthedocs.io](https://gecko-docs.readthedocs.io/).**
> This README only covers install and a quick orientation; that site is
> the place to actually learn or look things up.

> **Disambiguation.** This is a new, from-scratch port of the
> [GECKO Toolbox](https://github.com/CompMetEng/GECKO) (MATLAB). It is unrelated to
> the earlier and separate `geckopy` Python package
> ([on PyPI as release 2.0.2 and earlier](https://pypi.org/project/geckopy/2.0.2/))
> described in Carrasco Muriel, Long & Sonnenschein, *Microbiology Spectrum*
> 11(6):e01705-23 (2023), [doi:10.1128/spectrum.01705-23](https://doi.org/10.1128/spectrum.01705-23).

## What this is

Genome-scale metabolic models (GEMs) describe an organism's chemistry as a
big network of reactions. A standard GEM lets each reaction run as fast as
its substrates allow. In reality, enzymes catalyse those reactions, and
each enzyme has a finite capacity (its kcat) and a finite supply (the
cell's total protein budget). An *ec*Model (enzyme-constrained model)
adds those limits to the network.

geckopy is a Python port of the [GECKO Toolbox](https://github.com/CompMetEng/GECKO)
(MATLAB). It builds on [cobrapy](https://github.com/opencobra/cobrapy) for
the standard modelling layer, then layers GECKO's enzyme-constraint machinery
on top: it fetches kcat values from BRENDA, predicts them with DLKcat, tunes
them against experimental growth/flux data with CMA-ES, applies custom
curations, and integrates proteomics measurements.

> **Status: beta (`4.0.0b1`).** All the MATLAB GECKO 3.2.5 functions used in
> the standard ecModel build are ported, and the yeast-GEM tutorial runs
> end-to-end. geckopy's major and minor version number (x.y.-) will follow
> the MATLAB GECKO numbering, justifying a jump to version 4.

## Install

geckopy depends on [raven-toolbox](https://github.com/CompMetEng/raven-toolbox)
(the Python port of the RAVEN Toolbox). Both are currently pre-releases on
PyPI, so a plain `pip install geckopy` will install the wrong version — `--pre` is
required:

```bash
pip install --pre geckopy
```

`raven-toolbox` doesn't need to be named separately: geckopy's own dependency
on it already specifies `>=3.0.0b1`, and a specifier that names a pre-release
opts pip into matching pre-releases for that package too, `--pre` or not. To
pin the exact versions this was written against instead of "whatever's
newest," name both explicitly:

```bash
pip install raven-toolbox==3.0.0b1 geckopy==4.0.0b1
```

Optional extras:

```bash
# Adds matplotlib (needed to render plots from the tutorial)
pip install --pre "geckopy[tutorial]"

# Adds cma (needed for CMA-ES kcat tuning, geckopy.kcat_tuning.evotune)
pip install --pre "geckopy[evotune]"

# Adds pytest + ruff (only needed if you're contributing)
pip install --pre "geckopy[dev]"
```

Requires Python 3.11 or newer.

## Quick start

A minimal end-to-end build, assuming you have a project folder set up the
way the tutorial expects (`model_adapter.toml` + `models/` + `data/`):

```python
from geckopy import (
    ModelAdapter,
    load_conventional_gem,
    load_uniprot_tsv,
    make_ec_model,
    save_ec_model,
)

# 1. The adapter holds organism-specific parameters (taxonomy id,
#    biomass reaction, average enzyme saturation, etc.). They live in
#    model_adapter.toml so they're easy to swap per organism.
adapter = ModelAdapter.from_folder("my_project")

# 2. Load the starting GEM (the regular metabolic model you want to
#    extend) and a cached UniProt query (for enzyme MW and sequence).
model = load_conventional_gem(adapter)
uniprot = load_uniprot_tsv(adapter.params.path / "data" / "uniprot.tsv")

# 3. Build the ecModel. This adds protein pseudo-metabolites, a shared
#    protein pool, and per-enzyme usage reactions on top of the GEM.
ec_model = make_ec_model(model, adapter, uniprot_db=uniprot)

# 4. Save in geckopy's YAML format.
save_ec_model(ec_model, "ecModel.yml", adapter=adapter)
```

Every public function (loaders, kcat fetchers, FBA/FVA helpers,
YAML I/O, ...) is reachable as `from geckopy import X` -- the
subpackages exist for code organisation but you don't have to use
them. See `geckopy.__all__` for the full list (~80 names).

The full protocol (which adds kcat curation from BRENDA and DLKcat,
proteomics integration, and Crabtree-effect simulation) is in
[`tutorials/full_ecModel/protocol.py`](tutorials/full_ecModel/protocol.py).
That script reproduces the MATLAB GECKO Nature Protocols tutorial in
Python, step by step, and is the easiest way to see geckopy in action.

Fitting kcats against measured growth rates and exchange fluxes (rather
than looking them up) is a separate workflow, `geckopy.kcat_tuning.evotune`
— see [`docs/evotune_kcat_tuning.md`](docs/evotune_kcat_tuning.md).

## On-disk format

geckopy reads and writes ecModels as **YAML**: a canonical,
human-readable format that is a strict superset of cobrapy's YAML
schema, with three extra top-level keys for the ec data (`ec-rxns`,
`ec-enzymes`, `gecko_light`). Standard cobrapy tools (escher, memote,
...) can read the cobra portion and silently ignore the GECKO
extensions. The same format is read and written by MATLAB GECKO /
RAVEN, so ecModels exchange between the two toolboxes with no
translator. The schema and the parsing live in raven-toolbox.

See [`docs/yaml_format.md`](docs/yaml_format.md) for the YAML
specification and how it differs from the legacy MATLAB / RAVEN format.

## Relationship to MATLAB GECKO

geckopy is a function-by-function port: algorithmic fidelity comes
first, Pythonic idiom second. Each ported source file carries a
`Ported from GECKO MATLAB: <path>` docstring header. For current
MATLAB ↔ Python behavioural differences, see [gecko-docs.readthedocs.io](https://gecko-docs.readthedocs.io/), the
shared documentation site for both toolboxes.
## License

MIT. See [`LICENSE`](LICENSE), or the `license` field of
[`pyproject.toml`](pyproject.toml).
