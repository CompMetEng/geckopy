<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CompMetEng/geckopy/develop/img/gecko-logo-white.png">
  <img src="https://raw.githubusercontent.com/CompMetEng/geckopy/develop/img/gecko-logo.png" alt="GECKO" width="400px">
</picture>

[![tests](https://github.com/CompMetEng/geckopy/actions/workflows/test.yml/badge.svg)](https://github.com/CompMetEng/geckopy/actions/workflows/test.yml)
[![Release](https://img.shields.io/github/v/release/CompMetEng/geckopy?include_prereleases)](https://github.com/CompMetEng/geckopy/releases/)
[![Documentation](https://readthedocs.org/projects/gecko-docs/badge/?version=latest)](https://gecko-docs.readthedocs.io/en/latest/)
[![GitHub Discussions](https://img.shields.io/github/discussions-search?query=repo%3ACompMetEng%2Fgeckopy&label=GitHub%20Discussions)](https://github.com/CompMetEng/geckopy/discussions)

**[Documentation](https://gecko-docs.readthedocs.io/en/latest/)** · [Installation](https://gecko-docs.readthedocs.io/en/latest/installation/) · [Guide](https://gecko-docs.readthedocs.io/en/latest/guide/) · [API reference](https://gecko-docs.readthedocs.io/en/latest/api/) · [Migrate](https://gecko-docs.readthedocs.io/en/latest/migrate.html) · [Citations](https://gecko-docs.readthedocs.io/en/latest/references.html)

# geckopy

Enzyme-constrained genome-scale metabolic modelling in Python: the Python port of the [GECKO Toolbox](https://github.com/CompMetEng/GECKO) (MATLAB). It builds on [cobrapy](https://github.com/opencobra/cobrapy) and [raven-toolbox](https://github.com/CompMetEng/raven-toolbox), and reads and writes the same ecModel YAML files as GECKO, so ecModels move between the two toolboxes without conversion.

An *ec*Model adds the limits of enzymes to a genome-scale model: each enzyme has a finite capacity (its kcat) and the cell has a finite protein budget. geckopy fetches kcats from BRENDA, predicts them with DLKcat, tunes them against growth and flux data with CMA-ES, applies custom curations and integrates proteomics.

> **Status: beta (`4.0.0b1`).** The yeast-GEM tutorial runs end to end. The major and minor version follow GECKO, which is why it starts at 4.

> **Not the older geckopy.** This is a new port of GECKO. It is unrelated to the earlier `geckopy` on PyPI (release 2.0.2 and earlier), described in Carrasco Muriel, Long & Sonnenschein, *Microbiology Spectrum* 11(6):e01705-23 (2023), [doi:10.1128/spectrum.01705-23](https://doi.org/10.1128/spectrum.01705-23).

## Install

Both geckopy and raven-toolbox are pre-releases, so pip needs `--pre`:

```bash
pip install --pre geckopy
```

Extras: `geckopy[tutorial]` (matplotlib), `geckopy[evotune]` (cma, for kcat tuning) and `geckopy[dev]` (pytest and ruff). Python 3.11 or newer.

## Quick start

A minimal build, for a project folder with `model_adapter.toml`, `models/` and `data/`:

```python
from geckopy import ModelAdapter, load_conventional_gem, load_uniprot_tsv, make_ec_model, save_ec_model

adapter = ModelAdapter.from_folder("my_project")
model = load_conventional_gem(adapter)
uniprot = load_uniprot_tsv(adapter.params.path / "data" / "uniprot.tsv")
ec_model = make_ec_model(model, adapter, uniprot_db=uniprot)
save_ec_model(ec_model, "ecModel.yml", adapter=adapter)
```

The guide covers each step: [Getting started](https://gecko-docs.readthedocs.io/en/latest/guide/getting-started.html), [Building an ecModel](https://gecko-docs.readthedocs.io/en/latest/guide/building-ec-model.html) and [Gathering kcats](https://gecko-docs.readthedocs.io/en/latest/guide/gathering-kcats.html). Every public function is available as `from geckopy import X`; `geckopy.__all__` lists them, and the [API reference](https://gecko-docs.readthedocs.io/en/latest/api/) documents them. The full protocol, with kcat curation, proteomics and Crabtree-effect simulation, is [`tutorials/full_ecModel/protocol.py`](tutorials/full_ecModel/protocol.py). Fitting kcats to measured growth and fluxes is described in [`docs/evotune_kcat_tuning.md`](docs/evotune_kcat_tuning.md).

## Documentation and parity with GECKO

The protocol, installation, troubleshooting and API reference for both toolboxes are at [gecko-docs](https://gecko-docs.readthedocs.io/). The ecModel YAML format is described in the [guide](https://gecko-docs.readthedocs.io/en/latest/guide/yaml-format.html) and in [`docs/yaml_format.md`](docs/yaml_format.md). geckopy ports GECKO function by function, and each source file names its MATLAB original. The [parity ledger](https://github.com/CompMetEng/raven-gecko-parity) records, for every public function, whether the two agree and why where they do not.

## Cite us

If you have used geckopy in your work, please cite the GECKO protocol:

> Chen Y, Gustafsson J, Tafur Rangel A, Anton M, Domenzain I, Kittikunapong C, Li F, Yuan L, Nielsen J & Kerkhoven EJ. Reconstruction, simulation and analysis of enzyme-constrained metabolic models using GECKO Toolbox 3.0. *Nature Protocols* **19**, 629-667 (2024). doi:[10.1038/s41596-023-00931-7](https://doi.org/10.1038/s41596-023-00931-7).

## License

MIT. See [`LICENSE`](LICENSE).
