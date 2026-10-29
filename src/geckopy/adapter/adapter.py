"""Project-level configuration for a geckopy ecModel.

A ``ModelAdapter`` bundles everything that's specific to *this*
ecModel project: the organism's name and taxonomy id, the path
to the starting GEM, the path to UniProt and other cached data,
the biomass reaction, the assumed enzyme saturation factor, and so
on. Downstream geckopy functions take an adapter as input rather
than dozens of separate keyword arguments.

A project lives in a folder containing:

- ``model_adapter.toml`` — the parameters (see
  ``geckopy.adapter.params.ModelParameters`` for the schema).
- ``models/`` — the starting GEM and any ecModel outputs.
- ``data/`` — cached databases (UniProt, BRENDA, ComplexPortal,
  proteomics, etc.).

Ported from GECKO MATLAB: src/geckomat/model_adapter/ModelAdapter.m.

Then:

.. code-block:: python

    adapter = ModelAdapter.from_folder("my_project")

For most users that's all you need. Subclass ``ModelAdapter``
only when an organism needs custom logic — typically a custom
``get_spontaneous_reactions`` or a custom gene-id mapping.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from .params import ModelParameters

if TYPE_CHECKING:
    import cobra


class ModelAdapter:
    """Holds the organism-specific parameters and overridable
    helpers for one ecModel project.

    Most users don't subclass this directly: they make a folder
    with a ``model_adapter.toml`` and call
    ``ModelAdapter.from_folder(folder)``.

    Subclass only when a particular organism needs custom logic
    (e.g. an unusual way to identify spontaneous reactions, or a
    bespoke gene-id mapping that the default UniProt-table-driven
    one can't handle).
    """

    def __init__(self, params: ModelParameters):
        self.params = params

    @classmethod
    def from_folder(cls, folder: str | Path) -> "ModelAdapter":
        """Load an adapter from a folder containing model_adapter.toml.

        Parameters
        ----------
        folder
            Path to the ecModel project folder. Must contain
            `model_adapter.toml`.

        Returns
        -------
        A ModelAdapter (or subclass) instance with parameters loaded
        from the TOML file.
        """
        folder = Path(folder).resolve()
        config_path = folder / "model_adapter.toml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No model_adapter.toml found in {folder}. "
                "See docs for the expected folder structure."
            )

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # The path field is injected from the folder location, not the TOML,
        # so that the TOML file is location-independent.
        data["path"] = folder
        params = ModelParameters.model_validate(data)

        # Resolve project paths relative to the folder if they are not absolute.
        if not params.conv_gem.is_absolute():
            params.conv_gem = folder / params.conv_gem
        if params.spontaneous.file is not None and not params.spontaneous.file.is_absolute():
            params.spontaneous.file = folder / params.spontaneous.file

        # Note: a subclass in the project's `adapter.py` is not auto-discovered;
        # import and instantiate your ModelAdapter subclass explicitly.
        return cls(params)

    # Overridable methods. The default implementations match the MATLAB
    # base class behavior. Subclass and override as needed.

    def get_spontaneous_reactions(self, model: "cobra.Model") -> list[str]:
        """Return IDs of reactions that proceed without enzyme catalysis.

        The default reads the ``[spontaneous]`` section of the adapter:
        reactions whose name contains ``name_contains``, and those listed
        in ``file``. Without that section it returns an empty list.
        Override in a subclass for other rules.
        """
        rules = self.params.spontaneous
        if rules.name_contains is None and rules.file is None:
            return []
        listed: set[str] = set()
        if rules.file is not None:
            with open(rules.file, "r", encoding="utf-8") as f:
                next(f)  # header
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if not parts[0]:
                        continue
                    if len(parts) == 1 or parts[1].strip().lower() in ("1", "true"):
                        listed.add(parts[0].strip('"'))
        return [
            r.id for r in model.reactions
            if r.id in listed
            or (rules.name_contains is not None and rules.name_contains in (r.name or ""))
        ]

    def get_uniprot_compatible_genes(self, in_genes: list[str]) -> list[str]:
        """Map model gene IDs to IDs usable in UniProt queries.

        The default returns the input unchanged. Override when the model
        uses gene identifiers that need transformation before UniProt
        lookup (for example, stripping a prefix).
        """
        return list(in_genes)

    def get_uniprot_ids_from_table(self, model_genes: list[str]) -> list[str]:
        """Map model gene IDs to UniProt IDs via data/uniprotConversion.tsv.

        If the conversion table is absent, returns the input unchanged.
        The table is a two-column TSV with header row: model gene ID,
        UniProt ID. The cells are read as MATLAB's ``textscan`` with ``%q``
        reads them (see :func:`geckopy.databases.uniprot_loader.read_quoted_tsv`),
        so a cell in double quotes is given without the quotes.
        """
        table_path = self.params.path / "data" / "uniprotConversion.tsv"
        if not table_path.is_file():
            return list(model_genes)

        from ..databases.uniprot_loader import read_quoted_tsv

        with open(table_path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
        # A gene listed twice takes its first row, as MATLAB's ismember does.
        mapping: dict[str, str] = {}
        for gene, uniprot_id in read_quoted_tsv(text, 2, header_lines=1):
            mapping.setdefault(gene, uniprot_id)

        return [mapping.get(g, g) for g in model_genes]

    def get_kegg_compatible_genes(self, in_genes: list[str]) -> list[str]:
        """Map model gene IDs to IDs usable in KEGG ``genes`` column lookups.

        The default returns the input unchanged. Override when the model
        uses gene identifiers that need transformation before KEGG
        lookup. Parallels ``get_uniprot_compatible_genes``.
        """
        return list(in_genes)

    def get_brenda_db_folder(self) -> Path:
        """Return the folder where the BRENDA database files are stored.

        Prefers `<project_path>/data/brenda/` (consistent with
        `get_phyl_dist_path`, and writable on read-only installs), where
        `geckopy brenda-refresh` writes. Without a project copy, the files
        of geckopy's BRENDA release on toolbox-assets are downloaded once
        and cached (see `geckopy.databases.fetch_brenda_asset`). Override
        to point elsewhere.
        """
        project = self.params.path / "data" / "brenda"
        if project.is_dir():
            return project
        from ..databases import fetch_brenda_asset

        return fetch_brenda_asset()

    def get_phyl_dist_path(self) -> Path:
        """Return the path to the phylogenetic distance structure file.

        Default: `<project_path>/data/PhylDist.mat`.
        """
        return self.params.path / "data" / "PhylDist.mat"
