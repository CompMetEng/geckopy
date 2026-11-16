"""Parameter schemas for ecModel adapters.

``KeggParams``, ``UniprotParams``, ``ComplexParams``, and the core fields
of ``ModelParameters`` mirror the ``obj.params.*`` fields set in GECKO
MATLAB's adapter template. ``EvotuneParams`` holds the evotune settings and
``OkpParams`` the OpenKineticsPredictor request settings.

Ported from GECKO MATLAB: src/deprecated/adapterTemplate.m.
"""
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KeggParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default="",
        description='KEGG organism code (e.g. "sce"); empty: KEGG is not used',
    )
    gene_id: str = Field(
        default="kegg",
        description=(
            "Field in the KEGG entry matching model gene IDs. "
            "'kegg' uses the default KEGG entry identifier; other options "
            "include 'NCBI-GeneID', 'UniProt', 'Ensembl'."
        ),
    )


class UniprotParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["taxonomy", "proteome"] = Field(
        default="taxonomy",
        description="Whether 'id' is a taxonomy ID or a proteome ID",
    )
    id: str = Field(
        default="",
        description=(
            "Taxonomy ID (e.g. '559292') or proteome ID (e.g. 'UP000002311'), "
            "matching 'type'; required unless data/uniprot.tsv is provided"
        ),
    )
    gene_id_field: str = Field(
        default="gene_oln",
        description="UniProt return field matching model gene IDs",
    )
    reviewed: bool = Field(
        default=False,
        description="Restrict queries to reviewed (Swiss-Prot) entries",
    )


class ComplexParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomic_id: Optional[int] = Field(
        default=None, description="Taxonomy ID for Complex Portal query"
    )


class SpontaneousParams(BaseModel):
    """Which reactions proceed without an enzyme.

    Both rules may be given; a reaction is spontaneous if either matches.
    GECKO (MATLAB) reads the same keys, so a project needs no code for this.
    """
    model_config = ConfigDict(extra="forbid")

    name_contains: Optional[str] = Field(
        default=None,
        description="Reactions whose name contains this text are spontaneous.",
    )
    file: Optional[Path] = Field(
        default=None,
        description=(
            "TSV of spontaneous reactions, relative to the project folder: "
            "a header row, reaction identifiers in the first column and, "
            "optionally, a second column where 1 or true marks the "
            "spontaneous ones."
        ),
    )


class MatlabParams(BaseModel):
    """Settings that only GECKO (MATLAB) reads; geckopy ignores them.

    They let one model_adapter.toml configure both tools.
    """
    model_config = ConfigDict(extra="forbid")

    make_model_anaerobic: Optional[str] = Field(
        default=None,
        description="MATLAB function that constrains the ecModel to anaerobic growth.",
    )
    change_protein_biomass: Optional[str] = Field(
        default=None,
        description="MATLAB function that changes the protein content of the biomass.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Further GECKO parameters, kept under their MATLAB names.",
    )


class KcatStageParams(BaseModel):
    """One ``[[kcat_assignment.stage]]``: assigns a kcat to each reaction that is still
    open, from the first listed source with a value (``select``) or from all
    listed sources with a value, weighted by their reliability
    (``combine``)."""
    model_config = ConfigDict(extra="forbid")

    action: Literal["select", "combine"] = Field(default="select", description="What the stage does.")
    sources: list[str] = Field(
        min_length=1,
        description=(
            "Source file names (custom, ec_tree, dlkcat, okp_catapro, ...), the OKP "
            "methods of [kcat_sources.okp] as okp, and for select the BRENDA tiers "
            "database_exact, database_top and database_bottom, which read brenda.tsv; "
            "for select the first listed wins."
        ),
    )
    criteria: Literal["max", "min", "median", "mean"] = Field(
        default="max", description="Among several values of one source for a reaction."
    )
    top_origin_limit: int = Field(default=4, description="Origin limit of database_top.")
    bottom_origin_limit: int = Field(default=6, description="Origin limit of database_bottom.")
    wildcard_limit: int = Field(default=3, description="Wildcard limit of database_bottom.")


class KcatAssignmentParams(BaseModel):
    """How kcats are assigned from the kcat source files (``collect_kcats``)."""
    model_config = ConfigDict(extra="forbid")

    preset: Optional[Literal["brenda_ec_tree", "ec_tree_homologs", "ec_tree_okp", "okp", "gecko_median", "gecko_classic"]] = Field(
        default=None,
        description=(
            "Kcat assignment preset; brenda_ec_tree is used when neither a preset nor "
            "stages are given."
        ),
    )
    isozymes: bool = Field(
        default=True, description="Fill reactions without a kcat from their isozymes after the stages."
    )
    standard: bool = Field(default=True, description="Give the remaining reactions the standard kcat.")
    stage: list[KcatStageParams] = Field(
        default_factory=list, description="Stages instead of a preset."
    )

    @model_validator(mode="after")
    def _preset_or_stages(self) -> "KcatAssignmentParams":
        if self.preset is not None and self.stage:
            raise ValueError("[kcat_assignment] gives both a preset and stages; give one of the two.")
        return self


class SourceGroupRule(BaseModel):
    """One trust-tier entry in ``EvotuneParams.source_groups``.

    ``ec.source`` holds the label of the source of a kcat, such as
    ``"custom"``, ``"brenda"``, ``"ec_tree"``, ``"dlkcat"``, ``"combined"`` or
    ``"okp_kinform_h"``. An entry of ``sources`` matches the origin of the
    label or the whole label, in either case without regard to case. An entry
    ending in ``*`` is a prefix pattern on the origin: ``"okp_*"`` covers
    ``okp_eitlem_kinetics`` and ``okp_kinform_h``.
    """
    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(
        description=(
            "ec.source labels belonging to this group; an entry ending in * "
            "is a prefix pattern."
        ),
    )
    sigma0_log: float = Field(
        description=(
            "Prior standard deviation of the natural log of the kcat for this "
            "group. Lower is more trusted, and narrows that group's proposal width."
        ),
    )


class EvotuneParams(BaseModel):
    """Hyperparameters for kcat tuning against experimental data.

    A kcat whose source is in no ``source_groups`` entry takes
    ``sigma0_log_default``.
    """
    model_config = ConfigDict(extra="forbid")

    sigma0_log_default: float = Field(
        default=0.5,
        description=(
            "Prior standard deviation of the natural log of the kcat for kcats "
            "whose source matches no source_groups entry."
        ),
    )
    source_groups: dict[str, SourceGroupRule] = Field(
        default_factory=lambda: {
            "custom": SourceGroupRule(sources=["custom"], sigma0_log=0.1),
            "brenda": SourceGroupRule(sources=["brenda"], sigma0_log=0.2),
            "ec_tree": SourceGroupRule(sources=["ec_tree"], sigma0_log=0.3),
            "combined": SourceGroupRule(sources=["combined"], sigma0_log=0.4),
        },
        description=(
            "Trust tiers: group name -> the ec.source labels it covers and its "
            "sigma0_log. Groups given in the TOML replace the defaults."
        ),
    )

    rmse_threshold: float = Field(
        default=0.2,
        description="Stop once the best RMSE reaches this; negative never stops early.",
    )
    max_generations: int = Field(
        default=150, description="Hard cap on CMA-ES generations.",
    )

    max_growth_weight: float = Field(
        default=1.0,
        description=(
            "Weight on the max-growth RMSE against the flux RMSE: "
            "(rmse_flux + w * rmse_max_growth) / (w + 1). At 2 the "
            "max-growth conditions count double. Same weighting as "
            "MATLAB's maxGrowthWeight."
        ),
    )

    # Weight on a Gaussian prior term in the selection objective:
    # rmse + w * mean((log(k/k0) / sigma0_log)**2). Because sigma0_log
    # already encodes per-source confidence, this charges more for
    # moving a trusted kcat than an unlabelled one, making parsimony
    # part of what the search optimises rather than something applied
    # afterwards.
    #
    # The default is what makes a tuned kcat reproducible. The fit is
    # flat along many directions, so an unpenalised search returns an
    # arbitrary point along each: on ecYeastGEM two seeds of the same
    # unpenalised run disagree by up to 27 773-fold on individual
    # kcats, and only 69% of their large corrections even agree on the
    # direction of the change. At 0.03 the worst disagreement is 5.7x,
    # every correction beyond two-fold agrees in direction, and the fit
    # gives up 4.6%. Set 0 to score on RMSE alone.
    prior_penalty_weight: float = Field(
        default=0.03,
        description=(
            "Weight on a prior term in the selection objective, "
            "rmse + w * mean((log(k/k0) / sigma0_log)**2). Keeps large "
            "corrections reproducible across seeds. 0 scores on RMSE alone."
        ),
    )

    tie_isozymes: bool = Field(
        default=True,
        description=(
            "Give isozyme copies of one reaction a single kcat when they "
            "share a prior value and a source, so the search cannot invent "
            "a distinction the kcat assignment never made. Tying must "
            "happen before the first generation is sampled, not applied "
            "to an already-tuned result."
        ),
    )


class OkpParams(BaseModel):
    """OpenKineticsPredictor settings (used by kcats_from_okp).

    The OKP methods are set in ``[kcat_sources.okp] methods``.

    The API key is intentionally NOT here (it is a secret): provide it as a
    function argument, the OKP_API_KEY environment variable, or the file
    ``<path>/data/okpApiKey.txt``.
    """
    model_config = ConfigDict(extra="forbid")

    handle_long_sequences: str = Field(
        default="truncate",
        description="How OKP handles sequences exceeding a method's max length.",
    )
    include_similarity_columns: bool = Field(
        default=True,
        description="Append per-row similarity-to-training-data columns.",
    )
    canonicalize_substrates: bool = Field(
        default=True,
        description="Canonicalize substrate SMILES server-side before prediction.",
    )


class ModelParameters(BaseModel):
    """Top-level parameters for an ecModel adapter."""
    model_config = ConfigDict(extra="forbid")

    path: Path = Field(description="Root folder of this ecModel project")
    project_format: Optional[int] = Field(
        default=None,
        description="Layout version of the project folder; geckopy upgrade and upgradeGECKOproject write 4.",
    )
    conv_gem: Path = Field(description="Path to the conventional GEM SBML file")

    org_name: str = Field(description="Scientific name of the organism")

    # No range constraints: callers intentionally pass out-of-range values
    # (e.g. an inflated f) for sensitivity experiments and test fixtures.
    sigma: float = Field(default=0.5, description="Average enzyme saturation factor")
    p_tot: float = Field(default=0.5, description="Total protein content [g/gDw]")
    f: float = Field(default=0.5, description="Fraction of enzymes in model [g/g]")
    gr_exp: Optional[float] = Field(
        default=None,
        description=(
            "Growth rate the model should reach when not limited by nutrient "
            "uptake [1/h]; no default, functions that need it fail when it is not set"
        ),
    )

    c_source: str = Field(
        default="", description="Reaction ID for preferred carbon source exchange"
    )
    bio_rxn: str = Field(
        default="", description="Reaction ID for the biomass pseudoreaction"
    )
    enzyme_comp: str = Field(
        default="cytoplasm",
        description="Compartment where protein pseudometabolites are placed",
    )

    kegg: KeggParams = Field(default_factory=KeggParams)
    uniprot: UniprotParams = Field(default_factory=UniprotParams)
    complex: ComplexParams = Field(default_factory=ComplexParams)
    evotune: EvotuneParams = Field(default_factory=EvotuneParams)
    okp: OkpParams = Field(default_factory=OkpParams)
    spontaneous: SpontaneousParams = Field(default_factory=SpontaneousParams)
    kcat_assignment: KcatAssignmentParams = Field(default_factory=KcatAssignmentParams)
    kcat_sources: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Settings per kcat source file, e.g. kcat_sources.brenda.aggregation.",
    )
    matlab: MatlabParams = Field(default_factory=MatlabParams)
