"""Parameter schemas for ecModel adapters.

``KeggParams``, ``UniprotParams``, ``ComplexParams``, and the core fields
of ``ModelParameters`` mirror the ``obj.params.*`` fields set in GECKO
MATLAB's adapter template. ``EvotuneParams`` holds the evotune settings; ``OkpParams`` is a
geckopy-only extension with no MATLAB equivalent.

Ported from GECKO MATLAB: src/geckomat/model_adapter/adapterTemplate.m.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KeggParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="sce", description="KEGG organism code")
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
        default="559292",
        description="Taxonomy ID (e.g. '559292') or proteome ID, matching 'type'",
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


class SourceGroupRule(BaseModel):
    """One trust-tier entry in ``EvotuneParams.source_groups``.

    ``ec.source`` holds literal strings like ``"dlkcat"``, ``"brenda"``,
    and -- for OpenKineticsPredictor kcats -- the raw predictor method
    name (e.g. ``"CataPro"``), not a generic ``"okp"`` tag. A rule
    matches a given ``ec.source`` value if it's listed in ``sources``,
    or (when ``match_okp`` is True) if it equals the project's
    configured OKP method (``OkpParams.method``).
    """
    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(
        default_factory=list,
        description="Literal ec.source strings belonging to this group.",
    )
    match_okp: bool = Field(
        default=False,
        description=(
            "Also match ec.source values equal to the project's "
            "configured OpenKineticsPredictor method (OkpParams.method)."
        ),
    )


class EvotuneParams(BaseModel):
    """Hyperparameters for kcat tuning against experimental data.

    Sources not matched by any ``source_groups`` entry fall back to the
    ``*_default`` fields (matching MATLAB's ``noKcatSource`` behaviour:
    ``sigma0logDefault`` etc. apply to every kcat first, then only ones
    with a recognised source get overridden by their group's value).
    """
    model_config = ConfigDict(extra="forbid")

    sigma0_log_default: float = Field(
        default=0.5,
        description=(
            "Prior standard deviation in log-space for kcats whose source "
            "matches no source_groups entry."
        ),
    )
    source_groups: dict[str, SourceGroupRule] = Field(
        default_factory=lambda: {
            "dlkcat": SourceGroupRule(sources=["dlkcat"]),
            "brenda": SourceGroupRule(sources=["brenda"]),
            "custom": SourceGroupRule(sources=["custom"]),
        },
        description="Trust tiers: group name -> which ec.source values it covers.",
    )
    sigma0_log_source: dict[str, float] = Field(
        default_factory=lambda: {"dlkcat": 0.4, "brenda": 0.2, "custom": 0.1},
        description=(
            "Prior log-space standard deviation per group. Lower is more "
            "trusted, and narrows that group's proposal width."
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
            "max-growth conditions count double. MATLAB weights the "
            "flux term instead, so pass 0.5 to reproduce its 2."
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
    # gives up 4.6%. Set 0 to score on RMSE alone; a run reproducing
    # MATLAB, which has no such term, must do so.
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

    @model_validator(mode="after")
    def _check_group_keys(self) -> "EvotuneParams":
        """The per-source dicts must have exactly ``source_groups``'
        keys, otherwise a downstream lookup silently falls back to a
        default."""
        group_names = set(self.source_groups)
        for name in ("sigma0_log_source",):
            keys = set(getattr(self, name))
            if keys != group_names:
                raise ValueError(
                    f"EvotuneParams.{name} keys {sorted(keys)} must match "
                    f"source_groups keys {sorted(group_names)}."
                )
        return self


class OkpParams(BaseModel):
    """OpenKineticsPredictor settings (used by submit/fetch_open_kinetics_predictor).

    The API key is intentionally NOT here (it is a secret): provide it as a
    function argument, the OKP_API_KEY environment variable, or the file
    ``<path>/data/okpApiKey.txt``.
    """
    model_config = ConfigDict(extra="forbid")

    method: str = Field(
        default="CataPro",
        description=(
            "Predictor method: CataPro, CatPred, DLKcat, EITLEM, "
            "KinForm-H, KinForm-L, UniKP (see GET /api/v1/methods/)."
        ),
    )
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
    conv_gem: Path = Field(description="Path to the conventional GEM SBML file")

    org_name: str = Field(description="Scientific name of the organism")

    # No range constraints: callers intentionally pass out-of-range values
    # (e.g. an inflated f) for sensitivity experiments and test fixtures.
    sigma: float = Field(default=0.5, description="Average enzyme saturation factor")
    p_tot: float = Field(default=0.5, description="Total protein content [g/gDw]")
    f: float = Field(default=0.5, description="Fraction of enzymes in model [g/g]")
    gr_exp: float = Field(default=0.41, description="Reference growth rate [1/h]")

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

    # kcat-aggregation defaults applied when the matching function is called
    # without an explicit aggregate/criteria argument. One field per function
    # so each can keep its MATLAB-GECKO default while still being centrally
    # overridable on the adapter. See docs/kcat_aggregation.md for the
    # empirical rationale for offering `median` as an alternative to `max`.
    kcat_aggregate_brenda: Literal["max", "median"] = Field(
        default="max",
        description=(
            "Default aggregation for `fuzzy_kcat_matching` when no `aggregate` "
            "is passed: collapse the BRENDA rows matched at one search level "
            "to a single kcat. `max` matches MATLAB GECKO; `median` is more "
            "robust to engineered-mutant/non-physiological-substrate tails."
        ),
    )
    kcat_aggregate_candidates: Literal["max", "min", "median", "mean"] = Field(
        default="max",
        description=(
            "Default aggregation for `apply_kcat_list` when no `criteria` is "
            "passed: collapse multiple candidate kcats for the same reaction "
            "(e.g. several EC tokens per reaction) to one. `max` matches "
            "MATLAB GECKO."
        ),
    )
    kcat_aggregate_isozymes: Literal["max", "median", "mean"] = Field(
        default="mean",
        description=(
            "Default aggregation for `fill_kcats_from_isozymes` when no "
            "`aggregate` is passed: combine known kcats across sibling "
            "isozymes to fill a missing entry. `mean` matches MATLAB GECKO."
        ),
    )

    kegg: KeggParams = Field(default_factory=KeggParams)
    uniprot: UniprotParams = Field(default_factory=UniprotParams)
    complex: ComplexParams = Field(default_factory=ComplexParams)
    evotune: EvotuneParams = Field(default_factory=EvotuneParams)
    okp: OkpParams = Field(default_factory=OkpParams)
