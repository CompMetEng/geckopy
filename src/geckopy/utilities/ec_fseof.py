"""Flux-Scanning with Enforced Objective Function for an ecModel.

Thin wrapper around :func:`raven_toolbox.analysis.fseof.fseof` that adds
ec-specific concerns on top of raven's general-purpose FSEOF:

- resolves the biomass reaction from ``model.adapter`` when not passed
  explicitly;
- optionally runs a carbon-source consistency check (warns when the
  carbon-source lower bound is tighter than its uptake at biomass-max);
- anchors the scan's floor to the production-target reaction's flux at
  biomass-optimum (matching ``ecFSEOF.m``'s ``iniTarget``), rather than
  raven-toolbox's default fixed-fraction-of-maximum floor;
- restricts targets (and the reported scan) to gene-associated
  reactions whose GPR doesn't reference the "standard" placeholder
  pseudogene, matching ``ecFSEOF.m``'s candidate-set restriction;
- fixes the target flux at each level (lower and upper bound), as
  ``ecFSEOF.m`` does;
- orders ``targets`` as ``ecFSEOF.m`` returns them: reaction targets first,
  then transport targets, each group sorted by score, with a ``transport``
  boolean column that marks the group;
- filters ``usage_prot_*`` rows out of the returned ``scan`` /
  ``targets`` DataFrames — those reactions are protein-pool plumbing,
  never engineering targets;
- labels the targets ``OE`` (over expression), ``KD`` (knock down) and
  ``KO`` (knock out), as ``ecFSEOF.m`` does, and builds the gene targets
  as ``ecFSEOF.m`` does: one action per gene, reconciled over its
  reactions with a gene essentiality test.

The selection method and output shape come from raven-toolbox: each
reaction's flux is regressed against the enforced product flux across
the scan, and a reaction is a target if its flux is not constant across
the scan and ``|correlation| >= correlation_threshold``. See
:class:`raven_toolbox.analysis.fseof.FSEOFResult` for the output schema.

Ported from GECKO MATLAB: ``src/flux_analysis/ecFSEOF.m``,
re-designed around :func:`raven_toolbox.analysis.fseof.fseof`.

Callers who want the result tables on disk can call
``result.targets.to_csv(...)`` themselves. Targets come back as one table,
reaction targets before transport targets (``ecFSEOF.m``'s ``rxnTargets``
and ``transportTargets``); the ``transport`` column tells them apart.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from raven_toolbox.analysis.fseof import FSEOFResult, fseof
from ..solver_support import optimize_model

if TYPE_CHECKING:
    from ..ec_model.ec_model import EcModel


logger = logging.getLogger(__name__)

_USAGE_PROT_PREFIX = "usage_prot_"
_STANDARD_GENE = "standard"
# The fraction of the optimum pFBA keeps, as in raven-toolbox's fseof (RAVEN's solveLP).
_PFBA_FRACTION = 0.999999
# raven-toolbox's labels and those of ecFSEOF.m.
_ACTIONS = {"amplify": "OE", "knockdown": "KD", "knockout": "KO"}
# Growth below this with a gene's usage blocked makes the gene essential, as in ecFSEOF.m.
_ESSENTIAL_GROWTH = 1e-8


@dataclass
class EcFSEOFResult(FSEOFResult):
    """The result of :func:`ec_fseof`: raven-toolbox's ``FSEOFResult`` with the
    ``target_type`` labels ``OE``, ``KD`` and ``KO`` of ``ecFSEOF.m``, and the gene
    targets of ``ecFSEOF.m`` in ``gene_table``."""

    gene_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def amplification(self) -> pd.DataFrame:
        """Targets labelled ``OE`` (flux rises with the enforced product)."""
        return self.targets[self.targets["target_type"] == "OE"].reset_index(drop=True)

    @property
    def knockout(self) -> pd.DataFrame:
        """Targets labelled ``KO`` or ``KD`` (flux falls as the enforced product rises)."""
        return self.targets[self.targets["target_type"].isin(["KO", "KD"])].reset_index(drop=True)

    @property
    def gene_targets(self) -> pd.DataFrame:
        """One row per gene of the targets, as ``ecFSEOF.m``'s ``geneTargets``: ``gene``,
        ``name``, ``slope`` (the mean absolute slope of its target reactions), ``action``,
        ``essentiality`` (``"essential"`` or empty) and ``reactions``, by slope."""
        return self.gene_table


def ec_fseof(
    model: "EcModel",
    prod_target_rxn: str,
    cs_rxn: Optional[str] = None,
    *,
    n_steps: int = 16,
    bio_rxn: Optional[str] = None,
    max_fraction: float = 0.9,
    correlation_threshold: float = 0.9,
    flux_eps: float = 1e-6,
) -> FSEOFResult:
    """Run Flux-Scanning with Enforced Objective Function on an ecModel.

    Enforces an increasing flux toward ``prod_target_rxn`` while
    optimising biomass, runs pFBA at each step, regresses every
    reaction's flux against the enforced product flux, and returns
    reactions with strong slope-vs-correlation signal as engineering
    targets.

    Parameters
    ----------
    model
        EcModel with the protein-pool / usage-rxn machinery.
    prod_target_rxn
        Reaction id for the production target (typically a
        product-exchange).
    cs_rxn
        Reaction id for the main carbon source. When passed, a warning
        is emitted if its lower bound is tighter than its uptake at
        biomass-max (a common mis-configuration). ``None`` skips the
        check.
    n_steps
        Number of enforced-flux levels in the scan. Default 16, as in
        ``ecFSEOF.m``.
    bio_rxn
        Biomass reaction id. Defaults to ``adapter.params.bio_rxn`` when
        ``model.adapter`` is set.
    max_fraction
        Top of the scan range as a fraction of the theoretical maximum
        product flux. Default 0.9. The floor of the scan range is the
        production target's flux at biomass-optimum, not a fraction of
        this ceiling.
    correlation_threshold
        A reaction is a target when ``|corr(flux, enforced)| >=`` this
        value. Default 0.9.
    flux_eps
        Tolerance below which the theoretical maximum of the production
        target flux is treated as zero (the scan then raises
        ``ValueError``). It gates only that check. Default 1e-6.

    Returns
    -------
    EcFSEOFResult
        raven-toolbox's FSEOF output dataclass, with ``usage_prot_*``
        rows filtered out of both ``scan`` and ``targets``. ``target_type``
        is ``OE`` (the absolute flux rises with production), ``KO`` (it
        falls to at most 1e-8 at the last level) or ``KD`` (it falls
        otherwise). ``targets`` has an added boolean column ``transport``
        and lists the reaction
        targets first, then the transport targets, each group in the
        order of the score ``|slope| * |correlation|`` rounded to 9 significant digits (equal scores in
        model order); a reaction is a transport target when a metabolite
        name occurs more than once among its metabolites, not counting
        the ``prot_*`` ones. A caller that takes the top N rows gets the
        top reaction targets, then transport targets. See
        :class:`raven_toolbox.analysis.fseof.FSEOFResult` for the full
        field list (``scan``, ``enforced``, ``targets``, plus the
        ``amplification`` / ``knockout`` properties).

        ``gene_targets`` follows ``ecFSEOF.m``'s ``geneTargets``: a gene
        whose target reactions have one action keeps it; with several
        actions it is ``OE`` when one of them is ``OE``, otherwise ``KO``
        when the gene is not essential and ``KD`` when it is. A gene is
        essential when growth is infeasible or below 1e-8 with the
        usage reactions of its enzymes fixed at 0, the production target
        fixed at the last enforced level and the biomass reaction
        maximised from a lower bound of 0.

    Raises
    ------
    ValueError
        If the production target cannot carry flux, or if its flux at
        maximum growth is not below the top of the scan (``max_fraction``
        of its maximum), as GECKO's ``ecFSEOF`` raises ``ecFSEOF:noRange``.
    """
    from ..adapter import resolve_param
    bio_rxn_id = resolve_param(
        model, bio_rxn, "bio_rxn",
        purpose="ec_fseof needs the biomass reaction id",
    )

    if cs_rxn is not None:
        _check_cs_consistency(model, bio_rxn_id, cs_rxn)

    min_target = _biomass_optimal_target_flux(model, bio_rxn_id, prod_target_rxn)
    max_target = _scan_ceiling(model, prod_target_rxn, max_fraction)
    # A blocked target (ceiling 0) is left to fseof, which raises its own error.
    if max_target > 0 and min_target >= max_target:
        raise ValueError(
            f"ec_fseof: the flux of {prod_target_rxn!r} at maximum growth ({min_target:g}) is not "
            f"below the top of the scan ({max_target:g}), so there is no range to scan."
        )

    result = fseof(
        model, prod_target_rxn,
        biomass_rxn=bio_rxn_id,
        n_steps=n_steps,
        max_fraction=max_fraction,
        correlation_threshold=correlation_threshold,
        flux_eps=flux_eps,
        min_target=min_target,
        equality=True,
    )
    result = _drop_usage_prot(result)
    result = _drop_ungated_reactions(result, model)
    result = _group_transport_targets(result, model)
    targets = result.targets.copy()
    targets["target_type"] = targets["target_type"].map(_ACTIONS)
    genes = _gene_targets(model, targets, bio_rxn_id, prod_target_rxn, result.enforced[-1])
    return EcFSEOFResult(scan=result.scan, enforced=result.enforced, targets=targets, gene_table=genes)


def _gene_targets(
    model: "EcModel", targets: pd.DataFrame, bio_rxn_id: str, prod_target_rxn_id: str, last_level: float,
) -> pd.DataFrame:
    """``ecFSEOF.m``'s ``geneTargets``: per gene of the targets, the mean absolute slope
    of its target reactions, one action and the result of the essentiality test."""
    columns = ["gene", "name", "slope", "action", "essentiality", "reactions"]
    target_genes = {g for genes in targets["genes"] for g in genes}
    if not target_genes:
        return pd.DataFrame(columns=columns)
    ec_genes = [g.lower() for g in model.ec.genes]
    rows = []
    with model:
        target = model.reactions.get_by_id(prod_target_rxn_id)
        target.bounds = (last_level, last_level)
        model.reactions.get_by_id(bio_rxn_id).lower_bound = 0.0
        model.objective = bio_rxn_id
        for gene in model.genes:
            if gene.id not in target_genes:
                continue
            own = targets[[gene.id in genes for genes in targets["genes"]]]
            usage = [
                f"{_USAGE_PROT_PREFIX}{enzyme}"
                for enzyme, ec_gene in zip(model.ec.enzymes, ec_genes) if ec_gene == gene.id.lower()
            ]
            with model:
                for rid in usage:
                    if rid in model.reactions:
                        model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
                growth = model.slim_optimize(error_value=float("nan"))
            essential = not np.isfinite(growth) or growth < _ESSENTIAL_GROWTH
            actions = set(own["target_type"])
            if len(actions) == 1:
                action = actions.pop()
            elif "OE" in actions:
                action = "OE"
            else:
                action = "KD" if essential else "KO"
            rows.append({
                "gene": gene.id,
                "name": gene.name,
                "slope": float(np.mean(np.abs(own["slope"].to_numpy(dtype=float)))),
                "action": action,
                "essentiality": "essential" if essential else "",
                "reactions": ";".join(own["reaction"]),
            })
    table = pd.DataFrame(rows, columns=columns)
    # A stable sort keeps genes with equal slopes in model order, as MATLAB's sort does.
    return table.sort_values("slope", ascending=False, ignore_index=True, kind="stable")


def _check_cs_consistency(
    model: "EcModel", bio_rxn_id: str, cs_rxn_id: str,
) -> None:
    """Warn when the carbon source can't deliver its biomass-max uptake.

    If the carbon-source reaction's lower bound is less negative than
    its biomass-max flux, the scan will silently flatten at the bound
    and the slopes mean less than they look.
    """
    cs_rxn_obj = model.reactions.get_by_id(cs_rxn_id)
    with model:
        model.objective = bio_rxn_id
        sol = optimize_model(model)
        if sol.status != "optimal":
            return
        cs_flux = float(sol.fluxes.get(cs_rxn_id, 0.0))
    if cs_rxn_obj.lower_bound < cs_flux:
        logger.warning(
            "ec_fseof: carbon source %r lower bound is %g but uptake "
            "at biomass-max is %g; consider tightening so the scan "
            "doesn't flatten at the bound.",
            cs_rxn_id, cs_rxn_obj.lower_bound, cs_flux,
        )


def _biomass_optimal_target_flux(
    model: "EcModel", bio_rxn_id: str, prod_target_rxn_id: str,
) -> float:
    """The production-target reaction's flux at biomass-optimum.

    Matches ``ecFSEOF.m``'s ``iniTarget``: the scan's floor is where the
    model actually is before any production is enforced, which can be
    near zero for a target with no baseline flux -- not a fixed fraction
    of the theoretical maximum.
    """
    with model:
        model.objective = bio_rxn_id
        sol = optimize_model(model)
        if sol.status != "optimal":
            return 0.0
        return float(sol.fluxes.get(prod_target_rxn_id, 0.0))


def _scan_ceiling(model: "EcModel", prod_target_rxn_id: str, max_fraction: float) -> float:
    """The top of the scan, as raven-toolbox's ``fseof`` and ``ecFSEOF.m`` take it:
    ``max_fraction`` of the target flux in the pFBA solution that maximises the target."""
    with model:
        model.objective = prod_target_rxn_id
        try:
            flux = pfba(model, fraction_of_optimum=_PFBA_FRACTION).fluxes[prod_target_rxn_id]
        except OptimizationError:
            return 0.0
    return float(flux) * max_fraction


def _gated_reaction(model: "EcModel", rxn_id: str) -> bool:
    """Whether a reaction can be an engineering target: it must have a
    GPR, and that GPR must not reference the "standard" placeholder
    pseudogene added by :func:`assign_standard_kcat` for reactions with
    no real enzyme assignment."""
    gpr = model.reactions.get_by_id(rxn_id).gene_reaction_rule
    return bool(gpr) and _STANDARD_GENE not in gpr


def _drop_ungated_reactions(result: FSEOFResult, model: "EcModel") -> FSEOFResult:
    """Restrict ``scan`` and ``targets`` to gene-associated, non-standard
    reactions, matching ``ecFSEOF.m``'s candidate-set restriction (a
    reaction with no GPR, or only the ec.kcat-standard-value pseudogene,
    can't be engineered)."""
    scan = result.scan[[_gated_reaction(model, rid) for rid in result.scan.index]]
    targets = result.targets[
        result.targets["reaction"].map(lambda rid: _gated_reaction(model, rid))
    ].reset_index(drop=True)
    return FSEOFResult(scan=scan, enforced=result.enforced, targets=targets)


def _group_transport_targets(result: FSEOFResult, model: "EcModel") -> FSEOFResult:
    """Add the ``transport`` column and list reaction targets before transport targets.

    As in ``ecFSEOF.m``, a reaction is a transport reaction when a metabolite name repeats
    among its metabolites after the ``prot_*`` metabolites are dropped. Within a group the
    reactions are ordered by their score rounded to 9 significant digits, equal scores in
    model order.
    """
    targets = result.targets.copy()
    targets["transport"] = [
        _is_transport(model.reactions.get_by_id(rid)) for rid in targets["reaction"]
    ]
    position = {rxn.id: i for i, rxn in enumerate(model.reactions)}
    targets["_model_order"] = [position[rid] for rid in targets["reaction"]]
    targets["_score"] = [float(f"{x:.9g}") for x in targets["score"]]
    targets = targets.sort_values(["transport", "_score", "_model_order"], ascending=[True, False, True], ignore_index=True)
    targets = targets.drop(columns=["_model_order", "_score"])
    return FSEOFResult(scan=result.scan, enforced=result.enforced, targets=targets)


def _is_transport(reaction) -> bool:
    names = [m.name for m in reaction.metabolites if not (m.name or "").startswith("prot_")]
    return len(names) != len(set(names))


def _drop_usage_prot(result: FSEOFResult) -> FSEOFResult:
    """Strip ``usage_prot_*`` rows from raven's ``scan`` and ``targets``.

    These reactions exist solely to draw the per-enzyme prot mets out of
    the shared pool; they have no metabolic meaning of their own and
    can't be engineered. Filtering at the boundary keeps the rest of
    raven's output structure intact (the ``gene_targets`` property and
    the ``amplification`` / ``knockout`` slices recompute correctly off
    the filtered ``targets``).
    """
    scan = result.scan[
        ~result.scan.index.astype(str).str.startswith(_USAGE_PROT_PREFIX)
    ]
    targets = result.targets[
        ~result.targets["reaction"].str.startswith(_USAGE_PROT_PREFIX)
    ].reset_index(drop=True)
    return FSEOFResult(scan=scan, enforced=result.enforced, targets=targets)
