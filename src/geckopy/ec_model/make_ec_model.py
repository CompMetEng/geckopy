"""Build an EcModel from a conventional GEM.

``make_ec_model`` is the top-level entry point for turning a plain
cobra metabolic model into an enzyme-constrained model. It runs
the 12-stage GECKO pipeline (preprocess, expand isozymes, look up
UniProt data, add protein pseudometabolites, set up the shared
pool, ...) and returns a populated ``EcModel``.

Note that kcat values are NOT set by this function. You usually
follow up with one of:

- ``geckopy.gather_kcats`` functions (fetch from BRENDA, DLKcat,
  custom files), then ``apply_kcat_constraints`` to push the
  values into the LP, or
- ``ec_model.pipeline.set_kcat.set_kcat_for_reactions`` for a
  small manual set.

The MATLAB tutorials in
``tutorials/full_ecModel/protocol.py`` show the full flow.

Ported from GECKO MATLAB: src/geckomat/change_model/makeEcModel.m.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import cobra
from raven_toolbox.utils.sort import sort_identifiers

from ..databases import UniprotDB, load_kegg_tsv, load_uniprot_tsv
from ..solver_support import repair_scipy_problem, switch_to_scipy
from .ec_model import EcModel
from .pipeline import (
    add_protein_pool_exchange_reaction,
    add_protein_pool_pseudometabolite,
    add_protein_pseudometabolites,
    add_protein_usage_reactions,
    allocate_ec_and_coupling_light,
    allocate_ec_for_catalyzed_reactions,
    build_rxn_enzyme_coupling,
    convert_to_irreversible,
    expand_model,
    invert_backwards_only_reactions,
    populate_enzyme_data,
    remove_pseudoreaction_gprs,
)

if TYPE_CHECKING:
    from ..adapter import ModelAdapter
    from ..databases.kegg_loader import KeggDB


logger = logging.getLogger(__name__)

_EXP_SUFFIX = re.compile(r"_EXP_\d+$")


def _check_inputs(model: cobra.Model, adapter: "ModelAdapter") -> None:
    """Refuse inputs that GECKO's ``makeEcModel`` refuses."""
    enzyme_comp = adapter.params.enzyme_comp
    if enzyme_comp not in set(model.compartments.values()):
        raise ValueError(
            f"Compartment {enzyme_comp} (specified in params.enzyme_comp) "
            "cannot be found in the compartment names of the model."
        )
    if any(met.id.startswith("prot_") for met in model.metabolites):
        raise ValueError(
            "The identifiers in model.metabolites are not allowed to start "
            "with 'prot_'."
        )
    if any(
        rxn.id.startswith(("usage_prot_", "prot_pool"))
        or rxn.id.endswith("_REV")
        or _EXP_SUFFIX.search(rxn.id)
        for rxn in model.reactions
    ):
        raise ValueError(
            "The identifiers in model.reactions are not allowed to start "
            "with 'usage_prot' or 'prot_pool', or end with '_REV' or "
            "'_EXP_[digit]'."
        )
    if not model.genes:
        raise ValueError(
            "None of the proteins in uniprot.tsv or kegg.tsv match the genes "
            "in the model: the model has no genes."
        )


def _is_exchange(rxn: cobra.Reaction) -> bool:
    """Whether a reaction only consumes or only produces metabolites.

    RAVEN's ``getExchangeRxns`` counts every reaction without a positive or
    without a negative stoichiometric coefficient, so ``A + B <=>`` is an
    exchange reaction as well as ``A <=>``.
    """
    coefficients = list(rxn.metabolites.values())
    return all(c <= 0 for c in coefficients) or all(c >= 0 for c in coefficients)


def make_ec_model(
    model: cobra.Model,
    adapter: "ModelAdapter",
    *,
    gecko_light: bool = False,
    uniprot_db: Optional[UniprotDB] = None,
    kegg_db: "Optional[KeggDB]" = None,
) -> EcModel:
    """Build an enzyme-constrained model from a conventional GEM.

    Runs the 12-stage GECKO pipeline end to end and returns the
    populated ``EcModel``. The output has:

    - Reactions split per isozyme where applicable (``_EXP_<N>``
      suffix) and per direction for reversible ones (``_REV``).
    - One ``prot_<uniprot>`` pseudo-metabolite per enzyme.
    - One ``usage_prot_<uniprot>`` reaction per enzyme drawing
      from the shared protein pool.
    - A ``prot_pool`` pseudo-metabolite and a
      ``prot_pool_exchange`` reaction acting as the global
      protein budget.
    - A populated ``model.ec`` substructure (kcats, MW, sequences,
      coupling matrix).

    The kcat values themselves are NOT filled in here — they stay
    at 0 (0 marks "no kcat assigned") until you populate them
    via the ``gather_kcats`` functions (BRENDA / DLKcat / custom
    files) and call ``apply_kcat_constraints``, or set them manually
    via ``set_kcat_for_reactions``.

    Parameters
    ----------
    model
        A conventional cobra.Model (the starting GEM). Left
        unchanged: the function operates on an internal copy, so the
        caller's model is never mutated (matching the MATLAB GECKO
        ``makeEcModel`` value semantics).
    adapter
        A loaded ModelAdapter. Carries organism parameters (taxonomy
        id, biomass reaction, sigma factor, ...) and the path to
        the UniProt cache.
    gecko_light
        Build a light ecModel instead of a full one. The light layout
        skips the isozyme split (cobra reactions stay singular), omits
        the per-enzyme ``prot_<id>`` pseudometabolites and
        ``usage_prot_<id>`` reactions, and keeps only the shared
        ``prot_pool``. The per-isozyme bookkeeping moves into ``ec``
        instead: each cobra reaction with N isozymes produces N rows
        in ``ec.rxns`` distinguished by a 3-digit counter prefix
        (e.g. ``001_RXN1``, ``002_RXN1``). Suitable for large GEMs
        (HumanGEM, Recon3D) where the full layout's LP size becomes
        impractical. See ``docs/gecko_light_status.md``.
    uniprot_db
        Pre-loaded UniprotDB. If None, the function looks for
        ``adapter.params.path / "data" / "uniprot.tsv"`` and loads
        it automatically.
    kegg_db
        Pre-loaded KeggDB. Optional fallback source for protein
        information and EC numbers. When provided, stage 7 fills
        genes that UniProt missed (with ``ec.enzymes`` set to the
        UniProt accession carried in the KEGG row, or to the KEGG
        gene id when that accession is empty), and the EC-number
        lookup (``fill_eccodes_from_database``) consults KEGG when
        the UniProt result is empty or ends with ``-``. If None, the
        function loads ``adapter.params.path / "data" / "kegg.tsv"``
        when that file exists, and uses no KEGG data otherwise.

    Returns
    -------
    EcModel
        The built ecModel. ``model.ec.kcat`` is all 0; fill it in
        before solving.

    Raises
    ------
    FileNotFoundError
        If ``uniprot_db`` is None and no ``uniprot.tsv`` is found
        under the adapter's data folder.
    ValueError
        As ``makeEcModel`` does, if the compartment named by
        ``adapter.params.enzyme_comp`` is not one of the model's
        compartment names (a compartment ID that differs from its name is
        refused), if a metabolite ID starts with ``prot_``, if a reaction ID
        starts with ``usage_prot_`` or ``prot_pool`` or ends in ``_REV`` or
        ``_EXP_<digits>``, or if the model has no genes or none of them is
        found in the databases. Also propagated from individual pipeline
        stages on inconsistencies (e.g. genes referenced in GPRs that
        don't exist in the model).

    Genes referenced in the model's GPRs but not found in UniProt are
    left enzyme-unconstrained: they are reported as a single warning
    summary, and each affected reaction is annotated via
    ``rxn.notes["geckopy_warning"]``.
    """
    if isinstance(model, EcModel) and model.ec.n_rxns > 0:
        raise ValueError(
            "make_ec_model was called on a model that already has a "
            "populated ec substructure. Run it only on a conventional GEM."
        )

    _check_inputs(model, adapter)

    # Work on a copy: the pipeline stages below mutate the model in place
    # (splitting reactions, adding pseudometabolites, ...), but the caller's
    # input GEM must be left untouched.
    model = model.copy()
    # On the scipy solver the pipeline runs on glpk and the result is converted at the end:
    # optlang grows the scipy matrix densely, so the thousands of edits of a genome-scale
    # build take hours there (see geckopy.solver_support).
    on_scipy = repair_scipy_problem(model)
    if on_scipy:
        try:
            model.solver = "glpk"
        except Exception:  # glpk unavailable: build on scipy
            on_scipy = False

    # Stages 1-4: preprocess on a plain cobra.Model. Stage 5 (isozyme
    # expansion) is full-only; light keeps isozymes singular and tracks
    # them per-row in ec instead.
    remove_pseudoreaction_gprs(model, adapter)
    invert_backwards_only_reactions(model)
    # Only reactions that carry flux in both directions (lb < 0 < ub) and
    # are not exchange reactions are split, as MATLAB's makeEcModel.m does:
    # it passes convertToIrrev the reversible reactions without
    # getExchangeRxns(model). A reaction with both bounds negative stays as
    # it is.
    reversible_rxns = [
        rxn.id
        for rxn in model.reactions
        if rxn.lower_bound < 0 < rxn.upper_bound and not _is_exchange(rxn)
    ]
    convert_to_irreversible(model, rxns=reversible_rxns)
    if not gecko_light:
        expand_model(model)
        # Sort reactions, so that reversible and isozymic reactions are kept
        # near each other. MATLAB makeEcModel does this here (full models
        # only), before the protein pseudoreactions are appended, so the
        # metabolic block comes out alphabetically and the protein block
        # follows in enzyme order.
        sort_identifiers(model)

    # Promote to EcModel for stages 6-12.
    ec_model = EcModel.from_cobra(model, adapter, gecko_light=gecko_light)

    if uniprot_db is None:
        uniprot_path = adapter.params.path / "data" / "uniprot.tsv"
        uniprot_db = load_uniprot_tsv(uniprot_path)

    if kegg_db is None:
        kegg_path = adapter.params.path / "data" / "kegg.tsv"
        if kegg_path.is_file():
            kegg_db = load_kegg_tsv(kegg_path, auto_download=False)

    if gecko_light:
        # Light: stage 7 (per-enzyme data) runs first so the coupling
        # builder can index ec.genes when emitting the per-isozyme rows.
        # Stage 6 is folded into the light helper, which writes both
        # ec.rxns (one row per isozyme, with the 3-digit counter prefix)
        # and ec.rxn_enz_mat in one pass.
        no_uniprot = populate_enzyme_data(
            ec_model, uniprot_db, kegg_db=kegg_db,
        )
        allocate_ec_and_coupling_light(ec_model)
    else:
        # Full: stage 6 (allocate empty slots) runs first; stage 7
        # populates per-enzyme arrays; stage 8 fills the coupling matrix
        # from cobra GPRs (one AND-clause per reaction after the stage-5
        # expansion).
        allocate_ec_for_catalyzed_reactions(ec_model)
        no_uniprot = populate_enzyme_data(
            ec_model, uniprot_db, kegg_db=kegg_db,
        )
        build_rxn_enzyme_coupling(ec_model)

    # Stages 9-12: protein pool machinery. Light skips the per-enzyme
    # pseudometabolites + usage reactions (no per-enzyme bookkeeping in
    # the LP) but keeps the shared pool exactly as full models do.
    if not gecko_light:
        add_protein_pseudometabolites(ec_model)
    add_protein_pool_pseudometabolite(ec_model)
    if not gecko_light:
        add_protein_usage_reactions(ec_model)
    add_protein_pool_exchange_reaction(ec_model)

    if no_uniprot:
        preview = ", ".join(no_uniprot[:5])
        more = "" if len(no_uniprot) <= 5 else f" (and {len(no_uniprot) - 5} more)"
        logger.warning(
            "%d gene(s) not found in UniProt and left enzyme-unconstrained: "
            "%s%s. Affected reactions are annotated via "
            "rxn.notes['geckopy_warning'].",
            len(no_uniprot),
            preview,
            more,
        )

    if on_scipy:
        switch_to_scipy(ec_model)

    # Catch any internal length/shape drift in the ec arrays before the
    # model is handed back (cheap, and surfaces build bugs early).
    ec_model.ec.validate()

    return ec_model
