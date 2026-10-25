# gecko-light status

GECKO supports two ways of building an ecModel: **full** and **light**.
This document explains the difference, the current state of light
support in geckopy (short version: implemented end-to-end), and the
deliberate gaps that remain.

## What full vs light means

When a metabolic reaction is catalysed by an enzyme, GECKO needs to
encode the protein cost of that catalysis. There are two ways:

- **Full ecModel.** For each enzyme, geckopy adds a dedicated
  pseudo-metabolite (`prot_<uniprot>`) and a dedicated usage reaction
  (`usage_prot_<uniprot>`). Reactions that are catalysed by multiple
  isozymes are split into one copy per isozyme. This gives you maximum
  bookkeeping flexibility: you can ask the solver "how much of enzyme
  P00350 is being used right now?" directly. The cost is a big LP — a
  yeast-GEM full ecModel has ~8000 reactions; a HumanGEM full one
  blows up to tens of thousands.
- **Light ecModel.** Skip the per-enzyme bookkeeping. There's only a
  single shared protein pool. Each catalysed cobra reaction gets one
  extra stoichiometric coefficient `MW_sum / (kcat * 3600)` against
  that pool. Isozymes are not split into separate cobra reactions;
  instead `apply_kcat_constraints` picks the lowest-cost isozyme
  (smallest `MW_sum / kcat`) for each reaction and writes that one
  coefficient. The LP stays roughly the same size as the starting GEM.
  The trade-off is that you can no longer answer per-enzyme questions
  in solver outputs.

Light is the right choice for genome-scale models too big for the
full layout to solve in reasonable time (HumanGEM, Recon3D). For
smaller models (yeast, *E. coli*), full is usually fine.

## TL;DR

**Light is implemented.** End-to-end:

```python
ec_model = make_ec_model(model, adapter, gecko_light=True)
set_kcat_for_reactions(ec_model, ["001_R2", "002_R2"], 5.0)   # one row per isozyme
apply_kcat_constraints(ec_model)                 # picks cheapest isozyme per cobra rxn
set_prot_pool_size(ec_model)
ec_model.optimize()
```

A worked example lives in `tutorials/light_ecModel/protocol.py`
(uses ecTestGEM for speed). A real-scale build on the unmodified
Human-GEM YAML runs under `tests/test_light_humangem_smoke.py` —
opt-in via `pytest -m smoke`; auto-skipped when the Human-GEM repo
isn't checked out next to geckopy.

## How the data shapes differ

|  | Full | Light |
|---|---|---|
| Isozyme expansion | reactions split, one copy per isozyme with a `_EXP_<N>` suffix | cobra reactions stay singular |
| `ec.rxns` | one entry per (reaction, isozyme) pair; ids match the cobra reactions exactly | duplicate entries per isozyme; ids carry a `###_` counter prefix (e.g. `001_R2`, `002_R2`) |
| Per-enzyme pseudo-metabolite `prot_<id>` | yes | no |
| Per-enzyme usage reaction `usage_prot_<id>` | yes | no |
| Shared protein pool | yes | yes (single global cost applied directly to each catalysed reaction's stoichiometry) |

## What's implemented

| File | Light behaviour |
|---|---|
| `src/geckopy/ec_model/make_ec_model.py` | dispatches stage 5 (isozyme split) and stages 9/11 (per-enzyme prot mets + usage rxns) off; calls `allocate_ec_and_coupling_light` instead of the full stages 6/8 |
| `src/geckopy/ec_model/pipeline/populate_ec.py` | `allocate_ec_and_coupling_light` builds `ec.rxns` (one row per isozyme with the `###_` prefix) and `ec.rxn_enz_mat` in one pass; `split_light_rxn_id` is the inverse |
| `src/geckopy/ec_model/pipeline/apply_kcat.py` | `_apply_kcat_constraints_light` groups ec rows by cobra reaction, picks the lowest-cost isozyme, writes `-MW_sum / (kcat * 3600)` as the `prot_pool` coefficient; same clear-then-write idempotency contract as full |
| `src/geckopy/ec_model/pipeline/set_kcat.py` | recognises the `###_` prefix as the light analogue of `_EXP_<N>`; bare base names broadcast to every isozyme row |
| `src/geckopy/ec_model/enzyme.py` | `Enzyme.reactions` strips the prefix and dedupes; `Enzyme.mw` setter passes ec.rxns ids (not cobra ids) to `apply_kcat_constraints`; `_repr_html_` catches the light NotImplementedError |
| `src/geckopy/gather_kcats/fuzzy_kcat_matching.py` | strips the `###_` prefix for cobra lookups |
| `src/geckopy/gather_kcats/get_standard_kcat.py` | `_ec_rxn_to_cobra_id` handles both forms |

Tests covering the light path:

- `tests/test_make_ec_model.py` — 7 light tests (flag, prefix, coupling rows, no per-enzyme mets, kcat init).
- `tests/test_apply_kcat.py` — 8 light tests (single isozyme, lowest-cost isozyme, clear-then-write, NaN-MW guard, all-zero-kcat warning).
- `tests/test_set_kcat.py` — 7 light tests (prefix recognition, base-name broadcast, apply-true → prot_pool, lowest-cost tie-breaking).
- `tests/test_enzyme.py` — 5 light tests (reactions returns cobra rxns, dedupes isozyme rows, mw setter, repr_html doesn't crash, kcats keys are prefixed).
- `tests/test_light_humangem_smoke.py` — full Human-GEM build (opt-in via `-m smoke`).

## What is intentionally unavailable on light models

| Surface | Why |
|---|---|
| `Enzyme.flux`, `Enzyme.cap_usage`, `Enzyme.upper_bound`, `Enzyme.shadow_price`, `Enzyme.prot_metabolite`, `Enzyme.usage_reaction` | reference `prot_<id>` / `usage_prot_<id>`, which don't exist in light |
| `Enzyme.concentration` setter (`Kcats.__setitem__`) | requires the per-enzyme usage upper-bound machinery |
| `constrain_enz_concs`, `flexibilize_enz_concs` | proteomics integration is per-enzyme; the shared pool can't represent it |
| `pfba_enzymes` | minimises per-enzyme usage reactions, which light doesn't have |
| `fill_kcats_from_isozymes` / `get_kcat_across_isozymes` | full-model isozyme aggregation; the light layout already represents isozymes as separate ec rows |

All of these raise `NotImplementedError` with a clear message rather
than silently returning meaningless values.

## What still works the same way on light

Anything that reads from `model.ec.*` arrays or operates on the cobra
LP without inspecting per-enzyme metabolites: BRENDA fuzzy matching,
DLKcat I/O, custom-kcat overrides, `assign_standard_kcat`, the kcat
sensitivity tuning loop (it operates on the cobra LP and reads
`ec.kcat` / `ec.mw`), and the `Enzyme` proxy's read-only metadata
(`mw`, `gene`, `sequence`, `reactions`, `kcats[...]`).

## Pointers

- Tutorial: `tutorials/light_ecModel/protocol.py`
- Smoke test: `tests/test_light_humangem_smoke.py`
- MATLAB parity reference: `GECKO/tutorials/light_ecModel/protocol.m`
