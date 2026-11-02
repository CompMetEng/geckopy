"""Compare snapshot max vs snapshot median over the bundled BRENDA TSVs.

The snapshot now ships both views (one row per (ec, substrate, organism)
in ``kcat_max``, another in ``kcat_median``). This script aggregates each
view across all triples per EC and reports how far the per-EC max-of-max
sits from the per-EC median-of-median. Trypsin (3.4.21.4) is excluded
because its BRENDA entries are known unreliable.

For the full raw-JSON spread, use ``kcat_aggregation_raw.py``.

Run from the repo root: ``python scripts/kcat_aggregation_compare.py``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from geckopy.databases import load_brenda_data

ROOT = Path(__file__).resolve().parent.parent
BRENDA_DIR = ROOT / "src" / "geckopy" / "data" / "brenda"

EXCLUDED_ECS = {"3.4.21.4"}

print("Loading bundled BRENDA snapshot ...")
brenda = load_brenda_data(BRENDA_DIR)
kcat_max = brenda.kcat_max
kcat_med = brenda.kcat_median
print(
    f"  triples (max view):    {len(kcat_max):,}, unique EC: {kcat_max['ec_code'].nunique()}"
)
print(
    f"  triples (median view): {len(kcat_med):,}, unique EC: {kcat_med['ec_code'].nunique()}"
)


def _per_ec(view: pd.DataFrame) -> dict[str, list[float]]:
    """Group per-EC kcat values, filtering trypsin + nonpositive."""
    sub = view[(view["kcat"] > 0) & (~view["ec_code"].isin(EXCLUDED_ECS))]
    return {ec: vals.tolist() for ec, vals in sub.groupby("ec_code")["kcat"]}


per_ec_max = _per_ec(kcat_max)
per_ec_med = _per_ec(kcat_med)

records = []
for ec in per_ec_max:
    mx = per_ec_max[ec]
    md = per_ec_med.get(ec, [])
    if len(mx) < 5:
        continue
    records.append({
        "ec": ec,
        "n_triples": len(mx),
        "kcat_maxmax": float(np.max(mx)),
        "kcat_maxmed": float(np.median(mx)),
        "kcat_medmed": float(np.median(md)) if md else float("nan"),
    })

df = pd.DataFrame(records)
df["log_maxmax_over_medmed"] = np.log10(df["kcat_maxmax"] / df["kcat_medmed"])
df["log_maxmed_over_medmed"] = np.log10(df["kcat_maxmed"] / df["kcat_medmed"])

print(
    f"\nPer-EC stats over {len(df)} EC codes (>=5 triples, excluding "
    f"{sorted(EXCLUDED_ECS)}):"
)

def _summary(col: str, label: str) -> None:
    qs = df[col].quantile([0.25, 0.5, 0.75]).round(3).tolist()
    print(
        f"  {label:<40}  quartiles {qs}  mean {df[col].mean():.3f}  "
        f"p90 {df[col].quantile(0.9):.3f}"
    )

_summary("log_maxmax_over_medmed", "log10(maxmax / medmed)  -- true gap")
_summary("log_maxmed_over_medmed", "log10(maxmed / medmed)  -- snap-only swap")

print(
    f"\n  fraction maxmax / medmed >= 10x median: "
    f"{(df['log_maxmax_over_medmed'] >= 1).mean() * 100:.1f}%"
)
print(
    f"  fraction maxmax / medmed >= 100x median: "
    f"{(df['log_maxmax_over_medmed'] >= 2).mean() * 100:.1f}%"
)
print(
    f"  fraction maxmax / medmed >= 1000x median: "
    f"{(df['log_maxmax_over_medmed'] >= 3).mean() * 100:.1f}%"
)

print("\nTop 20 best-measured ECs (excluding trypsin):")
top = df.sort_values("n_triples", ascending=False).head(20).copy()
out = top[["ec", "n_triples", "kcat_maxmax", "kcat_maxmed", "kcat_medmed"]].round(2)
out["maxmax/medmed"] = (out["kcat_maxmax"] / out["kcat_medmed"]).round(1)
print(out.to_string(index=False))
