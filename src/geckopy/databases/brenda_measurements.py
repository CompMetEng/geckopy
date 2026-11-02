"""Load the BRENDA measurement table: one row per wild-type kcat measurement.

``measurements.tsv`` sits next to ``kcat.tsv``, ``sa.tsv`` and ``mw.tsv`` in
the BRENDA folder (see ``ModelAdapter.get_brenda_db_folder``). It holds the
wild-type BRENDA kcats with an EC number after the outlier screen, with the
substrate, organism, UniProt accession and sequence id of each measurement.
GECKO reads the same file (``loadBrendaMeasurements``).
"""
from __future__ import annotations

import lzma
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["ec", "substrate", "substrate_inchikey", "organism", "taxid", "uniprot", "seq_id",
           "kcat", "temperature", "ph", "pmid"]


def _measurement_path(folder: Path) -> Path:
    for name in ("measurements.tsv.xz", "measurements.tsv"):
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(
        f"No BRENDA measurement table (measurements.tsv) in {folder}. It comes with geckopy's "
        f"BRENDA files; a project's data/brenda folder needs its own copy."
    )


def load_brenda_measurements(folder: str | Path) -> pd.DataFrame:
    """The BRENDA measurement table of a BRENDA folder.

    Returns
    -------
    pandas.DataFrame
        Columns as in the file (``ec``, ``substrate``, ``substrate_inchikey``,
        ``organism``, ``taxid``, ``uniprot``, ``seq_id``, ``kcat``,
        ``temperature``, ``ph``, ``pmid``) plus ``log10_kcat``. The release
        line of the file (e.g. ``BRENDA release 2026.1``) is in
        ``attrs["release"]``.
    """
    path = _measurement_path(Path(folder))
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.readline()
        table = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in ("ec", "kcat") if c not in table.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    table["kcat"] = pd.to_numeric(table["kcat"], errors="coerce")
    table = table[(table["kcat"] > 0) & table["ec"].notna()].reset_index(drop=True)
    table["ec"] = table["ec"].str.strip()
    table["log10_kcat"] = np.log10(table["kcat"].to_numpy(dtype=float))
    m = re.match(r"^#\s*(BRENDA release \S+)", first)
    table.attrs["release"] = m.group(1) if m else ""
    return table
