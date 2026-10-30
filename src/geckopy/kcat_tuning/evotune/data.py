"""Load experimental data for evotune (CMA-ES) kcat tuning.

The flux and max-growth files, ``evotuneFluxData.tsv`` and
``evotuneMaxGrowth.tsv``, share the column layout
``Condition Ptot grRate <met> (<rxn>) ... evotuneRMSEweight source`` with
regular flux data, so both reuse
:func:`~geckopy.databases.flux_data.load_flux_data` unchanged; only
``evotuneZeroExch.tsv`` (a single ``Rxns`` id column) needs a loader of
its own.

All three files are optional: a missing file yields ``None`` (or an empty
list for ``zero_flux``), never an exception.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...databases.flux_data import FluxData, load_flux_data

if TYPE_CHECKING:
    from ...adapter import ModelAdapter

_FLUX_DATA_FILENAME = "evotuneFluxData.tsv"
_MAX_GROWTH_FILENAME = "evotuneMaxGrowth.tsv"
_ZERO_EXCH_FILENAME = "evotuneZeroExch.tsv"


@dataclass
class EvotuneData:
    """Experimental data used by evotune kcat tuning.

    Attributes
    ----------
    flux_data
        Per-condition growth/exchange-flux measurements, from
        ``evotuneFluxData.tsv``. ``None`` if that file isn't present.
    max_grate
        Per-condition maximum-growth-rate measurements (one active
        carbon source per condition, at ``-1000`` i.e. unconstrained),
        from ``evotuneMaxGrowth.tsv``. ``None`` if that file isn't
        present.
    zero_flux
        Reaction IDs assumed to carry zero flux, from
        ``evotuneZeroExch.tsv``. Empty if that file isn't present.
    """

    flux_data: Optional[FluxData] = None
    max_grate: Optional[FluxData] = None
    zero_flux: list[str] = field(default_factory=list)


def load_evotune_data(adapter: "ModelAdapter") -> EvotuneData:
    """Load the three evotune-kcat-tuning experimental data files.

    Each file is optional and resolved as
    ``<adapter.params.path>/data/<filename>``; a missing file yields
    ``None`` (``flux_data``/``max_grate``) or an empty list
    (``zero_flux``) rather than raising.

    Parameters
    ----------
    adapter
        A ``ModelAdapter`` whose ``params.path`` locates the ``data/``
        folder.

    Returns
    -------
    EvotuneData
    """
    data_dir = Path(adapter.params.path) / "data"
    return EvotuneData(
        flux_data=_load_optional_flux_data(data_dir / _FLUX_DATA_FILENAME),
        max_grate=_load_optional_flux_data(data_dir / _MAX_GROWTH_FILENAME),
        zero_flux=_load_zero_exch(data_dir / _ZERO_EXCH_FILENAME),
    )


def _load_optional_flux_data(path: Path) -> Optional[FluxData]:
    if not path.is_file():
        return None
    return load_flux_data(path)


def _load_zero_exch(path: Path) -> list[str]:
    """Parse ``evotuneZeroExch.tsv``: a single ``Rxns`` header, then
    one reaction ID per line."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rxn_ids = [line.strip() for line in lines[1:] if line.strip()]
    return rxn_ids
