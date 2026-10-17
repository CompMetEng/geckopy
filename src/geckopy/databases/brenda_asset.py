"""The BRENDA files of geckopy, from a CompMetEng/toolbox-assets release.

geckopy does not ship BRENDA data. The files of release ``brenda-2026.1-1``
(BRENDA 2026.1, CC BY 4.0) are downloaded on first use, checked against
their SHA-256 checksums, decompressed and cached by raven-toolbox
(``raven_toolbox.data.ensure_data_file``). GECKO reads the same release
(``fetchBrendaAsset``). A project can use its own BRENDA files instead, in
``data/brenda`` (``geckopy brenda-refresh``).
"""
from __future__ import annotations

import gzip
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable

BRENDA_ASSET = "brenda-2026.1-1"

#: The release's files as the toolbox-assets manifest lists them.
_REGISTRY: dict = {
    "brenda": {
        "version": BRENDA_ASSET,
        "files": {
            "kcat.tsv.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/kcat.tsv.gz",
                "sha256": "a6f448f81cb9bc62c829bbfb8043ec7e873ce58e70362e9bc3b94d0a1f01bbda",
            },
            "kcatCombineParams.tsv": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/kcatCombineParams.tsv",
                "sha256": "11fa0cdd3dd81feb70a3db0aa321b5140dd6551a666ed1c2672f3317cc889431",
            },
            "measurements.tsv.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/measurements.tsv.gz",
                "sha256": "88bc0d7ead734f04c69e51c5a105660037146ec73c85dc12f5d5ca74bf9ba31a",
            },
            "mw.tsv.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/mw.tsv.gz",
                "sha256": "99b6cacdf28d36826986fe0de578306b39ebb58f504afb7d9af1268218bf7c61",
            },
            "sa.tsv.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/sa.tsv.gz",
                "sha256": "e93bc0991e1fd817fd213a8ef507cf8d56bc58429beb8ff7a4e0c600e73acbc4",
            },
            "sequences.fasta.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/sequences.fasta.gz",
                "sha256": "3be2bf96068a2f05dbf791f2ca56dca1b083b8e19316cb475edd5d910a7fdc9d",
            },
            "substrates.tsv.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/brenda-2026.1-1/substrates.tsv.gz",
                "sha256": "bbd672ad55978dc01d5c3b66fd5461d67e55336dfb088e3f1b3445d4dcf8818f",
            },
        },
    },
}

#: The files the BRENDA folder holds (decompressed).
TABLES = ("kcat.tsv", "sa.tsv", "mw.tsv", "measurements.tsv")


def fetch_brenda_asset(files: Iterable[str] = TABLES) -> Path:
    """Download (once) the BRENDA files and return the folder that holds them.

    Parameters
    ----------
    files
        File names without ``.gz`` (default: ``kcat.tsv``, ``sa.tsv``,
        ``mw.tsv`` and ``measurements.tsv``); further files of the release
        are ``sequences.fasta``, ``substrates.tsv`` and
        ``kcatCombineParams.tsv``.

    Returns
    -------
    pathlib.Path
        The cache folder with the files, decompressed.
    """
    available = _REGISTRY["brenda"]["files"]
    folder = None
    for name in files:
        packed = name + ".gz" if name + ".gz" in available else name
        if packed not in available:
            raise FileNotFoundError(f"{name} is not a file of {BRENDA_ASSET}: {sorted(available)}")
        path = _download(packed)
        folder = path.parent
        if packed.endswith(".gz"):
            target = folder / name
            if not target.is_file():
                # A file of its own, so processes unpacking at once do not clash.
                fd, tmp = tempfile.mkstemp(dir=folder, prefix=name + ".", suffix=".part")
                with gzip.open(path, "rb") as src, os.fdopen(fd, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(tmp, target)
    return folder


def _download(name: str) -> Path:
    """The cached file, downloading it if needed. When another process
    finishes the same download first, its file is used."""
    from raven_toolbox.data import ensure_data_file

    for attempt in range(3):
        try:
            return ensure_data_file("brenda", name, registry=_REGISTRY)
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")
