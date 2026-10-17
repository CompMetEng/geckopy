"""Hatchling build hook: xz-compress the BRENDA snapshot into the wheel.

kcat.tsv/sa.tsv/mw.tsv stay plain text in the repo -- diffable, and what
`geckopy brenda-refresh` regenerates. They're also most of the installed
package's size (~6MB of ~6.4MB), so only the built wheel ships
xz-compressed copies instead; src/geckopy/databases/brenda_loader.py
looks for the .xz variant first and falls back to plain text.
"""
from __future__ import annotations

import lzma
import shutil
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_BRENDA_FILES = ("kcat.tsv", "sa.tsv", "mw.tsv", "measurements.tsv")


class BrendaCompressHook(BuildHookInterface):
    def initialize(self, version, build_data):
        brenda_dir = Path(self.root) / "src" / "geckopy" / "data" / "brenda"
        self._tmpdir = tempfile.mkdtemp(prefix="geckopy-brenda-xz-")
        for name in _BRENDA_FILES:
            src = brenda_dir / name
            dst = Path(self._tmpdir) / f"{name}.xz"
            with open(src, "rb") as f_in, lzma.open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            build_data["force_include"][str(dst)] = f"geckopy/data/brenda/{name}.xz"

    def finalize(self, version, build_data, artifact_path):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
