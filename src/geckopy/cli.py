"""Command-line interface for geckopy.

The `init` subcommand ports GECKO MATLAB's project-scaffolding logic
(`src/geckomat/utilities/startGECKOproject.m`): create the project
folder tree and write out a template adapter. The `brenda-refresh`,
`uniprot-download`, and `kegg-download` subcommands are new tooling
with no MATLAB equivalent.

Ported from GECKO MATLAB: src/geckomat/utilities/startGECKOproject.m
(the `init` subcommand only).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from geckopy.adapter.template import generate_template_toml
from geckopy.databases.brenda import parse_brenda_json
from geckopy.databases.brenda.aggregate import aggregate_and_write
from geckopy.databases.brenda.download import (
    BrendaDownloadError,
    ensure_brenda_json,
)
from geckopy.databases.kegg_download import download_kegg
from geckopy.databases.uniprot_download import download_uniprot


ADAPTER_PY_STUB = '''\
"""Custom adapter for this ecModel.

Uncomment and customize the class below to override default behavior,
typically when your model needs a non-standard way to identify
spontaneous reactions or map gene IDs. Then import and use it:

    from myModel.adapter import MyAdapter
    adapter = MyAdapter.from_folder("myModel")

If you only need to change parameter values, edit model_adapter.toml
and use the base ModelAdapter directly.
"""
# from geckopy import ModelAdapter
#
#
# class MyAdapter(ModelAdapter):
#     def get_spontaneous_reactions(self, model):
#         # Return a list of reaction IDs for spontaneous reactions.
#         return []
'''


def cmd_init(args: argparse.Namespace) -> int:
    folder = Path(args.folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        print(f"Error: {folder} already exists and is not empty.", file=sys.stderr)
        return 1

    folder.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "models", "output"):
        (folder / sub).mkdir(exist_ok=True)

    toml_text = generate_template_toml(project_name=folder.name)
    (folder / "model_adapter.toml").write_text(toml_text)
    (folder / "adapter.py").write_text(ADAPTER_PY_STUB)

    print(f"Created ecModel project at {folder}")
    print(
        f"Next: edit {folder / 'model_adapter.toml'} to set conv_gem and "
        f"org_name, then place your GEM SBML file in {folder / 'models'}."
    )
    return 0


# The raw BRENDA bulk-JSON download is a disposable, re-fetchable, multi-
# hundred-MB cache -- same OS user-cache-dir convention as raven-toolbox's
# own binary cache (raven_toolbox.binaries._cache_dir), not the project
# directory or the install tree (which may be read-only).
def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "geckopy" / "brenda"


_DEFAULT_CACHE_DIR = _default_cache_dir()
# The aggregated TSVs are a chosen project output, not a cache -- default to
# a project-relative location (resolved against the current working
# directory).
_DEFAULT_OUT_DIR = Path("data") / "brenda"


def cmd_brenda_refresh(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    try:
        json_path = ensure_brenda_json(cache_dir)
    except BrendaDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Parsing {json_path} ...")
    rows = list(parse_brenda_json(json_path))
    if not rows:
        print("Error: parser yielded no rows.", file=sys.stderr)
        return 1

    import json
    with json_path.open("r", encoding="utf-8") as fh:
        release = json.load(fh).get("release", "unknown")

    paths = aggregate_and_write(rows, out_dir, release=release)
    print(f"BRENDA release {release} aggregated to {out_dir}:")
    for kind, path in paths.items():
        # Each file has a "#" release comment plus a TSV column-header
        # row; both are skipped to report just the data row count.
        n = sum(1 for _ in path.open("r", encoding="utf-8")) - 2
        print(f"  {path.name}: {n} rows")
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    """List, or with --apply carry out, the steps that bring a GECKO 3
    project to the GECKO 4 layout (see :func:`geckopy.upgrade.upgrade_project`)."""
    import pandas as pd

    from .adapter import ModelAdapter
    from .upgrade import upgrade_project
    from .utilities.load_ec_model import load_ec_model

    ec_model = None
    if args.ec_model:
        adapter = ModelAdapter.from_folder(args.folder)
        ec_model = load_ec_model(Path(args.ec_model).resolve(), adapter=adapter)
    report = upgrade_project(args.folder, dry_run=not args.apply, ec_model=ec_model)
    if report.empty:
        print(f"{args.folder} has the current GECKO 4 layout.")
        return 0
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(report.to_string(index=False))
    if not args.apply and (report["status"] == "to do").any():
        print("Run geckopy upgrade with --apply to carry out the steps; originals go to legacy/<date>/.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geckopy", description="Command-line interface for geckopy."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser(
        "init", help="Scaffold a new ecModel project folder"
    )
    init_parser.add_argument(
        "folder",
        help="Target folder (will be created; must not already exist or be empty)",
    )
    init_parser.set_defaults(func=cmd_init)

    brenda_parser = sub.add_parser(
        "brenda-refresh",
        help="Rebuild kcat.tsv / sa.tsv / mw.tsv from the BRENDA bulk JSON",
    )
    brenda_parser.add_argument(
        "--cache-dir", default=str(_DEFAULT_CACHE_DIR),
        help=(
            "Directory holding the BRENDA bulk-JSON tarball or unpacked .json "
            f"(default: {_DEFAULT_CACHE_DIR})"
        ),
    )
    brenda_parser.add_argument(
        "--out-dir", default=str(_DEFAULT_OUT_DIR),
        help=f"Output directory for the three TSVs (default: {_DEFAULT_OUT_DIR})",
    )
    brenda_parser.set_defaults(func=cmd_brenda_refresh)

    uniprot_parser = sub.add_parser(
        "uniprot-download",
        help="Download an organism-specific UniProt TSV via the REST API",
    )
    uniprot_parser.add_argument(
        "uniprot_id",
        help="UniProt query identifier (e.g. NCBI taxonomy id 559292)",
    )
    uniprot_parser.add_argument(
        "out_path", help="Destination TSV path",
    )
    uniprot_parser.add_argument(
        "--id-type", default="taxonomy_id",
        help="UniProt query field for uniprot_id (default: taxonomy_id)",
    )
    uniprot_parser.add_argument(
        "--gene-id-field", default="gene_oln",
        help="UniProt field for the Gene Names column (default: gene_oln)",
    )
    uniprot_parser.add_argument(
        "--include-unreviewed", action="store_true",
        help="Include unreviewed (TrEMBL) entries (default: reviewed only)",
    )
    uniprot_parser.set_defaults(func=cmd_uniprot_download)

    kegg_parser = sub.add_parser(
        "kegg-download",
        help="Download organism-specific protein info via the KEGG REST API",
    )
    kegg_parser.add_argument(
        "kegg_id", help="KEGG organism code (e.g. sce)",
    )
    kegg_parser.add_argument(
        "out_path", help="Destination CSV path",
    )
    kegg_parser.add_argument(
        "--gene-id-field", default="kegg",
        help="KEGG entry field to use as the gene matching key (default: kegg)",
    )
    kegg_parser.set_defaults(func=cmd_kegg_download)

    upgrade_parser = sub.add_parser(
        "upgrade", help="Bring a GECKO 3 project to the GECKO 4 layout (a dry run without --apply)"
    )
    upgrade_parser.add_argument("folder", help="Project folder")
    upgrade_parser.add_argument("--apply", action="store_true", help="Carry out the steps")
    upgrade_parser.add_argument(
        "--ec-model", help="The full ecModel (YAML) the custom kcats and stored predictions are for",
    )
    upgrade_parser.set_defaults(func=cmd_upgrade)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_uniprot_download(args: argparse.Namespace) -> int:
    out = Path(args.out_path).resolve()
    download_uniprot(
        args.uniprot_id, out,
        id_type=args.id_type,
        gene_id_field=args.gene_id_field,
        reviewed=not args.include_unreviewed,
    )
    print(f"Wrote {out}")
    return 0


def cmd_kegg_download(args: argparse.Namespace) -> int:
    out = Path(args.out_path).resolve()
    download_kegg(
        args.kegg_id, out, gene_id_field=args.gene_id_field,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
