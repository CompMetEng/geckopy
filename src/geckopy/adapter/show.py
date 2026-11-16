"""Show the resolved parameters of an adapter and where each value comes from.

One line per parameter, in the order of ``adapterParameters.tsv``, with the
value in effect and its origin: ``file`` (set in model_adapter.toml),
``preset <name>`` (given by the kcat assignment preset) or ``default``. The
kcat assignment stages, the ``[kcat_sources]`` settings and the evotune source
groups are shown as resolved. GECKO (MATLAB) has the same function as
``showAdapterParams``; the command line is ``geckopy adapter show``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .adapter import ModelAdapter

_TABLE = Path(__file__).with_name("adapterParameters.tsv")


def _keys() -> list[str]:
    with open(_TABLE, newline="", encoding="utf-8") as f:
        return [r["toml_key"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


def _text(value: Any) -> str:
    """A value as ``adapter show`` prints it: strings bare, booleans true/false,
    lists comma-joined, floats with ``%g`` (whole-valued ones without decimals)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _line(key: str, value: Any, origin: str) -> str:
    if value is None:
        return f"{key} =  [not set]"
    return f"{key} = {_text(value)}  [{origin}]"


#: The settings of a stage that are shown when set (file) or not at their default (preset).
_STAGE_DEFAULTS = {"criteria": "max", "top_origin_limit": 4, "bottom_origin_limit": 6, "wildcard_limit": 3}


def _stage_line(i: int, stage, shown: list[str], origin: str) -> str:
    parts = [f"{stage.action}: {', '.join(stage.sources)}"]
    parts.extend(f"{k}={_text(getattr(stage, k))}" for k in _STAGE_DEFAULTS if k in shown)
    return f"kcat_assignment.stage[{i}] = {'; '.join(parts)}  [{origin}]"


def format_adapter_params(adapter: "ModelAdapter") -> str:
    """The resolved parameters of ``adapter`` as text (see :func:`show_adapter_params`)."""
    from ..gather_kcats.kcat_assignment import DEFAULT_OKP_METHODS, kcat_assignment_config

    params = adapter.params
    config = kcat_assignment_config(adapter)
    preset_origin = f"preset {config.preset}" if config.preset else "file"
    out: list[str] = []
    for key in _keys():
        owner: Any = params
        *parents, leaf = key.split(".")
        for name in parents:
            owner = getattr(owner, name)
        value = getattr(owner, leaf)
        origin = "file" if leaf in owner.model_fields_set else "default"
        if key == "kcat_assignment.preset":
            if params.kcat_assignment.stage:
                out.append("kcat_assignment.preset = none (stages are given)  [file]")
            else:
                out.append(_line(key, config.preset, origin))
        elif key == "kcat_assignment.stage":
            for i, stage in enumerate(config.stages, start=1):
                if params.kcat_assignment.stage:
                    shown = [k for k in _STAGE_DEFAULTS if k in params.kcat_assignment.stage[i - 1].model_fields_set]
                else:
                    shown = [k for k, d in _STAGE_DEFAULTS.items() if getattr(stage, k) != d]
                out.append(_stage_line(i, stage, shown, "file" if params.kcat_assignment.stage else preset_origin))
        elif key == "kcat_sources":
            sources = {name: dict(keys) for name, keys in config.sources.items()}
            if "methods" not in sources.get("okp", {}):
                sources.setdefault("okp", {})["methods"] = list(DEFAULT_OKP_METHODS)
            order = ["brenda", "ec_tree", "homologs", "okp"]
            for name in sorted(sources, key=lambda n: (order.index(n), "") if n in order else (len(order), n)):
                keys = sources[name]
                for k in sorted(keys):
                    v = keys[k]
                    if k in params.kcat_sources.get(name, {}):
                        where = "file"
                    elif name == "okp" and k == "methods":
                        where = "default"
                    else:
                        where = preset_origin
                    out.append(_line(f"kcat_sources.{name}.{k}", v, where))
        elif key == "evotune.source_groups":
            for name, rule in value.items():
                out.append(f"evotune.source_groups.{name} = {_text(rule.sources)}; sigma0_log {_text(rule.sigma0_log)}  [{origin}]")
        else:
            out.append(_line(key, value, origin))
    return "\n".join(out)


def show_adapter_params(adapter: "ModelAdapter") -> None:
    """Print every adapter parameter with its value and where it comes from.

    One line per item, in the order of ``adapterParameters.tsv``:
    ``<key> = <value>  [<origin>]``. The origin is ``file`` (set in
    model_adapter.toml), ``default``, ``preset <name>`` (given by the kcat
    assignment preset) or ``not set`` (no value: ``<key> =  [not set]``).
    The kcat assignment stages (``kcat_assignment.stage[<i>]``), the
    ``kcat_sources.<name>.<key>`` settings and the evotune source groups, in
    matching order with their ``sigma0_log``, are shown as resolved.
    """
    print(format_adapter_params(adapter))
