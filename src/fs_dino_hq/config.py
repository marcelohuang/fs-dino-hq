"""JSON run configurations shared by the training commands.

A configuration file holds argument names (with underscores) as keys; values
given on the command line take precedence. String values may reference
environment variables as ``${NAME}``, which keeps machine-specific locations
out of the files, for example::

    {"synthetic_root": "${DATA_DIR}/photo", "save_dir": "${OUTPUT_DIR}/stage2_five_shot"}

A referenced variable that is not set is an error, as is a key that is not
an option of the command. Values are checked like command-line values where
argparse would not do it for defaults: ``choices``, true/false for flags, and
a JSON list for options taking several values (a single string is accepted
as a one-element list). ``null`` leaves an option at ``None``. Relative paths
are resolved against the working directory, not the configuration file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any, source: str = "config") -> Any:
    """Replace ``${NAME}`` in strings (recursively in lists and dicts)."""
    if isinstance(value, str):
        def sub(m):
            name = m.group(1)
            if name not in os.environ:
                raise KeyError(f"{source}: environment variable {name} is not set "
                               f"(referenced in {value!r})")
            return os.environ[name]
        return _VAR.sub(sub, value)
    if isinstance(value, list):
        return [expand_env(v, source) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, source) for k, v in value.items()}
    return value


def load_config(path, allowed_keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Read a JSON configuration, check its keys and expand ``${NAME}`` references."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a JSON object")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}   # "_comment" keys
    if allowed_keys is not None:
        unknown = sorted(set(cfg) - set(allowed_keys))
        if unknown:
            raise ValueError(f"unknown keys in {path}: {unknown}")
    return expand_env(cfg, str(path))


def check_config_values(parser: argparse.ArgumentParser, cfg: Dict[str, Any],
                        source: str = "config") -> Dict[str, Any]:
    """Validate configuration values against the parser's actions.

    argparse applies ``type`` to string defaults but checks neither
    ``choices`` nor flags, so without this a configuration could set a value
    the command line would reject (or the string "false" for a flag, which is
    true). Returns ``cfg`` with single strings for multi-value options wrapped
    in a list and string list items converted with the option's ``type``.
    """
    out = dict(cfg)
    for action in parser._actions:
        if action.dest not in out or out[action.dest] is None:
            continue
        value = out[action.dest]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                               argparse.BooleanOptionalAction)):
            if not isinstance(value, bool):
                raise ValueError(f"{source}: {action.dest} must be true or false, got {value!r}")
            continue
        if action.nargs == 0:              # other constant-storing actions
            continue
        multi = action.nargs in ("+", "*") or isinstance(action.nargs, int)
        if multi:
            if isinstance(value, str) and action.nargs in ("+", "*"):
                value = [value]
            if not isinstance(value, list):
                raise ValueError(f"{source}: {action.dest} takes a list, got {value!r}")
            if isinstance(action.nargs, int) and len(value) != action.nargs:
                raise ValueError(f"{source}: {action.dest} takes {action.nargs} values, got {value!r}")
            if action.type is not None:    # argparse converts string defaults, not list items
                value = [action.type(v) if isinstance(v, str) else v for v in value]
            out[action.dest] = value
        if action.choices is not None:
            for v in (value if multi else [value]):
                if isinstance(v, str) and action.type is not None:
                    v = action.type(v)
                if v not in action.choices:
                    raise ValueError(f"{source}: {action.dest}={v!r} is not one of "
                                     f"{list(action.choices)}")
    return out


def parse_with_config(parser: argparse.ArgumentParser, argv=None,
                      allowed_keys: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse ``argv`` with defaults taken from ``--config`` (CLI values win).

    The parser must define ``--config``. ``allowed_keys`` defaults to the
    parser's own destinations. Errors in the configuration raise ``KeyError``
    (unset environment variable) or ``ValueError`` (unknown key, bad value).
    """
    pre, _ = parser.parse_known_args(argv)
    if getattr(pre, "config", None):
        keys = allowed_keys or {a.dest for a in parser._actions} - {"help", "config"}
        cfg = load_config(pre.config, keys)
        parser.set_defaults(**check_config_values(parser, cfg, str(pre.config)))
    return parser.parse_args(argv)
