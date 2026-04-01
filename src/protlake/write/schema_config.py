"""
Helpers for loading a user schema from a JSON or YAML config file.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pyarrow as pa
import yaml


_PRIMITIVE_TYPES: Dict[str, pa.DataType] = {
    "binary": pa.binary(),
    "bool": pa.bool_(),
    "date32": pa.date32(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "string": pa.string(),
}


def _parse_type(type_spec: Any) -> pa.DataType:
    if isinstance(type_spec, str):
        try:
            return _PRIMITIVE_TYPES[type_spec]
        except KeyError as exc:
            raise ValueError(f"Unsupported schema type '{type_spec}'") from exc

    if not isinstance(type_spec, dict):
        raise ValueError(f"Invalid schema type spec: {type_spec!r}")

    if "list" in type_spec:
        return pa.list_(_parse_type(type_spec["list"]))
    if "large_list" in type_spec:
        return pa.large_list(_parse_type(type_spec["large_list"]))
    if "struct" in type_spec:
        return pa.struct([_parse_field(field_cfg) for field_cfg in type_spec["struct"]])
    if "timestamp" in type_spec:
        ts_cfg = type_spec["timestamp"]
        if isinstance(ts_cfg, str):
            return pa.timestamp(ts_cfg)
        if isinstance(ts_cfg, dict):
            return pa.timestamp(ts_cfg["unit"], tz=ts_cfg.get("tz"))
        raise ValueError(f"Invalid timestamp schema spec: {ts_cfg!r}")

    raise ValueError(f"Unsupported schema type spec: {type_spec!r}")


def _parse_field(field_cfg: Dict[str, Any]) -> pa.Field:
    if "name" not in field_cfg or "type" not in field_cfg:
        raise ValueError("Each schema field must define 'name' and 'type'")
    return pa.field(
        field_cfg["name"],
        _parse_type(field_cfg["type"]),
        nullable=field_cfg.get("nullable", True),
    )


def load_schema_config(path: str) -> pa.Schema:
    with open(path, "r") as f:
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".json":
            payload = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(f)
        else:
            raise ValueError("Schema config must use a .json, .yaml, or .yml extension")

    if not isinstance(payload, dict) or "fields" not in payload:
        raise ValueError("Schema config must be an object with a 'fields' list")
    if not isinstance(payload["fields"], list) or not payload["fields"]:
        raise ValueError("Schema config 'fields' must be a non-empty list")

    return pa.schema([_parse_field(field_cfg) for field_cfg in payload["fields"]])
