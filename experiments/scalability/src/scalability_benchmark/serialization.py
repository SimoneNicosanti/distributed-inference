"""Canonical domain records, including mappings with structured keys."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from scalability_benchmark.randomness import fingerprint


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if all(isinstance(k, str) for k in value):
            return {k: jsonable(v) for k, v in sorted(value.items())}
        return sorted(
            [[jsonable(k), jsonable(v)] for k, v in value.items()],
            key=lambda pair: fingerprint(pair[0]),
        )
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted([jsonable(v) for v in value], key=fingerprint)
    return (
        value
        if value is None or isinstance(value, (str, int, float, bool))
        else str(value)
    )
