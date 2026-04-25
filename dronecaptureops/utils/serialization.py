"""Serialization helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Convert pydantic models and nested values into JSON-serializable data."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value
