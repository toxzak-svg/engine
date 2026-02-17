from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .io import read_json

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


class SchemaValidationError(ValueError):
    pass


def load_schema(schema_name: str) -> dict[str, Any]:
    return read_json(SCHEMA_DIR / f"{schema_name}.schema.json")


def validate_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "enum" in schema:
        if instance not in schema["enum"]:
            raise SchemaValidationError(f"{path}: value {instance!r} not in enum {schema['enum']!r}")

    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(instance, expected_type, path)

    if isinstance(expected_type, list):
        # The simple implementation only checks shared keywords if one branch matches.
        if not any(_is_type(instance, t) for t in expected_type):
            raise SchemaValidationError(f"{path}: expected one of {expected_type}, got {type(instance).__name__}")
        branch_type = next(t for t in expected_type if _is_type(instance, t))
        expected_type = branch_type

    if expected_type == "object":
        _validate_object(instance, schema, path)
    elif expected_type == "array":
        _validate_array(instance, schema, path)
    elif expected_type in {"string", "number", "integer"}:
        _validate_scalar(instance, schema, path, expected_type)


def _validate_type(instance: Any, expected_type: str | list[str], path: str) -> None:
    if isinstance(expected_type, list):
        if any(_is_type(instance, item) for item in expected_type):
            return
        raise SchemaValidationError(f"{path}: expected one of {expected_type}, got {type(instance).__name__}")
    if not _is_type(instance, expected_type):
        raise SchemaValidationError(f"{path}: expected type {expected_type}, got {type(instance).__name__}")


def _is_type(instance: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(instance, dict)
    if expected_type == "array":
        return isinstance(instance, list)
    if expected_type == "string":
        return isinstance(instance, str)
    if expected_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected_type == "number":
        return (isinstance(instance, (int, float)) and not isinstance(instance, bool) and math.isfinite(float(instance)))
    if expected_type == "boolean":
        return isinstance(instance, bool)
    return False


def _validate_object(instance: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(instance, dict):
        raise SchemaValidationError(f"{path}: expected object")

    required = schema.get("required", [])
    for key in required:
        if key not in instance:
            raise SchemaValidationError(f"{path}.{key}: missing required key")

    properties = schema.get("properties", {})
    additional_properties = schema.get("additionalProperties", True)

    for key, value in instance.items():
        if key in properties:
            validate_schema(value, properties[key], f"{path}.{key}")
            continue
        if additional_properties is False:
            raise SchemaValidationError(f"{path}.{key}: additional properties are not allowed")
        if isinstance(additional_properties, dict):
            validate_schema(value, additional_properties, f"{path}.{key}")


def _validate_array(instance: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(instance, list):
        raise SchemaValidationError(f"{path}: expected array")

    min_items = schema.get("minItems")
    if min_items is not None and len(instance) < min_items:
        raise SchemaValidationError(f"{path}: expected at least {min_items} items, got {len(instance)}")
    max_items = schema.get("maxItems")
    if max_items is not None and len(instance) > max_items:
        raise SchemaValidationError(f"{path}: expected at most {max_items} items, got {len(instance)}")

    if schema.get("uniqueItems"):
        seen = set()
        for item in instance:
            marker = repr(item)
            if marker in seen:
                raise SchemaValidationError(f"{path}: duplicate item {item!r} in uniqueItems array")
            seen.add(marker)

    items_schema = schema.get("items")
    if items_schema is not None:
        for idx, item in enumerate(instance):
            validate_schema(item, items_schema, f"{path}[{idx}]")


def _validate_scalar(instance: Any, schema: dict[str, Any], path: str, expected_type: str) -> None:
    if expected_type == "string":
        min_length = schema.get("minLength")
        if min_length is not None and len(instance) < min_length:
            raise SchemaValidationError(f"{path}: expected minLength {min_length}, got {len(instance)}")
    if expected_type in {"number", "integer"}:
        numeric = float(instance)
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and numeric < minimum:
            raise SchemaValidationError(f"{path}: expected minimum {minimum}, got {numeric}")
        if maximum is not None and numeric > maximum:
            raise SchemaValidationError(f"{path}: expected maximum {maximum}, got {numeric}")
