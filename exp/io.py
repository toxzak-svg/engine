from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_yaml_or_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {file_path}, got {type(data).__name__}")
        return data
    except ModuleNotFoundError:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {file_path}, got {type(data).__name__}")
        return data


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {file_path}, got {type(data).__name__}")
    return data


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
