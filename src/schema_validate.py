"""Validate a response dict against data/output_schema.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "output_schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_response(response: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns (is_valid, error_message). error_message is '' if valid."""
    try:
        jsonschema.validate(instance=response, schema=_SCHEMA)
        return True, ""
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e.message)
