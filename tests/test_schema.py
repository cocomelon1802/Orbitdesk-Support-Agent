import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema_validate import validate_response


def test_valid_response_passes():
    response = {
        "classification": "answerable",
        "answer": "Viewers cannot create API credentials; ask an Owner or Admin.",
        "sources": [{"source_id": "KB-005", "passage": "Only Owners and Admins can create credentials."}],
        "confidence": 0.9,
        "requires_human": False,
        "reason": "Directly answerable from KB-005.",
        "clarification_question": None,
        "warnings": [],
    }
    ok, err = validate_response(response)
    assert ok, err


def test_missing_required_field_fails():
    response = {
        "classification": "answerable",
        "answer": "Some answer.",
        "sources": [],
        "confidence": 0.9,
        "requires_human": False,
        # "reason" missing
    }
    ok, err = validate_response(response)
    assert not ok
    assert "reason" in err


def test_invalid_classification_value_fails():
    response = {
        "classification": "maybe",
        "answer": "Some answer.",
        "sources": [],
        "confidence": 0.9,
        "requires_human": False,
        "reason": "test",
    }
    ok, _ = validate_response(response)
    assert not ok
