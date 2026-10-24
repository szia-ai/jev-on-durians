"""Shared fixtures: made-up opinions and recorded (made-up) API responses under tests/fixtures/."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from benchmarking import load_opinions

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def opinions() -> pd.DataFrame:
    """Six made-up opinions with gold labels, two writers who reviewed two durians."""
    return load_opinions(FIXTURES / "opinions.csv")


@pytest.fixture
def systemone_response() -> dict[str, Any]:
    """A System One response (Jev / Ollama shape) for the 18 questions."""
    return json.loads((FIXTURES / "systemone_response.json").read_text(encoding="utf-8"))


@pytest.fixture
def decisions_response() -> dict[str, Any]:
    """An OpenAI Decisions API response for the 18 questions, with 1-based level values."""
    return json.loads((FIXTURES / "decisions_response.json").read_text(encoding="utf-8"))


@pytest.fixture
def llm_response() -> dict[str, Any]:
    """A stored structured-output answer (`answer` = a DurianAnswer as a dict)."""
    return json.loads((FIXTURES / "llm_response.json").read_text(encoding="utf-8"))
