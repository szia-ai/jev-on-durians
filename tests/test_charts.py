"""The seven charts render from tiny tables and are at least 1600 px wide (needs a Chrome)."""

from pathlib import Path

import numpy as np
import pandas as pd

from charts import CHART_FILES, png_size, save_all


def test_save_all_writes_seven_wide_pngs(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "label": ["Jev", "Local LLM", "GPT", "Kev"],
            "family": ["decision_cloud", "llm_local", "llm_cloud", "decision_local"],
            "local": [False, True, False, True],
            "headline": [0.8, 0.6, 0.75, 0.7],
            "usd_per_1m_opinions": [42.0, 0.0, 900.0, 0.0],
            "p50_s": [0.4, 12.0, 3.0, 0.3],
            "p95_s": [0.9, 20.0, 5.0, 0.5],
        },
        index=pd.Index(["jev", "ollama_llm", "gpt", "kev"], name="solution"),
    )
    by_question = pd.DataFrame(
        {"aspects F1": [0.9, 0.7, 0.8, 0.75], "variety": [0.8, float("nan"), 0.9, 0.7]},
        index=pd.Index(summary["label"]),
    )
    by_language = pd.DataFrame(
        {
            "English": [0.9, 0.7, 0.8, 0.8],
            "Chinese": [0.7, 0.5, 0.8, 0.6],
            "Malay": [0.8, 0.6, 0.7, 0.6],
        },
        index=pd.Index(summary["label"]),
    )
    points = pd.DataFrame(
        {
            "label": ["Jev", "Jev", "GPT"],
            "family": ["decision_cloud", "decision_cloud", "llm_cloud"],
            "opinion_id": ["a", "b", "a"],
            "real_score": [0.9, 0.6, 0.9],
            "overall_rating": [0.8, 0.7, 0.95],
        }
    )
    history = pd.DataFrame(
        {
            "when": ["2020", "2021", "2022"],
            "kind": ["field", "h2h", "writer"],
            "musang_king": [1500, 1480, 1470],
            "black_thorn": [1504, 1520, 1525],
            "red_prawn": [1500, 1500, 1510],
        }
    )
    finals = {"Jev": {"musang_king": 1450, "black_thorn": 1530, "red_prawn": 1520}}

    figures = save_all(
        summary, by_question, by_language, points, history, finals, tmp_path, "2026-10-23", "M2"
    )
    assert list(figures) == list(CHART_FILES)
    for file in CHART_FILES.values():
        width, height = png_size(tmp_path / file)
        assert width >= 1600, file
        assert height >= 900, file


def test_charts_tolerate_empty_input(tmp_path: Path) -> None:
    columns = ["label", "family", "local", "headline", "usd_per_1m_opinions", "p50_s", "p95_s"]
    summary = pd.DataFrame(columns=columns).astype({"local": bool, "headline": float})
    empty_table = pd.DataFrame()
    points = pd.DataFrame(columns=["label", "family", "opinion_id", "real_score", "overall_rating"])
    history = pd.DataFrame(columns=["when", "kind", "musang_king", "black_thorn", "red_prawn"])
    figures = save_all(summary, empty_table, empty_table, points, history, {}, tmp_path, "d")
    assert len(figures) == 7
    assert np.all([png_size(tmp_path / file)[0] >= 1600 for file in CHART_FILES.values()])
