"""results/summary.md and the README results block."""

from pathlib import Path

import pandas as pd
import pytest

from report import (
    RESULTS_END,
    RESULTS_START,
    RunInfo,
    markdown_table,
    observations,
    read_observations,
    replace_results_block,
    results_block,
    summary_markdown,
    write_results,
)


@pytest.fixture
def summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["Jev", "Local LLM"],
            "family": ["decision_cloud", "llm_local"],
            "local": [False, True],
            "headline": [0.8, 0.6],
            "aspects_f1": [0.9, 0.7],
            "overall_sentiment_acc": [0.8, 0.5],
            "compares_f1": [0.7, 0.6],
            "preferred_acc": [1.0, float("nan")],
            "variety_acc": [0.8, 0.6],
            "headline_en": [0.9, 0.7],
            "headline_zh": [0.7, 0.5],
            "headline_ms": [0.8, 0.6],
            "p50_s": [0.4, 12.0],
            "p95_s": [0.9, 20.0],
            "usd_run": [0.0031, 0.0],
            "usd_per_1m_opinions": [42.0, 0.0],
        },
        index=pd.Index(["jev", "ollama_llm"], name="solution"),
    )


@pytest.fixture
def elo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "musang_king": [1430, 1480],
            "black_thorn": [1580, 1520],
            "red_prawn": [1613, 1530],
            "king": ["red_prawn", "red_prawn"],
            "same_king": [True, True],
            "same_ranking": [True, False],
            "h2h": [7, 5],
            "writer": [21, 21],
            "field": [83, 83],
        },
        index=pd.Index(["gold", "jev"], name="rater"),
    )


def test_markdown_table_formats_cells() -> None:
    frame = pd.DataFrame(
        {"a": [1.5, float("nan")], "ok": [True, False], "name": ["x", "y"]},
        index=pd.Index(["r1", "r2"], name="row"),
    )
    assert markdown_table(frame).splitlines() == [
        "| row | a | ok | name |",
        "|---|---|---|---|",
        "| r1 | 1.50 | yes | x |",
        "| r2 | - | no | y |",
    ]


def test_replace_results_block_keeps_the_rest() -> None:
    readme = f"# Title\n\n{RESULTS_START}\nold\n{RESULTS_END}\n\n## Credits\n"
    updated = replace_results_block(readme, "new block")
    assert updated == f"# Title\n\n{RESULTS_START}\nnew block\n{RESULTS_END}\n\n## Credits\n"
    assert replace_results_block(updated, "third") == updated.replace("new block", "third")
    with pytest.raises(ValueError, match="markers"):
        replace_results_block("no markers here", "x")


def test_observations_are_plain_sentences(summary: pd.DataFrame, elo: pd.DataFrame) -> None:
    notes = observations(summary, elo)
    assert 3 <= len(notes) <= 5
    assert notes[0].startswith("Jev reads durians best")
    assert any("Red Prawn" in note for note in notes)
    assert any("Chinese is the hardest language" in note for note in notes)


def test_results_block_and_summary(summary: pd.DataFrame, elo: pd.DataFrame) -> None:
    info = RunInfo(83, "full", "2026-10-23 22:00 +08", {"kev": "model not pulled"})
    ratings = pd.DataFrame(
        {"musang_king": [0.6, 0.7], "black_thorn": [0.8, 0.75], "red_prawn": [0.85, 0.7]},
        index=pd.Index(["gold", "jev"], name="rater"),
    )
    block = results_block(summary, elo, info, [Path("assets/charts/01_cost_vs_accuracy.png")])
    assert "Measured on 83 opinions, 2026-10-23 22:00 +08" in block
    assert "| 1 | Jev | decision, cloud | 0.80 |" in block
    assert "| gold | 1430 | 1580 | 1613 |" not in block  # part 2 stays in summary.md
    assert "Mean overall rating per durian" not in block
    assert "[cost vs accuracy](assets/charts/01_cost_vs_accuracy.png)" in block
    assert "`kev` (model not pulled)" in block
    assert "Jev reads durians best" in block  # generated observations by default
    curated = results_block(summary, elo, info, [], notes=["Hand-written note one."])
    assert "- Hand-written note one." in curated and "reads durians best" not in curated
    confidence = pd.DataFrame(
        {"Jev": ["0.97 (500)", "0.80 (40)"]},
        index=pd.Index(["≥ 0.95", "0.90 - 0.95"], name="confidence"),
    )
    with_confidence = results_block(summary, elo, info, [], confidence, confidence)
    assert "Accuracy by confidence bin" in with_confidence
    assert "manual review" in with_confidence
    assert with_confidence.count("| ≥ 0.95 | 0.97 (500) |") == 2
    empty = pd.DataFrame(index=pd.Index([], name="confidence"))
    assert "Accuracy by confidence bin" not in results_block(summary, elo, info, [], empty)
    text = summary_markdown(
        summary,
        elo,
        info,
        main_only=summary,
        ratings=ratings,
        confidence=confidence,
        cutoffs=confidence,
        notes=["Curated."],
    )
    assert text.startswith("# Results summary")
    assert "Check on the 70 main-set opinions only" in text
    assert "## Skipped solutions" in text
    assert "## Confidence and a manual-review cutoff" in text
    assert text.count("| ≥ 0.95 | 0.97 (500) |") == 2
    assert "- Curated." in text and "reads durians best" not in text
    assert "| gold | 1430 | 1580 | 1613 |" in text and "| jev | 0.70 | 0.75 | 0.70 |" in text


def test_write_results(tmp_path: Path, summary: pd.DataFrame, elo: pd.DataFrame) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{RESULTS_START}\n_not run_\n{RESULTS_END}\nend\n", encoding="utf-8")
    info = RunInfo(5, "smoke", "2026-10-23 22:00 +08", {}, machine="MacBook Air, Apple M2, 24 GB")
    write_results(readme, tmp_path / "results" / "summary.md", summary, elo, info, [])
    assert (tmp_path / "results" / "summary.md").exists()
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("end\n")
    assert "_not run_" not in text and "Measured on 5 opinions" in text
    assert "local models ran on MacBook Air, Apple M2, 24 GB" in text


def test_read_observations(tmp_path: Path) -> None:
    path = tmp_path / "observations.md"
    path.write_text("# notes\n\n- First one.\n-\nplain line\n- Second one.  \n", encoding="utf-8")
    assert read_observations(path) == ["First one.", "Second one."]
