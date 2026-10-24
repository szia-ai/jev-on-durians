"""Normalisation, costs, the cache and the runner - all offline, with made-up responses."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import benchmarking
from benchmarking import (
    RawCall,
    Solution,
    cache_path,
    confidence_llm,
    confidence_openai_decisions,
    confidence_systemone,
    cost_usd,
    load_opinions,
    parse_llm,
    parse_openai_decisions,
    parse_systemone,
    run_benchmark,
    score_from_probabilities,
    smoke_subset,
)
from schema import QUESTIONS


def test_load_opinions_types(opinions: pd.DataFrame) -> None:
    assert len(opinions) == 6
    assert opinions["names_own_variety"].dtype == bool
    assert opinions["gold_compares"].tolist() == [1, 1, 0, 0, 0, 0]
    assert opinions["real_score"].notna().sum() == 2
    assert opinions.loc[3, "published"] == ""


def test_smoke_subset_covers_every_durian_and_language() -> None:
    subset = smoke_subset(load_opinions())
    assert len(subset) == 5
    assert set(subset["variety"]) == {"musang_king", "black_thorn", "red_prawn"}
    assert set(subset["language"]) == {"en", "zh", "ms"}


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"0": 0.0, "1": 0.5, "2": 0.5}, 0.75),
        ({"1": 0.0, "2": 0.5, "3": 0.5}, 0.75),
        ({"2": 1.0, "0": 0.0, "1": 0.0}, 1.0),
        ([2.0, 2.0], 0.5),
        ([1.0], 0.0),
        ({"a": 1.0, "b": 0.0}, 0.0),
    ],
)
def test_score_from_probabilities(
    probabilities: dict[str, float] | list[float], expected: float
) -> None:
    assert score_from_probabilities(probabilities) == pytest.approx(expected)


def test_parse_systemone(systemone_response: dict[str, Any]) -> None:
    answers = parse_systemone(systemone_response)
    assert answers.smell_mentioned == pytest.approx(0.9)
    assert answers.taste_mentioned == pytest.approx(0.2)
    assert answers.overall_sentiment == "mixed"
    assert answers.overall_rating == pytest.approx(8.5 / 9)
    assert answers.humour == pytest.approx(2.5 / 3)


def test_parse_openai_decisions_matches_systemone(
    decisions_response: dict[str, Any], systemone_response: dict[str, Any]
) -> None:
    assert parse_openai_decisions(decisions_response) == parse_systemone(systemone_response)


def test_parse_openai_decisions_refusal(decisions_response: dict[str, Any]) -> None:
    decisions_response["answers"][0] = {"type": "refusal", "name": "smell_mentioned"}
    with pytest.raises(ValueError, match="refused"):
        parse_openai_decisions(decisions_response)


def test_confidences(
    systemone_response: dict[str, Any],
    decisions_response: dict[str, Any],
    llm_response: dict[str, Any],
) -> None:
    systemone = confidence_systemone(systemone_response)
    assert systemone["smell_mentioned"] == pytest.approx(0.9)
    assert systemone["taste_mentioned"] == pytest.approx(0.8)
    assert systemone["overall_sentiment"] == pytest.approx(0.7)
    assert systemone["overall_rating"] == pytest.approx(0.5)
    assert confidence_openai_decisions(decisions_response) == pytest.approx(systemone)
    assert set(confidence_llm(llm_response).values()) == {1.0}


def test_parse_llm(llm_response: dict[str, Any]) -> None:
    answers = parse_llm(llm_response)
    assert answers.smell_mentioned == 1.0
    assert answers.taste_mentioned == 0.0
    assert answers.overall_sentiment == "mixed"
    assert answers.overall_rating == pytest.approx(8 / 9)
    assert answers.humour == pytest.approx(2 / 3)


def test_parse_llm_rejects_unknown_level(llm_response: dict[str, Any]) -> None:
    llm_response["answer"]["humour"] = "hilarious"
    with pytest.raises(ValueError):
        parse_llm(llm_response)


def test_cost_usd() -> None:
    assert cost_usd("jev", {"input_tokens": 1_000_000, "output_tokens": 0}) == pytest.approx(0.042)
    assert cost_usd("ollama_kev", {"input_tokens": 5000, "output_tokens": 0}) == 0.0
    luna = cost_usd(
        "gpt_6_luna", {"input_tokens": 1000, "output_tokens": 200, "cached_input_tokens": 400}
    )
    assert luna == pytest.approx((600 * 0.10 + 400 * 0.01 + 200 * 0.50) / 1e6)
    assert cost_usd("jev", {}) == 0.0


def _fake_solution(response: dict[str, Any], counter: dict[str, int]) -> Solution:
    def call(quote: str) -> RawCall:
        counter["calls"] += 1
        usage = {"input_tokens": 100, "output_tokens": 0, "cached_input_tokens": 0}
        return RawCall({"state": quote}, response, 0.25, usage)

    return Solution(
        "fake",
        "Fake",
        "decision_cloud",
        "fake-1",
        "jev",
        call,
        parse_systemone,
        confidence_systemone,
        lambda: None,
    )


def test_cache_path_is_namespaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmarking, "CACHE_DIR", tmp_path)
    solution = Solution(
        "kev",
        "Kev",
        "decision_local",
        "kev:4b-q8_0",
        "ollama_kev",
        lambda q: None,
        parse_systemone,
        confidence_systemone,
        lambda: None,
    )
    assert (
        cache_path(solution, "mk-en-01")
        == tmp_path / "kev" / "kev_4b-q8_0" / "v3" / "mk-en-01.json"
    )


def test_runner_caches_and_force_rerun_bypasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opinions: pd.DataFrame,
    systemone_response: dict[str, Any],
) -> None:
    monkeypatch.setattr(benchmarking, "CACHE_DIR", tmp_path)
    counter = {"calls": 0}
    solution = _fake_solution(systemone_response, counter)

    first = run_benchmark(opinions, [solution])
    assert counter["calls"] == 6
    assert not first["cached"].any()
    assert first["error"].isna().all()
    assert first["latency_s"].tolist() == [0.25] * 6
    assert first["cost_usd"].iloc[0] == pytest.approx(100 * 0.042 / 1e6)
    assert first["overall_sentiment"].tolist() == ["mixed"] * 6
    assert first["overall_sentiment_confidence"].tolist() == pytest.approx([0.7] * 6)
    assert set(QUESTIONS[0].name for _ in [0]) <= set(first.columns)

    second = run_benchmark(opinions, [solution])
    assert counter["calls"] == 6
    assert second["cached"].all()

    third = run_benchmark(opinions, [solution], force_rerun=True)
    assert counter["calls"] == 12
    assert not third["cached"].any()


def test_runner_records_errors_and_stops_a_failing_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opinions: pd.DataFrame
) -> None:
    monkeypatch.setattr(benchmarking, "CACHE_DIR", tmp_path)

    def broken(quote: str) -> RawCall:
        raise ValueError("boom")

    solution = Solution(
        "broken",
        "Broken",
        "llm_cloud",
        "x",
        "jev",
        broken,
        parse_systemone,
        confidence_systemone,
        lambda: None,
    )
    results = run_benchmark(opinions, [solution], max_consecutive_errors=2)
    assert results["error"].notna().all()
    assert results["error"].iloc[0] == "ValueError: boom"
    assert results["error"].iloc[-1] == "stopped after 2 consecutive errors"
    assert results["overall_rating"].isna().all()
    assert not list(tmp_path.iterdir())


def test_runner_warms_up_local_models_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opinions: pd.DataFrame,
    systemone_response: dict[str, Any],
) -> None:
    monkeypatch.setattr(benchmarking, "CACHE_DIR", tmp_path)
    counter = {"calls": 0}
    solution = _fake_solution(systemone_response, counter)
    solution.family = "decision_local"
    run_benchmark(opinions, [solution])
    assert counter["calls"] == 7


def test_runner_stops_over_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opinions: pd.DataFrame,
    systemone_response: dict[str, Any],
) -> None:
    monkeypatch.setattr(benchmarking, "CACHE_DIR", tmp_path)
    counter = {"calls": 0}
    solution = _fake_solution(systemone_response, counter)
    with pytest.raises(RuntimeError, match="over the budget"):
        run_benchmark(opinions, [solution], budget_usd=0.00001)
    assert counter["calls"] == 3


def test_runner_with_no_solutions_has_the_columns(opinions: pd.DataFrame) -> None:
    results = run_benchmark(opinions, [])
    assert results.empty
    assert "error" in results.columns


def test_retryable_errors() -> None:
    class Status(Exception):
        status_code = 429

    class Other(Exception):
        status_code = 400

    assert benchmarking._retryable(Status())
    assert not benchmarking._retryable(Other())
    assert not benchmarking._retryable(ValueError("x"))


def test_registry_ids_are_unique_and_priced() -> None:
    solutions = benchmarking.registry(local_llm="qwen3:8b")
    ids = [solution.id for solution in solutions]
    assert len(ids) == len(set(ids))
    prices = benchmarking.load_prices()["solutions"]
    for solution in solutions:
        assert solution.price_key in prices, solution.id
        if solution.id != "ollama_llm":
            assert prices[solution.price_key]["model"] == solution.model, solution.id
    assert "ollama_clef_flash" in ids
