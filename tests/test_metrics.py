"""Accuracy, time and money tables on made-up answers."""

import numpy as np
import pandas as pd
import pytest

from metrics import (
    accuracy,
    accuracy_table,
    agreement_matrix,
    annotate,
    answer_confidences,
    confidence_matrix,
    cost_table,
    cutoff_table,
    f1_binary,
    macro_f1,
    rating_points,
    review_curve,
    spearman,
    summary_table,
    time_table,
)
from schema import QUESTIONS


def test_spearman_on_ranks() -> None:
    a = pd.Series([0.1, 0.4, 0.4, 0.9])
    assert spearman(a, pd.Series([1.0, 2.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert spearman(a, pd.Series([4.0, 3.0, 2.0, 1.0])) == pytest.approx(-0.9486833, abs=1e-6)
    assert np.isnan(spearman(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])))


def test_f1_and_accuracy_basics() -> None:
    gold = pd.Series([True, True, False, False])
    assert f1_binary(gold, pd.Series([True, False, False, False])) == pytest.approx(2 / 3)
    assert f1_binary(gold, gold) == 1.0
    assert np.isnan(f1_binary(pd.Series([False]), pd.Series([False])))
    assert accuracy(pd.Series(["a", "b"]), pd.Series(["a", "a"])) == 0.5
    assert np.isnan(accuracy(pd.Series([], dtype=str), pd.Series([], dtype=str)))
    assert macro_f1(pd.Series(["a", "b"]), pd.Series(["a", "b"]), ("a", "b", "c")) == 1.0


def perfect_results(opinions: pd.DataFrame, solution: str) -> pd.DataFrame:
    """Answers that reproduce the gold labels exactly (scores are made up)."""
    rows = []
    for opinion in opinions.itertuples(index=False):
        row = {
            "solution": solution,
            "model": solution,
            "opinion_id": opinion.id,
            "cached": False,
            "latency_s": 0.5,
            "input_tokens": 1000,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cost_usd": 0.001,
            "error": None,
        }
        for question in QUESTIONS:
            row[f"{question.name}_confidence"] = 1.0
            if question.name.endswith("_mentioned"):
                row[question.name] = float(getattr(opinion, f"gold_{question.name[:-10]}"))
            elif question.name.endswith("_sentiment") and question.kind == "score":
                row[question.name] = 0.5
            elif question.name == "overall_rating":
                row[question.name] = 0.8
            elif question.name == "humour":
                row[question.name] = 0.0
            elif question.name == "compares":
                row[question.name] = float(opinion.gold_compares)
            elif question.name == "preferred":
                row[question.name] = opinion.gold_preferred
            elif question.name == "overall_sentiment":
                row[question.name] = opinion.gold_overall_sentiment
            else:
                row[question.name] = opinion.variety if opinion.names_own_variety else "not_named"
        rows.append(row)
    return pd.DataFrame(rows)


def test_perfect_answers_score_one(opinions: pd.DataFrame) -> None:
    table = accuracy_table(perfect_results(opinions, "perfect"), opinions)
    row = table.loc["perfect"]
    assert row["aspects_f1"] == 1.0
    assert row["overall_sentiment_acc"] == 1.0
    assert row["compares_f1"] == 1.0
    assert row["preferred_acc"] == 1.0
    assert row["variety_acc"] == 1.0
    assert row["headline"] == 1.0
    assert row["answer_acc"] == 1.0
    assert row["n_answered"] == 6 and row["n_errors"] == 0
    assert row["headline_zh"] == 1.0
    assert row["n_rated"] == 2
    assert np.isnan(row["rating_spearman"])
    assert row["rating_mae"] == pytest.approx(np.mean([abs(0.9 - 0.8), abs(0.6 - 0.8)]))


def test_one_wrong_answer_lowers_the_right_metric(opinions: pd.DataFrame) -> None:
    results = perfect_results(opinions, "flawed")
    results.loc[0, "overall_sentiment"] = "negative"
    results.loc[1, "taste_mentioned"] = 0.0
    results.loc[2, "error"] = "HTTPError: 500"
    table = accuracy_table(results, opinions)
    row = table.loc["flawed"]
    assert row["n_answered"] == 5 and row["n_errors"] == 1
    assert row["overall_sentiment_acc"] == pytest.approx(4 / 5)
    assert row["answer_acc"] == pytest.approx(1 - 2 / 45)  # 45 scored answers, two wrong
    assert row["aspects_f1"] < 1.0
    assert row["compares_f1"] == 1.0


def test_time_cost_and_summary_tables(opinions: pd.DataFrame) -> None:
    results = perfect_results(opinions, "a")
    results.loc[0, "latency_s"] = 2.0
    assert time_table(results).loc["a", "p50_s"] == 0.5
    assert time_table(results).loc["a", "p95_s"] > 0.5
    costs = cost_table(results)
    assert costs.loc["a", "usd_run"] == pytest.approx(0.006)
    assert costs.loc["a", "usd_per_1m_opinions"] == pytest.approx(1000.0)
    solutions = pd.DataFrame(
        {"label": ["A"], "family": ["decision_cloud"], "model": ["a"], "local": [False]},
        index=pd.Index(["a"], name="solution"),
    )
    summary = summary_table(results, opinions, solutions)
    assert list(summary.index) == ["a"]
    assert summary.loc["a", "label"] == "A"
    assert summary.loc["a", "headline"] == 1.0
    assert summary.loc["a", "usd_per_1m_opinions"] == pytest.approx(1000.0)


def test_agreement_and_rating_points(opinions: pd.DataFrame) -> None:
    a = perfect_results(opinions, "a")
    b = perfect_results(opinions, "b")
    b["humour"] = np.arange(len(b)) / 10
    a["humour"] = np.arange(len(a)) / 5
    agreement = agreement_matrix(pd.concat([a, b]), "humour")
    assert agreement.loc["a", "b"] == pytest.approx(1.0)
    points = rating_points(pd.concat([a, b]), opinions)
    assert len(points) == 4
    assert set(points.columns) == {"real_score", "solution", "opinion_id", "overall_rating"}


def test_confidence_matrix_and_cutoffs(opinions: pd.DataFrame) -> None:
    results = perfect_results(opinions, "a")
    results.loc[0, "overall_sentiment"] = "negative"  # one wrong answer, given with low confidence
    results.loc[0, "overall_sentiment_confidence"] = 0.55
    results.loc[1, "compares_confidence"] = 0.65  # a right answer, given with modest confidence
    labels = pd.Series({"a": "Solution A"})  # a Series, as the notebook passes it
    confidences = answer_confidences(results, opinions, labels)
    assert list(confidences.columns) == [
        "solution",
        "opinion_id",
        "question",
        "confidence",
        "correct",
    ]
    assert set(confidences["solution"]) == {"Solution A"}
    assert (
        len(confidences) == 6 * 6 + 6 + 6 + 2 + 4
    )  # aspects, sentiment, compares, preferred, variety
    assert confidences["correct"].sum() == len(confidences) - 1

    accuracy, counts = confidence_matrix(confidences)
    assert list(accuracy.index)[:2] == ["≥ 0.95", "0.90 - 0.95"]
    assert accuracy.loc["0.50 - 0.60", "Solution A"] == 0.0
    assert counts.loc["0.50 - 0.60", "Solution A"] == 1
    assert accuracy.loc["0.60 - 0.70", "Solution A"] == 1.0
    assert accuracy.loc["≥ 0.95", "Solution A"] == 1.0
    assert counts.loc["≥ 0.95", "Solution A"] == len(confidences) - 2
    assert counts.loc["< 0.50", "Solution A"] == 0

    cutoff_accuracy, share = cutoff_table(confidences)
    assert cutoff_accuracy.loc["all answers", "Solution A"] == pytest.approx(
        1 - 1 / len(confidences)
    )
    assert cutoff_accuracy.loc["≥ 0.60", "Solution A"] == 1.0
    assert share.loc["≥ 0.60", "Solution A"] == pytest.approx(1 - 1 / len(confidences))
    assert share.loc["≥ 0.95", "Solution A"] == pytest.approx(1 - 2 / len(confidences))

    shown = annotate(cutoff_accuracy, share, "{:.0%}")
    assert shown.loc["≥ 0.60", "Solution A"] == "1.00 (98%)"
    assert annotate(accuracy, counts).loc["< 0.50", "Solution A"] == "-"


def test_confidence_tables_without_decision_models() -> None:
    nothing = pd.DataFrame(columns=["solution", "opinion_id", "question", "confidence", "correct"])
    accuracy, counts = confidence_matrix(nothing)
    assert accuracy.empty and list(accuracy.index)[0] == "≥ 0.95"
    cutoff_accuracy, share = cutoff_table(nothing)
    assert cutoff_accuracy.empty and share.empty
    assert annotate(accuracy, counts).empty


def test_review_curve_removes_the_least_confident_first() -> None:
    confidences = pd.DataFrame(
        {
            "solution": ["a"] * 10,
            "opinion_id": [str(i) for i in range(10)],
            "question": ["compares"] * 10,
            "confidence": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99],
            "correct": [False, False, True, True, True, True, True, True, True, True],
        }
    )
    curve = review_curve(confidences, shares=(0.0, 0.1, 0.2, 0.3))
    assert curve["accuracy"].tolist() == pytest.approx([0.8, 8 / 9, 1.0, 1.0])
    assert curve["kept"].tolist() == [10, 9, 8, 7]
    assert review_curve(confidences.iloc[:0]).empty
