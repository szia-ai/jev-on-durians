"""The Durian Elo on a fixed fixture gives a fixed ranking."""

import pandas as pd
import pytest

from elo import (
    EloRow,
    build_matches,
    durian_elo,
    elo_table,
    expected_score,
    gold_rows,
    named_opponent,
    ratings_by_durian,
    solution_rows,
)


def test_expected_score() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) == pytest.approx(0.9091, abs=1e-4)


@pytest.mark.parametrize(
    ("quote", "own", "expected"),
    [
        ("Not as creamy as the red prawn durian", "musang_king", "red_prawn"),
        ("黑刺比猫山王好吃", "black_thorn", "musang_king"),
        ("吃過黑剌榴槤，就覺得貓山王普通", "musang_king", "black_thorn"),
        ("not so creamy compared to mau sang", "red_prawn", "musang_king"),
        ("Musang King is the best", "musang_king", None),
        ("just a durian", "red_prawn", None),
    ],
)
def test_named_opponent(quote: str, own: str, expected: str | None) -> None:
    assert named_opponent(quote, own) == expected


def test_gold_rows_and_matches(opinions: pd.DataFrame) -> None:
    rows = gold_rows(opinions)
    assert [row.score for row in rows] == [0.9, 1.0, 0.0, 1.0, 0.6, 0.5]
    assert rows[0].named_opponent == "black_thorn"
    assert rows[3].published is None
    kinds = [match.kind for match in build_matches(rows)]
    assert kinds.count("h2h") == 2
    assert kinds.count("field") == 6
    assert kinds.count("writer") == 3


def test_gold_elo_fixed_ranking(opinions: pd.DataFrame) -> None:
    result = durian_elo(gold_rows(opinions))
    assert result.matches == {"h2h": 2, "writer": 3, "field": 6}
    assert result.rating["musang_king"] == pytest.approx(1512.60, abs=0.01)
    assert result.rating["black_thorn"] == pytest.approx(1494.24, abs=0.01)
    assert result.rating["red_prawn"] == pytest.approx(1500.99, abs=0.01)
    assert result.ranking == ["musang_king", "red_prawn", "black_thorn"]
    assert result.king == "musang_king"
    assert len(result.history) == 11
    assert list(result.history.columns) == [
        "when",
        "kind",
        "musang_king",
        "black_thorn",
        "red_prawn",
    ]


def test_undated_opinions_play_last() -> None:
    rows = [
        EloRow("a", "red_prawn", "w1", None, 1.0, False, "none", None),
        EloRow("b", "musang_king", "w2", "2020-01-01", 0.0, False, "none", None),
    ]
    history = durian_elo(rows).history
    assert history["when"].tolist() == ["2020-01-01", "9999"]


def test_ratings_by_durian(opinions: pd.DataFrame) -> None:
    gold = gold_rows(opinions)
    table = ratings_by_durian(gold, {"half": [row for row in gold if row.variety != "red_prawn"]})
    assert list(table.index) == ["gold", "half"]
    assert table.loc["gold", "musang_king"] == pytest.approx(0.75)
    assert table.loc["gold", "black_thorn"] == pytest.approx(0.75)
    assert table.loc["gold", "red_prawn"] == pytest.approx(0.5)
    assert pd.isna(table.loc["half", "red_prawn"])


def test_solution_rows_and_elo_table(opinions: pd.DataFrame) -> None:
    answers = pd.DataFrame(
        {
            "opinion_id": opinions["id"],
            "overall_rating": [0.2, 0.9, 0.9, 0.9, 0.2, 0.9],
            "compares": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
            "preferred": ["black_thorn", "none", "none", "none", "none", "none"],
        }
    )
    answers.loc[3, "overall_rating"] = float("nan")
    rows = solution_rows(opinions, answers)
    assert len(rows) == 5
    assert rows[0].compares and rows[0].preferred == "black_thorn"
    gold = durian_elo(gold_rows(opinions))
    solution = durian_elo(rows)
    table = elo_table(gold, {"flipped": solution})
    assert list(table.index) == ["gold", "flipped"]
    assert table.loc["gold", "same_king"] and table.loc["gold", "same_ranking"]
    assert table.loc["flipped", "king"] != "musang_king"
    assert not table.loc["flipped", "same_king"]
