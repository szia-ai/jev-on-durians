"""The Durian Elo: three durians, three kinds of matches, played in date order (part 2)."""

import dataclasses
from collections import defaultdict
from itertools import combinations
from typing import Literal

import pandas as pd

DURIANS = ("musang_king", "black_thorn", "red_prawn")
START_RATING = 1500.0
K_HEAD_TO_HEAD = 32
K_SAME_WRITER = 24
K_FIELD = 8
DRAW_GAP = 0.1
UNDATED = "9999"
SENTIMENT_SCORE = {"positive": 1.0, "mixed": 0.5, "neutral": 0.5, "negative": 0.0}

# The names in config/durian.yaml plus the spellings that occur in the quotes (黑剌, "mau sang").
DURIAN_ALIASES: dict[str, tuple[str, ...]] = {
    "musang_king": (
        "musang king",
        "mao shan wang",
        "mao shan",
        "mau sang",
        "msw",
        "d197",
        "猫山王",
        "貓山王",
    ),
    "black_thorn": ("black thorn", "blackthorn", "ochee", "duri hitam", "d200", "黑刺", "黑剌"),
    "red_prawn": (
        "red prawn",
        "ang hay",
        "ang hae",
        "udang merah",
        "hong xia",
        "d175",
        "红虾",
        "紅蝦",
    ),
}

MatchKind = Literal["h2h", "writer", "field"]


@dataclasses.dataclass(frozen=True)
class EloRow:
    """One opinion as the Elo sees it: who it is about, when, and how it scores."""

    id: str
    variety: str
    writer_id: str
    published: str | None
    score: float
    compares: bool
    preferred: str
    named_opponent: str | None


@dataclasses.dataclass(frozen=True)
class Match:
    when: str
    kind: MatchKind
    durian_a: str
    durian_b: str | None
    result: float


@dataclasses.dataclass
class EloResult:
    rating: dict[str, float]
    history: pd.DataFrame
    matches: dict[MatchKind, int]

    @property
    def ranking(self) -> list[str]:
        return sorted(self.rating, key=lambda durian: self.rating[durian], reverse=True)

    @property
    def king(self) -> str:
        return self.ranking[0]


def expected_score(rating_a: float, rating_b: float) -> float:
    """The classic Elo expectation of durian A against durian B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def named_opponent(quote: str, own_variety: str) -> str | None:
    """
    Find another of the three durians named in a quote (alias match, case-insensitive).

    Args:
        quote: The opinion text in its original language.
        own_variety: The durian the opinion is about; it is never returned.

    Returns:
        The first other durian whose alias occurs in the quote, or None.

    Example:
        >>> named_opponent("Not as creamy as the red prawn durian", "musang_king")
        'red_prawn'
    """
    text = quote.lower()
    for durian in DURIANS:
        if durian != own_variety and any(alias in text for alias in DURIAN_ALIASES[durian]):
            return durian
    return None


def build_matches(rows: list[EloRow]) -> list[Match]:
    """
    Turn the opinions into matches: head-to-head, same writer and field.

    Args:
        rows: One row per opinion.

    Returns:
        The matches, not yet sorted. Undated opinions sort last.
    """
    matches: list[Match] = []
    for row in rows:
        when = row.published or UNDATED
        is_head_to_head = (
            row.compares
            and row.preferred in DURIANS
            and row.named_opponent is not None
            and row.preferred in (row.variety, row.named_opponent)
        )
        if is_head_to_head:
            loser = row.named_opponent if row.preferred == row.variety else row.variety
            matches.append(Match(when, "h2h", row.preferred, loser, 1.0))
        matches.append(Match(when, "field", row.variety, None, row.score))

    by_writer: dict[str, dict[str, list[EloRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_writer[row.writer_id][row.variety].append(row)
    for varieties in by_writer.values():
        for durian_a, durian_b in combinations(sorted(varieties), 2):
            mean_a = sum(r.score for r in varieties[durian_a]) / len(varieties[durian_a])
            mean_b = sum(r.score for r in varieties[durian_b]) / len(varieties[durian_b])
            when = max(r.published or UNDATED for r in varieties[durian_a] + varieties[durian_b])
            if abs(mean_a - mean_b) < DRAW_GAP:
                result = 0.5
            else:
                result = 1.0 if mean_a > mean_b else 0.0
            matches.append(Match(when, "writer", durian_a, durian_b, result))
    return matches


def durian_elo(rows: list[EloRow]) -> EloResult:
    """
    Play all matches in date order and return the final ratings and the rating history.

    Args:
        rows: One row per opinion (see `gold_rows` and `solution_rows`).

    Returns:
        The final rating per durian, a history frame with one line per match (when, kind and the
        three ratings after it), and the number of matches per kind.
    """
    rating = {durian: START_RATING for durian in DURIANS}
    history: list[dict[str, float | str]] = []
    counts: dict[MatchKind, int] = {"h2h": 0, "writer": 0, "field": 0}
    for match in sorted(build_matches(rows), key=lambda m: m.when):
        counts[match.kind] += 1
        if match.kind == "field":
            rating[match.durian_a] += K_FIELD * (
                match.result - expected_score(rating[match.durian_a], START_RATING)
            )
        else:
            k = K_HEAD_TO_HEAD if match.kind == "h2h" else K_SAME_WRITER
            assert match.durian_b is not None
            delta = k * (
                match.result - expected_score(rating[match.durian_a], rating[match.durian_b])
            )
            rating[match.durian_a] += delta
            rating[match.durian_b] -= delta
        history.append({"when": match.when, "kind": match.kind, **rating})
    return EloResult(rating=rating, history=pd.DataFrame(history), matches=counts)


def gold_rows(opinions: pd.DataFrame) -> list[EloRow]:
    """
    Build the Elo rows from the gold labels.

    The score is the writer's own `real_score` where there is one, else the gold overall
    sentiment mapped positive 1, mixed or neutral 0.5, negative 0.

    Args:
        opinions: The dataset as loaded by `benchmarking.load_opinions`.

    Returns:
        One row per opinion.
    """
    rows = []
    for opinion in opinions.to_dict("records"):
        if pd.notna(opinion["real_score"]):
            score = float(opinion["real_score"])
        else:
            score = SENTIMENT_SCORE[opinion["gold_overall_sentiment"]]
        rows.append(
            EloRow(
                id=opinion["id"],
                variety=opinion["variety"],
                writer_id=opinion["writer_id"],
                published=opinion["published"] or None,
                score=score,
                compares=bool(opinion["gold_compares"]),
                preferred=opinion["gold_preferred"],
                named_opponent=named_opponent(opinion["quote"], opinion["variety"]),
            )
        )
    return rows


def solution_rows(opinions: pd.DataFrame, answers: pd.DataFrame) -> list[EloRow]:
    """
    Build the Elo rows from one solution's answers.

    The score is the solution's `overall_rating`, `compares` is its probability at 0.5 and
    `preferred` its chosen option; the durian and the named opponent are the same as for gold.

    Args:
        opinions: The dataset as loaded by `benchmarking.load_opinions`.
        answers: The runner's rows for one solution, with the answer columns and `opinion_id`.

    Returns:
        One row per answered opinion; opinions without an answer are left out.
    """
    answered = answers.dropna(subset=["overall_rating"]).set_index("opinion_id")
    rows = []
    for opinion in opinions.to_dict("records"):
        if opinion["id"] not in answered.index:
            continue
        answer = answered.loc[opinion["id"]]
        rows.append(
            EloRow(
                id=opinion["id"],
                variety=opinion["variety"],
                writer_id=opinion["writer_id"],
                published=opinion["published"] or None,
                score=float(answer["overall_rating"]),
                compares=float(answer["compares"]) >= 0.5,
                preferred=str(answer["preferred"]),
                named_opponent=named_opponent(opinion["quote"], opinion["variety"]),
            )
        )
    return rows


def ratings_by_durian(gold: list[EloRow], per_solution: dict[str, list[EloRow]]) -> pd.DataFrame:
    """
    The mean overall score each rater gave each durian, 0..1.

    The gold row is the people's score the Elo uses: the writer's own score where there is one,
    else the gold sentiment mapped to 1 / 0.5 / 0. A solution's row is its mean `overall_rating`.

    Args:
        gold: The rows from `gold_rows`.
        per_solution: Solution label -> the rows from `solution_rows`.

    Returns:
        One row per rater (gold first), one column per durian, NaN where a rater has no opinion.
    """
    rows = []
    for rater, elo_rows in [("gold", gold), *per_solution.items()]:
        row: dict[str, str | float] = {"rater": rater}
        for durian in DURIANS:
            scores = [elo_row.score for elo_row in elo_rows if elo_row.variety == durian]
            row[durian] = sum(scores) / len(scores) if scores else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("rater")


def elo_table(gold: EloResult, per_solution: dict[str, EloResult]) -> pd.DataFrame:
    """
    Summarise the gold Elo and every solution's Elo in one table.

    Args:
        gold: The Elo from the gold labels.
        per_solution: The Elo per solution id.

    Returns:
        One row per rater (gold first) with the three ratings, the king, and whether the king and
        the full ranking match the gold's.
    """
    rows = []
    for name, result in [("gold", gold), *per_solution.items()]:
        rows.append(
            {
                "rater": name,
                **{durian: round(result.rating[durian]) for durian in DURIANS},
                "king": result.king,
                "same_king": result.king == gold.king,
                "same_ranking": result.ranking == gold.ranking,
                "h2h": result.matches["h2h"],
                "writer": result.matches["writer"],
                "field": result.matches["field"],
            }
        )
    return pd.DataFrame(rows).set_index("rater")
