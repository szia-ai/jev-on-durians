"""Part 1 of the notebook: accuracy against the gold labels, time and money, per solution."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

ASPECTS = ("smell", "taste", "texture", "value", "quality", "aftermath")
SENTIMENTS = ("positive", "mixed", "negative", "neutral")
LANGUAGES = {"en": "English", "zh": "Chinese", "ms": "Malay"}
HEADLINE_PARTS = ("aspects_f1", "overall_sentiment_acc", "compares_f1", "variety_acc")
QUESTION_LABELS = {
    "aspects_f1": "aspects F1",
    "overall_sentiment_acc": "overall sentiment",
    "compares_f1": "compares",
    "preferred_acc": "preferred",
    "variety_acc": "variety",
    "rating_spearman": "rating ρ",
}
GOLD_QUESTIONS = (
    *(f"{aspect}_mentioned" for aspect in ASPECTS),
    "overall_sentiment",
    "compares",
    "preferred",
    "variety",
)
CONFIDENCE_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
CUTOFFS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SCORE_KEYS = (
    *QUESTION_LABELS,
    "overall_sentiment_macro_f1",
    "rating_mae",
    "n_rated",
    "headline",
)


def f1_binary(gold: pd.Series, predicted: pd.Series) -> float:
    """
    F1 of the positive class for two boolean series of equal length.

    Args:
        gold: The true labels.
        predicted: The predicted labels.

    Returns:
        2TP / (2TP + FP + FN), or NaN when there is neither a true nor a predicted positive.

    Example:
        >>> f1_binary(pd.Series([True, True, False]), pd.Series([True, False, False]))
        0.6666666666666666
    """
    gold_true = gold.to_numpy(dtype=bool)
    predicted_true = predicted.to_numpy(dtype=bool)
    true_positive = int((gold_true & predicted_true).sum())
    false_positive = int((~gold_true & predicted_true).sum())
    false_negative = int((gold_true & ~predicted_true).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else float("nan")


def accuracy(gold: pd.Series, predicted: pd.Series) -> float:
    """The share of equal labels, or NaN for empty inputs."""
    if len(gold) == 0:
        return float("nan")
    return float((gold.to_numpy() == predicted.to_numpy()).mean())


def macro_f1(gold: pd.Series, predicted: pd.Series, classes: tuple[str, ...]) -> float:
    """The mean of the per-class F1 scores, skipping classes that never occur."""
    scores = [f1_binary(gold == label, predicted == label) for label in classes]
    defined = [score for score in scores if not np.isnan(score)]
    return float(np.mean(defined)) if defined else float("nan")


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman's ρ (Pearson on average ranks, no scipy needed), or NaN with fewer than 3 points."""
    if len(a) < 3:
        return float("nan")
    return float(a.astype(float).rank().corr(b.astype(float).rank()))


def merge_gold(opinions: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """
    Join one solution's answered rows onto the opinions (inner join on the opinion id).

    The model's `variety` answer becomes `variety_answer`; `variety` stays the gold column.

    Args:
        opinions: The dataset as loaded by `benchmarking.load_opinions`.
        answers: The runner's rows for one solution.

    Returns:
        One row per answered opinion with both the gold and the answer columns.
    """
    answered = answers[answers["error"].isna()]
    return opinions.merge(answered, left_on="id", right_on="opinion_id", suffixes=("", "_answer"))


def score_answers(merged: pd.DataFrame) -> dict[str, float]:
    """
    Score one solution against the gold labels.

    Args:
        merged: The frame from `merge_gold`.

    Returns:
        aspects micro-F1 (threshold 0.5), overall-sentiment accuracy and macro-F1, compares F1,
        preferred accuracy (where gold compares), variety accuracy (where the quote names its
        durian), the rating's Spearman ρ and MAE against the writers' own scores, and the
        headline: the mean of aspects F1, overall-sentiment accuracy, compares F1 and variety
        accuracy over the parts that are defined.
    """
    if merged.empty:
        return {key: float("nan") for key in SCORE_KEYS}
    gold_mentioned = pd.concat([merged[f"gold_{a}"] == 1 for a in ASPECTS], ignore_index=True)
    predicted_mentioned = pd.concat(
        [merged[f"{a}_mentioned"] >= 0.5 for a in ASPECTS], ignore_index=True
    )
    comparing = merged[merged["gold_compares"] == 1]
    named = merged[merged["names_own_variety"]]
    rated = merged.dropna(subset=["real_score"])
    scores = {
        "aspects_f1": f1_binary(gold_mentioned, predicted_mentioned),
        "overall_sentiment_acc": accuracy(
            merged["gold_overall_sentiment"], merged["overall_sentiment"]
        ),
        "overall_sentiment_macro_f1": macro_f1(
            merged["gold_overall_sentiment"], merged["overall_sentiment"], SENTIMENTS
        ),
        "compares_f1": f1_binary(merged["gold_compares"] == 1, merged["compares"] >= 0.5),
        "preferred_acc": accuracy(comparing["gold_preferred"], comparing["preferred"]),
        "variety_acc": accuracy(named["variety"], named["variety_answer"]),
        "rating_spearman": spearman(rated["real_score"], rated["overall_rating"]),
        "rating_mae": float((rated["real_score"] - rated["overall_rating"]).abs().mean())
        if len(rated)
        else float("nan"),
        "n_rated": float(len(rated)),
    }
    parts = [scores[part] for part in HEADLINE_PARTS if not np.isnan(scores[part])]
    scores["headline"] = float(np.mean(parts)) if parts else float("nan")
    return scores


def accuracy_table(results: pd.DataFrame, opinions: pd.DataFrame) -> pd.DataFrame:
    """
    Score every solution in the runner's results, overall and per language.

    Args:
        results: The runner's frame (one row per opinion x solution).
        opinions: The dataset; pass a subset (e.g. `set == "main"`) to score only part of it.

    Returns:
        One row per solution with the scores from `score_answers`, `n_answered`, `n_errors`,
        and `headline_en` / `headline_zh` / `headline_ms`.
    """
    rows = []
    for solution, answers in results.groupby("solution", sort=False):
        merged = merge_gold(opinions, answers)
        row: dict[str, float | str] = {
            "solution": str(solution),
            "n_answered": float(len(merged)),
            "n_errors": float(answers["error"].notna().sum()),
            **score_answers(merged),
        }
        for language in LANGUAGES:
            subset = merged[merged["language"] == language]
            row[f"headline_{language}"] = score_answers(subset)["headline"]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["solution", *SCORE_KEYS]).set_index("solution")
    return pd.DataFrame(rows).set_index("solution")


def time_table(results: pd.DataFrame) -> pd.DataFrame:
    """Seconds per opinion per solution: p50, p95 and mean of the answered rows."""
    answered = results[results["error"].isna()]
    grouped = answered.groupby("solution", sort=False)["latency_s"]
    return pd.DataFrame(
        {"p50_s": grouped.median(), "p95_s": grouped.quantile(0.95), "mean_s": grouped.mean()}
    )


def cost_table(results: pd.DataFrame) -> pd.DataFrame:
    """Tokens and USD per solution: for the run and extrapolated to 1M opinions."""
    answered = results[results["error"].isna()]
    grouped = answered.groupby("solution", sort=False)
    table = grouped[["input_tokens", "output_tokens", "cost_usd"]].sum()
    table = table.rename(columns={"cost_usd": "usd_run"})
    table["usd_per_1m_opinions"] = table["usd_run"] / grouped.size() * 1_000_000
    return table


def summary_table(
    results: pd.DataFrame, opinions: pd.DataFrame, solutions: pd.DataFrame
) -> pd.DataFrame:
    """
    The part-1 table: one row per solution with accuracy, time and money.

    Args:
        results: The runner's frame.
        opinions: The dataset.
        solutions: One row per solution id with `label`, `family`, `model` and `local`.

    Returns:
        The solutions frame joined with the accuracy, time and cost tables, in registry order.
    """
    ran = solutions.loc[solutions.index.isin(results["solution"])]
    return (
        ran.join(accuracy_table(results, opinions))
        .join(time_table(results))
        .join(cost_table(results))
    )


def by_question_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The accuracy-by-question heatmap data: labelled solutions x labelled questions."""
    table = summary[list(QUESTION_LABELS)].rename(columns=QUESTION_LABELS)
    table.index = summary["label"]
    return table


def by_language_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The accuracy-by-language heatmap data: labelled solutions x English / Chinese / Malay."""
    columns = {f"headline_{code}": name for code, name in LANGUAGES.items()}
    table = summary[list(columns)].rename(columns=columns)
    table.index = summary["label"]
    return table


def agreement_matrix(results: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    Solution x solution Spearman agreement on one score question (no gold needed).

    Args:
        results: The runner's frame.
        question: A score column, e.g. "humour" or "taste_sentiment".

    Returns:
        A symmetric correlation matrix over the opinions every pair answered.
    """
    answered = results[results["error"].isna()]
    wide = answered.pivot(index="opinion_id", columns="solution", values=question).astype(float)
    return wide.rank().corr()


def rating_points(results: pd.DataFrame, opinions: pd.DataFrame) -> pd.DataFrame:
    """The writers' own scores next to each solution's overall rating (the rated opinions only)."""
    rated = opinions.dropna(subset=["real_score"])[["id", "real_score"]]
    answered = results[results["error"].isna()][["solution", "opinion_id", "overall_rating"]]
    return rated.merge(answered, left_on="id", right_on="opinion_id").drop(columns="id")


def _gold_and_answer(
    merged: pd.DataFrame, question: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """The rows a gold question is scored on, with their gold labels and the answers given."""
    if question.endswith("_mentioned"):
        gold = merged[f"gold_{question.removesuffix('_mentioned')}"] == 1
        return merged, gold, merged[question] >= 0.5
    if question == "compares":
        return merged, merged["gold_compares"] == 1, merged["compares"] >= 0.5
    if question == "overall_sentiment":
        return merged, merged["gold_overall_sentiment"], merged["overall_sentiment"]
    if question == "preferred":
        subset = merged[merged["gold_compares"] == 1]
        return subset, subset["gold_preferred"], subset["preferred"]
    subset = merged[merged["names_own_variety"]]
    return subset, subset["variety"], subset["variety_answer"]


def answer_confidences(
    results: pd.DataFrame, opinions: pd.DataFrame, labels: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """
    One row per gold-scored answer: who answered what, how confidently, and whether it was right.

    Confidence is the probability of the answer given (the larger of p and 1 - p for yes/no, the
    chosen option's probability for choices); LLMs carry 1.0 because they give no probability.

    Args:
        results: The runner's frame.
        opinions: The dataset.
        labels: Solution id -> display label; ids are kept when None.

    Returns:
        Columns `solution`, `opinion_id`, `question`, `confidence`, `correct`.
    """
    columns = ["solution", "opinion_id", "question", "confidence", "correct"]
    frames = []
    for solution, answers in results.groupby("solution", sort=False):
        merged = merge_gold(opinions, answers)
        name = str(solution) if labels is None else str(labels.get(str(solution), solution))
        for question in GOLD_QUESTIONS:
            subset, gold, answer = _gold_and_answer(merged, question)
            frames.append(
                pd.DataFrame(
                    {
                        "solution": name,
                        "opinion_id": subset["opinion_id"].to_numpy(),
                        "question": question,
                        "confidence": subset[f"{question}_confidence"].to_numpy(dtype=float),
                        "correct": gold.to_numpy() == answer.to_numpy(),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _bin_labels(edges: tuple[float, ...]) -> list[str]:
    labels = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if low <= 0:
            labels.append(f"< {high:.2f}")
        elif high >= 1:
            labels.append(f"≥ {low:.2f}")
        else:
            labels.append(f"{low:.2f} - {high:.2f}")
    return labels


def confidence_matrix(
    confidences: pd.DataFrame, edges: tuple[float, ...] = CONFIDENCE_EDGES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accuracy per confidence bin (rows, most confident first) x solution (columns), with counts.

    Args:
        confidences: From `answer_confidences`.
        edges: Bin edges from 0 to 1; the last bin includes 1.0.

    Returns:
        The accuracy matrix and the matching matrix of answer counts.
    """
    labels = _bin_labels(edges)
    solutions = list(dict.fromkeys(confidences["solution"]))
    if not solutions:
        empty = pd.DataFrame(index=pd.Index(labels[::-1], name="confidence"))
        return empty, empty.copy()
    bins = [*edges[:-1], max(edges[-1], float(np.nextafter(1.0, 2)))]
    binned = pd.cut(
        confidences["confidence"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    frame = confidences.assign(bin=binned)
    accuracy = frame.pivot_table(
        index="bin", columns="solution", values="correct", aggfunc="mean", observed=False
    )
    counts = frame.pivot_table(
        index="bin", columns="solution", values="correct", aggfunc="count", observed=False
    )
    accuracy = accuracy.reindex(index=labels[::-1], columns=solutions).astype(float)
    counts = counts.reindex(index=labels[::-1], columns=solutions).fillna(0).astype(int)
    accuracy.index.name = counts.index.name = "confidence"
    return accuracy, counts


def cutoff_table(
    confidences: pd.DataFrame, cutoffs: tuple[float, ...] = CUTOFFS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    What you keep when you accept answers at or above a cutoff and send the rest to a person.

    Args:
        confidences: From `answer_confidences`.
        cutoffs: The thresholds to try; 0 means "accept everything".

    Returns:
        The accuracy among the accepted answers (rows = cutoffs, columns = solutions) and the
        share of answers accepted.
    """
    solutions = list(dict.fromkeys(confidences["solution"]))
    if not solutions:
        empty = pd.DataFrame(index=pd.Index([], name="cutoff"))
        return empty, empty.copy()
    totals = confidences.groupby("solution").size()
    accuracy_rows: dict[str, pd.Series] = {}
    share_rows: dict[str, pd.Series] = {}
    for cutoff in cutoffs:
        accepted = confidences[confidences["confidence"] >= cutoff]
        label = "all answers" if cutoff <= 0 else f"≥ {cutoff:.2f}"
        accuracy_rows[label] = accepted.groupby("solution")["correct"].mean()
        share_rows[label] = accepted.groupby("solution").size() / totals
    accuracy = pd.DataFrame(accuracy_rows).T.reindex(columns=solutions).astype(float)
    share = pd.DataFrame(share_rows).T.reindex(columns=solutions).fillna(0.0).astype(float)
    accuracy.index.name = share.index.name = "cutoff"
    return accuracy, share


def annotate(
    values: pd.DataFrame, detail: pd.DataFrame, detail_format: str = "{:d}"
) -> pd.DataFrame:
    """
    Render two matrices as one for display: "0.95 (412)" or "0.95 (82%)", "-" without a value.

    Args:
        values: The matrix shown first, formatted with two decimals.
        detail: Same shape; shown in brackets.
        detail_format: Format for the bracketed value, e.g. "{:d}" for counts or "{:.0%}".

    Returns:
        A frame of strings with the same index and columns.
    """

    def cell(value: float, extra: float) -> str:
        if pd.isna(value):
            return "-"
        return f"{value:.2f} ({detail_format.format(extra)})"

    cells = {
        column: [cell(v, d) for v, d in zip(values[column], detail[column], strict=True)]
        for column in values.columns
    }
    return pd.DataFrame(cells, index=values.index)
