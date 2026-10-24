"""`results/summary.md` and the timestamped results block of `README.md`."""

import dataclasses
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"
TIMEZONE = ZoneInfo("Asia/Singapore")
PART1_COLUMNS = {
    "label": "solution",
    "family": "family",
    "headline": "headline",
    "answer_acc": "answer acc",
    "aspects_f1": "aspects F1",
    "overall_sentiment_acc": "sentiment acc",
    "compares_f1": "compares F1",
    "preferred_acc": "preferred acc",
    "variety_acc": "variety acc",
    "p50_s": "p50 s",
    "p95_s": "p95 s",
    "usd_run": "USD run",
    "usd_per_1m_opinions": "USD / 1M opinions",
}
LANGUAGE_COLUMNS = {
    "label": "solution",
    "headline_en": "English",
    "headline_zh": "Chinese",
    "headline_ms": "Malay",
}
ELO_COLUMNS = {
    "musang_king": "Musang King",
    "black_thorn": "Black Thorn",
    "red_prawn": "Red Prawn",
    "king": "king",
    "same_king": "same king",
    "same_ranking": "same ranking",
}
RATINGS_CAPTION = (
    "Mean overall rating per durian, 0..1 (gold = the writer's own score where given, else "
    "positive 1 / mixed or neutral 0.5 / negative 0):"
)
CONFIDENCE_CAPTION = (
    "Accuracy by confidence bin for the decision models (confidence = the probability of the "
    "answer given; the number of gold-scored answers in brackets; LLMs give no probability and "
    "are left out):"
)
CUTOFF_CAPTION = (
    "Accept only answers at or above a cutoff and send the rest for manual review: answer "
    "accuracy of the accepted answers (the 'answer acc' of part 1 when everything is accepted), "
    "with the share of answers accepted in brackets:"
)
FAMILY_NAMES = {
    "decision_cloud": "decision, cloud",
    "decision_local": "decision, local",
    "llm_cloud": "LLM, cloud",
    "llm_local": "LLM, local",
    "reference": "reference LLM",
}


@dataclasses.dataclass
class RunInfo:
    """What the results describe: how many opinions, when, and which solutions were skipped."""

    n_opinions: int
    mode: str
    timestamp: str
    skipped: dict[str, str]
    machine: str = ""


def now_singapore() -> str:
    """The current time in Singapore as `YYYY-MM-DD HH:MM SGT`."""
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")


def _cell(value: object, float_format: str) -> str:
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if isinstance(value, float | np.floating):
        return "-" if np.isnan(value) else float_format.format(value)
    return str(value)


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.2f}") -> str:
    """
    Render a frame as a Markdown table (the index becomes the first column).

    Args:
        frame: The table; column names become the header.
        float_format: Format for float cells; NaN renders as "-".

    Returns:
        The table as Markdown text.

    Example:
        >>> markdown_table(pd.DataFrame({"a": [1.5]}, index=pd.Index(["x"], name="row")))
        '| row | a |\\n|---|---|\\n| x | 1.50 |'
    """
    header = [str(frame.index.name or ""), *map(str, frame.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in frame.iterrows():
        cells = [str(index), *(_cell(value, float_format) for value in row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pick(summary: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    table = summary[list(columns)].rename(columns=columns)
    if "family" in table.columns:
        table["family"] = table["family"].map(FAMILY_NAMES)
    table.index = pd.Index(range(1, len(table) + 1), name="#")
    return table


def part1_markdown(summary: pd.DataFrame) -> str:
    """The part-1 table (accuracy, time, money) as Markdown, best headline first."""
    ordered = summary.sort_values("headline", ascending=False, na_position="last")
    money = _pick(ordered, PART1_COLUMNS)
    money["USD run"] = money["USD run"].map(
        lambda value: f"{value:.4f}" if not np.isnan(value) else "-"
    )
    return markdown_table(money)


def language_markdown(summary: pd.DataFrame) -> str:
    """The headline per language as Markdown."""
    ordered = summary.sort_values("headline", ascending=False, na_position="last")
    return markdown_table(_pick(ordered, LANGUAGE_COLUMNS))


def elo_markdown(elo_table: pd.DataFrame) -> str:
    """The Durian Elo table (gold first, then every solution) as Markdown."""
    table = elo_table[list(ELO_COLUMNS)].rename(columns=ELO_COLUMNS)
    return markdown_table(table, float_format="{:.0f}")


def ratings_markdown(ratings: pd.DataFrame) -> str:
    """The mean overall rating per durian (gold first, then every solution) as Markdown."""
    table = ratings.rename(columns={durian: ELO_COLUMNS[durian] for durian in ratings.columns})
    return markdown_table(table)


def observations(summary: pd.DataFrame, elo_table: pd.DataFrame) -> list[str]:
    """
    Three to five plain-language observations drawn from the numbers.

    Args:
        summary: The part-1 table with `label`, `family`, `local`, `headline`,
            `usd_per_1m_opinions`, `p50_s` and the per-language headlines.
        elo_table: From `elo.elo_table`.

    Returns:
        Sentences for the README; fewer when a comparison has nothing to compare.
    """
    notes: list[str] = []
    scored = summary.dropna(subset=["headline"])
    if scored.empty:
        return ["No solution produced a scoreable answer in this run."]
    best = scored.sort_values("headline", ascending=False).iloc[0]
    notes.append(
        f"{best['label']} reads durians best in this run: headline accuracy {best['headline']:.2f}."
    )
    decision = scored[scored["family"].str.startswith("decision")]
    llm = scored[scored["family"].str.startswith("llm")]
    if not decision.empty and not llm.empty:
        best_decision = decision.sort_values("headline", ascending=False).iloc[0]
        best_llm = llm.sort_values("headline", ascending=False).iloc[0]
        gap = best_llm["headline"] - best_decision["headline"]
        notes.append(
            f"The best decision model ({best_decision['label']}, {best_decision['headline']:.2f}) "
            f"is {abs(gap):.2f} {'behind' if gap > 0 else 'ahead of'} the best LLM "
            f"({best_llm['label']}, {best_llm['headline']:.2f})."
        )
    paid = scored[~scored["local"] & (scored["usd_per_1m_opinions"] > 0)]
    if len(paid) >= 2:
        cheapest = paid.sort_values("usd_per_1m_opinions").iloc[0]
        priciest = paid.sort_values("usd_per_1m_opinions").iloc[-1]
        ratio = priciest["usd_per_1m_opinions"] / cheapest["usd_per_1m_opinions"]
        notes.append(
            f"A million opinions would cost USD {cheapest['usd_per_1m_opinions']:,.0f} with "
            f"{cheapest['label']} and USD {priciest['usd_per_1m_opinions']:,.0f} with "
            f"{priciest['label']} - {ratio:,.0f}x more, at list prices."
        )
    timed = scored.dropna(subset=["p50_s"])
    if len(timed) >= 2:
        fastest = timed.sort_values("p50_s").iloc[0]
        slowest = timed.sort_values("p50_s").iloc[-1]
        notes.append(
            f"{fastest['label']} answers an opinion in {fastest['p50_s']:.2f} s (p50); "
            f"{slowest['label']} takes {slowest['p50_s']:.2f} s."
        )
    languages = {
        "English": best["headline_en"],
        "Chinese": best["headline_zh"],
        "Malay": best["headline_ms"],
    }
    defined = {name: value for name, value in languages.items() if not np.isnan(value)}
    if len(defined) >= 2:
        weakest = min(defined.items(), key=lambda item: item[1])[0]
        strongest = max(defined.items(), key=lambda item: item[1])[0]
        notes.append(
            f"For {best['label']}, {weakest} is the hardest language ({defined[weakest]:.2f}) and "
            f"{strongest} the easiest ({defined[strongest]:.2f})."
        )
    solutions = elo_table.drop(index="gold", errors="ignore")
    if not solutions.empty:
        agree = int(solutions["same_king"].sum())
        king = str(elo_table.loc["gold", "king"]).replace("_", " ").title()
        notes.append(
            f"The people's Durian Elo crowns {king}; {agree} of {len(solutions)} solutions "
            "crown the same king."
        )
    return notes[:5]


def summary_markdown(
    summary: pd.DataFrame,
    elo_table: pd.DataFrame,
    info: RunInfo,
    main_only: pd.DataFrame | None = None,
    ratings: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    cutoffs: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> str:
    """
    The full `results/summary.md` text.

    Args:
        summary: The part-1 table.
        elo_table: From `elo.elo_table`.
        info: Run size, mode, timestamp and skipped solutions.
        main_only: The part-1 table scored on the 70 main opinions only, if computed.
        ratings: From `elo.ratings_by_durian`, shown under the Elo table when given.
        confidence: The annotated accuracy-by-confidence-bin matrix, when given.
        cutoffs: The annotated cutoff table, when given.
        notes: Hand-written observations; the generated ones are used when None.

    Returns:
        Markdown with the part-1 tables, the confidence tables, the Elo table, the ratings and
        the skipped solutions.
    """
    parts = [
        "# Results summary",
        "",
        f"{info.n_opinions} opinions · mode {info.mode} · {info.timestamp}"
        + (f" · local models on {info.machine}" if info.machine else ""),
        "",
        "## Part 1: accuracy, time, money",
        "",
        part1_markdown(summary),
        "",
        "Headline per language:",
        "",
        language_markdown(summary),
    ]
    if main_only is not None and not main_only.empty:
        parts += ["", "Check on the 70 main-set opinions only:", "", part1_markdown(main_only)]
    has_confidence = confidence is not None and not confidence.empty
    has_cutoffs = cutoffs is not None and not cutoffs.empty
    if has_confidence or has_cutoffs:
        parts += ["", "## Confidence and a manual-review cutoff"]
    if has_confidence:
        parts += ["", CONFIDENCE_CAPTION, "", markdown_table(confidence)]
    if has_cutoffs:
        parts += ["", CUTOFF_CAPTION, "", markdown_table(cutoffs)]
    parts += ["", "## Part 2: the Durian Elo", "", elo_markdown(elo_table), ""]
    if ratings is not None:
        parts += [RATINGS_CAPTION, "", ratings_markdown(ratings), ""]
    parts += ["## Observations", "", *(f"- {note}" for note in _notes(notes, summary, elo_table))]
    if info.skipped:
        parts += ["", "## Skipped solutions", ""]
        parts += [f"- `{solution}`: {reason}" for solution, reason in info.skipped.items()]
    return "\n".join(parts) + "\n"


def _notes(notes: list[str] | None, summary: pd.DataFrame, elo_table: pd.DataFrame) -> list[str]:
    return notes if notes is not None else observations(summary, elo_table)


def read_observations(path: Path) -> list[str]:
    """
    Read hand-written observations: the `- ` bullet lines of a Markdown file.

    Args:
        path: The file, e.g. `results/observations.md`; other lines are ignored.

    Returns:
        The bullet texts, in file order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line[2:].strip() for line in lines if line.startswith("- ") and line[2:].strip()]


def results_block(
    summary: pd.DataFrame,
    elo_table: pd.DataFrame,
    info: RunInfo,
    chart_paths: list[Path],
    confidence: pd.DataFrame | None = None,
    cutoffs: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> str:
    """
    The Markdown that goes between the RESULTS markers of `README.md`.

    Part 2 (the Elo, the ratings and the Elo chart) stays in `results/summary.md`; the README gets
    part 1, the confidence tables, the other chart links and the observations.

    Args:
        summary: The part-1 table.
        elo_table: From `elo.elo_table` (only for the generated observations).
        info: Run size, mode, timestamp, machine and skipped solutions.
        chart_paths: The chart PNGs, relative to the repo root or absolute.
        confidence: The annotated accuracy-by-confidence-bin matrix, shown after part 1.
        cutoffs: The annotated cutoff table, shown with it.
        notes: Hand-written observations; the generated ones are used when None.

    Returns:
        Markdown: timestamp line, the part-1 table, the confidence tables, chart links,
        observations.
    """
    links = ", ".join(
        f"[{path.stem[3:].replace('_', ' ')}]({Path('assets/charts') / path.name})"
        for path in chart_paths
        if "elo" not in path.name  # part 2 stays out of the README
    )
    parts = [
        f"_Measured on {info.n_opinions} opinions, {info.timestamp} (mode: {info.mode})"
        + (f"; local models ran on {info.machine}._" if info.machine else "._"),
        "",
        "**Part 1 - accuracy, time, money** (headline = mean of aspects F1, sentiment accuracy, "
        "compares F1 and variety accuracy; answer acc = the share of all gold-scored answers that "
        "are right, the measure the confidence tables filter; prices from `config/prices.yaml`; "
        "local models cost nothing per call):",
        "",
        part1_markdown(summary),
        "",
        *(
            [CONFIDENCE_CAPTION, "", markdown_table(confidence), ""]
            if confidence is not None and not confidence.empty
            else []
        ),
        *(
            [CUTOFF_CAPTION, "", markdown_table(cutoffs), ""]
            if cutoffs is not None and not cutoffs.empty
            else []
        ),
        f"Charts: {links}.",
        "",
        "**Observations**",
        "",
        *(f"- {note}" for note in _notes(notes, summary, elo_table)),
    ]
    if info.skipped:
        skipped = "; ".join(f"`{solution}` ({reason})" for solution, reason in info.skipped.items())
        parts += ["", f"Skipped in this run: {skipped}."]
    return "\n".join(parts)


def replace_results_block(readme_text: str, block: str) -> str:
    """
    Replace the text between the RESULTS markers, keeping the markers and the rest of the file.

    Args:
        readme_text: The current README text.
        block: The new Markdown for the block.

    Returns:
        The updated README text.

    Raises:
        ValueError: When a marker is missing.
    """
    start = readme_text.find(RESULTS_START)
    end = readme_text.find(RESULTS_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"README needs the {RESULTS_START} and {RESULTS_END} markers")
    return (
        readme_text[: start + len(RESULTS_START)] + "\n" + block.strip() + "\n" + readme_text[end:]
    )


def write_results(
    readme_path: Path,
    summary_path: Path,
    summary: pd.DataFrame,
    elo_table: pd.DataFrame,
    info: RunInfo,
    chart_paths: list[Path],
    main_only: pd.DataFrame | None = None,
    ratings: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    cutoffs: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> None:
    """
    Write `results/summary.md` and the results block of `README.md`.

    Args:
        readme_path: The README to update in place.
        summary_path: Where to write the summary Markdown.
        summary: The part-1 table.
        elo_table: From `elo.elo_table`.
        info: Run size, mode, timestamp, machine and skipped solutions.
        chart_paths: The chart PNGs.
        main_only: The part-1 table on the main set only, if computed.
        ratings: From `elo.ratings_by_durian`, if computed.
        confidence: The annotated accuracy-by-confidence-bin matrix, if computed.
        cutoffs: The annotated cutoff table, if computed.
        notes: Hand-written observations; the generated ones are used when None.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        summary_markdown(summary, elo_table, info, main_only, ratings, confidence, cutoffs, notes),
        encoding="utf-8",
    )
    block = results_block(
        summary, elo_table, info, chart_paths, confidence=confidence, cutoffs=cutoffs, notes=notes
    )
    readme_path.write_text(
        replace_results_block(readme_path.read_text(encoding="utf-8"), block), encoding="utf-8"
    )
