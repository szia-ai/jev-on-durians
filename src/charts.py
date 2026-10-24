"""The seven charts of the notebook in the szia.ai style: Plotly figures, saved as 1600 px PNGs."""

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PAPER = "#F6F6F3"
INK = "#101010"
STONE = "#DCDBD3"
GRAPHITE = "#5E5E57"
SKY = "#B9D2F6"
CORAL = "#F6B1A4"
AQUA = "#ADE1DB"
SUN = "#F5F899"
DURIAN_COLORS = {"musang_king": "#FF7A3C", "black_thorn": GRAPHITE, "red_prawn": "#FF6FB0"}
DURIAN_NAMES = {
    "musang_king": "Musang King",
    "black_thorn": "Black Thorn",
    "red_prawn": "Red Prawn",
}
FAMILY_FILLS = {
    "decision_cloud": SKY,
    "decision_local": AQUA,
    "llm_cloud": CORAL,
    "llm_local": AQUA,
    "reference": CORAL,
}
FAMILY_LEGEND = {SKY: "decision models (cloud)", CORAL: "LLMs (cloud)", AQUA: "local models"}
SEQUENTIAL = [[0.0, PAPER], [0.5, SKY], [1.0, "#6F98DD"]]
FONT = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
WIDTH = 1600
HEIGHT = 1000
MARKER_SIZE = 24
MARKER = dict(size=MARKER_SIZE, line=dict(color=INK, width=1.5))
CHART_FILES = {
    "cost_vs_accuracy": "01_cost_vs_accuracy.png",
    "time_per_opinion": "02_time_per_opinion.png",
    "accuracy_by_question": "03_accuracy_by_question.png",
    "accuracy_by_language": "04_accuracy_by_language.png",
    "rating_vs_real": "05_rating_vs_real.png",
    "durian_elo": "06_durian_elo.png",
    "time_vs_accuracy": "07_time_vs_accuracy.png",
}


def _figure(title: str, date: str, x_title: str = "", y_title: str = "") -> go.Figure:
    axis = dict(gridcolor=STONE, zeroline=False, linecolor=INK, ticks="outside", tickcolor=INK)
    fig = go.Figure()
    fig.update_layout(
        width=WIDTH,
        height=HEIGHT,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=FONT, color=INK, size=24),
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", font=dict(size=34)),
        margin=dict(l=100, r=60, t=110, b=130),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=22)),
        xaxis=dict(title=dict(text=x_title), **axis),
        yaxis=dict(title=dict(text=y_title), **axis),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK, size=18)),
    )
    fig.add_annotation(
        text=f"Jev on Durians · szia.ai · {date}",
        xref="paper",
        yref="paper",
        x=1,
        y=-0.12,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font=dict(size=17, color=GRAPHITE),
    )
    return fig


def _save(fig: go.Figure, path: Path) -> Path:
    """Write the figure as a PNG of WIDTH x HEIGHT pixels (kaleido, which needs a Chrome)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=WIDTH, height=HEIGHT, scale=1)
    return path


def _empty_note(fig: go.Figure, text: str) -> None:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)


def _family_traces(fig: go.Figure, families: pd.Series) -> None:
    """Invisible points that give the legend one entry per family colour."""
    for fill in dict.fromkeys(FAMILY_FILLS[family] for family in families):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=22, color=fill, symbol="square", line=dict(color=INK, width=1.5)),
                name=FAMILY_LEGEND[fill],
            )
        )


def _place_labels(
    fig: go.Figure,
    x: pd.Series,
    y: pd.Series,
    labels: pd.Series,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_log: bool = False,
    font_size: int = 21,
) -> None:
    """
    Label points without overlaps: each label tries positions around its point at growing
    distances, skipping any that would cover another label or marker; a label that had to move
    far gets a thin leader line.

    Args:
        fig: The figure; its margins and the ranges below give the pixel geometry.
        x: Point x values (data units).
        y: Point y values (data units).
        labels: One label per point.
        x_range: The x axis range as set on the figure (log10 units when `x_log`).
        y_range: The y axis range as set on the figure.
        x_log: Whether the x axis is logarithmic.
        font_size: Label font size in pixels.
    """
    margin = fig.layout.margin
    left, top = float(margin.l), float(margin.t)
    plot_width = WIDTH - left - float(margin.r)
    plot_height = HEIGHT - top - float(margin.b)
    x_values = np.log10(x.to_numpy(dtype=float)) if x_log else x.to_numpy(dtype=float)
    px = left + (x_values - x_range[0]) / (x_range[1] - x_range[0]) * plot_width
    py = (
        top
        + plot_height
        - (y.to_numpy(dtype=float) - y_range[0]) / (y_range[1] - y_range[0]) * plot_height
    )
    radius = MARKER_SIZE / 2 + 3
    clearance = radius + 22  # a label must not crowd someone else's point
    markers = [
        (cx - clearance, cy - clearance, cx + clearance, cy + clearance)
        for cx, cy in zip(px, py, strict=True)
    ]
    taken: list[tuple[float, float, float, float]] = []
    char_width, line_height = font_size * 0.56, font_size * 1.3
    directions = [(1, -1), (1, 1), (-1, -1), (-1, 1), (1, 0), (-1, 0), (0, -1), (0, 1)]

    def overlaps(box: tuple[float, float, float, float], own: int) -> bool:
        others = [marker for index, marker in enumerate(markers) if index != own]
        return any(
            box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]
            for other in [*taken, *others]
        )

    points = zip(px, py, x, y, labels, strict=True)
    for own, (cx, cy, x_data, y_data, label) in enumerate(points):
        width, height = len(str(label)) * char_width, line_height
        choice = None
        for distance in (16, 34, 60, 95, 140, 200):
            for sx, sy in directions:
                ax, ay = cx + sx * distance, cy + sy * distance
                x0 = ax if sx > 0 else ax - width if sx < 0 else ax - width / 2
                y0 = ay - height if sy < 0 else ay if sy > 0 else ay - height / 2
                box = (x0, y0, x0 + width, y0 + height)
                inside = (
                    left <= x0
                    and box[2] <= left + plot_width
                    and top <= y0
                    and box[3] <= top + plot_height
                )
                if inside and not overlaps(box, own):
                    choice = (sx * distance, sy * distance, sx, sy, box, distance)
                    break
            if choice:
                break
        if choice is None:
            choice = (16, -16, 1, -1, (cx + 16, cy - 16 - height, cx + 16 + width, cy - 16), 16)
        dx, dy, sx, sy, box, distance = choice
        taken.append(box)
        x_pos = float(np.log10(x_data)) if x_log else x_data  # log axes take log10 positions
        fig.add_annotation(
            x=x_pos,
            y=y_data,
            text=str(label),
            xshift=dx,
            yshift=-dy,
            xanchor="left" if sx > 0 else "right" if sx < 0 else "center",
            yanchor="bottom" if sy < 0 else "top" if sy > 0 else "middle",
            showarrow=False,
            font=dict(size=font_size),
        )
        if distance > 20:  # a leader line from the marker's edge to the nearest point of the label
            bx, by = min(max(cx, box[0]), box[2]), min(max(cy, box[1]), box[3])
            vx, vy = bx - cx, by - cy
            norm = float(np.hypot(vx, vy)) or 1.0
            sx0, sy0 = cx + vx / norm * radius, cy + vy / norm * radius

            def to_x(value: float) -> float:
                return x_range[0] + (value - left) / plot_width * (x_range[1] - x_range[0])

            def to_y(value: float) -> float:
                return y_range[0] + (top + plot_height - value) / plot_height * (
                    y_range[1] - y_range[0]
                )

            fig.add_shape(  # unlike annotations, shapes on a log axis take raw values
                type="line",
                x0=10 ** to_x(sx0) if x_log else to_x(sx0),
                y0=to_y(sy0),
                x1=10 ** to_x(bx) if x_log else to_x(bx),
                y1=to_y(by),
                line=dict(color=GRAPHITE, width=1.2),
                layer="above",
            )


def _log_ticks(low: float, high: float) -> list[float]:
    """Ticks at 1, 2 and 5 times a power of ten inside [low, high]."""
    ticks = []
    for exponent in range(int(np.floor(np.log10(low))) - 1, int(np.ceil(np.log10(high))) + 2):
        for mantissa in (1, 2, 5):
            value = mantissa * 10.0**exponent
            if low <= value <= high:
                ticks.append(value)
    return ticks


def png_size(path: Path) -> tuple[int, int]:
    """
    Width and height of a PNG from its header, without an image library.

    Args:
        path: The PNG file.

    Returns:
        (width, height) in pixels.

    Raises:
        ValueError: When the file is not a PNG.
    """
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def cost_vs_accuracy(summary: pd.DataFrame, path: Path, date: str) -> go.Figure:
    """
    Chart 1: USD per 1M opinions (log x, local models at the left edge as "free") vs headline.

    Args:
        summary: The part-1 table (`label`, `family`, `local`, `headline`, `usd_per_1m_opinions`).
        path: Where to save the PNG.
        date: The date for the source line.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        "Cost vs accuracy: what a million durian opinions would cost",
        date,
        "USD per 1M opinions (list prices; local models run on your own hardware)",
        "headline accuracy",
    )
    paid = summary[~summary["local"] & (summary["usd_per_1m_opinions"] > 0)]
    lowest_paid = float(paid["usd_per_1m_opinions"].min()) if not paid.empty else 100.0
    free_x = 10 ** (np.floor(np.log10(lowest_paid)) - 1)
    x = summary["usd_per_1m_opinions"].where(~summary["local"], free_x).astype(float)
    top = max(float(x.max()) * 6, free_x * 100) if len(x) else free_x * 100
    for fill in dict.fromkeys(FAMILY_FILLS[family] for family in summary["family"]):
        mask = summary["family"].map(FAMILY_FILLS) == fill
        rows = summary[mask]
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=rows["headline"],
                mode="markers",
                text=rows["label"],
                marker=dict(color=fill, **MARKER),
                name=FAMILY_LEGEND[fill],
                customdata=rows["usd_per_1m_opinions"],
                hovertemplate="%{text}<br>USD %{customdata:,.0f} per 1M opinions<br>"
                "headline %{y:.2f}<extra></extra>",
            )
        )
    ticks = [free_x, *[10.0**k for k in range(int(np.log10(free_x)) + 1, int(np.log10(top)) + 1)]]
    x_range = (float(np.log10(free_x / 2)), float(np.log10(top)))
    fig.update_xaxes(
        type="log",
        range=list(x_range),
        tickvals=ticks,
        ticktext=["free", *[f"{tick:g}" for tick in ticks[1:]]],
    )
    fig.update_yaxes(range=[0, 1.02])
    _place_labels(fig, x, summary["headline"], summary["label"], x_range, (0, 1.02), x_log=True)
    fig.update_layout(legend=dict(x=0.98, y=0.04, xanchor="right", yanchor="bottom"))
    _save(fig, path)
    return fig


def time_per_opinion(summary: pd.DataFrame, path: Path, date: str) -> go.Figure:
    """
    Chart 2: p50 seconds per opinion as horizontal bars, a whisker to p95, fastest on top.

    Args:
        summary: The part-1 table (`label`, `family`, `p50_s`, `p95_s`).
        path: Where to save the PNG.
        date: The date for the source line.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        "Seconds per opinion (18 questions in one request)",
        date,
        "seconds per opinion: bar = p50, whisker to p95 (cloud calls include the round trip)",
    )
    ordered = summary.dropna(subset=["p50_s"]).sort_values("p50_s", ascending=False)
    if ordered.empty:
        _empty_note(fig, "no timed solution in this run")
        _save(fig, path)
        return fig
    fills = [FAMILY_FILLS[family] for family in ordered["family"]]
    fig.add_trace(
        go.Bar(
            y=ordered["label"],
            x=ordered["p50_s"],
            orientation="h",
            marker=dict(color=fills, line=dict(color=INK, width=1.2)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ordered["p95_s"] - ordered["p50_s"],
                arrayminus=[0] * len(ordered),
                color=INK,
                thickness=2.5,
                width=10,
            ),
            showlegend=False,
            customdata=ordered["p95_s"],
            hovertemplate="%{y}<br>p50 %{x:.2f} s · p95 %{customdata:.2f} s<extra></extra>",
        )
    )
    for label, p50, p95 in zip(ordered["label"], ordered["p50_s"], ordered["p95_s"], strict=True):
        fig.add_annotation(
            x=p95, y=label, text=f"{p50:.2f} s", xanchor="left", xshift=10, showarrow=False
        )
    _family_traces(fig, ordered["family"])
    fig.update_xaxes(range=[0, float(ordered["p95_s"].max()) * 1.35])
    fig.update_yaxes(showgrid=False, ticks="")
    fig.update_layout(
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"), margin=dict(l=40)
    )
    _save(fig, path)
    return fig


def _heatmap(fig: go.Figure, table: pd.DataFrame) -> None:
    values = table.to_numpy(dtype=float)
    if values.size == 0:
        _empty_note(fig, "nothing to show")
        return
    text = [["-" if np.isnan(value) else f"{value:.2f}" for value in row] for row in values]
    fig.add_trace(
        go.Heatmap(
            z=values,
            x=[str(column) for column in table.columns],
            y=[str(index) for index in table.index],
            colorscale=SEQUENTIAL,
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=22, color=INK),
            showscale=False,
            xgap=3,
            ygap=3,
            hoverongaps=False,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", showline=False)
    fig.update_xaxes(
        showgrid=False,
        ticks="",
        showline=False,
        tickangle=-25 if len(table.columns) > 4 else 0,
        side="bottom",
    )


def accuracy_by_question(by_question: pd.DataFrame, path: Path, date: str) -> go.Figure:
    """
    Chart 3: heatmap of solutions x questions (aspects F1, sentiment, compares, preferred,
    variety, rating ρ).

    Args:
        by_question: Labelled solutions as rows, labelled questions as columns, values 0..1.
        path: Where to save the PNG.
        date: The date for the source line.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        "Accuracy by question",
        date,
        "F1 or accuracy against the gold labels; rating ρ against the writers' own scores",
    )
    _heatmap(fig, by_question)
    fig.update_layout(margin=dict(l=40))
    _save(fig, path)
    return fig


def accuracy_by_language(by_language: pd.DataFrame, path: Path, date: str) -> go.Figure:
    """
    Chart 4: heatmap of solutions x languages (headline accuracy).

    Args:
        by_language: Labelled solutions as rows, English / Chinese / Malay as columns.
        path: Where to save the PNG.
        date: The date for the source line.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        "Headline accuracy by language", date, "headline accuracy per language of the opinion"
    )
    _heatmap(fig, by_language)
    fig.update_layout(margin=dict(l=40))
    _save(fig, path)
    return fig


def rating_vs_real(points: pd.DataFrame, path: Path, date: str) -> go.Figure:
    """
    Chart 5: the writers' own scores (x) vs each solution's overall rating (y), with the diagonal.

    Args:
        points: One row per rated opinion x solution: `label`, `family`, `opinion_id`,
            `real_score`, `overall_rating`.
        path: Where to save the PNG.
        date: The date for the source line.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        "Overall rating vs the writers' own scores (8 rated opinions)",
        date,
        "the writer's own score, normalised to 0..1",
        "the solution's overall rating, 0..1",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=INK, dash="dash", width=1.5))
    symbols = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star"]
    for index, (label, group) in enumerate(points.groupby("label", sort=False)):
        fig.add_trace(
            go.Scatter(
                x=group["real_score"],
                y=group["overall_rating"],
                mode="markers",
                marker=dict(
                    color=FAMILY_FILLS[str(group["family"].iloc[0])],
                    symbol=symbols[index % len(symbols)],
                    **MARKER,
                ),
                name=str(label),
                customdata=group["opinion_id"] if "opinion_id" in group else group.index,
                hovertemplate="%{customdata}<br>writer %{x:.2f} · rating %{y:.2f}<extra></extra>",
            )
        )
    if points.empty:
        _empty_note(fig, "no rated opinion in this run")
    fig.update_xaxes(range=[0, 1.02])
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"))
    _save(fig, path)
    return fig


def durian_elo(
    gold_history: pd.DataFrame,
    final_by_solution: dict[str, dict[str, float]],
    path: Path,
    date: str,
) -> go.Figure:
    """
    Chart 6: the gold Elo of the three durians over the matches, and every solution's final Elo as
    dots on the right.

    Args:
        gold_history: The gold `EloResult.history` (one line per match with the three ratings).
        final_by_solution: Solution label -> final rating per durian.
        path: Where to save the PNG.
        date: The date for the source line.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        "The Durian Elo: the people's ranking, and what each solution saw",
        date,
        "gold matches in date order (2009 → 2026), then each solution's final Elo",
        "Elo rating (all start at 1500)",
    )
    n_matches = len(gold_history)
    x = np.arange(1, n_matches + 1)
    ratings = [1500.0]
    for durian, color in DURIAN_COLORS.items():
        series = gold_history[durian] if n_matches else pd.Series(dtype=float)
        ratings += [float(value) for value in series]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series,
                mode="lines",
                line=dict(color=color, width=4),
                name=DURIAN_NAMES[durian],
                hovertemplate=f"{DURIAN_NAMES[durian]}<br>match %{{x}}: %{{y:.0f}}<extra></extra>",
            )
        )
        if n_matches:
            fig.add_annotation(
                x=n_matches,
                y=float(series.iloc[-1]),
                text=f"{series.iloc[-1]:.0f}",
                xanchor="left",
                xshift=8,
                showarrow=False,
                font=dict(color=color, size=20),
            )
    gap = max(n_matches * 0.08, 1)
    for index, (label, rating) in enumerate(final_by_solution.items()):
        position = n_matches + gap * (index + 1.5)
        ratings += [float(value) for value in rating.values()]
        fig.add_trace(
            go.Scatter(
                x=[position] * len(DURIAN_COLORS),
                y=[rating[durian] for durian in DURIAN_COLORS],
                mode="markers",
                marker=dict(color=list(DURIAN_COLORS.values()), **MARKER),
                showlegend=False,
                text=[f"{label}<br>{DURIAN_NAMES[durian]}" for durian in DURIAN_COLORS],
                hovertemplate="%{text}: %{y:.0f}<extra></extra>",
            )
        )
    low, high = min(ratings), max(ratings)
    span = max(high - low, 50.0)
    if final_by_solution:
        for index, label in enumerate(final_by_solution):
            fig.add_annotation(
                x=n_matches + gap * (index + 1.5),
                y=low - span * 0.43,
                text=label.replace(" (", "<br>("),
                textangle=-90,
                yanchor="bottom",
                showarrow=False,
                font=dict(size=17),
            )
        fig.add_shape(
            type="line",
            x0=n_matches + gap * 0.75,
            x1=n_matches + gap * 0.75,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=STONE, width=2, dash="dot"),
        )
        fig.update_yaxes(range=[low - span * 0.45, high + span * 0.06])
    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper", y0=1500, y1=1500, line=dict(color=STONE, width=2)
    )
    step = max(int(np.ceil(n_matches / 25)) * 5, 1)
    fig.update_xaxes(tickvals=list(range(0, n_matches + 1, step)), showgrid=False)
    fig.update_layout(
        legend=dict(title=dict(text="gold labels"), x=0.02, y=0.98, xanchor="left", yanchor="top")
    )
    _save(fig, path)
    return fig


def time_vs_accuracy(summary: pd.DataFrame, path: Path, date: str, machine: str) -> go.Figure:
    """
    Chart 7: seconds per opinion (p50, log x) vs headline accuracy for every solution.

    Args:
        summary: The part-1 table (`label`, `family`, `headline`, `p50_s`, `p95_s`).
        path: Where to save the PNG.
        date: The date for the source line.
        machine: What the local models ran on, for the title.

    Returns:
        The figure (also saved to `path`).
    """
    fig = _figure(
        f"Time vs accuracy (local models on {machine})",
        date,
        "seconds per opinion, p50 (cloud calls include the round trip from Singapore)",
        "headline accuracy",
    )
    timed = summary.dropna(subset=["p50_s", "headline"])
    if timed.empty:
        _empty_note(fig, "no timed solution in this run")
        _save(fig, path)
        return fig
    for fill in dict.fromkeys(FAMILY_FILLS[family] for family in timed["family"]):
        mask = timed["family"].map(FAMILY_FILLS) == fill
        rows = timed[mask]
        fig.add_trace(
            go.Scatter(
                x=rows["p50_s"],
                y=rows["headline"],
                mode="markers",
                text=rows["label"],
                marker=dict(color=fill, **MARKER),
                name=FAMILY_LEGEND[fill],
                customdata=rows["p95_s"],
                hovertemplate="%{text}<br>p50 %{x:.2f} s · p95 %{customdata:.2f} s<br>"
                "headline %{y:.2f}<extra></extra>",
            )
        )
    low, high = float(timed["p50_s"].min()) / 3, float(timed["p50_s"].max()) * 6
    ticks = _log_ticks(low, high)
    x_range = (float(np.log10(low)), float(np.log10(high)))
    fig.update_xaxes(
        type="log", range=list(x_range), tickvals=ticks, ticktext=[f"{tick:g}" for tick in ticks]
    )
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(legend=dict(x=0.98, y=0.04, xanchor="right", yanchor="bottom"))
    _place_labels(
        fig, timed["p50_s"], timed["headline"], timed["label"], x_range, (0, 1.02), x_log=True
    )
    _save(fig, path)
    return fig


def save_all(
    summary: pd.DataFrame,
    by_question: pd.DataFrame,
    by_language: pd.DataFrame,
    points: pd.DataFrame,
    gold_history: pd.DataFrame,
    final_by_solution: dict[str, dict[str, float]],
    out_dir: Path,
    date: str,
    machine: str = "local machine",
) -> dict[str, go.Figure]:
    """
    Draw and save the seven charts.

    Args:
        summary: The part-1 table.
        by_question: From `metrics.by_question_table`.
        by_language: From `metrics.by_language_table`.
        points: From `metrics.rating_points`, with `label` and `family` added.
        gold_history: The gold Elo history.
        final_by_solution: Solution label -> final Elo per durian.
        out_dir: The folder for the PNGs, e.g. `assets/charts`.
        date: The date for the source lines.
        machine: What the local models ran on, for chart 7's title.

    Returns:
        The figures by chart name, in the order of `CHART_FILES`.
    """
    files = {name: out_dir / file for name, file in CHART_FILES.items()}
    return {
        "cost_vs_accuracy": cost_vs_accuracy(summary, files["cost_vs_accuracy"], date),
        "time_per_opinion": time_per_opinion(summary, files["time_per_opinion"], date),
        "accuracy_by_question": accuracy_by_question(
            by_question, files["accuracy_by_question"], date
        ),
        "accuracy_by_language": accuracy_by_language(
            by_language, files["accuracy_by_language"], date
        ),
        "rating_vs_real": rating_vs_real(points, files["rating_vs_real"], date),
        "durian_elo": durian_elo(gold_history, final_by_solution, files["durian_elo"], date),
        "time_vs_accuracy": time_vs_accuracy(summary, files["time_vs_accuracy"], date, machine),
    }
