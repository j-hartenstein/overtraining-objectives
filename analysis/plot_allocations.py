#!/usr/bin/env python3
"""Plot within-generation allocation contrasts and release-time token intensity."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from reportlab.pdfgen import canvas


DEVELOPER_ORDER = ["Meta", "Google DeepMind", "Ai2", "Alibaba", "DeepSeek"]
DEVELOPER_COLORS = {
    "Meta": "#2864A7",
    "Google DeepMind": "#B84A4A",
    "Ai2": "#34845B",
    "Alibaba": "#8554A3",
    "DeepSeek": "#D17828",
}
DEVELOPER_MARKERS = {
    "Meta": "circle",
    "Google DeepMind": "square",
    "Ai2": "triangle",
    "Alibaba": "diamond",
    "DeepSeek": "cross",
}
SIZE_COLORS = {
    "small": "#2F6B9A",
    "medium": "#E69F00",
    "large": "#F0E442",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_csv", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


def set_color(c: canvas.Canvas, value: str, stroke: bool = True, fill: bool = False) -> None:
    rgb = hex_rgb(value)
    if stroke:
        c.setStrokeColorRGB(*rgb)
    if fill:
        c.setFillColorRGB(*rgb)


def marker(
    c: canvas.Canvas,
    x: float,
    y: float,
    kind: str,
    color: str,
    radius: float = 3.4,
    outline: str | None = None,
) -> None:
    set_color(c, color, stroke=False, fill=kind != "cross")
    set_color(c, outline or color, stroke=True, fill=False)
    c.setLineWidth(1)
    if kind == "circle":
        c.circle(x, y, radius, stroke=1, fill=1)
    elif kind == "square":
        c.rect(x - radius, y - radius, 2 * radius, 2 * radius, stroke=1, fill=1)
    elif kind == "triangle":
        path = c.beginPath()
        path.moveTo(x, y + radius * 1.2)
        path.lineTo(x - radius * 1.1, y - radius)
        path.lineTo(x + radius * 1.1, y - radius)
        path.close()
        c.drawPath(path, stroke=1, fill=1)
    elif kind == "diamond":
        path = c.beginPath()
        path.moveTo(x, y + radius * 1.25)
        path.lineTo(x - radius, y)
        path.lineTo(x, y - radius * 1.25)
        path.lineTo(x + radius, y)
        path.close()
        c.drawPath(path, stroke=1, fill=1)
    else:
        c.line(x - radius, y - radius, x + radius, y + radius)
        c.line(x - radius, y + radius, x + radius, y - radius)


def log_map(value: float, domain: tuple[float, float], target: tuple[float, float]) -> float:
    lo, hi = map(math.log10, domain)
    return target[0] + (math.log10(value) - lo) / (hi - lo) * (target[1] - target[0])


def date_map(value: datetime, domain: tuple[datetime, datetime], target: tuple[float, float]) -> float:
    return target[0] + (value - domain[0]).total_seconds() / (domain[1] - domain[0]).total_seconds() * (target[1] - target[0])


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    rows = []
    for row in values:
        architecture = row["architecture"].lower()
        if row["eligible_for_dn"].lower() != "true":
            continue
        if "moe" in architecture or "mixture-of-experts" in architecture:
            continue
        rows.append(
            {
                "developer": row["developer"],
                "generation": row["generation"],
                "model": row["model"],
                "date": datetime.fromisoformat(row["release_date"]),
                "generation_date": datetime.fromisoformat(row["generation_date"]),
                "n": float(row["total_params_b"]),
                "d": float(row["training_tokens_t"]),
                "tokens_per_parameter": 1000 * float(row["training_tokens_t"]) / float(row["total_params_b"]),
            }
        )
    return rows


def within_generation(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["developer"], row["generation"], row["generation_date"])].append(row)
    result = []
    for (developer, generation, date), models in groups.items():
        if len(models) < 2:
            continue
        models.sort(key=lambda r: r["n"])
        result.append(
            {
                "developer": developer,
                "generation": generation,
                "date": date,
                "contrast": models[0]["tokens_per_parameter"] / models[-1]["tokens_per_parameter"],
            }
        )
    return sorted(result, key=lambda r: (DEVELOPER_ORDER.index(r["developer"]), r["date"]))


def release_groups(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["developer"], row["generation"], row["generation_date"])].append(row)
    result = []
    for (developer, generation, date), models in groups.items():
        result.append(
            {
                "developer": developer,
                "generation": generation,
                "date": date,
                "models": sorted(models, key=lambda model: model["n"]),
            }
        )
    return sorted(result, key=lambda row: row["date"])


def size_bin(n: float) -> str:
    if n < 10:
        return "small"
    if n <= 32:
        return "medium"
    return "large"


def draw_left(c: canvas.Canvas, rows: list[dict], box: tuple[float, float, float, float]) -> None:
    x0, y0, width, height = box
    data = within_generation(rows)
    x_domain = (0.8, 70)
    separator_margin = 40
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0 - 36, y0 + height + 17, "A. Allocation within model generations")

    c.setStrokeColorRGB(0.15, 0.15, 0.15)
    c.setLineWidth(0.65)
    c.rect(x0, y0, width, height, stroke=1, fill=0)
    for tick in (1, 2, 5, 10, 20, 50):
        x = log_map(tick, x_domain, (x0, x0 + width))
        c.setStrokeColorRGB(0.87, 0.87, 0.87)
        c.setLineWidth(0.35)
        c.line(x, y0, x, y0 + height)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 6.8)
        c.drawCentredString(x, y0 - 11, f"{tick:g}×")

    x_reference = log_map(1, x_domain, (x0, x0 + width))
    c.setStrokeColorRGB(0.25, 0.25, 0.25)
    c.setLineWidth(0.8)
    c.setDash(3, 2)
    c.line(x_reference, y0, x_reference, y0 + height)
    c.setDash()
    c.setFont("Helvetica", 6.5)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(x_reference + 3, y0 + height - 9, "same token intensity")

    n = len(data)
    row_gap = height / (n + 1)
    prior_developer = None
    for index, row in enumerate(data):
        y = y0 + height - (index + 1) * row_gap
        if prior_developer is not None and row["developer"] != prior_developer:
            c.setStrokeColorRGB(0.85, 0.85, 0.85)
            c.setLineWidth(0.45)
            c.line(x0 - separator_margin, y + row_gap / 2, x0 + width, y + row_gap / 2)
        c.setFillColorRGB(0.18, 0.18, 0.18)
        c.setFont("Helvetica", 6.7)
        display_generation = {"DeepSeek LLM": "DeepSeek"}.get(row["generation"], row["generation"])
        c.drawRightString(x0 - 5, y - 2.4, display_generation)
        x = log_map(row["contrast"], x_domain, (x0, x0 + width))
        marker(c, x, y, DEVELOPER_MARKERS[row["developer"]], DEVELOPER_COLORS[row["developer"]], radius=3.2)
        c.setFillColorRGB(0.18, 0.18, 0.18)
        c.setFont("Helvetica", 6.4)
        contrast_label = f"{row['contrast']:.1f}×"
        if x + 5 + c.stringWidth(contrast_label, "Helvetica", 6.4) > x0 + width - 2:
            c.drawRightString(x - 5, y - 2.2, contrast_label)
        else:
            c.drawString(x + 5, y - 2.2, contrast_label)
        prior_developer = row["developer"]

    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica", 7.2)
    c.drawCentredString(x0 + width / 2, y0 - 25, "Smallest sibling's token intensity (largest = 1)")


def draw_right(c: canvas.Canvas, rows: list[dict], box: tuple[float, float, float, float]) -> None:
    x0, y0, width, height = box
    data = release_groups(rows)
    x_domain = (datetime(2022, 12, 29), datetime(2025, 12, 20))
    y_domain = (0.8, 130)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0 - 36, y0 + height + 31, "B. Token intensity at release")

    c.setFont("Helvetica", 5.7)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    size_legend_y = y0 + height + 19.5
    c.drawString(x0 - 36, size_legend_y - 2, "Size:")
    legend_x = x0 - 13
    for label, key in (("<10B", "small"), ("10-32B", "medium"), (">32B", "large")):
        set_color(c, SIZE_COLORS[key], stroke=False, fill=True)
        c.setStrokeColorRGB(0.12, 0.12, 0.12)
        c.rect(legend_x, size_legend_y - 3, 6, 6, stroke=1, fill=1)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.drawString(legend_x + 9, size_legend_y - 2, label)
        legend_x += 45

    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(x0 - 36, y0 + height + 8, "Shape:")
    legend_x = x0 - 8
    for developer in DEVELOPER_ORDER:
        marker(c, legend_x, y0 + height + 10, DEVELOPER_MARKERS[developer], "#555555", radius=2.2)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        label = {"Google DeepMind": "Google", "Alibaba": "Qwen"}.get(developer, developer)
        c.drawString(legend_x + 4, y0 + height + 7.5, label)
        legend_x += {"Meta": 33, "Google DeepMind": 40, "Ai2": 25, "Alibaba": 34, "DeepSeek": 0}[developer]

    c.setStrokeColorRGB(0.15, 0.15, 0.15)
    c.setLineWidth(0.65)
    c.rect(x0, y0, width, height, stroke=1, fill=0)
    for tick in (1, 3, 10, 30, 100):
        y = log_map(tick, y_domain, (y0, y0 + height))
        c.setStrokeColorRGB(0.88, 0.88, 0.88)
        c.setLineWidth(0.35)
        c.line(x0, y, x0 + width, y)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.setFont("Helvetica", 6.6)
        c.drawRightString(x0 - 4, y - 2.2, f"{tick}×")

    y_chinchilla = log_map(1, y_domain, (y0, y0 + height))
    c.setStrokeColorRGB(0.25, 0.25, 0.25)
    c.setLineWidth(0.8)
    c.setDash(3, 2)
    c.line(x0, y_chinchilla, x0 + width, y_chinchilla)
    c.setDash()
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.setFont("Helvetica", 6.3)
    c.drawRightString(x0 + width - 4, y_chinchilla + 4, "Chinchilla")

    for year in (2023, 2024, 2025):
        x = date_map(datetime(year, 1, 1), x_domain, (x0, x0 + width))
        c.setStrokeColorRGB(0.9, 0.9, 0.9)
        c.setLineWidth(0.3)
        c.line(x, y0, x, y0 + height)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.setFont("Helvetica", 6.6)
        if year == 2023:
            c.drawString(x, y0 - 11, str(year))
        else:
            c.drawCentredString(x, y0 - 11, str(year))

    # Spread nearby release ladders just enough to distinguish them while
    # preserving their chronological order and keeping each ladder vertical.
    raw_x = [date_map(release["date"], x_domain, (x0, x0 + width)) for release in data]
    display_x = raw_x.copy()
    cluster_start = 0
    for cluster_end in range(1, len(data) + 1):
        closes_cluster = cluster_end == len(data) or raw_x[cluster_end] - raw_x[cluster_end - 1] >= 8
        if not closes_cluster:
            continue
        cluster_size = cluster_end - cluster_start
        if cluster_size > 1:
            center = sum(raw_x[cluster_start:cluster_end]) / cluster_size
            midpoint = (cluster_size - 1) / 2
            for index in range(cluster_size):
                display_x[cluster_start + index] = center + (index - midpoint) * 7
        cluster_start = cluster_end

    for release, x in zip(data, display_x):
        points = [
            {**model, "multiple": model["tokens_per_parameter"] / 20}
            for model in release["models"]
        ]
        if len(points) > 1:
            ys = [log_map(point["multiple"], y_domain, (y0, y0 + height)) for point in points]
            c.setStrokeColorRGB(0.52, 0.52, 0.52)
            c.setLineWidth(0.75)
            c.line(x, min(ys), x, max(ys))

        duplicates: dict[float, list[dict]] = defaultdict(list)
        for point in points:
            duplicates[round(point["multiple"], 6)].append(point)
        point_offsets: dict[str, float] = {}
        for duplicate_points in duplicates.values():
            midpoint = (len(duplicate_points) - 1) / 2
            for index, point in enumerate(sorted(duplicate_points, key=lambda item: item["n"])):
                point_offsets[point["model"]] = (index - midpoint) * 6.8
        for point in sorted(points, key=lambda item: item["n"], reverse=True):
            marker(
                c,
                x + point_offsets[point["model"]],
                log_map(point["multiple"], y_domain, (y0, y0 + height)),
                DEVELOPER_MARKERS[release["developer"]],
                SIZE_COLORS[size_bin(point["n"])],
                radius=3.2,
                outline="#222222",
            )

    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica", 7.2)
    c.drawCentredString(x0 + width / 2, y0 - 25, "Release date")
    c.saveState()
    c.translate(x0 - 29, y0 + height / 2)
    c.rotate(90)
    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica", 7)
    c.drawCentredString(0, 0, "Training tokens per parameter (relative to Chinchilla)")
    c.restoreState()


def main() -> None:
    parsed = args()
    rows = load_rows(parsed.panel_csv)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 560, 320
    c = canvas.Canvas(str(parsed.output), pagesize=(page_width, page_height))
    c.setTitle("Allocation contrasts and token intensity over time")
    c.setAuthor("Anonymous")
    c.translate(-14.4, 0)
    draw_left(c, rows, (80, 50, 194, 224))
    draw_right(c, rows, (326, 50, 219, 210))
    c.save()
    print(f"Plotted {len(rows)} dense models to {parsed.output}")


if __name__ == "__main__":
    main()
