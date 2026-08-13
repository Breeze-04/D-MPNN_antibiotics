"""Create transparent SVG diagnostics for the MIC regression baseline.

The script uses only the Python standard library, so it does not require
Matplotlib or another plotting dependency.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


BLUE = "#2F6B9A"
DARK = "#24292F"
GREY = "#7A828A"
LIGHT_GREY = "#C7CDD3"

WIDTH = 1440
HEIGHT = 610
PLOT_TOP = 112
PLOT_HEIGHT = 390
PLOT_WIDTH = 530
LEFT_X = 92
RIGHT_X = 805


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("experiments/outputs/baseline_smoke/test_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/baseline_smoke/baseline_diagnostics.svg"),
    )
    return parser.parse_args()


def calculate_metrics(actual: list[float], predicted: list[float]) -> tuple[float, float, float]:
    residuals = [pred - obs for obs, pred in zip(actual, predicted)]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    mae = sum(abs(value) for value in residuals) / len(residuals)
    mean_actual = sum(actual) / len(actual)
    denominator = sum((value - mean_actual) ** 2 for value in actual)
    r2 = 1 - sum(value * value for value in residuals) / denominator
    return rmse, mae, r2


def linear_scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    fraction = (value - source_min) / (source_max - source_min)
    return target_min + fraction * (target_max - target_min)


def format_tick(value: float) -> str:
    rounded = round(value)
    return str(rounded) if math.isclose(value, rounded, abs_tol=1e-8) else f"{value:.1f}"


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: str = "normal",
    rotate: int | None = None,
    fill: str = DARK,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}"{transform}>'
        f"{html.escape(text)}</text>"
    )


def draw_axes(
    parts: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    n_ticks: int = 6,
) -> None:
    bottom = y0 + height
    parts.append(f'<line x1="{x0}" y1="{bottom}" x2="{x0 + width}" y2="{bottom}" class="axis"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{bottom}" class="axis"/>')

    for index in range(n_ticks):
        fraction = index / (n_ticks - 1)
        x = x0 + fraction * width
        y = bottom - fraction * height
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{bottom}" class="grid"/>')
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + width}" y2="{y:.2f}" class="grid"/>')
        parts.append(svg_text(x, bottom + 24, format_tick(x_value), size=12, anchor="middle", fill=GREY))
        parts.append(svg_text(x0 - 13, y + 4, format_tick(y_value), size=12, anchor="end", fill=GREY))

    parts.append(svg_text(x0 + width / 2, bottom + 58, x_label, size=14, anchor="middle"))
    parts.append(svg_text(x0 - 64, y0 + height / 2, y_label, size=14, anchor="middle", rotate=-90))


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".svg":
        raise ValueError("Output must use the .svg extension")

    actual: list[float] = []
    predicted: list[float] = []
    with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"log2_mic", "pred_log2_mic"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Prediction CSV is missing required columns: {missing}")
        for row in reader:
            observed = float(row["log2_mic"])
            estimate = float(row["pred_log2_mic"])
            if math.isfinite(observed) and math.isfinite(estimate):
                actual.append(observed)
                predicted.append(estimate)

    if not actual:
        raise ValueError("No finite target/prediction pairs are available")

    residuals = [pred - obs for obs, pred in zip(actual, predicted)]
    rmse, mae, r2 = calculate_metrics(actual, predicted)

    shared_min = min(min(actual), min(predicted))
    shared_max = max(max(actual), max(predicted))
    shared_padding = 0.05 * (shared_max - shared_min)
    shared_min -= shared_padding
    shared_max += shared_padding

    observed_padding = 0.05 * (max(actual) - min(actual))
    observed_min = min(actual) - observed_padding
    observed_max = max(actual) + observed_padding
    residual_limit = 1.08 * max(abs(min(residuals)), abs(max(residuals)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; }",
        f".axis {{ stroke: {GREY}; stroke-width: 1.2; }}",
        f".grid {{ stroke: {LIGHT_GREY}; stroke-width: 0.7; stroke-opacity: 0.42; }}",
        f".point {{ fill: {BLUE}; fill-opacity: 0.60; }}",
        "</style>",
        svg_text(WIDTH / 2, 35, "D-MPNN baseline test-set diagnostics", size=21, anchor="middle", weight="bold"),
        svg_text(WIDTH / 2, 59, "E. coli ATCC 25922 MIC regression · n = 331 test molecules", size=13, anchor="middle", fill=GREY),
        svg_text(LEFT_X, 91, "A  Observed vs predicted", size=16, weight="bold"),
        svg_text(RIGHT_X, 91, "B  Residuals vs observed", size=16, weight="bold"),
    ]

    draw_axes(
        parts,
        x0=LEFT_X,
        y0=PLOT_TOP,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        x_min=shared_min,
        x_max=shared_max,
        y_min=shared_min,
        y_max=shared_max,
        x_label="Observed log2(MIC [µg/mL])",
        y_label="Predicted log2(MIC [µg/mL])",
    )
    left_bottom = PLOT_TOP + PLOT_HEIGHT
    parts.append(
        f'<line x1="{LEFT_X}" y1="{left_bottom}" x2="{LEFT_X + PLOT_WIDTH}" y2="{PLOT_TOP}" '
        f'stroke="{DARK}" stroke-width="1.6" stroke-dasharray="7 5"/>'
    )
    for observed, estimate in zip(actual, predicted):
        x = linear_scale(observed, shared_min, shared_max, LEFT_X, LEFT_X + PLOT_WIDTH)
        y = linear_scale(estimate, shared_min, shared_max, left_bottom, PLOT_TOP)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" class="point"/>')

    parts.append(f'<line x1="{LEFT_X + 18}" y1="{PLOT_TOP + 25}" x2="{LEFT_X + 58}" y2="{PLOT_TOP + 25}" stroke="{DARK}" stroke-width="1.6" stroke-dasharray="7 5"/>')
    parts.append(svg_text(LEFT_X + 68, PLOT_TOP + 30, "Ideal: y = x", size=12))
    metric_x = LEFT_X + PLOT_WIDTH - 18
    parts.extend(
        [
            svg_text(metric_x, left_bottom - 76, f"n = {len(actual)}", size=13, anchor="end"),
            svg_text(metric_x, left_bottom - 56, f"RMSE = {rmse:.3f}", size=13, anchor="end"),
            svg_text(metric_x, left_bottom - 36, f"MAE = {mae:.3f}", size=13, anchor="end"),
            svg_text(metric_x, left_bottom - 16, f"R² = {r2:.3f}", size=13, anchor="end", weight="bold"),
        ]
    )

    draw_axes(
        parts,
        x0=RIGHT_X,
        y0=PLOT_TOP,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        x_min=observed_min,
        x_max=observed_max,
        y_min=-residual_limit,
        y_max=residual_limit,
        x_label="Observed log2(MIC [µg/mL])",
        y_label="Residual: predicted − observed",
    )
    right_bottom = PLOT_TOP + PLOT_HEIGHT

    reference_lines = [
        (0.0, DARK, "7 5", 1.6),
        (1.0, GREY, "2 4", 1.0),
        (-1.0, GREY, "2 4", 1.0),
        (2.0, LIGHT_GREY, "7 4 2 4", 1.0),
        (-2.0, LIGHT_GREY, "7 4 2 4", 1.0),
    ]
    for value, color, dash, line_width in reference_lines:
        y = linear_scale(value, -residual_limit, residual_limit, right_bottom, PLOT_TOP)
        parts.append(
            f'<line x1="{RIGHT_X}" y1="{y:.2f}" x2="{RIGHT_X + PLOT_WIDTH}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="{line_width}" stroke-dasharray="{dash}"/>'
        )

    for observed, residual in zip(actual, residuals):
        x = linear_scale(observed, observed_min, observed_max, RIGHT_X, RIGHT_X + PLOT_WIDTH)
        y = linear_scale(residual, -residual_limit, residual_limit, right_bottom, PLOT_TOP)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" class="point"/>')

    legend_x = RIGHT_X + PLOT_WIDTH - 192
    legend_y = PLOT_TOP + 23
    legend_items = [
        (DARK, "7 5", "Zero residual"),
        (GREY, "2 4", "±1 log₂ (about 2-fold)"),
        (LIGHT_GREY, "7 4 2 4", "±2 log₂ (about 4-fold)"),
    ]
    for index, (color, dash, label) in enumerate(legend_items):
        y = legend_y + index * 22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 38}" y2="{y}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
        parts.append(svg_text(legend_x + 47, y + 4, label, size=11))

    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")

    print(f"Saved: {args.output.resolve()}")
    print(f"n={len(actual)}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")


if __name__ == "__main__":
    main()
