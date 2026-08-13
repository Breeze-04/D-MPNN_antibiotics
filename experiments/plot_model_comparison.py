"""Create a transparent SVG comparing baseline and attention experiments.

This script does not train a model and has no plotting-library dependency. It reads
the saved Chemprop experiment artifacts and writes SVG directly with Python's
standard library.

The figure contains:
  A1. Validation R2 versus epoch.
  A2. Training and validation loss versus epoch.
  B. Test-set R2, RMSE, and MAE comparisons.

Before plotting, the script verifies that the two runs used the same split and the
same main training settings. It also selects the Lightning metrics.csv whose test
metrics match each run's metrics.json, avoiding accidental use of an older log.
"""

from __future__ import annotations

'''
python experiments/plot_model_comparison.py `
  --baseline-dir experiments/outputs/baseline_smoke `
  --attention-dir experiments/outputs/attention_smoke `
  --output experiments/outputs/model_comparison.svg
'''


import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASELINE_COLOR = "#2878B5"
ATTENTION_COLOR = "#E67E22"
TEXT_COLOR = "#24303A"
MUTED_COLOR = "#64727E"
GRID_COLOR = "#D9E0E5"
LIGHT_COLOR = "#EEF2F5"


@dataclass
class RunData:
    name: str
    directory: Path
    config: dict
    test_metrics: dict[str, float]
    test_n: int
    log_path: Path
    curves: dict[str, list[tuple[int, float]]]
    best_epoch: int
    best_val_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and attention Chemprop runs in one SVG."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("experiments/outputs/baseline_smoke"),
        help="Directory containing the baseline config, metrics, splits, and logs.",
    )
    parser.add_argument(
        "--attention-dir",
        type=Path,
        default=Path("experiments/outputs/attention_smoke"),
        help="Directory containing the attention config, metrics, splits, and logs.",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Optional explicit baseline Lightning metrics.csv.",
    )
    parser.add_argument(
        "--attention-log",
        type=Path,
        default=None,
        help="Optional explicit attention Lightning metrics.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/model_comparison.svg"),
        help="Output SVG path.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def log_test_metrics(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    found: dict[str, float] = {}
    for row in rows:
        for key, column in (
            ("r2", "test/r2"),
            ("rmse", "test/rmse"),
            ("mae", "test/mae"),
        ):
            value = as_float(row.get(column))
            if value is not None:
                found[key] = value
    return found


def metrics_match(observed: dict[str, float], expected: dict[str, float]) -> bool:
    return all(
        key in observed and math.isclose(observed[key], expected[key], rel_tol=2e-4, abs_tol=2e-4)
        for key in ("r2", "rmse", "mae")
    )


def select_log(
    run_dir: Path, expected: dict[str, float], explicit: Path | None
) -> tuple[Path, list[dict[str, str]]]:
    if explicit is not None:
        rows = read_csv_rows(explicit)
        observed = log_test_metrics(rows)
        if not metrics_match(observed, expected):
            raise ValueError(
                f"Explicit log does not match {run_dir / 'metrics.json'}: {explicit}"
            )
        return explicit, rows

    candidates: list[tuple[int, int, Path, list[dict[str, str]]]] = []
    for path in run_dir.glob("logs/version_*/metrics.csv"):
        rows = read_csv_rows(path)
        observed = log_test_metrics(rows)
        if not metrics_match(observed, expected):
            continue
        epochs = [
            int(float(row["epoch"]))
            for row in rows
            if row.get("epoch", "").strip() != ""
        ]
        version_text = path.parent.name.removeprefix("version_")
        version = int(version_text) if version_text.isdigit() else -1
        candidates.append((max(epochs, default=-1), version, path, rows))

    if not candidates:
        raise FileNotFoundError(
            f"No metrics.csv under {run_dir / 'logs'} matches metrics.json. "
            "Use the corresponding --baseline-log or --attention-log explicitly."
        )

    _, _, path, rows = max(candidates, key=lambda item: (item[0], item[1]))
    return path, rows


def extract_curves(rows: Iterable[dict[str, str]]) -> dict[str, list[tuple[int, float]]]:
    by_epoch: dict[int, dict[str, float]] = {}
    columns = {
        "train_loss": "train_loss_epoch",
        "val_loss": "val_loss",
        "val_r2": "val/r2",
    }
    for row in rows:
        epoch_text = row.get("epoch", "").strip()
        if not epoch_text:
            continue
        epoch = int(float(epoch_text)) + 1  # Lightning is zero-based; display is one-based.
        record = by_epoch.setdefault(epoch, {})
        for curve_name, column in columns.items():
            value = as_float(row.get(column))
            if value is not None:
                record[curve_name] = value

    curves = {
        name: [(epoch, values[name]) for epoch, values in sorted(by_epoch.items()) if name in values]
        for name in columns
    }
    missing = [name for name, points in curves.items() if not points]
    if missing:
        raise ValueError(f"Training log is missing required curves: {', '.join(missing)}")
    return curves


def load_run(name: str, run_dir: Path, explicit_log: Path | None) -> RunData:
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    expected = {key: float(metrics["macro_average"][key]) for key in ("r2", "rmse", "mae")}
    per_target = metrics.get("per_target", {})
    if not per_target:
        raise ValueError(f"No per-target metrics found in {run_dir / 'metrics.json'}")
    target_details = next(iter(per_target.values()))
    test_n = int(target_details["n"])

    log_path, rows = select_log(run_dir, expected, explicit_log)
    curves = extract_curves(rows)
    best_epoch, best_val_loss = min(curves["val_loss"], key=lambda point: point[1])
    return RunData(
        name=name,
        directory=run_dir,
        config=config,
        test_metrics=expected,
        test_n=test_n,
        log_path=log_path,
        curves=curves,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )


def split_map(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Split file is empty: {path}")
    if "source_row" not in rows[0] or "split" not in rows[0]:
        raise ValueError(f"Split file must contain source_row and split columns: {path}")
    result: dict[str, str] = {}
    for row in rows:
        source_row = row["source_row"].strip()
        split = row["split"].strip()
        if source_row in result and result[source_row] != split:
            raise ValueError(f"Conflicting split assignments for source_row={source_row}: {path}")
        result[source_row] = split
    return result


def validate_comparability(baseline: RunData, attention: RunData) -> None:
    controlled_fields = (
        "data_path",
        "smiles_column",
        "target_columns",
        "split_type",
        "split_sizes",
        "seed",
        "epochs",
        "patience",
        "batch_size",
        "num_workers",
        "message_hidden_dim",
        "depth",
        "ffn_hidden_dim",
        "ffn_num_layers",
        "dropout",
        "accelerator",
    )
    differences = [
        field
        for field in controlled_fields
        if baseline.config.get(field) != attention.config.get(field)
    ]
    if differences:
        details = ", ".join(
            f"{field}: {baseline.config.get(field)!r} != {attention.config.get(field)!r}"
            for field in differences
        )
        raise ValueError(f"Runs are not controlled comparisons ({details}).")

    baseline_splits = split_map(baseline.directory / "splits.csv")
    attention_splits = split_map(attention.directory / "splits.csv")
    if baseline_splits != attention_splits:
        raise ValueError("Baseline and attention split assignments are not identical.")
    if baseline.test_n != attention.test_n:
        raise ValueError(
            f"Test sizes differ: baseline n={baseline.test_n}, attention n={attention.test_n}."
        )

    baseline_arch = str(baseline.config.get("architecture", ""))
    attention_arch = str(attention.config.get("architecture", ""))
    if "MeanAggregation" not in baseline_arch:
        raise ValueError("Baseline config does not identify MeanAggregation.")
    if "AttentiveAggregation" not in attention_arch:
        raise ValueError("Attention config does not identify AttentiveAggregation.")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<style>",
            "text { font-family: Arial, Helvetica, sans-serif; }",
            ".axis { stroke: #65747F; stroke-width: 1.2; }",
            ".grid { stroke: #D9E0E5; stroke-width: 1; }",
            "</style>",
        ]

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        width: float = 1.0,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'
        )

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        stroke: str,
        width: float = 2.5,
        dash: str | None = None,
    ) -> None:
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{point_text}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        stroke: str | None = None,
        width: float = 1.0,
    ) -> None:
        stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        size: int = 14,
        weight: int | str = 400,
        anchor: str = "start",
        fill: str = TEXT_COLOR,
        rotate: float | None = None,
    ) -> None:
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}"{transform}>{esc(value)}</text>'
        )

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def tick_values(low: float, high: float, count: int = 5) -> list[float]:
    if math.isclose(low, high):
        high = low + 1.0
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def rounded_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def draw_line_chart(
    svg: Svg,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    series: list[tuple[list[tuple[int, float]], str, str | None]],
    x_max: int,
    y_min: float,
    y_max: float,
    best_markers: list[tuple[int, float, str]],
    show_x_label: bool,
) -> None:
    svg.text(x, y - 18, title, size=16, weight=700)
    plot_left = x + 68
    plot_right = x + width - 12
    plot_top = y
    plot_bottom = y + height - 45

    def sx(epoch: float) -> float:
        return plot_left + (epoch - 1) / max(x_max - 1, 1) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for value in tick_values(y_min, y_max):
        py = sy(value)
        svg.line(plot_left, py, plot_right, py, GRID_COLOR)
        svg.text(plot_left - 10, py + 5, f"{value:.2f}", size=12, anchor="end", fill=MUTED_COLOR)

    x_ticks = sorted({1, *[round(1 + (x_max - 1) * index / 4) for index in range(1, 5)]})
    for epoch in x_ticks:
        px = sx(epoch)
        svg.line(px, plot_bottom, px, plot_bottom + 5, MUTED_COLOR)
        svg.text(px, plot_bottom + 22, epoch, size=12, anchor="middle", fill=MUTED_COLOR)

    svg.line(plot_left, plot_top, plot_left, plot_bottom, MUTED_COLOR, width=1.2)
    svg.line(plot_left, plot_bottom, plot_right, plot_bottom, MUTED_COLOR, width=1.2)
    svg.text(x + 18, (plot_top + plot_bottom) / 2, y_label, size=13, anchor="middle", rotate=-90)
    if show_x_label:
        svg.text((plot_left + plot_right) / 2, plot_bottom + 42, "Epoch", size=13, anchor="middle")

    for points, color, dash in series:
        svg.polyline([(sx(epoch), sy(value)) for epoch, value in points], color, dash=dash)

    for epoch, value, color in best_markers:
        px, py = sx(epoch), sy(value)
        svg.line(px, plot_top, px, plot_bottom, color, width=1.2, dash="4 5", opacity=0.45)
        svg.circle(px, py, 5.0, "white", stroke=color, width=2.5)
        svg.text(px + 7, py - 8, f"best e{epoch}", size=11, fill=color)


def draw_metric_block(
    svg: Svg,
    x: float,
    y: float,
    width: float,
    name: str,
    direction: str,
    baseline_value: float,
    attention_value: float,
    delta_text: str,
) -> None:
    svg.text(x, y, name, size=18, weight=700)
    svg.text(x + width, y, direction, size=12, anchor="end", fill=MUTED_COLOR)

    axis_left = x + 104
    axis_right = x + width - 18
    axis_y = y + 38
    upper = rounded_upper(max(baseline_value, attention_value) * 1.08)
    svg.line(axis_left, axis_y, axis_right, axis_y, GRID_COLOR)
    for value in tick_values(0.0, upper, 4):
        px = axis_left + value / upper * (axis_right - axis_left)
        svg.line(px, axis_y - 4, px, axis_y + 4, MUTED_COLOR)
        svg.text(px, axis_y - 9, f"{value:.2f}", size=10, anchor="middle", fill=MUTED_COLOR)

    rows = (
        ("Baseline", baseline_value, BASELINE_COLOR, y + 70),
        ("Attention", attention_value, ATTENTION_COLOR, y + 106),
    )
    for label, value, color, row_y in rows:
        px = axis_left + value / upper * (axis_right - axis_left)
        svg.text(axis_left - 12, row_y + 5, label, size=13, anchor="end")
        svg.line(axis_left, row_y, px, row_y, color, width=5, opacity=0.30)
        svg.circle(px, row_y, 6, color)
        label_x = min(px + 11, axis_right - 2)
        anchor = "start" if px + 58 <= axis_right else "end"
        if anchor == "end":
            label_x = px - 10
        svg.text(label_x, row_y + 5, f"{value:.4f}", size=13, weight=700, anchor=anchor, fill=color)

    svg.text(axis_left, y + 137, delta_text, size=13, weight=700, fill=ATTENTION_COLOR)


def render_svg(baseline: RunData, attention: RunData) -> str:
    svg = Svg(1600, 800)
    svg.text(60, 42, "Baseline vs Attention", size=27, weight=700)
    svg.text(
        60,
        68,
        "D-MPNN molecular MIC regression · controlled comparison on the same test split",
        size=14,
        fill=MUTED_COLOR,
    )

    # Compact legend: color denotes model; line style denotes data split.
    legend_y = 96
    svg.line(65, legend_y, 101, legend_y, BASELINE_COLOR, width=3)
    svg.text(110, legend_y + 5, "Baseline", size=13)
    svg.line(205, legend_y, 241, legend_y, ATTENTION_COLOR, width=3)
    svg.text(250, legend_y + 5, "Attention", size=13)
    svg.line(365, legend_y, 401, legend_y, TEXT_COLOR, width=2.5)
    svg.text(410, legend_y + 5, "Validation", size=13)
    svg.line(520, legend_y, 556, legend_y, TEXT_COLOR, width=2.5, dash="7 5")
    svg.text(565, legend_y + 5, "Training", size=13)
    svg.text(795, legend_y + 5, "○ best validation-loss epoch", size=12, anchor="end", fill=MUTED_COLOR)

    svg.text(60, 132, "A  Learning dynamics", size=20, weight=700)
    all_r2 = baseline.curves["val_r2"] + attention.curves["val_r2"]
    r2_values = [value for _, value in all_r2]
    r2_min = min(0.0, math.floor(min(r2_values) * 10) / 10)
    r2_max = max(0.1, math.ceil(max(r2_values) * 10) / 10)
    max_epoch = max(
        baseline.curves["val_r2"][-1][0], attention.curves["val_r2"][-1][0]
    )

    baseline_best_r2 = dict(baseline.curves["val_r2"]).get(baseline.best_epoch)
    attention_best_r2 = dict(attention.curves["val_r2"]).get(attention.best_epoch)
    r2_markers = []
    if baseline_best_r2 is not None:
        r2_markers.append((baseline.best_epoch, baseline_best_r2, BASELINE_COLOR))
    if attention_best_r2 is not None:
        r2_markers.append((attention.best_epoch, attention_best_r2, ATTENTION_COLOR))

    draw_line_chart(
        svg,
        x=60,
        y=165,
        width=755,
        height=250,
        title="A1  Validation R²",
        y_label="Validation R²",
        series=[
            (baseline.curves["val_r2"], BASELINE_COLOR, None),
            (attention.curves["val_r2"], ATTENTION_COLOR, None),
        ],
        x_max=max_epoch,
        y_min=r2_min,
        y_max=r2_max,
        best_markers=r2_markers,
        show_x_label=False,
    )

    loss_values = [
        value
        for run in (baseline, attention)
        for curve in ("train_loss", "val_loss")
        for _, value in run.curves[curve]
    ]
    loss_max = rounded_upper(max(loss_values) * 1.03)
    draw_line_chart(
        svg,
        x=60,
        y=475,
        width=755,
        height=250,
        title="A2  Normalized MSE loss",
        y_label="Loss",
        series=[
            (baseline.curves["val_loss"], BASELINE_COLOR, None),
            (attention.curves["val_loss"], ATTENTION_COLOR, None),
            (baseline.curves["train_loss"], BASELINE_COLOR, "7 5"),
            (attention.curves["train_loss"], ATTENTION_COLOR, "7 5"),
        ],
        x_max=max_epoch,
        y_min=0.0,
        y_max=loss_max,
        best_markers=[
            (baseline.best_epoch, baseline.best_val_loss, BASELINE_COLOR),
            (attention.best_epoch, attention.best_val_loss, ATTENTION_COLOR),
        ],
        show_x_label=True,
    )

    svg.line(850, 125, 850, 735, GRID_COLOR, width=1.2)
    right_x, right_width = 900, 630
    svg.text(right_x, 132, "B  Test-set performance", size=20, weight=700)
    svg.text(
        right_x + right_width,
        132,
        f"n = {baseline.test_n}",
        size=13,
        anchor="end",
        fill=MUTED_COLOR,
    )

    b = baseline.test_metrics
    a = attention.test_metrics
    r2_delta = a["r2"] - b["r2"]
    rmse_reduction = (b["rmse"] - a["rmse"]) / b["rmse"] * 100
    mae_reduction = (b["mae"] - a["mae"]) / b["mae"] * 100
    draw_metric_block(
        svg,
        right_x,
        175,
        right_width,
        "Test R²",
        "higher is better ↑",
        b["r2"],
        a["r2"],
        f"Attention change: {r2_delta:+.4f}",
    )
    svg.line(right_x, 326, right_x + right_width, 326, LIGHT_COLOR)
    draw_metric_block(
        svg,
        right_x,
        360,
        right_width,
        "Test RMSE",
        "lower is better ↓",
        b["rmse"],
        a["rmse"],
        f"Attention reduction: {rmse_reduction:.2f}%",
    )
    svg.line(right_x, 511, right_x + right_width, 511, LIGHT_COLOR)
    draw_metric_block(
        svg,
        right_x,
        545,
        right_width,
        "Test MAE",
        "lower is better ↓",
        b["mae"],
        a["mae"],
        f"Attention reduction: {mae_reduction:.2f}%",
    )

    svg.line(60, 754, 1530, 754, GRID_COLOR)
    svg.text(
        60,
        780,
        "Same data split and training protocol; checkpoints selected by validation loss. "
        "Positive R² change and RMSE/MAE reductions favor Attention.",
        size=12,
        fill=MUTED_COLOR,
    )
    return svg.finish()


def main() -> None:
    args = parse_args()
    baseline = load_run("Baseline", args.baseline_dir, args.baseline_log)
    attention = load_run("Attention", args.attention_dir, args.attention_log)
    validate_comparability(baseline, attention)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(baseline, attention), encoding="utf-8")

    print(f"Baseline log: {baseline.log_path}")
    print(f"Attention log: {attention.log_path}")
    print(f"Saved comparison figure: {output}")


if __name__ == "__main__":
    main()
