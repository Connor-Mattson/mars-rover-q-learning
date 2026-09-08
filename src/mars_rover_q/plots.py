"""Saved matplotlib figures for the reward-comparison experiment.

Learning curves carry uncertainty bands across seeds. Shaping reward is plotted
separately from base mission return and never mixed into it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .experiment import read_training_csv
from .metrics import mean_ci, rolling_success_rate
from .rewards import RewardMode

REWARD_COLOURS: dict[str, str] = {
    RewardMode.SPARSE.value: "#3c7dd9",
    RewardMode.NAIVE_DENSE.value: "#d95f3c",
    RewardMode.POTENTIAL.value: "#3ca66d",
}

REWARD_MARKERS: dict[str, str] = {
    RewardMode.SPARSE.value: "o",
    RewardMode.NAIVE_DENSE.value: "s",
    RewardMode.POTENTIAL.value: "^",
}


def _colour(reward_mode: str) -> str:
    return REWARD_COLOURS.get(reward_mode, "#777777")


def learning_curves(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    window: int = 100,
) -> Path:
    """Trailing success rate per scenario and reward mode, with 95% CI bands."""
    scenarios = sorted({row["scenario"] for row in rows})
    modes = sorted({row["reward_mode"] for row in rows})
    fig, axes = plt.subplots(
        1, max(1, len(scenarios)), figsize=(5.2 * max(1, len(scenarios)), 4.0), squeeze=False
    )

    for ax, scenario in zip(axes[0], scenarios, strict=False):
        for mode in modes:
            cell = [r for r in rows if r["scenario"] == scenario and r["reward_mode"] == mode]
            curves = []
            for row in cell:
                records = read_training_csv(Path(row["run_dir"]) / "training_metrics.csv")
                if records:
                    curves.append(rolling_success_rate(records, window))
            if not curves:
                continue
            length = min(len(c) for c in curves)
            stacked = np.vstack([c[:length] for c in curves])
            means = stacked.mean(axis=0)
            intervals = [mean_ci(stacked[:, i]) for i in range(length)]
            low = np.array([iv.low if np.isfinite(iv.low) else iv.mean for iv in intervals])
            high = np.array([iv.high if np.isfinite(iv.high) else iv.mean for iv in intervals])
            x = np.arange(length)
            ax.plot(x, means, label=mode, color=_colour(mode), linewidth=1.6)
            ax.fill_between(x, low, high, color=_colour(mode), alpha=0.18, linewidth=0)
        ax.set_title(scenario)
        ax.set_xlabel("training episode")
        ax.set_ylabel(f"success rate (trailing {window})")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(title="reward mode", fontsize=8)

    fig.suptitle("Learning curves across seeds (mean with 95% CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _grouped_bars(
    ax: Any,
    aggregated: Sequence[dict[str, Any]],
    metric: str,
    scenarios: Sequence[str],
    modes: Sequence[str],
) -> None:
    width = 0.8 / max(1, len(modes))
    positions = np.arange(len(scenarios))
    for offset, mode in enumerate(modes):
        means, errs = [], []
        for scenario in scenarios:
            entry = next(
                (e for e in aggregated if e["scenario"] == scenario and e["reward_mode"] == mode),
                None,
            )
            mean = float(entry[f"{metric}_mean"]) if entry else np.nan
            high = float(entry[f"{metric}_ci_high"]) if entry else np.nan
            means.append(mean)
            errs.append(high - mean if np.isfinite(high) and np.isfinite(mean) else 0.0)
        ax.bar(
            positions + offset * width - 0.4 + width / 2,
            means,
            width=width * 0.9,
            yerr=errs,
            capsize=3,
            label=mode,
            color=_colour(mode),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def comparison_bars(output_path: Path, aggregated: Sequence[dict[str, Any]]) -> Path:
    """Success rate, delivered value, base return, and loop behaviour side by side."""
    scenarios = sorted({e["scenario"] for e in aggregated})
    modes = sorted({e["reward_mode"] for e in aggregated})
    panels = (
        ("eval_success_rate", "evaluation success rate"),
        ("eval_mean_delivered_value", "mean delivered scientific value"),
        ("eval_mean_base_return", "mean BASE mission return (no shaping)"),
        ("eval_repeated_edge_fraction", "repeated-edge fraction (loop behaviour)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        _grouped_bars(ax, aggregated, metric, scenarios, modes)
        ax.set_title(title, fontsize=10)
    axes[0][0].legend(title="reward mode", fontsize=8)
    fig.suptitle("Reward-condition comparison (mean with 95% CI across seeds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def failure_modes(output_path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Stacked failure-reason counts per scenario and reward mode."""
    scenarios = sorted({row["scenario"] for row in rows})
    modes = sorted({row["reward_mode"] for row in rows})
    fig, axes = plt.subplots(
        1, max(1, len(scenarios)), figsize=(4.6 * max(1, len(scenarios)), 3.8), squeeze=False
    )
    for ax, scenario in zip(axes[0], scenarios, strict=False):
        battery, step_limit, success = [], [], []
        for mode in modes:
            cell = [r for r in rows if r["scenario"] == scenario and r["reward_mode"] == mode]
            episodes = sum(1 for _ in cell) or 1
            battery.append(sum(r["failure_battery_depleted"] for r in cell) / episodes)
            step_limit.append(sum(r["failure_step_limit"] for r in cell) / episodes)
            success.append(float(np.mean([r["eval_success_rate"] for r in cell])) if cell else 0.0)
        x = np.arange(len(modes))
        ax.bar(x, battery, label="battery depleted", color="#d95f3c")
        ax.bar(x, step_limit, bottom=battery, label="step limit", color="#f2b134")
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=8, rotation=15)
        ax.set_title(scenario, fontsize=10)
        ax.set_ylabel("failed evaluation episodes per seed")
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Failure-mode distribution (evaluation)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_experiment_plots(
    output_root: Path,
    rows: Sequence[dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
) -> list[Path]:
    """Write every experiment figure into ``output_root/plots``."""
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = [
        learning_curves(plots_dir / "learning_curves.png", rows),
        comparison_bars(plots_dir / "reward_comparison.png", aggregated),
        failure_modes(plots_dir / "failure_modes.png", rows),
    ]
    return written


__all__ = [
    "REWARD_COLOURS",
    "comparison_bars",
    "failure_modes",
    "learning_curves",
    "write_experiment_plots",
]
