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
from numpy.typing import NDArray

from .curriculum import CurriculumStrategy, curriculum_arm_label
from .experiment import CHECKPOINT_CSV, read_checkpoint_csv, read_training_csv
from .metrics import canonical_start_records, mean_ci, rolling_success_rate
from .rewards import RewardMode
from .sweep import BASELINE_CURRICULUM, BudgetSeries, arm_of_row, budget_curve_series

NDArray_f = NDArray[np.float64]


def _format_budget(value: float) -> str:
    """``20000 -> "20k"``. Episode budgets, not powers of ten."""
    if value >= 1000 and value % 1000 == 0:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _budget_axis(ax: Any, budgets: NDArray_f) -> None:
    """Log-scale the x-axis and tick it at the budgets that were actually measured.

    Matplotlib's default log ticks read ``2 x 10^1``, which is the wrong vocabulary
    for an episode count. Ticking only at the measured budgets is also the honest
    labelling: nothing was sampled between them, and the curve there is drawn
    interpolation, not data.
    """
    ax.set_xscale("log")
    if budgets.size:
        ax.set_xticks(budgets)
        ax.set_xticklabels([_format_budget(b) for b in budgets], fontsize=8)
        # Room on the right for the direct labels, which sit outside the data area.
        ax.set_xlim(float(budgets.min()) / 1.3, float(budgets.max()) * 2.4)
    ax.minorticks_off()


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


#: The curriculum sweep's palette: one colour for the control arm and one per
#: sampling strategy. Deliberately disjoint from ``REWARD_COLOURS`` -- these figures
#: encode a different factor, and reusing a reward mode's hue for "no curriculum"
#: would make one colour mean two things across the artifact set.
CURRICULUM_BASELINE_COLOUR = "#7d6bb5"
CURRICULUM_TREATMENT_COLOUR = "#c9762e"
CURRICULUM_STRATEGY_COLOURS: dict[str, str] = {
    CurriculumStrategy.GROWING.value: CURRICULUM_TREATMENT_COLOUR,
    CurriculumStrategy.SLIDING.value: "#2f8f83",
    CurriculumStrategy.VISIT_WEIGHTED.value: "#b5476b",
}
#: Recessive ink for the zero rule and the target line: reference geometry, not data.
REFERENCE_INK = "#8a8a86"


def _colour(reward_mode: str) -> str:
    return REWARD_COLOURS.get(reward_mode, "#777777")


#: ``group_by`` value that splits the learning curves by experimental arm rather
#: than by a single column. The arm is a pair -- anneal fraction and sampling
#: strategy -- and grouping on the fraction alone would draw three strategies as
#: one curve, which is the pooling the whole sweep is designed to avoid.
CURRICULUM_ARM_GROUP = "curriculum_arm"


def _group_key(row: dict[str, Any], group_by: str) -> Any:
    """The series a per-seed row belongs to under ``group_by``."""
    return arm_of_row(row) if group_by == CURRICULUM_ARM_GROUP else row[group_by]


def _curriculum_colour(fraction: float, strategy: str = CurriculumStrategy.GROWING.value) -> str:
    if fraction <= BASELINE_CURRICULUM:
        return CURRICULUM_BASELINE_COLOUR
    return CURRICULUM_STRATEGY_COLOURS.get(strategy, CURRICULUM_TREATMENT_COLOUR)


def curriculum_label(fraction: float, strategy: str = CurriculumStrategy.GROWING.value) -> str:
    """Legend text for a curriculum condition.

    Thin wrapper over :func:`mars_rover_q.curriculum.curriculum_arm_label`, which
    is where the arm vocabulary lives; the figures and the analysis must not be
    able to disagree about what an arm is called.
    """
    return curriculum_arm_label(fraction, strategy)


def _finite_band(
    series: BudgetSeries,
) -> tuple[NDArray_f, NDArray_f, NDArray_f, NDArray_f]:
    """Budgets, means, and CI bounds with unusable points dropped.

    A budget whose cell produced no seeds carries ``nan``; matplotlib would draw a
    break in the line but ``fill_between`` would silently drop the band around it,
    so both are filtered together and the gap stays visible in the marker sequence.
    """
    x = np.asarray(series.budgets, dtype=np.float64)
    mean = np.asarray(series.means, dtype=np.float64)
    low = np.asarray(series.ci_low, dtype=np.float64)
    high = np.asarray(series.ci_high, dtype=np.float64)
    keep = np.isfinite(mean)
    # A single seed has a defined mean but an undefined interval; collapse the band
    # onto the line there rather than dropping the point.
    low = np.where(np.isfinite(low), low, mean)
    high = np.where(np.isfinite(high), high, mean)
    return x[keep], mean[keep], low[keep], high[keep]


def learning_curves(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    window: int = 100,
    group_by: str = "reward_mode",
) -> Path:
    """Greedy performance on the canonical mission against training episodes.

    Where checkpoint data exists (``eval_checkpoints.csv``), that is what is
    plotted: every ``budget / n`` episodes each run was paused and scored over a
    fixed batch of greedy episodes from the canonical lander start. Both conditions
    are then measured on the same mission, at the same training-episode counts,
    with the same evaluation budget, and the x-axis is literally training episodes.

    This matters most for the curriculum arm. Scoring it by its own training
    episodes is not comparable, because those episodes start mid-mission; scoring
    it by the subset that happened to start at the lander leaves a thin, high-
    epsilon sample, and indexing that subset by its position within itself
    compresses the anneal toward the origin and makes the curriculum look
    instantaneous. A checkpoint curve has none of those problems.

    Runs predating the checkpoints fall back to the trailing success rate over
    canonical-start training episodes, positioned at each episode's *true* index
    so the two conditions still share an honest x-axis.
    """
    scenarios = sorted({row["scenario"] for row in rows})
    by_curriculum = group_by == CURRICULUM_ARM_GROUP
    modes = sorted({_group_key(row, group_by) for row in rows})
    checkpointed = any((Path(row["run_dir"]) / CHECKPOINT_CSV).exists() for row in rows)
    fig, axes = plt.subplots(
        1, max(1, len(scenarios)), figsize=(5.6 * max(1, len(scenarios)), 4.2), squeeze=False
    )

    for ax, scenario in zip(axes[0], scenarios, strict=False):
        for mode in modes:
            cell = [
                r for r in rows if r["scenario"] == scenario and _group_key(r, group_by) == mode
            ]
            if not cell:
                continue
            grid, stacked = _aligned_curves(cell, window, checkpointed)
            if grid.size == 0:
                continue
            colour = _curriculum_colour(*mode) if by_curriculum else _colour(str(mode))
            label = curriculum_label(*mode) if by_curriculum else str(mode)
            # One interval per grid point, over the seeds that actually cover it.
            # The mean comes from the same place, so a point no seed reached is
            # `nan` (drawn as a gap) rather than a warning from `np.nanmean` on an
            # empty slice.
            intervals = [mean_ci(stacked[np.isfinite(stacked[:, i]), i]) for i in range(grid.size)]
            means = np.array([iv.mean for iv in intervals])
            low = np.array([iv.low if np.isfinite(iv.low) else iv.mean for iv in intervals])
            high = np.array([iv.high if np.isfinite(iv.high) else iv.mean for iv in intervals])
            ax.plot(grid, means, label=label, color=colour, linewidth=1.8)
            ax.fill_between(grid, low, high, color=colour, alpha=0.18, linewidth=0)
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("training episodes")
        ax.set_ylabel(
            "greedy success rate (canonical start)"
            if checkpointed
            else f"success rate (trailing {window})"
        )
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, loc="lower right")
    subtitle = (
        "greedy evaluation from the lander, measured during training"
        if checkpointed
        else "trailing success rate over canonical-start training episodes"
    )
    fig.suptitle(f"Learning curves across seeds, mean with 95% CI -- {subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _aligned_curves(
    cell: Sequence[dict[str, Any]], window: int, checkpointed: bool
) -> tuple[NDArray_f, NDArray_f]:
    """Per-seed curves resampled onto one shared training-episode grid.

    Seeds do not share x positions -- a fallback curve's canonical episodes land
    wherever the curriculum happened to draw them -- so each seed is interpolated
    onto a common grid before averaging. Outside a seed's measured range the value
    is ``nan`` rather than a held-constant edge, so the band narrows where seeds
    are missing instead of pretending they agreed.
    """
    series: list[tuple[NDArray_f, NDArray_f]] = []
    for row in cell:
        run_dir = Path(row["run_dir"])
        if checkpointed and (run_dir / CHECKPOINT_CSV).exists():
            points = read_checkpoint_csv(run_dir / CHECKPOINT_CSV)
            if points:
                xs = np.array([p.episode for p in points], dtype=np.float64)
                ys = np.array([p.success_rate for p in points], dtype=np.float64)
                series.append((xs, ys))
            continue
        records = canonical_start_records(read_training_csv(run_dir / "training_metrics.csv"))
        if records:
            xs = np.array([r.episode for r in records], dtype=np.float64)
            series.append((xs, rolling_success_rate(records, window)))

    if not series:
        return np.array([]), np.array([])

    highest = min(float(xs[-1]) for xs, _ in series)
    grid = np.linspace(0.0, highest, num=min(400, max(2, int(highest) + 1)))
    stacked = np.vstack([np.interp(grid, xs, ys, left=np.nan, right=np.nan) for xs, ys in series])
    return grid, stacked


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


def budget_curves(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = "eval_mean_base_return",
    ylabel: str = "mean BASE mission return (no shaping)",
    analysis: dict[str, Any] | None = None,
) -> Path:
    """The headline sweep figure: metric against training budget, one panel per scenario.

    The x-axis is log-scaled because the budget ladder is log-spaced; on a linear
    axis the 10k-to-20k gap would dominate the plot and the interesting separation
    at the cheap end would be squeezed into the left margin.

    Each condition is a line with a 95% CI band across seeds. Those bands overlapping
    does *not* mean the conditions are indistinguishable -- the arms are paired by
    seed, and :func:`paired_difference_curves` is the panel that answers that. When
    ``analysis`` is supplied, each panel also draws the shared performance target as
    a recessive rule, so the budget each curve needs to cross it is readable directly
    off the x-axis.
    """
    series = budget_curve_series(rows, metric)
    scenarios = sorted({s.scenario for s in series})
    fig, axes = plt.subplots(
        1, max(1, len(scenarios)), figsize=(5.6 * max(1, len(scenarios)), 4.3), squeeze=False
    )

    for ax, scenario in zip(axes[0], scenarios, strict=False):
        panel = sorted((s for s in series if s.scenario == scenario), key=lambda s: s.arm)
        all_budgets = (
            np.unique(np.concatenate([np.asarray(s.budgets, dtype=np.float64) for s in panel]))
            if panel
            else np.array([])
        )
        for index, entry in enumerate(panel):
            x, mean, low, high = _finite_band(entry)
            if x.size == 0:
                continue
            colour = _curriculum_colour(entry.curriculum_fraction, entry.curriculum_strategy)
            ax.plot(x, mean, label=entry.label, color=colour, linewidth=2.0, marker="o", ms=5)
            ax.fill_between(x, low, high, color=colour, alpha=0.18, linewidth=0)
            # Direct label at the right end, offset vertically per series so that two
            # conditions sitting on the same value do not print on top of each other.
            ax.annotate(
                entry.label,
                (x[-1], mean[-1]),
                textcoords="offset points",
                xytext=(8, 8 if index % 2 == 0 else -12),
                fontsize=7,
                color=colour,
                va="center",
                annotation_clip=False,
            )
        if analysis is not None:
            group = _analysis_group(analysis, scenario)
            if group is not None and np.isfinite(group["target"]):
                ax.axhline(
                    group["target"], color=REFERENCE_INK, linewidth=1.0, linestyle=":", zorder=0
                )
                ax.annotate(
                    f"target {group['target']:.0f}",
                    (0.01, group["target"]),
                    xycoords=("axes fraction", "data"),
                    textcoords="offset points",
                    xytext=(0, 4),
                    fontsize=7,
                    color=REFERENCE_INK,
                )
        _budget_axis(ax, all_budgets)
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("training episode budget")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(
        "Curriculum vs no curriculum across training budgets (mean with 95% CI)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _analysis_group(analysis: dict[str, Any], scenario: str) -> dict[str, Any] | None:
    return next((g for g in analysis.get("groups", []) if g["scenario"] == scenario), None)


def paired_difference_curves(output_path: Path, analysis: dict[str, Any]) -> Path:
    """Within-seed ``curriculum - baseline`` difference against budget, with its CI.

    One series per panel, so no legend box is needed -- the title names it. The
    zero rule is the reference the whole figure is read against: a point whose
    interval clears zero is a difference the seeds agree on, and those are drawn
    with filled markers. Hollow markers are differences the data does not
    distinguish from no effect, which is a result and not an omission.
    """
    groups = analysis.get("groups", [])
    scenarios = sorted({g["scenario"] for g in groups})
    fig, axes = plt.subplots(
        1, max(1, len(scenarios)), figsize=(5.6 * max(1, len(scenarios)), 4.1), squeeze=False
    )

    for ax, scenario in zip(axes[0], scenarios, strict=False):
        group = _analysis_group(analysis, scenario)
        entries = sorted(group["paired"], key=lambda e: e["budget"]) if group else []
        x = np.array([e["budget"] for e in entries], dtype=np.float64)
        mean = np.array([e["mean"] for e in entries], dtype=np.float64)
        low = np.array([e["ci_low"] for e in entries], dtype=np.float64)
        high = np.array([e["ci_high"] for e in entries], dtype=np.float64)
        significant = np.array([bool(e["excludes_zero"]) for e in entries], dtype=bool)

        ax.axhline(0.0, color=REFERENCE_INK, linewidth=1.2, zorder=0)
        if x.size:
            errors = np.vstack(
                [
                    np.where(np.isfinite(low), mean - low, 0.0),
                    np.where(np.isfinite(high), high - mean, 0.0),
                ]
            )
            ax.errorbar(
                x,
                mean,
                yerr=np.abs(errors),
                color=CURRICULUM_TREATMENT_COLOUR,
                linewidth=2.0,
                capsize=3,
                zorder=2,
            )
            # Fill encodes "the paired interval clears zero"; it is a second channel
            # on top of position, never the only way to read significance.
            for keep, marker_face in (
                (significant, CURRICULUM_TREATMENT_COLOUR),
                (~significant, "white"),
            ):
                if keep.any():
                    ax.plot(
                        x[keep],
                        mean[keep],
                        linestyle="none",
                        marker="o",
                        ms=7,
                        color=CURRICULUM_TREATMENT_COLOUR,
                        markerfacecolor=marker_face,
                        markeredgewidth=1.6,
                        zorder=3,
                    )
        _budget_axis(ax, x)
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("training episode budget")
        ax.set_ylabel("paired difference (curriculum - baseline)", fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Within-seed effect of the curriculum, with paired 95% CI "
        "(filled = interval excludes zero)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_experiment_plots(
    output_root: Path,
    rows: Sequence[dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
    analysis: dict[str, Any] | None = None,
) -> list[Path]:
    """Write every experiment figure into ``output_root/plots``.

    When ``analysis`` is present the grid swept budget or curriculum, so the two
    sweep figures are added and the reward-comparison bars are restricted to the
    largest budget. Those bars group by (scenario, reward mode) alone; pooling five
    budgets into one bar would average a 1k-episode agent with a 20k-episode one and
    report the result as a single condition.
    """
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    bar_rows: Sequence[dict[str, Any]] = rows
    bar_aggregated: Sequence[dict[str, Any]] = aggregated
    if analysis is not None and rows:
        largest = max(int(r["train_episodes"]) for r in rows)
        bar_rows = [r for r in rows if int(r["train_episodes"]) == largest]
        bar_aggregated = [e for e in aggregated if int(e["train_episodes"]) == largest]

    written = [
        learning_curves(
            plots_dir / "learning_curves.png",
            bar_rows,
            group_by=CURRICULUM_ARM_GROUP if analysis is not None else "reward_mode",
        ),
        comparison_bars(plots_dir / "reward_comparison.png", bar_aggregated),
        failure_modes(plots_dir / "failure_modes.png", bar_rows),
    ]
    if analysis is not None:
        written.append(budget_curves(plots_dir / "budget_curves.png", rows, analysis=analysis))
        written.append(paired_difference_curves(plots_dir / "paired_difference.png", analysis))
    return written


__all__ = [
    "CURRICULUM_ARM_GROUP",
    "CURRICULUM_BASELINE_COLOUR",
    "CURRICULUM_STRATEGY_COLOURS",
    "CURRICULUM_TREATMENT_COLOUR",
    "REWARD_COLOURS",
    "budget_curves",
    "comparison_bars",
    "failure_modes",
    "learning_curves",
    "paired_difference_curves",
    "write_experiment_plots",
]
