"""Per-run diagnostic figures, written into ``<run>/figs``.

These answer three questions about a single training run that no learning curve
answers: *how much of the state space did this run ever write to*, *where on the
map did the experience behind that table actually come from*, and *what did the
table end up believing about each of those states*. The first two are coverage
diagnostics, not performance results -- a run can post a high success rate from
the lander while leaving nine tenths of its table untouched, and those figures are
what make that visible.

The two top-level figures count different things and one contains the other: a
state is *reached* when an update was applied to it, and *learned* when that
update actually moved a value. Under a sparse reward most updates carry a zero TD
error, so a cell can collect thousands of visits and still leave every one of its
states at ``initial_q``. Reached is always the superset; neither figure should be
read as the other.

The third figure is a set, not a single image. Both top-level maps sum battery
away, which is the only way to fit the table on one page but also throws away the
axis the mission actually turns on -- the same cell is a good place to stand with
a full battery and a fatal one with eight percent left. ``figs/q_by_battery/``
holds one map per battery *level* with nothing marginalised: every panel is a table
row, painted by its learned value, labelled with the update count behind that value,
and arrowed with the action that value came from. Under the dense encoding a level
is one exact battery reading; under the binned one it is the range of readings that
share a row, which each frame names. Every frame in the set shares one colour scale,
so the frames can be flipped through as a sequence and compared to each other.

Figures are drawn from the finished table and the recorded visit counts, so they
describe the run that happened; nothing here re-steps the environment. Like
:mod:`mars_rover_q.plots` and :mod:`mars_rover_q.animation` this is reporting
machinery -- the learning path never imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    TwoSlopeNorm,
    to_rgb,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from numpy.typing import NDArray

from .actions import MOVE_ACTIONS, Action
from .metrics import greedy_actions, learned_state_mask
from .plots import REFERENCE_INK
from .scenario import Scenario, Terrain
from .state import (
    COLLECTABLE_SAMPLES,
    NUM_PAYLOAD_STATES,
    SAMPLE_LABELS,
    BatteryBinning,
    SampleType,
    StateEncoder,
)

FIGS_DIRNAME = "figs"
COVERAGE_FIG = "state_coverage.png"
EXPERIENCE_FIG = "experience_heatmaps.png"
BATTERY_FIGS_DIRNAME = "q_by_battery"
BATTERY_FIG_STEM = "battery"

#: The sequential blue ramp, light -> dark, steps 100-700. One hue: these figures
#: encode magnitude, and a magnitude scale that changes hue invites the reader to
#: look for a category boundary that is not there.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

#: The diverging ramp for learned values, dark warm -> pale -> dark blue, with the
#: pale step landing exactly on zero. Q-values have a real, meaningful midpoint --
#: zero is both the table's initial value and the boundary between "this state is
#: worth being in" and "this state is a liability" -- and a sequential ramp would
#: put that boundary at an arbitrary point in the middle of one hue. Blue against
#: orange is also the safest diverging pair to hand a red-green colourblind reader,
#: and it keeps the positive arm in the same hue the coverage figures use.
DIVERGING_Q: tuple[str, ...] = (
    "#7a3508",
    "#9c4a0f",
    "#bc6019",
    "#d47a30",
    "#e79f5e",
    "#f0bf90",
    "#f7dcc0",
    "#f7f4ef",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

#: Reserved for "not data": a wall the rover can never occupy, and a state the
#: trainer never reached. Both are grey so that no amount of blue is ever
#: confused for either, and they differ from each other in lightness as well as
#: in the legend.
WALL_INK = REFERENCE_INK
UNTOUCHED_INK = "#f0efec"

TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#52514e"

#: Letter cues for the mission landmarks, matching the Pygame renderer's glyphs so
#: the same cell is called the same thing in a figure and in a replay.
LANDER_GLYPH = "L"
SAMPLE_GLYPHS: dict[SampleType, str] = {
    SampleType.BASALT: "B",
    SampleType.HYDRATED_MINERAL: "H",
    SampleType.BIOSIGNATURE: "X",
}

#: The greedy action as a single character. Arrows rather than the renderer's
#: ``N``/``S``/``E``/``W`` letters: a static map is read by scanning for the shape of
#: a route, and an arrow points where the rover would go without being decoded first.
#: ``COLLECT`` is not a direction and does not get one.
ACTION_ARROWS: dict[Action, str] = {
    Action.NORTH: "\u2191",
    Action.SOUTH: "\u2193",
    Action.EAST: "\u2192",
    Action.WEST: "\u2190",
    Action.COLLECT: "O",
}

PAYLOAD_TITLES: dict[SampleType, str] = {
    SampleType.NONE: "Empty bay",
    SampleType.BASALT: "Carrying basalt core",
    SampleType.HYDRATED_MINERAL: "Carrying hydrated mineral",
    SampleType.BIOSIGNATURE: "Carrying biosignature candidate",
}

PAYLOAD_BAR_LABELS: dict[SampleType, str] = {
    SampleType.NONE: "empty bay",
    SampleType.BASALT: "basalt core",
    SampleType.HYDRATED_MINERAL: "hydrated mineral",
    SampleType.BIOSIGNATURE: "biosignature",
}


def blue_ramp() -> LinearSegmentedColormap:
    """The sequential ramp as a matplotlib colormap."""
    return LinearSegmentedColormap.from_list("mars_sequential_blue", list(SEQUENTIAL_BLUE))


def diverging_q_ramp() -> LinearSegmentedColormap:
    """The diverging value ramp as a matplotlib colormap, zero at its midpoint."""
    return LinearSegmentedColormap.from_list("mars_diverging_q", list(DIVERGING_Q))


def plural(count: float, noun: str) -> str:
    """``1 updates`` reads as a bug; ``1 update`` does not."""
    return noun[:-1] if count == 1 and noun.endswith("s") else noun


def compact_count(value: float) -> str:
    """``1_904_221 -> "1.9M"``. Cell labels have room for four characters."""
    number = float(value)
    if number >= 1_000_000:
        return f"{number / 1e6:.1f}M"
    if number >= 10_000:
        return f"{number / 1e3:.0f}k"
    if number >= 1_000:
        return f"{number / 1e3:.1f}k"
    return f"{number:.0f}"


def compact_value(value: float) -> str:
    """``-83.2227 -> "-83.2"``. Significant digits scaled to the magnitude.

    Two decimals on a value of 160 is five characters of noise in a cell that has
    room for five characters total, and rounding a value of 0.04 to ``0.0`` would
    paint a state the run genuinely learned something about as one it did not.
    """
    number = float(value)
    magnitude = abs(number)
    if magnitude >= 100:
        return f"{number:.0f}"
    if magnitude >= 10:
        return f"{number:.1f}"
    if magnitude >= 0.005:
        return f"{number:.2f}"
    # A value this small is a state the run brushed against without learning
    # anything measurable, which is worth telling apart from a state still holding
    # exactly the value the table started with -- but not at the price of an
    # exponent in a label five characters wide.
    return "0" if number == 0.0 else "~0"


@dataclass(frozen=True, slots=True)
class PayloadCoverage:
    """Learned-state counts for one payload slice of the table."""

    payload: SampleType
    learned: int
    total: int
    #: States in this slice the rover can actually be in with a move still to make.
    #: Varies by payload: the further the carried sample is from the lander, the
    #: more of the battery axis is already spent by the time the bay holds it.
    reachable: int

    @property
    def fraction(self) -> float:
        """Share of this slice's states carrying a learned value."""
        return self.learned / self.total if self.total else 0.0

    @property
    def reachable_fraction(self) -> float:
        """Share of this slice's occupiable states carrying a learned value."""
        return self.learned / self.reachable if self.reachable else 0.0


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Whole-table coverage plus its per-payload breakdown."""

    total: int
    learned: int
    reachable: int
    initial_q: float
    by_payload: tuple[PayloadCoverage, ...]

    @property
    def fraction(self) -> float:
        """Share of all encodable states carrying a learned value."""
        return self.learned / self.total if self.total else 0.0

    @property
    def reachable_fraction(self) -> float:
        """Share of the states the rover can occupy carrying a learned value."""
        return self.learned / self.reachable if self.reachable else 0.0


def coverage_report(
    scenario: Scenario,
    q_table: NDArray[np.float64],
    *,
    initial_q: float = 0.0,
    binning: BatteryBinning | None = None,
) -> CoverageReport:
    """Count the states this run wrote to, whole-table and per payload.

    ``reachable`` is :func:`occupiable_state_mask`: the states the rover can be in
    with a move still to make. It is the ceiling a run could conceivably reach, and
    it matters twice over. A third of ``safe_corridor``'s grid is wall, so coverage
    against every encodable state understates the run by exactly that third. And
    the ceiling is *not* the same for all four payload slices -- carrying a sample
    means having already paid for the trip out to it, so the charge left is capped
    by how far that sample is from the lander. Both denominators are reported
    rather than picking one.
    """
    encoder = _encoder_for(scenario, binning)
    mask = learned_state_mask(q_table, initial_q)
    learned_grid = encoder.grid_view(mask)
    occupiable = occupiable_state_mask(scenario, encoder)
    total_per_payload = encoder.rows * encoder.cols * encoder.battery_levels

    by_payload = tuple(
        PayloadCoverage(
            payload=SampleType(index),
            learned=int(learned_grid[:, :, :, index].sum()),
            total=total_per_payload,
            reachable=int(occupiable[:, :, :, index].sum()),
        )
        for index in range(NUM_PAYLOAD_STATES)
    )
    return CoverageReport(
        total=encoder.num_states,
        learned=int(mask.sum()),
        reachable=int(occupiable.sum()),
        initial_q=initial_q,
        by_payload=by_payload,
    )


@dataclass(frozen=True, slots=True)
class ExperienceField:
    """The per-cell, per-payload quantity the map figure paints.

    ``counts`` is ``(rows, cols, payloads)`` with battery already summed away.
    ``log`` says whether the colour scale is logarithmic, which is the right
    default for visit counts: an empty-bay cell beside the lander collects
    hundreds of thousands of updates while a far corner collects a handful, and on
    a linear scale every panel but the first would be blank.
    """

    counts: NDArray[np.float64]
    scale_label: str
    cell_noun: str
    log: bool


def visit_field(
    scenario: Scenario,
    visit_counts: NDArray[np.int64],
    *,
    binning: BatteryBinning | None = None,
) -> ExperienceField:
    """Q-updates per cell and payload, battery summed away."""
    grid = _encoder_for(scenario, binning).grid_view(np.asarray(visit_counts))
    return ExperienceField(
        counts=np.asarray(grid.sum(axis=2), dtype=np.float64),
        scale_label="Q-updates made in this cell (log scale)",
        cell_noun="updates",
        log=True,
    )


def learned_battery_field(
    scenario: Scenario,
    q_table: NDArray[np.float64],
    *,
    initial_q: float = 0.0,
    binning: BatteryBinning | None = None,
) -> ExperienceField:
    """Fallback for runs saved before visit counts were recorded.

    Counts, per cell and payload, how many of the battery levels hold a learned
    value. It is a coarser picture than the visit counts -- it saturates at the
    battery level count and cannot tell one update from ten thousand -- so it is
    labelled differently rather than passed off as the same measurement.
    """
    encoder = _encoder_for(scenario, binning)
    grid = encoder.grid_view(learned_state_mask(q_table, initial_q))
    return ExperienceField(
        counts=np.asarray(grid.sum(axis=2), dtype=np.float64),
        scale_label=(
            f"battery levels in this cell holding a learned value (of {encoder.battery_levels})"
        ),
        cell_noun="learned levels",
        log=False,
    )


@dataclass(frozen=True, slots=True)
class QValueField:
    """The finished table as a map, with battery kept as an axis rather than summed.

    Every array is ``(rows, cols, battery_levels, payloads)``, indexed exactly the
    way :meth:`StateEncoder.grid_view` lays the table out, so one index into all
    three names a single fully specified state -- no marginalisation, nothing
    averaged.

    ``values`` is the greedy value of each state, ``max_a Q(s, a)``: the one number
    the table is actually asked for at decision time, and the one a reader means by
    "what did it learn about this square". ``actions`` is the action that value came
    from, and ``updates`` is how many Q-updates landed on that state -- the evidence
    behind the value, and the difference between a considered estimate and a single
    lucky backup. ``reached`` marks the states the trainer touched at all;
    ``has_counts`` is false for runs saved before visit counts were recorded, where
    ``updates`` is all zero and must not be shown.

    ``decided`` marks the states whose best action is *unique*. It is the only
    condition under which ``actions`` says anything: where two or more actions share
    the maximum, ``argmax`` returns the lowest-numbered of them, and drawing that as
    the policy would invent a preference the table does not hold. This is stricter
    than :func:`mars_rover_q.metrics.tied_state_fraction`, which counts only rows
    that are flat all the way across -- a row can be split at the top and still be
    undecided about where to drive.
    """

    values: NDArray[np.float64]
    actions: NDArray[np.int64]
    updates: NDArray[np.float64]
    reached: NDArray[np.bool_]
    decided: NDArray[np.bool_]
    has_counts: bool
    #: How axis 2 maps onto real battery readings. Under the dense encoding level
    #: ``k`` *is* battery ``k``; under a coarser one it is a range, and every label
    #: on these figures has to say which.
    binning: BatteryBinning

    @property
    def battery_levels(self) -> int:
        """Number of battery bins, i.e. how many frames the set holds."""
        return int(self.values.shape[2])


def q_value_field(
    scenario: Scenario,
    q_table: NDArray[np.float64],
    *,
    visit_counts: NDArray[np.int64] | None = None,
    initial_q: float = 0.0,
    binning: BatteryBinning | None = None,
) -> QValueField:
    """Greedy values and update counts per fully specified state.

    ``reached`` prefers the visit counts, which record every update including the
    ones that moved nothing. Without them it falls back to the learned mask, which
    is a strictly smaller set -- a state updated only with a zero TD error is
    indistinguishable from one never visited once the counts are gone -- so the
    fallback under-reports coverage rather than guessing at it.
    """
    encoder = _encoder_for(scenario, binning)
    table = np.asarray(q_table, dtype=np.float64)
    best = table.max(axis=1)
    values = encoder.grid_view(best)
    actions = encoder.grid_view(greedy_actions(table))
    decided = encoder.grid_view((table == best[:, None]).sum(axis=1) == 1)

    counts = None if visit_counts is None or not visit_counts.size else np.asarray(visit_counts)
    if counts is None:
        updates = np.zeros_like(values)
        reached = encoder.grid_view(learned_state_mask(q_table, initial_q))
    else:
        updates = np.asarray(encoder.grid_view(counts), dtype=np.float64)
        reached = updates > 0

    return QValueField(
        values=np.asarray(values, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.int64),
        updates=updates,
        reached=np.asarray(reached, dtype=np.bool_),
        decided=np.asarray(decided, dtype=np.bool_),
        has_counts=counts is not None,
        binning=encoder.binning,
    )


def q_value_norm(field: QValueField) -> TwoSlopeNorm:
    """One colour scale for the whole battery set, pinned to zero at its midpoint.

    Derived once from every reached state rather than per frame, which is the
    property that makes the frames a sequence instead of sixty-one unrelated
    pictures: a cell that darkens as the battery drains has to mean the value fell,
    and it cannot mean that if each frame renormalises to its own extremes.

    The two arms are scaled independently. A run whose losses bottom out at -83
    while its wins reach +160 would spend half a symmetric scale on values no state
    holds, so the negative arm is stretched to its own minimum and the colour bar is
    ticked at both ends and at zero to say so.
    """
    reached = field.values[field.reached]
    lowest = float(reached.min()) if reached.size else 0.0
    highest = float(reached.max()) if reached.size else 0.0
    # TwoSlopeNorm requires vmin < 0 < vmax strictly, and a table that learned
    # nothing -- or learned only in one direction -- satisfies neither.
    margin = max(abs(lowest), abs(highest), 1.0) * 0.02
    return TwoSlopeNorm(vmin=min(lowest, -margin), vcenter=0.0, vmax=max(highest, margin))


def battery_q_figure(
    output_path: Path,
    scenario: Scenario,
    field: QValueField,
    level: int,
    *,
    norm: TwoSlopeNorm,
    subtitle: str = "",
) -> Path:
    """One frame of the battery set: the whole map at a single battery level.

    Nothing on this page is summed or averaged over the map. Each cell is one table
    row, and its two labels are that row's greedy value and the number of updates
    that produced it. Under the dense encoding a row is one exact state; under a
    binned one it is every state in the frame's battery range, which the title and
    the gauge both name.
    """
    ramp = diverging_q_ramp()
    walls = ~_traversable_mask(scenario)
    binning = field.binning
    capacity = binning.capacity
    low, high = binning.span(level)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.0))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.818, bottom=0.155, wspace=0.10, hspace=0.16)

    payloads = (SampleType.NONE, *COLLECTABLE_SAMPLES)
    for axis, payload in zip(axes.flat, payloads, strict=True):
        _draw_q_panel(axis, scenario, field, payload, level, walls=walls, norm=norm, ramp=ramp)

    fig.text(
        0.05,
        0.962,
        f"What the table learned at battery {binning.label(level)} of {capacity}",
        fontsize=16,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    if subtitle:
        fig.text(0.05, 0.938, subtitle, fontsize=9.5, color=TEXT_SECONDARY)
    # Two fixed lines, wrapped by hand. Matplotlib's ``wrap`` breaks on the canvas
    # width rather than on the text's own left margin, which puts the break past the
    # right edge of every panel underneath it.
    second_line = (
        "The number in parentheses is how many Q-updates produced that value. "
        f"All {field.battery_levels} frames share this colour scale."
        if field.has_counts
        else "This run recorded no visit counts, so the updates behind each value "
        f"are not shown. All {field.battery_levels} frames share this colour scale."
    )
    first_line = (
        "Each cell is one exact state, painted by the value of its best action; "
        "the arrow is that action."
        if binning.is_dense
        else f"Each cell is one table row -- every battery reading from {low} to "
        f"{high} shares it -- painted by the value of its best action; the arrow "
        "is that action."
    )
    fig.text(0.05, 0.914, first_line, fontsize=9.5, color=TEXT_SECONDARY)
    fig.text(0.05, 0.896, second_line, fontsize=9.5, color=TEXT_SECONDARY)

    _draw_battery_gauge(fig, low, high, capacity)

    glyph_note = "     ".join(
        [f"{LANDER_GLYPH}  lander"]
        + [f"{SAMPLE_GLYPHS[sample]}  {SAMPLE_LABELS[sample]}" for sample in COLLECTABLE_SAMPLES]
    )
    fig.text(0.5, 0.130, glyph_note, fontsize=9, color=TEXT_SECONDARY, ha="center")
    arrows = " ".join(ACTION_ARROWS[move] for move in MOVE_ACTIONS)
    fig.text(
        0.5,
        0.110,
        f"{arrows}  drive     {ACTION_ARROWS[Action.COLLECT]}  collect     "
        "no arrow  ·  two or more actions tied at the top, so the greedy pick is a "
        "coin toss",
        fontsize=9,
        color=TEXT_SECONDARY,
        ha="center",
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=UNTOUCHED_INK,
                edgecolor="#c9c7c1",
                hatch="///",
                label="never updated at this battery level",
            ),
            Patch(facecolor=WALL_INK, label="wall"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.072),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    # Ticked at both ends and at zero rather than at even intervals: with the arms
    # scaled independently an evenly spaced tick sequence would imply a linearity
    # the scale does not have.
    lowest, highest = _norm_bounds(norm)
    bar_axis = fig.add_axes((0.22, 0.038, 0.56, 0.015))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=ramp),
        cax=bar_axis,
        orientation="horizontal",
        ticks=[lowest, lowest / 2, 0.0, highest / 2, highest],
    )
    colorbar.set_label(
        "Value of this state's best action  ·  arms scaled independently about zero",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        labelpad=6,
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5, length=3, colors=TEXT_SECONDARY)
    colorbar.ax.xaxis.set_major_formatter(lambda value, _pos: compact_value(value))

    return _save(fig, output_path)


def write_battery_q_figures(
    figs_dir: Path,
    scenario: Scenario,
    field: QValueField,
    *,
    subtitle: str = "",
) -> list[Path]:
    """Write one value map per battery level into ``figs_dir/q_by_battery``.

    The frames are named with a zero-padded *level* index so that a directory
    listing, an image viewer's arrow keys, and ``ffmpeg`` all walk them in the order
    the battery actually drains. Under the dense encoding the level is the battery
    reading and the names are unchanged; under a binned one it is the bin index, and
    the reading range it covers is on the frame itself rather than in the filename,
    which has to stay sortable.
    """
    battery_dir = Path(figs_dir) / BATTERY_FIGS_DIRNAME
    battery_dir.mkdir(parents=True, exist_ok=True)

    norm = q_value_norm(field)
    width = len(str(field.battery_levels - 1))
    return [
        battery_q_figure(
            battery_dir / f"{BATTERY_FIG_STEM}_{level:0{width}d}.png",
            scenario,
            field,
            level,
            norm=norm,
            subtitle=subtitle,
        )
        for level in range(field.battery_levels)
    ]


def state_coverage_figure(output_path: Path, report: CoverageReport, *, subtitle: str = "") -> Path:
    """Hero coverage number over a bullet bar per payload.

    The bars are one measure across four categories, so they take one hue, not
    four; the tick behind each bar is the reachable ceiling, which is what makes a
    short bar readable as "the run never got there" rather than "the map is small".
    """
    ramp = blue_ramp()
    bar_colour = ramp(0.72)
    track_colour = UNTOUCHED_INK

    fig = plt.figure(figsize=(10.0, 6.2))
    fig.patch.set_facecolor("white")

    fig.text(
        0.06, 0.94, "Learned state coverage", fontsize=16, fontweight="bold", color=TEXT_PRIMARY
    )
    if subtitle:
        fig.text(0.06, 0.895, subtitle, fontsize=9.5, color=TEXT_SECONDARY)

    fig.text(
        0.06,
        0.775,
        f"{report.learned:,} of {report.total:,} states  ({report.fraction:.1%})",
        fontsize=22,
        fontweight="bold",
        color=bar_colour,
    )
    fig.text(
        0.06,
        0.725,
        f"hold a value this run wrote. Against the {report.reachable:,} states the rover can "
        f"actually occupy, {report.reachable_fraction:.1%}.",
        fontsize=10,
        color=TEXT_SECONDARY,
    )
    fig.text(
        0.06,
        0.685,
        "A state counts as learned when at least one of its five action values has moved off "
        f"the initial value of {report.initial_q:g}.",
        fontsize=9,
        color=TEXT_SECONDARY,
    )

    ax = fig.add_axes((0.22, 0.155, 0.64, 0.45))
    rows = list(report.by_payload)
    positions = np.arange(len(rows))[::-1]

    for position, row in zip(positions, rows, strict=True):
        ax.barh(position, row.total, height=0.62, color=track_colour, zorder=1)
        ax.barh(position, row.learned, height=0.62, color=bar_colour, zorder=3)
        ax.plot(
            [row.reachable, row.reachable],
            [position - 0.36, position + 0.36],
            color=REFERENCE_INK,
            linewidth=1.6,
            zorder=4,
        )
        ax.text(
            row.total * 1.02,
            position,
            f"{row.learned:,}  ({row.reachable_fraction:.1%} of reachable)",
            va="center",
            ha="left",
            fontsize=9.5,
            color=TEXT_PRIMARY,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([PAYLOAD_BAR_LABELS[row.payload] for row in rows], fontsize=10)
    slice_total = rows[0].total if rows else 1
    ax.set_xlim(0, slice_total * 1.30)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("states in the payload slice", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_title(
        "By payload in the sample bay", fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10
    )
    # The axis is padded past the track only to make room for the direct labels;
    # ticking that padding would advertise a range no bar can occupy.
    ax.set_xticks([tick for tick in ax.get_xticks() if 0 <= tick <= slice_total])
    ax.spines["bottom"].set_bounds(0, slice_total)
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:,.0f}")
    _recede_axes(ax)
    ax.tick_params(axis="y", length=0)

    fig.legend(
        handles=[
            Patch(facecolor=bar_colour, label="learned"),
            Patch(facecolor=track_colour, label="never written"),
            Line2D([], [], color=REFERENCE_INK, linewidth=1.6, label="reachable ceiling"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.54, 0.015),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    return _save(fig, output_path)


def experience_heatmap_figure(
    output_path: Path,
    scenario: Scenario,
    field: ExperienceField,
    *,
    subtitle: str = "",
) -> Path:
    """One map panel per payload, on a colour scale shared across all four.

    Sharing the scale is the whole point of the figure: the empty-bay panel
    outweighs the three carrying panels by orders of magnitude, and four
    independently scaled panels would hide exactly that. The log scale is what
    keeps the carrying panels legible under a shared maximum.
    """
    ramp = blue_ramp()
    walls = ~_traversable_mask(scenario)
    highest = float(field.counts.max()) if field.counts.size else 0.0
    norm: Normalize = (
        LogNorm(vmin=1.0, vmax=max(highest, 10.0))
        if field.log
        # Not vmin=0: zero is painted as "never reached", so letting the ramp's
        # lightest step mean zero would spend the most legible end of the scale on
        # a value no cell is ever drawn with.
        else Normalize(vmin=1.0, vmax=max(highest, 2.0))
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.0))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.855, bottom=0.155, wspace=0.10, hspace=0.16)

    payloads = (SampleType.NONE, *COLLECTABLE_SAMPLES)
    for axis, payload in zip(axes.flat, payloads, strict=True):
        _draw_panel(axis, scenario, field, payload, walls=walls, norm=norm, ramp=ramp)

    fig.text(
        0.05,
        0.955,
        "Where the experience came from",
        fontsize=16,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    if subtitle:
        fig.text(0.05, 0.928, subtitle, fontsize=9.5, color=TEXT_SECONDARY)
    fig.text(
        0.05,
        0.902,
        "One panel per payload in the sample bay, battery level summed away. "
        "All four panels share the colour scale below.",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )

    glyph_note = "     ".join(
        [f"{LANDER_GLYPH}  lander"]
        + [f"{SAMPLE_GLYPHS[sample]}  {SAMPLE_LABELS[sample]}" for sample in COLLECTABLE_SAMPLES]
    )
    fig.text(0.5, 0.118, glyph_note, fontsize=9, color=TEXT_SECONDARY, ha="center")
    fig.legend(
        handles=[
            Patch(facecolor=UNTOUCHED_INK, edgecolor="#dcdad4", label="never reached"),
            Patch(facecolor=WALL_INK, label="wall"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.083),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    bar_axis = fig.add_axes((0.22, 0.038, 0.56, 0.015))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=ramp), cax=bar_axis, orientation="horizontal"
    )
    colorbar.set_label(field.scale_label, fontsize=9.5, color=TEXT_SECONDARY, labelpad=6)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5, length=3, colors=TEXT_SECONDARY)
    # A log axis defaults to 10^n labels; these are counts, and a reader of a count
    # should not have to evaluate an exponent to read the legend.
    colorbar.ax.xaxis.set_major_formatter(lambda value, _pos: compact_count(value))
    colorbar.ax.xaxis.set_minor_formatter(lambda _value, _pos: "")

    return _save(fig, output_path)


def write_run_figures(
    run_dir: Path,
    scenario: Scenario,
    q_table: NDArray[np.float64],
    *,
    visit_counts: NDArray[np.int64] | None = None,
    initial_q: float = 0.0,
    subtitle: str = "",
    battery_frames: bool = True,
    binning: BatteryBinning | None = None,
) -> list[Path]:
    """Write every per-run figure into ``run_dir/figs`` and return their paths.

    The two summary figures land at the top of ``figs``; the per-battery value maps
    land in the ``q_by_battery`` subdirectory beneath it, one frame per battery
    reading. The returned list is in that order, so a caller that wants to report
    the summaries separately from the set can slice it at
    :data:`BATTERY_FIGS_DIRNAME`.

    ``visit_counts`` is optional so that a run saved before they were recorded
    still gets every figure; the map then paints learned battery levels per cell
    and says so on its colour bar rather than silently plotting a different
    quantity under the same label, and the battery frames drop their update counts
    rather than printing a zero they cannot support.

    ``battery_frames`` exists because the set is one figure per battery level and
    costs proportionally more than the two summaries put together; a caller redrawing
    figures in a loop over many runs may not want it every time.

    ``binning`` is the battery axis ``q_table`` was written under, and every figure
    here reads the table through it. It defaults to dense because that is what a run
    with nothing recorded about its encoding is; callers holding a manifest should
    pass the run's own.
    """
    figs_dir = Path(run_dir) / FIGS_DIRNAME
    figs_dir.mkdir(parents=True, exist_ok=True)

    report = coverage_report(scenario, q_table, initial_q=initial_q, binning=binning)
    field = (
        visit_field(scenario, visit_counts, binning=binning)
        if visit_counts is not None and visit_counts.size
        else learned_battery_field(scenario, q_table, initial_q=initial_q, binning=binning)
    )
    paths = [
        state_coverage_figure(figs_dir / COVERAGE_FIG, report, subtitle=subtitle),
        experience_heatmap_figure(figs_dir / EXPERIENCE_FIG, scenario, field, subtitle=subtitle),
    ]
    if battery_frames:
        values = q_value_field(
            scenario,
            q_table,
            visit_counts=visit_counts,
            initial_q=initial_q,
            binning=binning,
        )
        paths.extend(write_battery_q_figures(figs_dir, scenario, values, subtitle=subtitle))
    return paths


def run_subtitle(config: dict[str, Any]) -> str:
    """The one-line condition summary printed under each figure's title."""
    parts = [
        str(config.get("scenario", "?")),
        f"{config.get('reward_mode', '?')} reward",
        f"{int(config.get('episodes', 0)):,} episodes",
        f"seed {config.get('seed', '?')}",
    ]
    fraction = float(config.get("curriculum_fraction", 0.0) or 0.0)
    parts.append("no curriculum" if fraction <= 0.0 else f"curriculum {fraction:g}")
    return "  ·  ".join(parts)


# -- internals ------------------------------------------------------------


def _encoder_for(scenario: Scenario, binning: BatteryBinning | None = None) -> StateEncoder:
    """The encoder a figure has to read a saved table through.

    ``binning`` defaults to dense rather than to the current training default: these
    functions are handed tables loaded from disk, and a table's battery axis is a
    property of the run that wrote it, never of the version drawing it.
    """
    return StateEncoder(scenario.rows, scenario.cols, scenario.battery_capacity, binning)


def _traversable_mask(scenario: Scenario) -> NDArray[np.bool_]:
    """``(rows, cols)`` boolean: which cells are not wall."""
    return np.asarray(scenario.grid != int(Terrain.WALL), dtype=np.bool_)


def occupiable_state_mask(scenario: Scenario, encoder: StateEncoder) -> NDArray[np.bool_]:
    """``(rows, cols, battery_levels, payloads)`` boolean: which states can be acted from.

    The ceiling every coverage figure is measured against. A state qualifies only
    when the rover could actually be sitting in it with a move still to make, which
    is the same three-part test :meth:`MarsRoverEnv._validated_start` applies to an
    injected start:

    * **The cell is not wall.**
    * **The battery arithmetic works out.** The rover leaves the lander with a full
      charge and pays the energy cost of every tile it enters, so the charge missing
      from a state is at least the cheapest route that explains it -- straight from
      the lander with an empty bay, and via the carried sample's own cell plus
      ``collect_energy_cost`` when the bay is full. A battery level is occupiable
      when *some* reading in its bin is payable that way. Wandering can always spend
      more; nothing can spend less.
    * **The state is not already terminal.** A flat battery ends the episode, and so
      does arriving on the lander with a sample aboard -- that is the delivery. Both
      are entered and never acted from, so no Q-update ever writes their rows.

    Note:
        This is per payload, and deliberately so. Sharing one traversable-cell count
        across all four slices treats a state the map makes impossible -- a full
        battery while carrying a sample fetched from the far end of the corridor --
        as a state the run failed to reach, which reads on the figure as a
        curriculum that never got to the valuable payloads. Distances are the static
        Dijkstra fields, so slip is ignored: this is a bound, not a prediction.
    """
    binning = encoder.binning
    capacity = scenario.battery_capacity
    mask = np.zeros(
        (encoder.rows, encoder.cols, encoder.battery_levels, NUM_PAYLOAD_STATES), dtype=np.bool_
    )
    for payload in (SampleType.NONE, *COLLECTABLE_SAMPLES):
        for cell in scenario.traversable_cells():
            if payload is SampleType.NONE:
                spent = scenario.distance(scenario.lander, cell)
            elif cell == scenario.lander:
                continue  # carrying a sample onto the lander is the delivery
            else:
                sample_cell = scenario.samples[payload].position
                spent = (
                    scenario.distance(scenario.lander, sample_cell)
                    + scenario.collect_energy_cost
                    + scenario.distance(sample_cell, cell)
                )
            if not isfinite(spent) or spent > capacity - 1:
                continue
            highest = int(capacity - spent)
            for level in range(encoder.battery_levels):
                low, high = binning.span(level)
                if low <= highest and high >= 1:
                    mask[cell[0], cell[1], level, int(payload)] = True
    return mask


def _recede_axes(axis: Any) -> None:
    """Grid and spines are reference geometry; keep them behind the data."""
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#dcdad4")
    axis.tick_params(labelsize=8.5, colors=TEXT_SECONDARY, length=3)
    axis.set_axisbelow(True)


def _save(figure: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path


def _draw_panel(
    axis: Any,
    scenario: Scenario,
    field: ExperienceField,
    payload: SampleType,
    *,
    walls: NDArray[np.bool_],
    norm: Normalize,
    ramp: LinearSegmentedColormap,
) -> None:
    """Paint one payload's map, its landmarks, and its per-cell labels."""
    counts = field.counts[:, :, int(payload)]
    rows, cols = counts.shape

    # Built as explicit RGB rather than a masked array so that "wall" and "never
    # reached" are chosen colours rather than whatever the colormap does at its
    # under-range end, where the lightest blue step is nearly the surface itself.
    image = np.empty((rows, cols, 3), dtype=np.float64)
    image[:] = to_rgb(UNTOUCHED_INK)
    touched = counts > 0
    if touched.any():
        image[touched] = ramp(norm(counts[touched]))[:, :3]
    image[walls] = to_rgb(WALL_INK)

    axis.imshow(image, interpolation="nearest", aspect="equal")
    _draw_cell_grid(axis, rows, cols)

    for row in range(rows):
        for col in range(cols):
            if walls[row, col]:
                continue
            ink = _ink_on(image[row, col])
            if counts[row, col] > 0:
                axis.text(
                    col,
                    row + 0.16,
                    compact_count(counts[row, col]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=ink,
                )
            glyph = _glyph_at(scenario, (row, col))
            if glyph:
                axis.text(
                    col - 0.36,
                    row - 0.30,
                    glyph,
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=ink,
                )

    reached = int(touched.sum())
    walkable = int((~walls).sum())
    axis.set_title(
        f"{PAYLOAD_TITLES[payload]}\n"
        f"{reached} of {walkable} cells reached  ·  "
        f"{compact_count(counts.sum())} {plural(counts.sum(), field.cell_noun)}",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        pad=8,
    )


def _value_span(values: NDArray[np.float64]) -> str:
    """The value range in a panel, phrased as a range only when it is one.

    ``values 0 to 0`` reads as a broken axis; ``every value 0`` reads as the finding
    it actually is -- a payload slice the run reached but learned nothing in.
    """
    lowest = compact_value(float(values.min()))
    highest = compact_value(float(values.max()))
    return f"every value {lowest}" if lowest == highest else f"values {lowest} to {highest}"


def _norm_bounds(norm: Normalize) -> tuple[float, float]:
    """A norm's explicit limits as plain floats; both are always set by construction."""
    return (
        float(norm.vmin if norm.vmin is not None else 0.0),
        float(norm.vmax if norm.vmax is not None else 0.0),
    )


def _draw_battery_gauge(figure: Any, low: int, high: int, capacity: int) -> None:
    """A slim charge bar under the header, so a frame is placeable at a glance.

    The frames are meant to be flipped through, and a reader three frames deep has
    lost track of where in the drain they are; a title number alone does not restore
    that, a bar filled to the same fraction the rover has left does.

    A frame covers a *range* of readings -- one reading wide under the dense
    encoding, wider under a binned one -- so the bar is filled solid to ``low``, the
    charge every state in the frame is guaranteed, and continues in a lighter tone
    to ``high``. Drawing the bin at its top instead would show a charge most of its
    states do not have.
    """
    # Deliberately short of the page width: a full battery drawn edge to edge reads
    # as a rule under the header rather than as a bar filled to its end.
    axis = figure.add_axes((0.05, 0.874, 0.34, 0.009))
    axis.barh(0, capacity, height=1.0, color=UNTOUCHED_INK)
    if high > low:
        axis.barh(0, high, height=1.0, color=TEXT_SECONDARY, alpha=0.35)
    # Not the wall grey: a bar drawn in the same ink as the map's walls invites the
    # reader to look for a relationship between the two, and there is none.
    axis.barh(0, low, height=1.0, color=TEXT_SECONDARY)
    axis.set_xlim(0, capacity)
    axis.set_ylim(-0.5, 0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_q_panel(
    axis: Any,
    scenario: Scenario,
    field: QValueField,
    payload: SampleType,
    level: int,
    *,
    walls: NDArray[np.bool_],
    norm: TwoSlopeNorm,
    ramp: LinearSegmentedColormap,
) -> None:
    """Paint one payload's map at one battery level, with both per-cell labels."""
    values = field.values[:, :, level, int(payload)]
    actions = field.actions[:, :, level, int(payload)]
    updates = field.updates[:, :, level, int(payload)]
    reached = field.reached[:, :, level, int(payload)] & ~walls
    decided = field.decided[:, :, level, int(payload)] & reached
    rows, cols = values.shape

    image = np.empty((rows, cols, 3), dtype=np.float64)
    image[:] = to_rgb(UNTOUCHED_INK)
    if reached.any():
        image[reached] = ramp(norm(values[reached]))[:, :3]
    image[walls] = to_rgb(WALL_INK)

    axis.imshow(image, interpolation="nearest", aspect="equal")
    _draw_cell_grid(axis, rows, cols)

    for row in range(rows):
        for col in range(cols):
            if walls[row, col]:
                continue
            if not reached[row, col]:
                # Hatched rather than merely pale: the diverging ramp's midpoint is
                # almost white, so a flat light fill for "no data" would sit a step
                # away from a genuine value of zero. Texture is not a colour and
                # cannot be mistaken for one.
                axis.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1.0,
                        1.0,
                        facecolor="none",
                        edgecolor="#c9c7c1",
                        hatch="///",
                        linewidth=0.0,
                        zorder=3,
                    )
                )
            ink = _ink_on(image[row, col])
            if reached[row, col]:
                axis.text(
                    col,
                    row + (0.02 if field.has_counts else 0.16),
                    compact_value(values[row, col]),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=ink,
                    zorder=4,
                )
                if field.has_counts:
                    axis.text(
                        col,
                        row + 0.30,
                        f"({compact_count(updates[row, col])})",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=ink,
                        alpha=0.85,
                        zorder=4,
                    )
            if decided[row, col]:
                # Top right, opposite the landmark cue: the two never collide, and an
                # arrow in a fixed corner can be scanned for the shape of a route
                # without reading anything else in the cell.
                axis.text(
                    col + 0.38,
                    row - 0.30,
                    ACTION_ARROWS[Action(int(actions[row, col]))],
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=ink,
                    zorder=4,
                )
            glyph = _glyph_at(scenario, (row, col))
            if glyph:
                axis.text(
                    col - 0.36,
                    row - 0.30,
                    glyph,
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=ink,
                    zorder=4,
                )

    walkable = int((~walls).sum())
    span = _value_span(values[reached]) if reached.any() else "nothing updated here"
    axis.set_title(
        f"{PAYLOAD_TITLES[payload]}\n"
        f"{int(reached.sum())} of {walkable} cells updated  ·  "
        f"{int(decided.sum())} decided  ·  {span}",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        pad=8,
    )


def _draw_cell_grid(axis: Any, rows: int, cols: int) -> None:
    """Hairlines between cells, and no axis furniture: the grid is a map, not a plot."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row in range(rows + 1):
        axis.axhline(row - 0.5, color="white", linewidth=1.0, zorder=2)
    for col in range(cols + 1):
        axis.axvline(col - 0.5, color="white", linewidth=1.0, zorder=2)
    axis.add_patch(
        Rectangle(
            (-0.5, -0.5),
            cols,
            rows,
            fill=False,
            edgecolor="#dcdad4",
            linewidth=1.0,
            zorder=5,
        )
    )
    axis.set_xlim(-0.5, cols - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)


def _glyph_at(scenario: Scenario, cell: tuple[int, int]) -> str:
    if scenario.lander == cell:
        return LANDER_GLYPH
    spec = scenario.sample_at(cell)
    return SAMPLE_GLYPHS[spec.sample_type] if spec is not None else ""


def _ink_on(rgb: Sequence[float]) -> str:
    """Readable text colour for a cell fill, by relative luminance."""
    red, green, blue = (float(channel) for channel in rgb[:3])
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 0.55 else TEXT_PRIMARY


__all__ = [
    "ACTION_ARROWS",
    "BATTERY_FIGS_DIRNAME",
    "BATTERY_FIG_STEM",
    "COVERAGE_FIG",
    "EXPERIENCE_FIG",
    "FIGS_DIRNAME",
    "CoverageReport",
    "ExperienceField",
    "PayloadCoverage",
    "QValueField",
    "battery_q_figure",
    "blue_ramp",
    "compact_count",
    "compact_value",
    "coverage_report",
    "diverging_q_ramp",
    "experience_heatmap_figure",
    "learned_battery_field",
    "occupiable_state_mask",
    "plural",
    "q_value_field",
    "q_value_norm",
    "run_subtitle",
    "state_coverage_figure",
    "visit_field",
    "write_battery_q_figures",
    "write_run_figures",
]
