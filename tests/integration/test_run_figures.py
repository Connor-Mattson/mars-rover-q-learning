"""Per-run coverage figures: the arithmetic behind them, and that they render.

The figures themselves are checked only for "a non-trivial PNG appeared" -- what
is worth asserting is the counting, which is what a reader actually takes away
from them.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from mars_rover_q.actions import Action
from mars_rover_q.experiment import load_run, save_run
from mars_rover_q.metrics import greedy_actions, learned_state_mask
from mars_rover_q.rewards import RewardMode
from mars_rover_q.run_figures import (
    ACTION_ARROWS,
    BATTERY_FIGS_DIRNAME,
    COVERAGE_FIG,
    EXPERIENCE_FIG,
    FIGS_DIRNAME,
    compact_count,
    compact_value,
    coverage_report,
    learned_battery_field,
    occupiable_state_mask,
    q_value_field,
    q_value_norm,
    run_subtitle,
    visit_field,
    write_run_figures,
)
from mars_rover_q.scenario import Scenario, Terrain, resolve_scenario
from mars_rover_q.state import (
    COLLECTABLE_SAMPLES,
    NUM_PAYLOAD_STATES,
    BatteryBinning,
    BatteryEncoding,
    RoverState,
    SampleType,
    StateEncoder,
)
from mars_rover_q.training import TrainConfig, TrainResult, train


@pytest.fixture
def corridor() -> Scenario:
    return resolve_scenario("safe_corridor")


@pytest.fixture
def quick_run(corridor: Scenario) -> TrainResult:
    config = TrainConfig(
        scenario="safe_corridor",
        reward_mode=RewardMode.SPARSE,
        seed=1,
        episodes=6,
        log_every=0,
    )
    return train(corridor, config, warn_on_stubs=False, stream=io.StringIO())


@pytest.fixture
def binning(corridor: Scenario, quick_run: TrainResult) -> BatteryBinning:
    """The battery axis ``quick_run``'s table was written under.

    Every figure reads a saved table through the binning of the run that wrote it,
    so a test holding a freshly trained table has to hand over the same one.
    """
    return corridor.battery_binning(quick_run.config.battery_encoding)


@pytest.fixture
def encoder(corridor: Scenario, binning: BatteryBinning) -> StateEncoder:
    """An encoder over ``quick_run``'s table."""
    return StateEncoder(corridor.rows, corridor.cols, corridor.battery_capacity, binning)


def test_coverage_report_counts_the_states_the_run_wrote_to(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    report = coverage_report(corridor, quick_run.q_table, binning=binning)

    assert report.total == quick_run.metadata["num_states"]
    assert 0 < report.learned < report.total
    assert report.learned == sum(row.learned for row in report.by_payload)
    assert report.fraction == pytest.approx(quick_run.metadata["learned_state_fraction"])


def test_coverage_reachable_ceiling_excludes_wall_cells(corridor: Scenario) -> None:
    """The ceiling is the honest second denominator; walls can never be occupied."""
    table = np.zeros((24_400, 5))
    report = coverage_report(corridor, table)
    walls = int((corridor.grid == int(Terrain.WALL)).sum())

    assert report.reachable < report.total
    assert report.total - report.reachable >= walls * (corridor.battery_capacity + 1) * (
        NUM_PAYLOAD_STATES
    )
    assert all(row.reachable < row.total for row in report.by_payload)


def test_coverage_ceiling_shrinks_with_the_carried_sample_s_distance(corridor: Scenario) -> None:
    """A far sample costs battery to fetch, so fewer of its states can ever exist.

    The bug this pins: one traversable-cell count shared across all four payload
    slices made the biosignature slice -- fetched from the far end of the corridor,
    so it can never be carried on a full charge -- look like states the run failed
    to reach rather than states the map forbids.
    """
    report = coverage_report(corridor, np.zeros((24_400, 5)))
    ceilings = {row.payload: row.reachable for row in report.by_payload}
    distances = {
        sample: corridor.distance(corridor.lander, corridor.samples[sample].position)
        for sample in COLLECTABLE_SAMPLES
    }
    furthest = max(distances, key=lambda sample: distances[sample])

    assert ceilings[SampleType.NONE] > max(ceilings[sample] for sample in COLLECTABLE_SAMPLES)
    assert ceilings[furthest] == min(ceilings[sample] for sample in COLLECTABLE_SAMPLES)


def test_coverage_ceiling_excludes_states_that_are_already_terminal(corridor: Scenario) -> None:
    """A delivery and a flat battery are entered, never acted from."""
    encoder = StateEncoder(
        corridor.rows,
        corridor.cols,
        corridor.battery_capacity,
        BatteryBinning.dense(corridor.battery_capacity),
    )
    occupiable = occupiable_state_mask(corridor, encoder)
    row, col = corridor.lander

    assert not occupiable[row, col, :, [int(sample) for sample in COLLECTABLE_SAMPLES]].any()
    assert occupiable[row, col, corridor.battery_capacity, int(SampleType.NONE)]
    assert not occupiable[:, :, 0, :].any()


def test_coverage_ceiling_bounds_what_a_real_run_learned(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    """No run can write a state the map forbids, so the ceiling must not be beaten."""
    report = coverage_report(corridor, quick_run.q_table, binning=binning)

    assert report.learned <= report.reachable
    assert all(row.learned <= row.reachable for row in report.by_payload)


def test_coverage_is_split_across_all_four_payload_slices(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    report = coverage_report(corridor, quick_run.q_table, binning=binning)

    assert [row.payload for row in report.by_payload] == [SampleType.NONE, *COLLECTABLE_SAMPLES]
    assert all(row.total == report.total // NUM_PAYLOAD_STATES for row in report.by_payload)
    assert report.by_payload[0].learned > 0


def test_visit_field_sums_battery_away_and_keeps_the_total(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    field = visit_field(corridor, quick_run.visit_counts, binning=binning)

    assert field.counts.shape == (corridor.rows, corridor.cols, NUM_PAYLOAD_STATES)
    assert field.counts.sum() == quick_run.total_env_steps
    assert field.log


def test_visit_field_lands_experience_on_the_cell_it_happened_in(corridor: Scenario) -> None:
    encoder_states = corridor.rows * corridor.cols * (corridor.battery_capacity + 1)
    counts = np.zeros(encoder_states * NUM_PAYLOAD_STATES, dtype=np.int64)
    encoder = StateEncoder(corridor.rows, corridor.cols, corridor.battery_capacity)
    state = RoverState(7, 3, 22, SampleType.HYDRATED_MINERAL)
    counts[encoder.encode(state)] = 9

    field = visit_field(corridor, counts)
    assert field.counts[7, 3, int(SampleType.HYDRATED_MINERAL)] == 9
    assert field.counts.sum() == 9


def test_walls_never_accumulate_experience(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    """The rover cannot stand on a wall, so no wall cell may carry a visit."""
    field = visit_field(corridor, quick_run.visit_counts, binning=binning)
    walls = corridor.grid == int(Terrain.WALL)

    assert field.counts[walls].sum() == 0


def test_fallback_field_saturates_at_the_battery_level_count(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    """The no-visit-counts fallback is a coarser measurement and is labelled as one."""
    field = learned_battery_field(corridor, quick_run.q_table, binning=binning)

    assert not field.log
    assert field.counts.max() <= binning.levels
    assert "battery levels" in field.scale_label


def test_write_run_figures_emits_both_summary_figures(
    tmp_path: Path, corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    paths = write_run_figures(
        tmp_path,
        corridor,
        quick_run.q_table,
        visit_counts=quick_run.visit_counts,
        subtitle=run_subtitle(quick_run.config.as_dict()),
        battery_frames=False,
        binning=binning,
    )

    assert [path.name for path in paths] == [COVERAGE_FIG, EXPERIENCE_FIG]
    for path in paths:
        assert path.parent.name == FIGS_DIRNAME
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > 10_000


@pytest.mark.parametrize(
    "encoding",
    [
        BatteryEncoding.AFFORDABILITY,
        # Sixty-one matplotlib frames; the marker is assigned from --durations.
        pytest.param(BatteryEncoding.DENSE, marks=pytest.mark.slow),
    ],
)
def test_battery_set_holds_one_frame_per_battery_level(
    tmp_path: Path, corridor: Scenario, encoding: BatteryEncoding
) -> None:
    """Battery is kept as an axis here, so the set is as long as that axis is.

    Both encodings are exercised: the dense one is the sixty-one-frame drain the set
    was designed around, and the binned default is the same set over four rows.
    """
    axis = corridor.battery_binning(encoding)
    rows = corridor.rows * corridor.cols * axis.levels * NUM_PAYLOAD_STATES
    # A spread of learned values rather than a zero table, so every frame has
    # something to paint and the colour scale is exercised alongside the naming.
    table = np.linspace(-100.0, 160.0, rows * 5).reshape(rows, 5)
    paths = write_run_figures(
        tmp_path, corridor, table, visit_counts=np.ones(rows, dtype=np.int64), binning=axis
    )
    frames = [path for path in paths if path.parent.name == BATTERY_FIGS_DIRNAME]

    assert paths[:2] == [
        tmp_path / FIGS_DIRNAME / COVERAGE_FIG,
        tmp_path / FIGS_DIRNAME / EXPERIENCE_FIG,
    ]
    assert len(frames) == axis.levels
    assert frames[0].parent.parent.name == FIGS_DIRNAME
    # Zero-padded so a directory listing and an image viewer walk the drain in order.
    assert [path.name for path in frames] == sorted(path.name for path in frames)
    width = len(str(axis.levels - 1))
    assert frames[0].name == f"battery_{0:0{width}d}.png"
    assert frames[-1].name == f"battery_{axis.levels - 1}.png"
    assert frames[-1].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert frames[-1].stat().st_size > 10_000


def test_battery_frames_are_opt_out(
    tmp_path: Path, corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    paths = write_run_figures(
        tmp_path, corridor, quick_run.q_table, battery_frames=False, binning=binning
    )

    assert len(paths) == 2
    assert not (tmp_path / FIGS_DIRNAME / BATTERY_FIGS_DIRNAME).exists()


def test_q_value_field_keeps_battery_as_an_axis_and_takes_the_greedy_value(
    corridor: Scenario,
    quick_run: TrainResult,
    binning: BatteryBinning,
    encoder: StateEncoder,
) -> None:
    field = q_value_field(
        corridor, quick_run.q_table, visit_counts=quick_run.visit_counts, binning=binning
    )
    state = RoverState(5, 1, corridor.battery_capacity - 1, SampleType.NONE)
    index = encoder.encode(state)
    level = binning.level_of(state.battery)

    assert field.values.shape == (
        corridor.rows,
        corridor.cols,
        binning.levels,
        NUM_PAYLOAD_STATES,
    )
    assert field.battery_levels == binning.levels
    assert field.binning == binning
    assert field.has_counts
    assert field.values[5, 1, level, int(SampleType.NONE)] == pytest.approx(
        quick_run.q_table[index].max()
    )
    assert field.updates[5, 1, level, int(SampleType.NONE)] == (quick_run.visit_counts[index])
    assert field.updates.sum() == quick_run.total_env_steps


def test_q_value_field_carries_the_action_the_value_came_from(
    corridor: Scenario,
    quick_run: TrainResult,
    binning: BatteryBinning,
    encoder: StateEncoder,
) -> None:
    field = q_value_field(
        corridor, quick_run.q_table, visit_counts=quick_run.visit_counts, binning=binning
    )
    state = RoverState(5, 1, corridor.battery_capacity - 1, SampleType.NONE)
    index = encoder.encode(state)
    level = binning.level_of(state.battery)

    assert field.actions.shape == field.values.shape
    assert (
        field.actions[5, 1, level, int(SampleType.NONE)]
        == (greedy_actions(quick_run.q_table)[index])
    )


def test_only_a_unique_best_action_counts_as_decided(corridor: Scenario) -> None:
    """A tie at the top makes argmax arbitrary, and an arrow drawn from it a lie."""
    encoder = StateEncoder(corridor.rows, corridor.cols, corridor.battery_capacity)
    table = np.zeros((encoder.num_states, 5))
    unique = encoder.encode(RoverState(5, 1, 30, SampleType.NONE))
    split = encoder.encode(RoverState(5, 2, 30, SampleType.NONE))
    table[unique, int(Action.EAST)] = 4.0
    # Two actions share the maximum: updated, non-flat, and still undecided.
    table[split, int(Action.NORTH)] = 4.0
    table[split, int(Action.SOUTH)] = 4.0

    field = q_value_field(corridor, table, visit_counts=np.ones(encoder.num_states, np.int64))

    assert field.decided[5, 1, 30, int(SampleType.NONE)]
    assert field.actions[5, 1, 30, int(SampleType.NONE)] == int(Action.EAST)
    assert not field.decided[5, 2, 30, int(SampleType.NONE)]
    # A flat row is undecided too -- that is the all-zero case, and it is the common one.
    assert not field.decided[5, 3, 30, int(SampleType.NONE)]


def test_every_action_has_an_arrow(corridor: Scenario, quick_run: TrainResult) -> None:
    """The panel indexes this by whatever argmax returned; a gap would be a KeyError."""
    assert set(ACTION_ARROWS) == set(Action)
    assert ACTION_ARROWS[Action.COLLECT] == "O"
    assert len(set(ACTION_ARROWS.values())) == len(Action)


def test_q_value_field_falls_back_to_the_learned_mask_without_visit_counts(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    """Without counts the update numbers are unknown, so they are not shown at all."""
    field = q_value_field(corridor, quick_run.q_table, binning=binning)

    assert not field.has_counts
    assert field.updates.sum() == 0
    assert field.reached.sum() > 0
    assert field.reached.sum() == int(learned_state_mask(quick_run.q_table).sum())


def test_walls_are_never_marked_reached(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    field = q_value_field(
        corridor, quick_run.q_table, visit_counts=quick_run.visit_counts, binning=binning
    )
    walls = corridor.grid == int(Terrain.WALL)

    assert not field.reached[walls].any()


def test_q_value_norm_is_shared_and_pinned_to_zero(
    corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    """One scale for the whole set, or a cell darkening between frames means nothing."""
    field = q_value_field(
        corridor, quick_run.q_table, visit_counts=quick_run.visit_counts, binning=binning
    )
    norm = q_value_norm(field)

    assert norm.vcenter == 0.0
    assert norm.vmin is not None and norm.vmax is not None
    assert norm.vmin < 0.0 < norm.vmax
    reached = field.values[field.reached]
    assert norm.vmin <= reached.min()
    assert norm.vmax >= reached.max()


def test_q_value_norm_survives_a_table_that_learned_nothing(corridor: Scenario) -> None:
    """TwoSlopeNorm demands vmin < 0 < vmax; an all-zero table supplies neither."""
    field = q_value_field(corridor, np.zeros((24_400, 5)), visit_counts=np.ones(24_400, np.int64))
    norm = q_value_norm(field)

    assert norm.vmin is not None and norm.vmax is not None
    assert norm.vmin < 0.0 < norm.vmax


def test_figures_render_for_a_run_saved_without_visit_counts(
    tmp_path: Path, corridor: Scenario, quick_run: TrainResult, binning: BatteryBinning
) -> None:
    """Older runs still get both figures; the map just paints the coarser quantity."""
    save_run(tmp_path / "run", corridor, quick_run, None)
    (tmp_path / "run" / "visit_counts.npy").unlink()
    loaded = load_run(tmp_path / "run")
    assert loaded.visit_counts is None

    paths = write_run_figures(
        tmp_path / "run",
        loaded.scenario,
        loaded.q_table,
        battery_frames=False,
        binning=binning,
    )
    assert all(path.exists() for path in paths)


def test_figures_survive_a_table_that_learned_nothing(tmp_path: Path, corridor: Scenario) -> None:
    """An all-zero table has no colour range at all; the scales must not collapse."""
    paths = write_run_figures(
        tmp_path,
        corridor,
        np.zeros((24_400, 5)),
        visit_counts=np.zeros(24_400, dtype=np.int64),
        battery_frames=False,
    )
    assert all(path.stat().st_size > 10_000 for path in paths)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0"), (7, "7"), (999, "999"), (1_500, "1.5k"), (52_900, "53k"), (1_904_221, "1.9M")],
)
def test_compact_count_keeps_cell_labels_short(value: int, expected: str) -> None:
    assert compact_count(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (159.99999, "160"),
        (-83.2227, "-83.2"),
        (4.6431, "4.64"),
        (0.0004, "~0"),
        (-0.0004, "~0"),
    ],
)
def test_compact_value_scales_its_digits_to_the_magnitude(value: float, expected: str) -> None:
    assert compact_value(value) == expected


def test_compact_value_separates_an_untouched_zero_from_a_negligible_one() -> None:
    """Both round to nothing, but only one of them means "nothing was ever written"."""
    assert compact_value(0.0) != compact_value(1e-9)


def test_run_subtitle_names_the_condition_rather_than_the_parameter() -> None:
    baseline = run_subtitle({"scenario": "safe_corridor", "episodes": 4000, "seed": 1})
    treated = run_subtitle({"scenario": "safe_corridor", "curriculum_fraction": 0.5})

    assert "no curriculum" in baseline
    assert "4,000 episodes" in baseline
    assert "curriculum 0.5" in treated
