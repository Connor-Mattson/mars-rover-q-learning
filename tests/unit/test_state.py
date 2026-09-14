"""State encoding: the map onto rows has to be exact, or every Q-value is misfiled.

Under the dense battery axis it is a bijection and every round trip is exact. Under a
binned one it is deliberately many-to-one on battery alone, and the tests below pin
down precisely which half of the round trip survives.
"""

from __future__ import annotations

import numpy as np
import pytest

from mars_rover_q.state import (
    COLLECTABLE_SAMPLES,
    NUM_PAYLOAD_STATES,
    BatteryBinning,
    RoverState,
    SampleType,
    StateEncoder,
)


def test_num_states_matches_the_product_of_the_factors() -> None:
    encoder = StateEncoder(rows=4, cols=5, battery_capacity=7)
    assert encoder.num_states == 4 * 5 * 8 * NUM_PAYLOAD_STATES


def test_encode_decode_round_trips_for_every_state() -> None:
    encoder = StateEncoder(rows=3, cols=4, battery_capacity=5)
    seen: set[int] = set()
    for row in range(3):
        for col in range(4):
            for battery in range(6):
                for carried in SampleType:
                    state = RoverState(row, col, battery, carried)
                    index = encoder.encode(state)
                    assert encoder.decode(index) == state
                    seen.add(index)
    assert seen == set(range(encoder.num_states))


def test_decode_encode_round_trips_for_every_index() -> None:
    encoder = StateEncoder(rows=2, cols=3, battery_capacity=4)
    for index in range(encoder.num_states):
        assert encoder.encode(encoder.decode(index)) == index


@pytest.mark.parametrize(
    "state",
    [
        RoverState(-1, 0, 3),
        RoverState(0, 9, 3),
        RoverState(0, 0, -1),
        RoverState(0, 0, 99),
    ],
)
def test_encode_rejects_out_of_range_states(state: RoverState) -> None:
    encoder = StateEncoder(rows=3, cols=3, battery_capacity=5)
    with pytest.raises(ValueError):
        encoder.encode(state)


@pytest.mark.parametrize("index", [-1, 10_000])
def test_decode_rejects_out_of_range_indices(index: int) -> None:
    encoder = StateEncoder(rows=3, cols=3, battery_capacity=5)
    with pytest.raises(ValueError):
        encoder.decode(index)


@pytest.mark.parametrize(
    ("rows", "cols", "battery"),
    [(0, 3, 5), (3, 0, 5), (3, 3, 0), (-2, 3, 5)],
)
def test_encoder_rejects_degenerate_dimensions(rows: int, cols: int, battery: int) -> None:
    with pytest.raises(ValueError):
        StateEncoder(rows, cols, battery)


def test_payload_states_are_distinct_rows() -> None:
    encoder = StateEncoder(rows=2, cols=2, battery_capacity=3)
    indices = {
        encoder.encode(RoverState(1, 1, 2, carried))
        for carried in (SampleType.NONE, *COLLECTABLE_SAMPLES)
    }
    assert len(indices) == NUM_PAYLOAD_STATES


def test_grid_view_agrees_with_decode_on_every_index() -> None:
    """The reshape is only a legitimate shortcut if it *is* the decode."""
    encoder = StateEncoder(rows=3, cols=4, battery_capacity=5)
    values = np.arange(encoder.num_states, dtype=np.int64)
    view = encoder.grid_view(values)

    assert view.shape == (3, 4, 6, NUM_PAYLOAD_STATES)
    for index in range(encoder.num_states):
        state = encoder.decode(index)
        assert view[state.row, state.col, state.battery, int(state.carried)] == index


def test_grid_view_sums_battery_away_per_cell_and_payload() -> None:
    encoder = StateEncoder(rows=2, cols=2, battery_capacity=3)
    values = np.zeros(encoder.num_states, dtype=np.int64)
    values[encoder.encode(RoverState(1, 0, 2, SampleType.BIOSIGNATURE))] = 5
    values[encoder.encode(RoverState(1, 0, 3, SampleType.BIOSIGNATURE))] = 7

    per_cell = encoder.grid_view(values).sum(axis=2)
    assert per_cell[1, 0, int(SampleType.BIOSIGNATURE)] == 12
    assert per_cell.sum() == 12


def test_grid_view_rejects_arrays_that_are_not_one_row_per_state() -> None:
    encoder = StateEncoder(rows=2, cols=2, battery_capacity=3)
    with pytest.raises(ValueError, match="1-D array"):
        encoder.grid_view(np.zeros((encoder.num_states, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="1-D array"):
        encoder.grid_view(np.zeros(encoder.num_states + 1, dtype=np.float64))


# -- battery binning ------------------------------------------------------


def test_the_dense_binning_is_the_identity_on_every_reading() -> None:
    """The dense encoding has to stay exactly what it was before binning existed."""
    binning = BatteryBinning.dense(12)
    assert binning.levels == 13
    assert binning.is_dense
    assert [binning.level_of(b) for b in range(13)] == list(range(13))
    assert all(binning.span(level) == (level, level) for level in range(13))
    assert binning.label(4) == "4"


def test_thresholds_become_the_lower_bound_of_each_bin() -> None:
    binning = BatteryBinning.from_thresholds(60, [9, 13, 33])
    assert binning.levels == 4
    assert not binning.is_dense
    assert [binning.span(level) for level in range(4)] == [(0, 8), (9, 12), (13, 32), (33, 60)]
    assert [binning.label(level) for level in range(4)] == ["0-8", "9-12", "13-32", "33-60"]
    assert [binning.level_of(b) for b in (0, 8, 9, 12, 13, 32, 33, 60)] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_binning_is_monotone_in_battery() -> None:
    """The figures read the axis as a drain sequence; a non-monotone map breaks that."""
    binning = BatteryBinning.from_thresholds(40, [7, 20, 31])
    levels = [binning.level_of(b) for b in range(41)]
    assert levels == sorted(levels)


def test_representative_is_the_worst_case_of_its_bin() -> None:
    """Decoding must not flatter a row by handing back more charge than it covers."""
    binning = BatteryBinning.from_thresholds(60, [9, 13, 33])
    assert [binning.representative(level) for level in range(4)] == [0, 9, 13, 33]


@pytest.mark.parametrize(
    ("thresholds", "expected"),
    [
        # Unreachable samples and duplicate round trips both collapse.
        ([13.0, 13.0, float("inf")], (13,)),
        # A threshold at or below empty, or above full, splits nothing.
        ([0, 13, 61], (13,)),
        # Fractional costs round up: the bin edge is the first affordable reading.
        ([12.4], (13,)),
        ([], ()),
    ],
)
def test_thresholds_are_deduplicated_rounded_and_clipped(
    thresholds: list[float], expected: tuple[int, ...]
) -> None:
    assert BatteryBinning.from_thresholds(60, thresholds).edges == expected


@pytest.mark.parametrize("edges", [(0,), (61,), (13, 9), (9, 9)])
def test_invalid_edges_are_rejected(edges: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        BatteryBinning(capacity=60, edges=edges)


def test_level_of_rejects_a_reading_off_the_scale() -> None:
    binning = BatteryBinning.dense(10)
    with pytest.raises(ValueError, match="outside"):
        binning.level_of(11)


def test_span_rejects_a_level_off_the_axis() -> None:
    binning = BatteryBinning.from_thresholds(60, [9, 13, 33])
    with pytest.raises(ValueError, match="outside"):
        binning.span(4)


def test_a_binned_encoder_collapses_the_readings_that_share_a_bin() -> None:
    binning = BatteryBinning.from_thresholds(20, [5, 9])
    encoder = StateEncoder(rows=3, cols=3, battery_capacity=20, binning=binning)

    assert encoder.num_states == 3 * 3 * 3 * NUM_PAYLOAD_STATES
    assert encoder.battery_levels == 3
    shared = {
        encoder.encode(RoverState(1, 1, battery, SampleType.NONE)) for battery in range(9, 21)
    }
    assert len(shared) == 1
    assert encoder.encode(RoverState(1, 1, 8, SampleType.NONE)) not in shared


def test_a_binned_encoder_still_round_trips_every_row() -> None:
    """``decode`` recovers the row, not the rover; encoding it again must return it."""
    binning = BatteryBinning.from_thresholds(20, [5, 9])
    encoder = StateEncoder(rows=3, cols=3, battery_capacity=20, binning=binning)
    for index in range(encoder.num_states):
        assert encoder.encode(encoder.decode(index)) == index


def test_a_binned_encoder_keeps_position_and_payload_exact() -> None:
    binning = BatteryBinning.from_thresholds(20, [5, 9])
    encoder = StateEncoder(rows=3, cols=3, battery_capacity=20, binning=binning)
    state = RoverState(2, 1, 17, SampleType.BIOSIGNATURE)
    decoded = encoder.decode(encoder.encode(state))

    assert (decoded.row, decoded.col, decoded.carried) == (2, 1, SampleType.BIOSIGNATURE)
    assert decoded.battery == 9


def test_an_encoder_rejects_a_binning_built_for_another_capacity() -> None:
    with pytest.raises(ValueError, match="does not match"):
        StateEncoder(rows=2, cols=2, battery_capacity=10, binning=BatteryBinning.dense(20))


def test_grid_view_axis_two_is_as_long_as_the_battery_axis() -> None:
    binning = BatteryBinning.from_thresholds(20, [5, 9])
    encoder = StateEncoder(rows=3, cols=3, battery_capacity=20, binning=binning)
    view = encoder.grid_view(np.zeros(encoder.num_states, dtype=np.int64))
    assert view.shape == (3, 3, 3, NUM_PAYLOAD_STATES)
