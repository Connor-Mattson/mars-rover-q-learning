"""State encoding: the bijection has to be exact, or every Q-value is misfiled."""

from __future__ import annotations

import pytest

from mars_rover_q.state import (
    COLLECTABLE_SAMPLES,
    NUM_PAYLOAD_STATES,
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
