"""The action set and its geometry."""

from __future__ import annotations

import pytest

from mars_rover_q.actions import (
    ACTION_DELTAS,
    DEFLECTIONS,
    MOVE_ACTIONS,
    NUM_ACTIONS,
    Action,
    delta,
    is_move,
)


def test_action_set_is_exactly_the_five_documented_actions() -> None:
    assert [a.name for a in Action] == ["NORTH", "SOUTH", "EAST", "WEST", "COLLECT"]
    assert NUM_ACTIONS == 5


def test_move_actions_exclude_collect() -> None:
    assert set(MOVE_ACTIONS) == set(Action) - {Action.COLLECT}
    assert is_move(Action.NORTH)
    assert not is_move(Action.COLLECT)


def test_opposite_moves_cancel() -> None:
    assert ACTION_DELTAS[Action.NORTH] == tuple(-v for v in ACTION_DELTAS[Action.SOUTH])
    assert ACTION_DELTAS[Action.EAST] == tuple(-v for v in ACTION_DELTAS[Action.WEST])


@pytest.mark.parametrize("action", MOVE_ACTIONS)
def test_deflections_are_perpendicular_and_distinct(action: Action) -> None:
    left, right = DEFLECTIONS[action]
    assert left != right
    assert left not in (
        action,
        *(a for a in MOVE_ACTIONS if delta(a) == tuple(-v for v in delta(action))),
    )
    dr, dc = delta(action)
    for deflected in (left, right):
        ddr, ddc = delta(deflected)
        assert dr * ddr + dc * ddc == 0  # perpendicular


def test_delta_rejects_collect() -> None:
    with pytest.raises(KeyError):
        delta(Action.COLLECT)
