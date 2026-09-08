"""The rover's discrete action set."""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class Action(IntEnum):
    """The five actions available to the rover on every step."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    COLLECT = 4


NUM_ACTIONS: Final[int] = len(Action)

MOVE_ACTIONS: Final[tuple[Action, ...]] = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)

#: (row, column) displacement produced by a successfully executed move.
ACTION_DELTAS: Final[dict[Action, tuple[int, int]]] = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
}

#: Ninety-degree deflections, as (left_of_heading, right_of_heading).
DEFLECTIONS: Final[dict[Action, tuple[Action, Action]]] = {
    Action.NORTH: (Action.WEST, Action.EAST),
    Action.SOUTH: (Action.EAST, Action.WEST),
    Action.EAST: (Action.NORTH, Action.SOUTH),
    Action.WEST: (Action.SOUTH, Action.NORTH),
}

ACTION_LABELS: Final[dict[Action, str]] = {
    Action.NORTH: "N",
    Action.SOUTH: "S",
    Action.EAST: "E",
    Action.WEST: "W",
    Action.COLLECT: "C",
}


def is_move(action: Action) -> bool:
    """Return ``True`` when ``action`` commands a drive rather than a collection."""
    return action in ACTION_DELTAS


def delta(action: Action) -> tuple[int, int]:
    """Return the (row, column) displacement for a move action.

    Raises:
        KeyError: if ``action`` is not one of the four move actions.
    """
    return ACTION_DELTAS[action]


__all__ = [
    "ACTION_DELTAS",
    "ACTION_LABELS",
    "DEFLECTIONS",
    "MOVE_ACTIONS",
    "NUM_ACTIONS",
    "Action",
    "delta",
    "is_move",
]
