"""Semantic rover state and its bijection with integer Q-table row IDs.

The MDP state is ``(row, column, battery_remaining, carried_sample)``. The map is
fixed for the duration of a training run, so the position already determines the
terrain type; terrain is deliberately *not* part of the state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class SampleType(IntEnum):
    """The payload the rover is carrying, or ``NONE``."""

    NONE = 0
    BASALT = 1
    HYDRATED_MINERAL = 2
    BIOSIGNATURE = 3


#: The three collectable sample types, in map/legend order.
COLLECTABLE_SAMPLES: Final[tuple[SampleType, ...]] = (
    SampleType.BASALT,
    SampleType.HYDRATED_MINERAL,
    SampleType.BIOSIGNATURE,
)

NUM_PAYLOAD_STATES: Final[int] = len(SampleType)

SAMPLE_LABELS: Final[dict[SampleType, str]] = {
    SampleType.NONE: "none",
    SampleType.BASALT: "basalt core",
    SampleType.HYDRATED_MINERAL: "hydrated mineral",
    SampleType.BIOSIGNATURE: "biosignature candidate",
}


@dataclass(frozen=True, slots=True)
class RoverState:
    """A fully observable rover state.

    Attributes:
        row: zero-based grid row.
        col: zero-based grid column.
        battery: remaining battery, an exact integer in ``[0, battery_capacity]``.
        carried: the payload currently locked in the sample bay.
    """

    row: int
    col: int
    battery: int
    carried: SampleType = SampleType.NONE

    @property
    def position(self) -> tuple[int, int]:
        """The ``(row, col)`` cell occupied by the rover."""
        return (self.row, self.col)


class StateEncoder:
    """Deterministic bijection between :class:`RoverState` and table row IDs.

    The encoding is mixed-radix with ``carried`` as the fastest varying digit::

        index = ((row * cols + col) * (battery_capacity + 1) + battery) * 4 + carried

    Every integer in ``range(num_states)`` decodes to a valid state, and
    ``decode(encode(s)) == s`` for every in-range state, so the table has no
    unreachable padding rows.
    """

    __slots__ = ("battery_capacity", "cols", "rows")

    def __init__(self, rows: int, cols: int, battery_capacity: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must be non-empty, got rows={rows}, cols={cols}")
        if battery_capacity <= 0:
            raise ValueError(f"battery_capacity must be positive, got {battery_capacity}")
        self.rows = rows
        self.cols = cols
        self.battery_capacity = battery_capacity

    @property
    def battery_levels(self) -> int:
        """Number of distinct battery readings, including empty."""
        return self.battery_capacity + 1

    @property
    def num_states(self) -> int:
        """Total number of encodable states, i.e. the Q-table row count."""
        return self.rows * self.cols * self.battery_levels * NUM_PAYLOAD_STATES

    def encode(self, state: RoverState) -> int:
        """Map a semantic state to its integer row ID.

        Raises:
            ValueError: if any field falls outside the encoder's range.
        """
        if not 0 <= state.row < self.rows:
            raise ValueError(f"row {state.row} outside [0, {self.rows})")
        if not 0 <= state.col < self.cols:
            raise ValueError(f"col {state.col} outside [0, {self.cols})")
        if not 0 <= state.battery <= self.battery_capacity:
            raise ValueError(f"battery {state.battery} outside [0, {self.battery_capacity}]")
        cell = state.row * self.cols + state.col
        return (cell * self.battery_levels + state.battery) * NUM_PAYLOAD_STATES + int(
            state.carried
        )

    def decode(self, index: int) -> RoverState:
        """Map an integer row ID back to its semantic state.

        Raises:
            ValueError: if ``index`` is outside ``range(num_states)``.
        """
        if not 0 <= index < self.num_states:
            raise ValueError(f"state index {index} outside [0, {self.num_states})")
        index, carried = divmod(index, NUM_PAYLOAD_STATES)
        cell, battery = divmod(index, self.battery_levels)
        row, col = divmod(cell, self.cols)
        return RoverState(row=row, col=col, battery=battery, carried=SampleType(carried))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"StateEncoder(rows={self.rows}, cols={self.cols}, "
            f"battery_capacity={self.battery_capacity}, num_states={self.num_states})"
        )


__all__ = [
    "COLLECTABLE_SAMPLES",
    "NUM_PAYLOAD_STATES",
    "SAMPLE_LABELS",
    "RoverState",
    "SampleType",
    "StateEncoder",
]
