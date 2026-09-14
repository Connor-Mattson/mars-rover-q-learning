"""Semantic rover state and its mapping onto integer Q-table row IDs.

The MDP state is ``(row, column, battery_remaining, carried_sample)``. The map is
fixed for the duration of a training run, so the position already determines the
terrain type; terrain is deliberately *not* part of the state.

The battery axis is the one factor the table does not have to carry at full
resolution. :class:`BatteryBinning` is the seam: it maps a raw battery reading onto
the table's battery axis, and the two constructors on it are the two encodings the
project supports -- :meth:`BatteryBinning.dense`, one row per reading, and a
threshold binning, a handful of rows covering the readings between the decisions
that actually turn on remaining charge. The *environment* always tracks exact
integer battery; only the table's view of it is coarsened.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from math import ceil, isfinite
from typing import Any, Final

from numpy.typing import NDArray


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


class BatteryEncoding(StrEnum):
    """How the Q-table's battery axis is resolved.

    ``AFFORDABILITY`` is the default. ``DENSE`` is the original one-row-per-reading
    encoding, kept because it is the honest baseline any claim about the binning has
    to be measured against.
    """

    AFFORDABILITY = "affordability"
    DENSE = "dense"


@dataclass(frozen=True, slots=True)
class BatteryBinning:
    """A monotone map from an exact battery reading onto the table's battery axis.

    ``edges`` are the *lower* bounds of every bin above the first, strictly
    ascending and each in ``[1, capacity]``. Bin ``k`` therefore covers
    ``[edges[k-1], edges[k])``, with bin ``0`` running from an empty battery up to
    the first edge and the last bin running to ``capacity``.

    The map is monotone by construction, which is the property the figures rely on:
    a higher bin always means more charge, so the frames still read as a drain
    sequence.

    Attributes:
        capacity: the scenario's full battery, the largest encodable reading.
        edges: the bin lower bounds, ascending and without duplicates.
    """

    capacity: int
    edges: tuple[int, ...]
    #: ``battery -> level``, precomputed. Encoding happens twice per environment
    #: step, so this is a lookup rather than a search.
    _levels: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if list(self.edges) != sorted(set(self.edges)):
            raise ValueError(f"edges must be strictly ascending and unique, got {self.edges}")
        for edge in self.edges:
            if not 1 <= edge <= self.capacity:
                raise ValueError(f"edge {edge} outside [1, {self.capacity}]")
        object.__setattr__(
            self,
            "_levels",
            tuple(bisect_right(self.edges, battery) for battery in range(self.capacity + 1)),
        )

    @classmethod
    def dense(cls, capacity: int) -> BatteryBinning:
        """One bin per battery reading: the identity map, and the original encoding."""
        return cls(capacity=capacity, edges=tuple(range(1, capacity + 1)))

    @classmethod
    def from_thresholds(cls, capacity: int, thresholds: Sequence[float]) -> BatteryBinning:
        """Bin at the given battery thresholds, rounded up and made unique.

        Thresholds outside ``[1, capacity]`` are dropped rather than clamped: a
        threshold at or below an empty battery, or above a full one, splits nothing,
        and clamping it would silently manufacture a bin boundary the caller did not
        ask for. Non-finite thresholds -- an unreachable sample's round trip -- are
        dropped for the same reason.
        """
        edges = sorted(
            {
                ceil(value)
                for value in thresholds
                if isfinite(value) and 1 <= ceil(value) <= capacity
            }
        )
        return cls(capacity=capacity, edges=tuple(edges))

    @property
    def levels(self) -> int:
        """Number of bins, i.e. the length of the table's battery axis."""
        return len(self.edges) + 1

    @property
    def is_dense(self) -> bool:
        """Whether every battery reading gets its own bin."""
        return self.levels == self.capacity + 1

    def level_of(self, battery: int) -> int:
        """The bin ``battery`` falls in.

        Raises:
            ValueError: if ``battery`` is outside ``[0, capacity]``.
        """
        if not 0 <= battery <= self.capacity:
            raise ValueError(f"battery {battery} outside [0, {self.capacity}]")
        return self._levels[battery]

    def span(self, level: int) -> tuple[int, int]:
        """The inclusive ``(lowest, highest)`` battery readings in ``level``.

        Raises:
            ValueError: if ``level`` is outside ``range(levels)``.
        """
        if not 0 <= level < self.levels:
            raise ValueError(f"battery level {level} outside [0, {self.levels})")
        low = 0 if level == 0 else self.edges[level - 1]
        high = self.capacity if level == self.levels - 1 else self.edges[level] - 1
        return (low, high)

    def representative(self, level: int) -> int:
        """The lowest battery reading in ``level``.

        This is what :meth:`StateEncoder.decode` puts in the state it hands back. It
        is the *worst case* of the bin, deliberately: a decoded state is used to ask
        "what does the table say about here", and rounding that question toward more
        charge than the rover might have would flatter the answer.
        """
        return self.span(level)[0]

    def label(self, level: int) -> str:
        """Human-readable range for ``level``, e.g. ``"13-32"`` or ``"7"``."""
        low, high = self.span(level)
        return str(low) if low == high else f"{low}-{high}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.is_dense:
            return f"BatteryBinning.dense(capacity={self.capacity})"
        spans = ", ".join(self.label(level) for level in range(self.levels))
        return f"BatteryBinning(capacity={self.capacity}, bins=[{spans}])"


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
    """Deterministic map from :class:`RoverState` onto table row IDs.

    The encoding is mixed-radix with ``carried`` as the fastest varying digit::

        index = ((row * cols + col) * battery_levels + battery_level) * 4 + carried

    where ``battery_level`` is ``binning.level_of(state.battery)``. Every integer in
    ``range(num_states)`` decodes to a valid state, so the table has no unreachable
    padding rows, and ``encode(decode(i)) == i`` always holds.

    The reverse round trip is where the two encodings part company. Under the dense
    binning the map is a bijection and ``decode(encode(s)) == s``; under any coarser
    one it is deliberately many-to-one -- every reading in a bin shares a row -- and
    ``decode`` returns the bin's lowest reading. That is the whole point of the
    binning, and callers that need the rover's exact charge must read it off the
    state or the environment, never off a decoded row.

    Args:
        rows: grid height.
        cols: grid width.
        battery_capacity: the scenario's full battery.
        binning: how raw readings map onto the battery axis. Defaults to
            :meth:`BatteryBinning.dense`, so an encoder built without one behaves
            exactly as it did before binning existed.

    Raises:
        ValueError: on a non-positive dimension, or a ``binning`` whose capacity
            disagrees with ``battery_capacity``.
    """

    __slots__ = ("battery_capacity", "binning", "cols", "rows")

    def __init__(
        self,
        rows: int,
        cols: int,
        battery_capacity: int,
        binning: BatteryBinning | None = None,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must be non-empty, got rows={rows}, cols={cols}")
        if battery_capacity <= 0:
            raise ValueError(f"battery_capacity must be positive, got {battery_capacity}")
        if binning is not None and binning.capacity != battery_capacity:
            raise ValueError(
                f"binning capacity {binning.capacity} does not match "
                f"battery_capacity {battery_capacity}"
            )
        self.rows = rows
        self.cols = cols
        self.battery_capacity = battery_capacity
        self.binning = binning if binning is not None else BatteryBinning.dense(battery_capacity)

    @property
    def battery_levels(self) -> int:
        """Length of the table's battery axis: one entry per bin."""
        return self.binning.levels

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
        level = self.binning.level_of(state.battery)
        return (cell * self.battery_levels + level) * NUM_PAYLOAD_STATES + int(state.carried)

    def decode(self, index: int) -> RoverState:
        """Map an integer row ID back to a semantic state.

        The battery of the returned state is the *lowest* reading in the row's bin,
        so under a coarse binning this recovers the row, not the rover: see the
        class docstring.

        Raises:
            ValueError: if ``index`` is outside ``range(num_states)``.
        """
        if not 0 <= index < self.num_states:
            raise ValueError(f"state index {index} outside [0, {self.num_states})")
        index, carried = divmod(index, NUM_PAYLOAD_STATES)
        cell, level = divmod(index, self.battery_levels)
        row, col = divmod(cell, self.cols)
        battery = self.binning.representative(level)
        return RoverState(row=row, col=col, battery=battery, carried=SampleType(carried))

    def grid_view(self, values: NDArray[Any]) -> NDArray[Any]:
        """View a per-state vector as ``(rows, cols, battery_levels, payloads)``.

        Axis 2 is the *binned* battery axis, so it is as long as
        ``binning.levels`` -- 61 entries under the dense encoding, 4 under the
        affordability one.

        The encoding is mixed-radix in exactly this order, so the reshape *is* the
        decode for whole-table quantities -- a visit-count vector, a coverage mask.
        Anything that wants to marginalise battery away, which every map figure
        does, sums axis 2 of this view instead of decoding every index one at a
        time.

        Returns a view into ``values`` when the array is contiguous, so callers
        must not write through it expecting the original to stay untouched.

        Raises:
            ValueError: if ``values`` is not a 1-D array of length
                :attr:`num_states`.
        """
        if values.ndim != 1 or values.shape[0] != self.num_states:
            raise ValueError(
                f"expected a 1-D array of {self.num_states} state values, got shape {values.shape}"
            )
        return values.reshape(self.rows, self.cols, self.battery_levels, NUM_PAYLOAD_STATES)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"StateEncoder(rows={self.rows}, cols={self.cols}, "
            f"battery_capacity={self.battery_capacity}, binning={self.binning!r}, "
            f"num_states={self.num_states})"
        )


__all__ = [
    "COLLECTABLE_SAMPLES",
    "NUM_PAYLOAD_STATES",
    "SAMPLE_LABELS",
    "BatteryBinning",
    "BatteryEncoding",
    "RoverState",
    "SampleType",
    "StateEncoder",
]
