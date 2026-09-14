"""Scenario loading, validation, and static geometry (shortest-path costs).

A scenario is a fixed tile map plus the mission parameters that make an episode
well defined. Scenarios are hand-authored JSON; they are never procedurally
regenerated between episodes, because tabular Q-values are tied to one map.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .state import COLLECTABLE_SAMPLES, BatteryBinning, BatteryEncoding, SampleType


class Terrain(IntEnum):
    """Tile types. ``WALL`` is impassable; everything else is traversable."""

    FLAT = 0
    ROUGH = 1
    SAND = 2
    ROCK = 3
    WALL = 4


TERRAIN_SYMBOLS: Final[dict[str, Terrain]] = {
    ".": Terrain.FLAT,
    "~": Terrain.ROUGH,
    "s": Terrain.SAND,
    "^": Terrain.ROCK,
    "#": Terrain.WALL,
}

SYMBOL_FOR_TERRAIN: Final[dict[Terrain, str]] = {v: k for k, v in TERRAIN_SYMBOLS.items()}

SAMPLE_KEYS: Final[dict[str, SampleType]] = {
    "basalt": SampleType.BASALT,
    "hydrated_mineral": SampleType.HYDRATED_MINERAL,
    "biosignature": SampleType.BIOSIGNATURE,
}

KEY_FOR_SAMPLE: Final[dict[SampleType, str]] = {v: k for k, v in SAMPLE_KEYS.items()}

#: Scientific values kept constant across scenarios so cross-scenario plots stay
#: interpretable. Scenario JSON may override them, but the suite in ``configs/``
#: deliberately does not.
DEFAULT_SAMPLE_VALUES: Final[dict[SampleType, int]] = {
    SampleType.BASALT: 40,
    SampleType.HYDRATED_MINERAL: 90,
    SampleType.BIOSIGNATURE: 160,
}

#: Movement outcome names, in the order used by the transition sampler.
OUTCOMES: Final[tuple[str, ...]] = ("forward", "stay", "left", "right")

UNREACHABLE: Final[float] = float("inf")

#: Share of the battery a round trip may consume to count as "affordable" for the
#: shaping subgoal heuristic.
AFFORDABLE_FRACTION: Final[float] = 0.8


@dataclass(frozen=True, slots=True)
class TerrainSpec:
    """Energy cost and movement outcome distribution of one terrain type."""

    energy_cost: int
    move_probabilities: dict[str, float]

    def probability_vector(self) -> NDArray[np.float64]:
        """Outcome probabilities ordered as :data:`OUTCOMES`."""
        return np.array([self.move_probabilities[name] for name in OUTCOMES], dtype=np.float64)


DEFAULT_TERRAIN: Final[dict[Terrain, TerrainSpec]] = {
    Terrain.FLAT: TerrainSpec(1, {"forward": 1.0, "stay": 0.0, "left": 0.0, "right": 0.0}),
    Terrain.ROUGH: TerrainSpec(2, {"forward": 0.75, "stay": 0.15, "left": 0.05, "right": 0.05}),
    Terrain.SAND: TerrainSpec(3, {"forward": 0.6, "stay": 0.3, "left": 0.05, "right": 0.05}),
    Terrain.ROCK: TerrainSpec(4, {"forward": 0.7, "stay": 0.1, "left": 0.1, "right": 0.1}),
    Terrain.WALL: TerrainSpec(0, {"forward": 0.0, "stay": 1.0, "left": 0.0, "right": 0.0}),
}


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """One collectable sample: where it sits and what it is worth."""

    sample_type: SampleType
    position: tuple[int, int]
    value: int


@dataclass(frozen=True, slots=True)
class ShapingSpec:
    """Per-scenario knobs for the intentionally naive dense reward."""

    closer_bonus: float = 2.0
    farther_penalty: float = -1.0


class ScenarioError(ValueError):
    """Raised when a scenario file is structurally or semantically invalid."""


@dataclass(slots=True)
class Scenario:
    """A validated, immutable-in-practice mission map.

    Attributes:
        name: scenario identifier, matching its file stem.
        description: one-line human summary shown by the CLI and renderer.
        grid: ``(rows, cols)`` array of :class:`Terrain` values.
        lander: the ``(row, col)`` home cell.
        samples: the three collectable samples keyed by type.
        battery_capacity: starting battery, also the encoder's battery range.
        max_steps: truncation limit in environment steps.
        collect_energy_cost: energy charged for any ``COLLECT``, valid or not.
        battery_penalty: base reward on battery depletion (negative).
        step_limit_penalty: base reward on step-limit truncation (negative).
        terrain: per-terrain costs and slip distributions.
        shaping: naive-dense shaping magnitudes for this map.
    """

    name: str
    description: str
    grid: NDArray[np.int8]
    lander: tuple[int, int]
    samples: dict[SampleType, SampleSpec]
    battery_capacity: int
    max_steps: int
    collect_energy_cost: int
    battery_penalty: float
    step_limit_penalty: float
    terrain: dict[Terrain, TerrainSpec]
    shaping: ShapingSpec = field(default_factory=ShapingSpec)
    source_path: Path | None = None
    _distance_cache: dict[tuple[int, int], NDArray[np.float64]] = field(
        default_factory=dict, repr=False
    )

    # -- geometry ---------------------------------------------------------

    @property
    def rows(self) -> int:
        """Number of grid rows."""
        return int(self.grid.shape[0])

    @property
    def cols(self) -> int:
        """Number of grid columns."""
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the tile map."""
        return (self.rows, self.cols)

    def terrain_at(self, cell: tuple[int, int]) -> Terrain:
        """Terrain type of ``cell``."""
        return Terrain(int(self.grid[cell[0], cell[1]]))

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        """Whether ``cell`` lies inside the map."""
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_traversable(self, cell: tuple[int, int]) -> bool:
        """Whether ``cell`` is inside the map and not a wall."""
        return self.in_bounds(cell) and self.terrain_at(cell) is not Terrain.WALL

    def energy_cost(self, cell: tuple[int, int]) -> int:
        """Energy charged for occupying ``cell`` after a transition."""
        return self.terrain[self.terrain_at(cell)].energy_cost

    def sample_at(self, cell: tuple[int, int]) -> SampleSpec | None:
        """The sample sitting on ``cell``, if any."""
        for spec in self.samples.values():
            if spec.position == cell:
                return spec
        return None

    def traversable_cells(self) -> list[tuple[int, int]]:
        """Every non-wall cell, in row-major order."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.is_traversable((r, c))
        ]

    # -- static shortest-path costs ---------------------------------------

    def distances_to(self, target: tuple[int, int]) -> NDArray[np.float64]:
        """Energy-weighted shortest-path cost from every cell to ``target``.

        Edge weight is the energy cost of the tile being *entered*, matching the
        environment's accounting rule. Slip is ignored: this is deterministic
        static geometry used for reward shaping, never a learned quantity.
        Unreachable and wall cells hold ``inf``.
        """
        cached = self._distance_cache.get(target)
        if cached is not None:
            return cached

        if not self.is_traversable(target):
            raise ScenarioError(f"distance target {target} is not traversable")

        dist = np.full(self.shape, UNREACHABLE, dtype=np.float64)
        dist[target] = 0.0
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, target)]
        while queue:
            cost, cell = heapq.heappop(queue)
            if cost > dist[cell]:
                continue
            row, col = cell
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                neighbour = (row + dr, col + dc)
                if not self.is_traversable(neighbour):
                    continue
                # Travelling neighbour -> cell charges the cost of entering `cell`.
                candidate = cost + self.energy_cost(cell)
                if candidate < dist[neighbour]:
                    dist[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))

        dist.setflags(write=False)
        self._distance_cache[target] = dist
        return dist

    def distance(self, source: tuple[int, int], target: tuple[int, int]) -> float:
        """Energy-weighted shortest-path cost from ``source`` to ``target``."""
        return float(self.distances_to(target)[source])

    def round_trip_cost(self, sample_type: SampleType) -> float:
        """Lander -> sample -> lander energy-weighted cost."""
        spec = self.samples[sample_type]
        return self.distance(self.lander, spec.position) + self.distance(spec.position, self.lander)

    def mission_costs(self) -> dict[SampleType, float]:
        """Minimum battery to collect and deliver each sample, starting from the lander.

        Lander -> sample -> lander on the energy-weighted shortest path, plus the
        ``COLLECT`` charge. :data:`UNREACHABLE` for a sample no route reaches.
        """
        return {
            sample: self.round_trip_cost(sample) + self.collect_energy_cost
            for sample in COLLECTABLE_SAMPLES
        }

    def battery_binning(
        self, encoding: BatteryEncoding | str = BatteryEncoding.AFFORDABILITY
    ) -> BatteryBinning:
        """The battery axis this scenario's Q-table uses.

        ``dense`` is one row per reading. ``affordability`` bins at
        :meth:`mission_costs` -- the exact battery at which each sample's mission
        stops being payable -- which puts a bin boundary at every charge level where
        the optimal action can change and none anywhere else. Between two adjacent
        thresholds the affordable set is constant, and with it the decision the
        rover faces; the readings in between differ only by a factor of ``gamma``
        that the discount already accounts for.

        On all three bundled scenarios the three samples give three distinct
        thresholds and therefore four bins, which is the shape the default is named
        for; a map whose samples cost the same, or one with an unreachable sample,
        gets correspondingly fewer.
        """
        resolved = BatteryEncoding(encoding)
        if resolved is BatteryEncoding.DENSE:
            return BatteryBinning.dense(self.battery_capacity)
        return BatteryBinning.from_thresholds(
            self.battery_capacity, list(self.mission_costs().values())
        )

    def heuristic_target_sample(self) -> SampleType:
        """The subgoal sample used by both shaping schemes.

        Documented heuristic: take the highest-value sample whose lander round trip
        fits inside :data:`AFFORDABLE_FRACTION` of the battery, breaking ties toward
        the cheaper trip; if nothing fits, fall back to the cheapest reachable
        sample. It is computed once from static geometry and never uses learned
        values, so it is a fixed part of the reward definition rather than something
        the agent influences.
        """
        budget = AFFORDABLE_FRACTION * self.battery_capacity
        costs = {s: self.round_trip_cost(s) for s in COLLECTABLE_SAMPLES}
        affordable = [s for s, cost in costs.items() if cost <= budget]
        if affordable:
            return max(affordable, key=lambda s: (self.samples[s].value, -costs[s]))
        return min(COLLECTABLE_SAMPLES, key=lambda s: (costs[s], -self.samples[s].value))

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-round-trippable snapshot of this scenario."""
        return {
            "name": self.name,
            "description": self.description,
            "grid": [
                "".join(SYMBOL_FOR_TERRAIN[Terrain(int(v))] for v in row) for row in self.grid
            ],
            "lander": list(self.lander),
            "samples": {
                KEY_FOR_SAMPLE[spec.sample_type]: {
                    "position": list(spec.position),
                    "value": spec.value,
                }
                for spec in self.samples.values()
            },
            "battery_capacity": self.battery_capacity,
            "max_steps": self.max_steps,
            "collect_energy_cost": self.collect_energy_cost,
            "battery_penalty": self.battery_penalty,
            "step_limit_penalty": self.step_limit_penalty,
            "terrain": {
                SYMBOL_FOR_TERRAIN[t]: {
                    "energy_cost": spec.energy_cost,
                    "move_probabilities": dict(spec.move_probabilities),
                }
                for t, spec in self.terrain.items()
            },
            "shaping": {
                "closer_bonus": self.shaping.closer_bonus,
                "farther_penalty": self.shaping.farther_penalty,
            },
        }


# -- loading and validation ----------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioError(message)


def _parse_grid(raw_rows: Any, name: str) -> NDArray[np.int8]:
    _require(
        isinstance(raw_rows, list) and len(raw_rows) > 0,
        f"{name}: 'grid' must be a non-empty list of strings",
    )
    _require(
        all(isinstance(row, str) for row in raw_rows),
        f"{name}: every 'grid' row must be a string",
    )
    widths = {len(row) for row in raw_rows}
    _require(len(widths) == 1, f"{name}: grid is not rectangular, row widths {sorted(widths)}")
    _require(widths.pop() > 0, f"{name}: grid rows must be non-empty")
    unknown = sorted({ch for row in raw_rows for ch in row} - set(TERRAIN_SYMBOLS))
    _require(not unknown, f"{name}: unknown terrain symbols {unknown}")
    return np.array(
        [[int(TERRAIN_SYMBOLS[ch]) for ch in row] for row in raw_rows],
        dtype=np.int8,
    )


def _parse_int(raw: Any, name: str, label: str, *, minimum: int) -> int:
    _require(
        isinstance(raw, int) and not isinstance(raw, bool) and raw >= minimum,
        f"{name}: '{label}' must be an integer >= {minimum}",
    )
    return int(raw)


def _parse_cell(raw: Any, name: str, label: str) -> tuple[int, int]:
    _require(
        isinstance(raw, list | tuple) and len(raw) == 2 and all(isinstance(v, int) for v in raw),
        f"{name}: '{label}' must be a [row, col] integer pair",
    )
    return (int(raw[0]), int(raw[1]))


def _parse_terrain(raw: Any, name: str) -> dict[Terrain, TerrainSpec]:
    specs = dict(DEFAULT_TERRAIN)
    if raw is None:
        return specs
    _require(isinstance(raw, dict), f"{name}: 'terrain' must be an object")
    for symbol, override in raw.items():
        _require(symbol in TERRAIN_SYMBOLS, f"{name}: unknown terrain symbol {symbol!r}")
        terrain = TERRAIN_SYMBOLS[symbol]
        base = specs[terrain]
        cost = override.get("energy_cost", base.energy_cost)
        _require(
            isinstance(cost, int) and cost >= 0,
            f"{name}: energy_cost for {symbol!r} must be a non-negative integer",
        )
        probs = dict(base.move_probabilities)
        probs.update(override.get("move_probabilities", {}))
        _require(
            set(probs) == set(OUTCOMES),
            f"{name}: move_probabilities for {symbol!r} must have keys {list(OUTCOMES)}",
        )
        _require(
            all(0.0 <= float(p) <= 1.0 for p in probs.values()),
            f"{name}: move_probabilities for {symbol!r} must lie in [0, 1]",
        )
        total = sum(float(p) for p in probs.values())
        _require(
            abs(total - 1.0) < 1e-9,
            f"{name}: move_probabilities for {symbol!r} sum to {total}, expected 1.0",
        )
        if terrain is not Terrain.WALL:
            _require(
                cost > 0,
                f"{name}: traversable terrain {symbol!r} must have a positive energy cost",
            )
        specs[terrain] = TerrainSpec(int(cost), {k: float(v) for k, v in probs.items()})
    return specs


def _parse_samples(raw: Any, name: str) -> dict[SampleType, SampleSpec]:
    _require(isinstance(raw, dict), f"{name}: 'samples' must be an object")
    _require(
        set(raw) == set(SAMPLE_KEYS),
        f"{name}: 'samples' must define exactly {sorted(SAMPLE_KEYS)}, got {sorted(raw)}",
    )
    samples: dict[SampleType, SampleSpec] = {}
    for key, entry in raw.items():
        sample_type = SAMPLE_KEYS[key]
        position = _parse_cell(entry.get("position"), name, f"samples.{key}.position")
        value = entry.get("value", DEFAULT_SAMPLE_VALUES[sample_type])
        _require(
            isinstance(value, int) and value > 0,
            f"{name}: samples.{key}.value must be a positive integer",
        )
        samples[sample_type] = SampleSpec(sample_type, position, int(value))
    positions = [spec.position for spec in samples.values()]
    _require(
        len(set(positions)) == 3, f"{name}: sample positions must be distinct, got {positions}"
    )
    return samples


def scenario_from_dict(payload: dict[str, Any], *, source_path: Path | None = None) -> Scenario:
    """Build and fully validate a :class:`Scenario` from a parsed JSON object.

    Raises:
        ScenarioError: on any structural or semantic problem. Validation is
            deterministic and reports the first failure it finds.
    """
    name = payload.get("name", source_path.stem if source_path else "<unnamed>")
    _require(isinstance(name, str) and bool(name), "scenario 'name' must be a non-empty string")

    grid = _parse_grid(payload.get("grid"), name)
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    terrain = _parse_terrain(payload.get("terrain"), name)

    battery_capacity = _parse_int(
        payload.get("battery_capacity"), name, "battery_capacity", minimum=1
    )
    max_steps = _parse_int(payload.get("max_steps"), name, "max_steps", minimum=1)
    collect_cost = _parse_int(
        payload.get("collect_energy_cost", 1), name, "collect_energy_cost", minimum=0
    )

    lander = _parse_cell(payload.get("lander"), name, "lander")
    samples = _parse_samples(payload.get("samples"), name)

    scenario = Scenario(
        name=name,
        description=str(payload.get("description", "")),
        grid=grid,
        lander=lander,
        samples=samples,
        battery_capacity=battery_capacity,
        max_steps=max_steps,
        collect_energy_cost=collect_cost,
        battery_penalty=float(payload.get("battery_penalty", -100.0)),
        step_limit_penalty=float(payload.get("step_limit_penalty", -100.0)),
        terrain=terrain,
        shaping=ShapingSpec(
            closer_bonus=float(payload.get("shaping", {}).get("closer_bonus", 2.0)),
            farther_penalty=float(payload.get("shaping", {}).get("farther_penalty", -1.0)),
        ),
        source_path=source_path,
    )

    _require(
        scenario.in_bounds(lander),
        f"{name}: lander {lander} outside the {rows}x{cols} grid",
    )
    _require(scenario.is_traversable(lander), f"{name}: lander {lander} sits on a wall")
    for key, spec in samples.items():
        label = KEY_FOR_SAMPLE[key]
        _require(
            scenario.in_bounds(spec.position),
            f"{name}: sample {label} at {spec.position} outside the grid",
        )
        _require(
            scenario.is_traversable(spec.position),
            f"{name}: sample {label} at {spec.position} sits on a wall",
        )
        _require(
            spec.position != lander,
            f"{name}: sample {label} may not share the lander cell",
        )

    reachable = scenario.distances_to(lander)
    unreachable = [
        cell for cell in scenario.traversable_cells() if not np.isfinite(reachable[cell])
    ]
    _require(
        not unreachable,
        f"{name}: traversable cells unreachable from the lander: {unreachable[:8]}",
    )

    _require(
        scenario.shaping.closer_bonus > 0.0,
        f"{name}: shaping.closer_bonus must be positive",
    )
    _require(
        scenario.shaping.farther_penalty < 0.0,
        f"{name}: shaping.farther_penalty must be negative",
    )
    return scenario


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario JSON file."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"scenario file not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return scenario_from_dict(payload, source_path=resolved)


def scenario_directory() -> Path:
    """The bundled ``configs/scenarios`` directory."""
    return Path(__file__).resolve().parents[2] / "configs" / "scenarios"


def resolve_scenario(reference: str | Path) -> Scenario:
    """Load a scenario by bare name (``safe_corridor``) or by file path."""
    candidate = Path(reference)
    if candidate.suffix == ".json" and candidate.exists():
        return load_scenario(candidate)
    bundled = scenario_directory() / f"{Path(reference).stem}.json"
    if bundled.exists():
        return load_scenario(bundled)
    raise FileNotFoundError(
        f"unknown scenario {reference!r}; expected a JSON path or one of "
        f"{sorted(p.stem for p in scenario_directory().glob('*.json'))}"
    )


def available_scenarios() -> list[str]:
    """Names of the bundled scenarios."""
    return sorted(p.stem for p in scenario_directory().glob("*.json"))


__all__ = [
    "AFFORDABLE_FRACTION",
    "DEFAULT_SAMPLE_VALUES",
    "DEFAULT_TERRAIN",
    "OUTCOMES",
    "SAMPLE_KEYS",
    "TERRAIN_SYMBOLS",
    "UNREACHABLE",
    "SampleSpec",
    "Scenario",
    "ScenarioError",
    "ShapingSpec",
    "Terrain",
    "TerrainSpec",
    "available_scenarios",
    "load_scenario",
    "resolve_scenario",
    "scenario_directory",
    "scenario_from_dict",
]
