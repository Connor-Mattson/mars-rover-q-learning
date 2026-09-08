"""Scenario parsing, validation, and static geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mars_rover_q.scenario import (
    Scenario,
    ScenarioError,
    Terrain,
    available_scenarios,
    load_scenario,
    resolve_scenario,
    scenario_from_dict,
)
from mars_rover_q.state import COLLECTABLE_SAMPLES, SampleType
from tests.conftest import TINY_SCENARIO, make_scenario


def test_all_three_bundled_scenarios_exist() -> None:
    assert available_scenarios() == [
        "risk_value_tradeoff",
        "safe_corridor",
        "shaping_trap",
    ]


def test_bundled_scenarios_are_valid(bundled_scenario: Scenario) -> None:
    assert 10 <= bundled_scenario.rows <= 14
    assert 10 <= bundled_scenario.cols <= 14
    assert bundled_scenario.battery_capacity > 0
    assert bundled_scenario.max_steps > 0
    assert set(bundled_scenario.samples) == set(COLLECTABLE_SAMPLES)


def test_bundled_scenarios_keep_the_default_sample_values(bundled_scenario: Scenario) -> None:
    values = {t: spec.value for t, spec in bundled_scenario.samples.items()}
    assert values == {
        SampleType.BASALT: 40,
        SampleType.HYDRATED_MINERAL: 90,
        SampleType.BIOSIGNATURE: 160,
    }


def test_bundled_scenarios_are_visually_distinct() -> None:
    grids = {
        name: tuple(resolve_scenario(name).to_dict()["grid"]) for name in available_scenarios()
    }
    assert len(set(grids.values())) == len(grids)


def test_every_sample_is_reachable_and_affordable(bundled_scenario: Scenario) -> None:
    for sample_type in COLLECTABLE_SAMPLES:
        trip = bundled_scenario.round_trip_cost(sample_type)
        assert np.isfinite(trip)
        assert trip < bundled_scenario.battery_capacity


def test_subgoal_heuristic_is_deterministic(bundled_scenario: Scenario) -> None:
    first = bundled_scenario.heuristic_target_sample()
    assert first is bundled_scenario.heuristic_target_sample()
    assert first in COLLECTABLE_SAMPLES


def test_distance_uses_the_cost_of_the_entered_tile() -> None:
    scenario = make_scenario(
        grid=["#####", "#...#", "#.#.#", "#...#", "#####"],
        terrain={
            ".": {
                "energy_cost": 1,
                "move_probabilities": {"forward": 1.0, "stay": 0.0, "left": 0.0, "right": 0.0},
            }
        },
    )
    assert scenario.distance((1, 1), (1, 1)) == 0.0
    assert scenario.distance((1, 1), (1, 2)) == 1.0
    # (1,1) -> (1,2) -> (1,3): two unit-cost tiles entered.
    assert scenario.distance((1, 1), (1, 3)) == 2.0


def test_distance_is_infinite_for_walls() -> None:
    scenario = make_scenario()
    assert not np.isfinite(scenario.distances_to(scenario.lander)[0, 0])


def test_distances_are_cached_and_read_only() -> None:
    scenario = make_scenario()
    first = scenario.distances_to(scenario.lander)
    assert first is scenario.distances_to(scenario.lander)
    with pytest.raises(ValueError):
        first[0, 0] = 1.0


def test_round_trip_through_to_dict(tmp_path: Path) -> None:
    original = resolve_scenario("safe_corridor")
    path = tmp_path / "copy.json"
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    reloaded = load_scenario(path)
    assert np.array_equal(reloaded.grid, original.grid)
    assert reloaded.lander == original.lander
    assert reloaded.samples == original.samples
    assert reloaded.battery_capacity == original.battery_capacity


def _broken(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(TINY_SCENARIO))
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"grid": ["####", "#..#", "#####"]}, "rectangular"),
        ({"grid": ["#####", "#.Z.#", "#####"]}, "unknown terrain symbols"),
        ({"battery_capacity": 0}, "battery_capacity"),
        ({"max_steps": -1}, "max_steps"),
        ({"lander": [0, 0]}, "wall"),
        ({"lander": [99, 99]}, "outside"),
        ({"collect_energy_cost": -1}, "collect_energy_cost"),
    ],
)
def test_validation_rejects_broken_scenarios(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ScenarioError, match=fragment):
        scenario_from_dict(_broken(**overrides))


def test_validation_requires_exactly_three_uniquely_typed_samples() -> None:
    payload = _broken()
    del payload["samples"]["biosignature"]
    with pytest.raises(ScenarioError, match="exactly"):
        scenario_from_dict(payload)


def test_validation_rejects_duplicate_sample_positions() -> None:
    payload = _broken()
    payload["samples"]["basalt"]["position"] = payload["samples"]["hydrated_mineral"]["position"]
    with pytest.raises(ScenarioError, match="distinct"):
        scenario_from_dict(payload)


def test_validation_rejects_unreachable_regions() -> None:
    payload = _broken(
        grid=["#######", "#..#..#", "#..#..#", "#######"],
        lander=[1, 1],
        samples={
            "basalt": {"position": [2, 1], "value": 40},
            "hydrated_mineral": {"position": [2, 2], "value": 90},
            "biosignature": {"position": [1, 2], "value": 160},
        },
    )
    with pytest.raises(ScenarioError, match="unreachable"):
        scenario_from_dict(payload)


def test_validation_rejects_probabilities_that_do_not_sum_to_one() -> None:
    payload = _broken(
        terrain={
            ".": {
                "move_probabilities": {
                    "forward": 0.5,
                    "stay": 0.2,
                    "left": 0.0,
                    "right": 0.0,
                }
            }
        }
    )
    with pytest.raises(ScenarioError, match="sum to"):
        scenario_from_dict(payload)


def test_validation_rejects_out_of_range_probabilities() -> None:
    payload = _broken(
        terrain={
            ".": {
                "move_probabilities": {
                    "forward": 1.4,
                    "stay": -0.4,
                    "left": 0.0,
                    "right": 0.0,
                }
            }
        }
    )
    with pytest.raises(ScenarioError, match=r"\[0, 1\]"):
        scenario_from_dict(payload)


def test_wall_tiles_are_not_traversable() -> None:
    scenario = make_scenario()
    assert scenario.terrain_at((0, 0)) is Terrain.WALL
    assert not scenario.is_traversable((0, 0))
    assert not scenario.is_traversable((-1, 0))
    assert scenario.is_traversable((1, 1))


def test_resolve_scenario_reports_unknown_names() -> None:
    with pytest.raises(FileNotFoundError, match="unknown scenario"):
        resolve_scenario("not_a_scenario")
