"""Contract tests for the five human-owned Q-learning functions.

These are EXPECTED TO FAIL until ``src/mars_rover_q/agent.py`` is implemented.
They are deselected from the default ``pytest`` run and are executed with::

    pytest -m human_todo tests/human_todo

Do not skip, weaken, xfail, or delete them. Every failure here should be a plain
assertion failure caused by a placeholder, never an import or fixture error.
"""

from __future__ import annotations

import numpy as np
import pytest

from mars_rover_q.actions import NUM_ACTIONS, Action
from mars_rover_q.agent import (
    calculate_target,
    calculate_td_error,
    initialize_q_table,
    select_action,
    update_q_value,
)

pytestmark = pytest.mark.human_todo


# -- initialize_q_table ---------------------------------------------------


def test_q_table_has_the_requested_shape() -> None:
    table = initialize_q_table(12, 5)
    assert table.shape == (12, 5)


def test_q_table_defaults_to_the_full_action_set() -> None:
    assert initialize_q_table(7).shape == (7, NUM_ACTIONS)


def test_q_table_uses_the_requested_dtype() -> None:
    assert initialize_q_table(3, 2).dtype == np.float64
    assert initialize_q_table(3, 2, dtype=np.float32).dtype == np.float32


def test_q_table_honours_a_zero_initial_value() -> None:
    assert np.all(initialize_q_table(4, 3, initial_value=0.0) == 0.0)


def test_q_table_honours_a_nonzero_initial_value() -> None:
    """Optimistic initialisation is a real exploration device; it must be honoured."""
    table = initialize_q_table(4, 3, initial_value=7.5)
    assert np.all(table == 7.5)


def test_q_table_honours_a_negative_initial_value() -> None:
    assert np.all(initialize_q_table(2, 2, initial_value=-3.25) == -3.25)


@pytest.mark.parametrize(
    ("num_states", "num_actions"),
    [(0, 5), (5, 0), (-1, 5), (5, -1), (0, 0)],
)
def test_q_table_rejects_invalid_dimensions(num_states: int, num_actions: int) -> None:
    with pytest.raises(ValueError):
        initialize_q_table(num_states, num_actions)


def test_q_tables_do_not_share_memory() -> None:
    first = initialize_q_table(3, 2)
    second = initialize_q_table(3, 2)
    first[0, 0] = 99.0
    assert second[0, 0] == 0.0


# -- calculate_target -----------------------------------------------------


def test_target_bootstraps_on_an_ordinary_transition() -> None:
    next_values = np.array([1.0, 5.0, -2.0, 0.0, 3.0])
    target = calculate_target(2.0, next_values, 0.9, terminated=False, truncated=False)
    assert target == pytest.approx(2.0 + 0.9 * 5.0)


def test_target_uses_the_best_successor_action_not_the_first() -> None:
    next_values = np.array([-10.0, -10.0, 4.0, -10.0, -10.0])
    target = calculate_target(0.0, next_values, 0.5, terminated=False, truncated=False)
    assert target == pytest.approx(2.0)


def test_target_does_not_bootstrap_across_termination() -> None:
    next_values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    target = calculate_target(-100.0, next_values, 0.99, terminated=True, truncated=False)
    assert target == pytest.approx(-100.0)


def test_target_does_not_bootstrap_across_truncation() -> None:
    next_values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    target = calculate_target(-100.0, next_values, 0.99, terminated=False, truncated=True)
    assert target == pytest.approx(-100.0)


def test_target_respects_the_discount_factor() -> None:
    next_values = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    low = calculate_target(0.0, next_values, 0.1, terminated=False, truncated=False)
    high = calculate_target(0.0, next_values, 0.99, terminated=False, truncated=False)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(9.9)


# -- calculate_td_error ---------------------------------------------------


def test_td_error_is_positive_when_the_target_exceeds_the_estimate() -> None:
    assert calculate_td_error(1.0, 4.0) == pytest.approx(3.0)


def test_td_error_is_zero_when_the_estimate_is_already_correct() -> None:
    assert calculate_td_error(2.5, 2.5) == pytest.approx(0.0)


def test_td_error_is_negative_when_the_estimate_is_too_high() -> None:
    assert calculate_td_error(6.0, 1.5) == pytest.approx(-4.5)


# -- update_q_value -------------------------------------------------------


def test_update_moves_the_selected_entry_by_the_scaled_error() -> None:
    table = np.zeros((3, 5), dtype=np.float64)
    update_q_value(table, 1, 2, 0.5, 4.0)
    assert table[1, 2] == pytest.approx(2.0)


def test_update_with_a_learning_rate_of_one_lands_on_the_target() -> None:
    table = np.zeros((2, 2), dtype=np.float64)
    target = 7.0
    error = calculate_td_error(float(table[0, 1]), target)
    update_q_value(table, 0, 1, 1.0, error)
    assert table[0, 1] == pytest.approx(target)


def test_update_scales_with_the_learning_rate() -> None:
    slow = np.zeros((1, 1), dtype=np.float64)
    fast = np.zeros((1, 1), dtype=np.float64)
    update_q_value(slow, 0, 0, 0.1, 10.0)
    update_q_value(fast, 0, 0, 0.4, 10.0)
    assert fast[0, 0] == pytest.approx(4.0 * slow[0, 0])
    assert slow[0, 0] == pytest.approx(1.0)


def test_update_touches_exactly_one_entry() -> None:
    table = np.arange(15, dtype=np.float64).reshape(3, 5)
    before = table.copy()
    update_q_value(table, 2, 3, 0.5, 6.0)
    changed = np.flatnonzero(table.ravel() != before.ravel())
    assert changed.tolist() == [2 * 5 + 3]


def test_update_accumulates_across_repeated_calls() -> None:
    table = np.zeros((1, 1), dtype=np.float64)
    update_q_value(table, 0, 0, 0.5, 2.0)
    update_q_value(table, 0, 0, 0.5, 2.0)
    assert table[0, 0] == pytest.approx(2.0)


# -- select_action --------------------------------------------------------


def greedy_table() -> np.ndarray:
    table = np.zeros((1, NUM_ACTIONS), dtype=np.float64)
    table[0, Action.EAST] = 5.0
    return table


def test_epsilon_zero_is_purely_greedy() -> None:
    rng = np.random.default_rng(0)
    table = greedy_table()
    chosen = {select_action(table, 0, 0.0, rng) for _ in range(50)}
    assert chosen == {int(Action.EAST)}


def test_epsilon_one_explores_every_action_uniformly() -> None:
    rng = np.random.default_rng(1)
    table = greedy_table()
    counts = np.bincount(
        [select_action(table, 0, 1.0, rng) for _ in range(4000)], minlength=NUM_ACTIONS
    )
    assert counts.min() > 0
    expected = 4000 / NUM_ACTIONS
    assert counts.max() < expected * 1.25
    assert counts.min() > expected * 0.75


def test_intermediate_epsilon_is_reproducible_under_a_fixed_seed() -> None:
    table = greedy_table()
    first = [select_action(table, 0, 0.3, np.random.default_rng(7)) for _ in range(1)]
    repeated = [select_action(table, 0, 0.3, np.random.default_rng(7)) for _ in range(1)]
    assert first == repeated

    rng = np.random.default_rng(11)
    sequence = [select_action(table, 0, 0.3, rng) for _ in range(200)]
    assert set(sequence) == set(range(NUM_ACTIONS))
    assert sequence.count(int(Action.EAST)) > 200 * 0.5


def test_ties_are_broken_randomly_not_by_lowest_index() -> None:
    rng = np.random.default_rng(3)
    table = np.zeros((1, NUM_ACTIONS), dtype=np.float64)
    chosen = {select_action(table, 0, 0.0, rng) for _ in range(200)}
    assert chosen == set(range(NUM_ACTIONS))


def test_partial_ties_are_broken_among_the_best_actions_only() -> None:
    rng = np.random.default_rng(5)
    table = np.zeros((1, NUM_ACTIONS), dtype=np.float64)
    table[0, Action.NORTH] = 3.0
    table[0, Action.WEST] = 3.0
    chosen = {select_action(table, 0, 0.0, rng) for _ in range(200)}
    assert chosen == {int(Action.NORTH), int(Action.WEST)}


def test_action_mask_restricts_greedy_selection() -> None:
    rng = np.random.default_rng(2)
    table = greedy_table()
    mask = np.ones(NUM_ACTIONS, dtype=np.bool_)
    mask[Action.EAST] = False
    table[0, Action.SOUTH] = 1.0
    chosen = {select_action(table, 0, 0.0, rng, mask) for _ in range(50)}
    assert chosen == {int(Action.SOUTH)}


def test_action_mask_restricts_exploration() -> None:
    rng = np.random.default_rng(4)
    table = greedy_table()
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[int(Action.NORTH)] = True
    mask[int(Action.COLLECT)] = True
    chosen = {select_action(table, 0, 1.0, rng, mask) for _ in range(200)}
    assert chosen == {int(Action.NORTH), int(Action.COLLECT)}


def test_an_empty_action_mask_is_rejected() -> None:
    rng = np.random.default_rng(6)
    table = greedy_table()
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    with pytest.raises(ValueError):
        select_action(table, 0, 0.0, rng, mask)


def test_selection_reads_the_requested_state_row() -> None:
    rng = np.random.default_rng(8)
    table = np.zeros((2, NUM_ACTIONS), dtype=np.float64)
    table[0, Action.NORTH] = 9.0
    table[1, Action.SOUTH] = 9.0
    assert select_action(table, 1, 0.0, rng) == int(Action.SOUTH)
