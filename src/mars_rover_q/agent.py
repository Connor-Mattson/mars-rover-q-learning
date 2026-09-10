"""Tabular Q-learning agent.

The five functions below -- table allocation, the one-step target, the TD error,
the in-place update, and epsilon-greedy action selection -- are the whole learning
algorithm. They were implemented by Connor; everything else in the package is
plumbing around them.

These algorithms live here and nowhere else -- not in the trainer, not in tests,
not in helper modules. The real training path calls exactly these functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .actions import NUM_ACTIONS

QTable = NDArray[np.float64]


# ---------------------------------------------------------------------------
# The learning algorithm
# ---------------------------------------------------------------------------


def initialize_q_table(
    num_states: int,
    num_actions: int = NUM_ACTIONS,
    *,
    initial_value: float = 0.0,
    dtype: type[np.floating] = np.float64,
) -> QTable:
    """Allocate the state-by-action value table.

    Args:
        num_states: number of encodable states; must be strictly positive.
        num_actions: number of actions; must be strictly positive.
        initial_value: the value every entry starts at. Optimistic initialisation
            (a value above any achievable return) is a legitimate exploration
            device, so this must be honoured exactly, not assumed to be zero.
        dtype: floating point type of the table.

    Returns:
        A newly allocated array of shape ``(num_states, num_actions)`` whose every
        entry equals ``initial_value``, using ``dtype``. The caller owns the array;
        no state is shared between calls.

    Raises:
        ValueError: if ``num_states`` or ``num_actions`` is zero or negative.

    Invariants:
        The returned array is independent of any previously returned array.
    """
    if num_states < 1:
        raise ValueError(
            "You must have at least 1 state in the MDP! "
            "Please call initialize_q_table with param num_states >= 1"
        )
    if num_actions < 1:
        raise ValueError(
            "You must have at least 1 action in the MDP! "
            "Please call initialize_q_table with param num_actions >= 1"
        )
    table = np.full((num_states, num_actions), initial_value, dtype=dtype)
    return table


def calculate_target(
    reward: float,
    next_state_values: NDArray[np.float64],
    gamma: float,
    terminated: bool,
    truncated: bool,
) -> float:
    """Compute the one-step off-policy Q-learning target for a transition.

    Args:
        reward: the reward received on this transition (base plus shaping).
        next_state_values: the row of the Q-table for the successor state, i.e. the
            estimated value of every action available there.
        gamma: discount factor in ``(0, 1]``.
        terminated: the successor state ended the episode (delivery, battery
            depletion).
        truncated: the episode hit its step limit.

    Returns:
        The scalar the current estimate should be moved toward.

    Notes:
        Q-learning is off-policy: the bootstrap uses the *best* successor action,
        not the one the behaviour policy will actually take. In this educational
        project both ``terminated`` and ``truncated`` are treated as episode
        boundaries that must not be bootstrapped across, so the target must not
        depend on ``next_state_values`` in either case. This convention is stated in
        the README; it makes truncated episodes slightly pessimistic, which is the
        trade accepted here for a simpler, honest learning rule.
    """
    if terminated or truncated:
        return float(reward)
    return float(reward + (gamma * (max(next_state_values))))


def calculate_td_error(current_estimate: float, target: float) -> float:
    """Compute the signed temporal-difference error.

    Args:
        current_estimate: the table's present value for the visited state-action
            pair.
        target: the value returned by :func:`calculate_target`.

    Returns:
        The signed discrepancy, defined so that a *positive* result means the
        outcome was better than the table currently believes and the entry should
        move up. Sign matters: it is what makes the update a correction rather than
        a drift.
    """
    return target - current_estimate


def update_q_value(
    q_table: QTable,
    state_index: int,
    action_index: int,
    learning_rate: float,
    td_error: float,
) -> None:
    """Apply the incremental update to exactly one table entry, in place.

    Args:
        q_table: the table to modify. Mutated in place; nothing is returned.
        state_index: row of the visited state.
        action_index: column of the taken action.
        learning_rate: step size in ``(0, 1]``. A learning rate of ``1`` should
            move the entry all the way to the target; a smaller one moves it
            proportionally less.
        td_error: the signed error from :func:`calculate_td_error`.

    Invariants:
        Exactly one entry changes. Every other entry of ``q_table`` -- including
        other actions of the same state -- must be bit-for-bit unchanged.
    """
    q_prev = q_table[state_index][action_index]
    q_table[state_index][action_index] = q_prev + (learning_rate * (td_error))
    return None


def select_action(
    q_table: QTable,
    state_index: int,
    epsilon: float,
    rng: np.random.Generator,
    action_mask: NDArray[np.bool_] | None = None,
) -> int:
    """Choose an action with an epsilon-greedy policy.

    Args:
        q_table: the current value table.
        state_index: row of the state to act in.
        epsilon: exploration probability in ``[0, 1]``. With ``epsilon == 0`` the
            choice is purely greedy; with ``epsilon == 1`` it is uniform over the
            legal actions.
        rng: the injected generator. Every random draw must come from it -- never
            from ``numpy.random`` module-level functions -- so runs are reproducible
            from their seed.
        action_mask: optional boolean array, ``True`` where an action is legal. When
            given, both exploration and greedy selection are restricted to legal
            actions. When ``None``, all actions are legal.

    Returns:
        The chosen action index.

    Notes:
        Exploration must be uniform across the legal actions, and ties among the
        greedy-best actions must be broken *randomly*. Always returning the lowest
        index on a tie biases a freshly initialised table -- where every action
        ties -- toward one direction and quietly cripples early exploration.

    Raises:
        ValueError: if ``action_mask`` marks no action as legal.
    """
    legal = (
        np.flatnonzero(action_mask).tolist()
        if action_mask is not None
        else list(range(q_table.shape[1]))
    )
    if len(legal) == 0:
        raise ValueError(
            "There are no valid actions for the action mask. "
            "Check that action_mask passed to select_action has len > 0!"
        )

    # Obtain Max Q Value
    q_vals = q_table[state_index]
    legal_actions = [(float(q_vals[i]), i) for i in legal]
    max_q = max(legal_actions)[0]
    max_a_candidates = [g[1] for g in legal_actions if g[0] == max_q]

    if rng.random() < epsilon:
        return int(rng.choice(legal))
    return int(rng.choice(max_a_candidates))


# ---------------------------------------------------------------------------
# Supporting machinery (already complete)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpsilonSchedule:
    """Linear exploration decay measured in episodes.

    ``epsilon`` falls from ``start`` to ``end`` over ``decay_episodes`` episodes and
    then stays flat. Evaluation uses ``epsilon = 0`` instead of this schedule.
    """

    start: float = 1.0
    end: float = 0.05
    decay_episodes: int = 2000

    def __post_init__(self) -> None:
        if not 0.0 <= self.end <= self.start <= 1.0:
            raise ValueError(
                f"require 0 <= end <= start <= 1, got start={self.start}, end={self.end}"
            )
        if self.decay_episodes <= 0:
            raise ValueError(f"decay_episodes must be positive, got {self.decay_episodes}")

    def value_at(self, episode: int) -> float:
        """Exploration rate for a zero-based episode index."""
        if episode >= self.decay_episodes:
            return self.end
        fraction = episode / self.decay_episodes
        return self.start + (self.end - self.start) * fraction


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    """Hyper-parameters of the tabular Q-learning agent."""

    learning_rate: float = 0.2
    gamma: float = 0.99
    initial_q: float = 0.0
    epsilon: EpsilonSchedule = EpsilonSchedule()

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(f"learning_rate must lie in (0, 1], got {self.learning_rate}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in (0, 1], got {self.gamma}")


#: Human-readable descriptions used by the teaching-state warning.
HUMAN_OWNED_FUNCTIONS: tuple[str, ...] = (
    "initialize_q_table",
    "calculate_target",
    "calculate_td_error",
    "update_q_value",
    "select_action",
)


def teaching_stub_status() -> tuple[str, ...]:
    """Probe the five human-owned functions and report which still look unfinished.

    This is a behavioural smoke check, not a grader: it exercises each function on
    a trivial input and reports the ones whose documented contract is obviously not
    met yet. It exists so that ``train`` can print a conspicuous warning instead of
    silently producing a meaningless Q-table.
    """
    pending: list[str] = []

    try:
        table = initialize_q_table(2, 3, initial_value=0.5)
        if table.shape != (2, 3) or not np.allclose(table, 0.5):
            pending.append("initialize_q_table")
    except Exception:
        pending.append("initialize_q_table")

    try:
        bootstrapped = calculate_target(1.0, np.array([10.0, 0.0, 0.0]), 0.9, False, False)
        boundary = calculate_target(1.0, np.array([10.0, 0.0, 0.0]), 0.9, True, False)
        if not (bootstrapped > boundary + 1e-9):
            pending.append("calculate_target")
    except Exception:
        pending.append("calculate_target")

    try:
        if abs(calculate_td_error(2.0, 5.0)) < 1e-12:
            pending.append("calculate_td_error")
    except Exception:
        pending.append("calculate_td_error")

    try:
        scratch = np.zeros((2, 2), dtype=np.float64)
        update_q_value(scratch, 0, 0, 0.5, 2.0)
        if scratch[0, 0] == 0.0:
            pending.append("update_q_value")
    except Exception:
        pending.append("update_q_value")

    try:
        probe_rng = np.random.default_rng(0)
        probe_table = np.zeros((1, NUM_ACTIONS), dtype=np.float64)
        drawn = {select_action(probe_table, 0, 1.0, probe_rng) for _ in range(50)}
        if len(drawn) < 2:
            pending.append("select_action")
    except Exception:
        pending.append("select_action")

    return tuple(pending)


__all__ = [
    "HUMAN_OWNED_FUNCTIONS",
    "EpsilonSchedule",
    "QLearningConfig",
    "QTable",
    "calculate_target",
    "calculate_td_error",
    "initialize_q_table",
    "select_action",
    "teaching_stub_status",
    "update_q_value",
]
