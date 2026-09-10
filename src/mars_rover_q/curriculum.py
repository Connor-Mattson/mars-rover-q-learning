"""An annealed curriculum over physically reachable start states.

Why this exists: with epsilon-greedy exploration from the lander alone, the
probability of stumbling onto a distant, high-value sample *and* carrying it home
before the battery runs out falls off exponentially with the length of the route.
On the bundled maps that leaves most of the state space untouched -- every action
of an untouched state still holds its initial value, so
:func:`mars_rover_q.agent.select_action` breaks a five-way tie at random and the
greedy policy renders as noise. :func:`mars_rover_q.metrics.tied_state_fraction`
measures exactly that.

The fix implemented here is a *start-state* curriculum, not a reward change and not
an MDP change. Early training episodes begin from states close to a finished
mission -- carrying a sample, a few tiles from home -- so the terminal reward is
reachable by chance and can propagate backwards. As training progresses the start
distribution widens toward harder states and finally collapses onto the canonical
lander start, so the tail of training is on-distribution.

Two rules keep this honest:

* Only *physically reachable* states are ever used as starts. A start state must be
  one the rover could have driven itself into from the lander, and one it can still
  finish the mission from. Teleporting the rover into a state it could never occupy
  would train values for a mission that does not exist.
* Evaluation never uses the curriculum. :mod:`mars_rover_q.evaluation` always resets
  to the canonical lander start, so reported success rates are comparable with runs
  that trained without a curriculum.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .scenario import UNREACHABLE, Scenario
from .state import COLLECTABLE_SAMPLES, RoverState, SampleType, StateEncoder


class CurriculumStrategy(StrEnum):
    """How a training episode's start state is drawn from the ranked pool.

    Every strategy answers the same two questions, and they differ only in the
    answers: which ranks are *admissible* at a given anneal progress, and how the
    draw is spread over them.

    ``GROWING``
        Uniform over the easiest ``floor(N * progress)`` states. The original
        schedule. Its support only ever widens, so a state admitted in the first
        hundred episodes keeps full weight for the rest of the anneal and the easy
        end is drawn far more often than the hard end it is supposed to be making
        way for.
    ``SLIDING``
        Uniform over a band of fixed width that *translates* from the easy end to
        the hard end. Solved states are retired rather than accumulated, so the
        draw tracks the frontier instead of averaging over everything behind it.
    ``VISIT_WEIGHTED``
        The ``GROWING`` admission rule -- deliberately identical, so the comparison
        isolates one variable -- with the uniform draw replaced by one that
        downweights states whose Q-values have already been updated many times.
    """

    GROWING = "growing"
    SLIDING = "sliding"
    VISIT_WEIGHTED = "visit_weighted"


CURRICULUM_STRATEGY_LABELS: Final[dict[CurriculumStrategy, str]] = {
    CurriculumStrategy.GROWING: "Growing Window",
    CurriculumStrategy.SLIDING: "Sliding Window",
    CurriculumStrategy.VISIT_WEIGHTED: "Visit-Weighted",
}

#: ``curriculum_fraction`` at or below which no curriculum runs at all. The control
#: arm of every comparison, and the reason a strategy name is meaningless there.
BASELINE_CURRICULUM: Final[float] = 0.0

#: Width of the :attr:`CurriculumStrategy.SLIDING` band, as a share of the pool.
#: A quarter is the widest band that still satisfies the "starts at the easy end"
#: invariant the growing schedule is held to, so the two strategies begin from
#: comparable support and diverge only in what happens afterwards.
DEFAULT_WINDOW_FRACTION: Final[float] = 0.25

#: Exponent on the visit-weighted tilt. ``1.0`` is inverse-linear: a state with
#: nine updates is drawn a tenth as often as an untouched one. ``0.0`` recovers the
#: uniform draw exactly, which makes it the control for the weighting itself.
DEFAULT_WEIGHT_EXPONENT: Final[float] = 1.0


def curriculum_arm_label(
    fraction: float, strategy: CurriculumStrategy | str = CurriculumStrategy.GROWING
) -> str:
    """Legend text for one experimental arm.

    Named by what the arm *is*, not by the parameters that switch it on: a reader
    of a figure has no reason to know that ``0.5`` is an anneal fraction, and the
    number invites the misreading that it is a performance value. A disabled
    curriculum has no strategy -- there is one control arm, not three -- so the
    strategy is dropped rather than printed as a distinction that does not exist.
    """
    if fraction <= BASELINE_CURRICULUM:
        return "No Curriculum"
    return f"{CURRICULUM_STRATEGY_LABELS[CurriculumStrategy(strategy)]} Curriculum"


def curriculum_arm_slug(
    fraction: float, strategy: CurriculumStrategy | str = CurriculumStrategy.GROWING
) -> str:
    """:func:`curriculum_arm_label` as a filename segment."""
    return curriculum_arm_label(fraction, strategy).lower().replace(" ", "-")


def canonical_start_state(scenario: Scenario) -> RoverState:
    """The mission's real start: parked on the lander, full battery, empty bay."""
    return RoverState(
        row=scenario.lander[0],
        col=scenario.lander[1],
        battery=scenario.battery_capacity,
        carried=SampleType.NONE,
    )


def start_state_difficulty(scenario: Scenario, state: RoverState) -> float:
    """Energy-weighted cost of *finishing* the mission from ``state``.

    Carrying a sample, that is the cost of driving home. Empty-handed, it is the
    cheapest remaining collect-and-return over the three samples, including the
    ``COLLECT`` charge. Distances are the same static, deterministic Dijkstra
    fields the shaping models use, so slip is ignored: this is a ranking, not a
    prediction.

    Returns:
        The remaining cost, or :data:`mars_rover_q.scenario.UNREACHABLE` when no
        delivery can be completed from ``state`` at all.

    Note:
        Lower is easier. The canonical start sits near the hard end of the range by
        construction -- the whole mission is still ahead of it.
    """
    if state.carried is not SampleType.NONE:
        return scenario.distance(state.position, scenario.lander)
    best = UNREACHABLE
    for sample in COLLECTABLE_SAMPLES:
        cell = scenario.samples[sample].position
        cost = (
            scenario.distance(state.position, cell)
            + scenario.collect_energy_cost
            + scenario.distance(cell, scenario.lander)
        )
        best = min(best, cost)
    return best


def enumerate_start_states(scenario: Scenario) -> list[RoverState]:
    """Every state the curriculum is allowed to start a training episode from.

    A state is admissible only when *both* halves of its history and future are
    physically possible on this map:

    * **It could have got there.** The rover starts every real episode on the
      lander with a full battery and pays the energy cost of each tile it enters
      (:meth:`Scenario.distance` is that exact cost, and already excludes the tile
      being left). So the energy already missing from ``state.battery`` must be at
      least the cheapest route that would explain the state: straight from the
      lander when the bay is empty, and via the carried sample's own cell -- plus
      ``scenario.collect_energy_cost`` for the ``COLLECT`` itself -- when it is
      not. Spending *more* than the cheapest route is always possible (the rover
      can wander); spending less is not.
    * **It can still finish.** The remaining battery must cover
      :func:`start_state_difficulty`. A start the rover provably cannot deliver
      from can only teach failure, and would drag the curriculum's easy end into
      states with no reachable reward at all.

    Excluded on top of that: wall and out-of-bounds cells; a flat battery, which is
    already terminal; and carrying a sample while parked on the lander, which the
    environment scores as a completed delivery the moment it steps.

    Args:
        scenario: the fixed map, its sample positions, battery capacity, and
            energy costs.

    Returns:
        A list of admissible :class:`RoverState` values, without duplicates and in
        any order -- :class:`StartStateCurriculum` sorts it by difficulty. The
        canonical start is itself admissible and belongs in the list.

    Invariants:
        Every returned state encodes without error, is non-terminal, and satisfies
        both conditions above. The scenario is not mutated.
    """
    valid_carries = [*list(COLLECTABLE_SAMPLES), SampleType.NONE]
    start_states = []

    for curr_cell in scenario.traversable_cells():
        for sample_type in valid_carries:
            # Determine how far we would have come since starting from the lander
            dist_from_start = scenario.distance(scenario.lander, curr_cell)
            if sample_type != SampleType.NONE:
                sample_cell = scenario.samples[sample_type].position
                dist_from_start = (
                    scenario.distance(scenario.lander, sample_cell)
                    + scenario.collect_energy_cost
                    + scenario.distance(sample_cell, curr_cell)
                )

            # Use the best-case distance from start to determine the best-case battery level
            best_case_battery = int(scenario.battery_capacity - dist_from_start)
            if best_case_battery <= 0:
                continue

            # Consider all possible battery values
            battery_level = best_case_battery
            solvable_w_battery = True
            while solvable_w_battery:
                candidate_state = RoverState(
                    row=curr_cell[0], col=curr_cell[1], battery=battery_level, carried=sample_type
                )
                candidate_difficulty = start_state_difficulty(scenario, candidate_state)
                reachable_w_battery_lvl = battery_level >= candidate_difficulty and (
                    candidate_difficulty is not UNREACHABLE
                )
                if reachable_w_battery_lvl and candidate_difficulty > 0.0:
                    start_states.append(candidate_state)
                else:
                    solvable_w_battery = False
                battery_level -= 1
    return start_states


def sample_start_state(
    ranked_pool: Sequence[RoverState],
    canonical: RoverState,
    progress: float,
    rng: np.random.Generator,
) -> RoverState:
    """Pick the start state for one training episode.

    This is the schedule itself: how quickly the start distribution widens from the
    easy end of the pool toward the hard one, and how the canonical start takes
    over by the end of the anneal. A schedule that widens too fast never gives the
    terminal reward time to propagate; one that widens too slowly spends the budget
    on states the agent has already solved.

    Args:
        ranked_pool: admissible start states sorted *easiest first* -- index ``0``
            is the cheapest state to finish from, the last index the hardest.
            Treat it as read-only.
        canonical: the real mission start, which the policy is ultimately evaluated
            from. It is itself a member of ``ranked_pool``, though not necessarily a
            hard one: difficulty is remaining cost-to-go, and from the lander the
            cheapest itinerary can be shorter than a long haul home with a sample.
        progress: how far through the anneal this episode is, in ``[0, 1]``.
        rng: the injected generator. Every draw must come from it, never from
            ``numpy.random`` module-level functions.

    Returns:
        The state the episode should begin from: either ``canonical`` or a member
        of ``ranked_pool``.

    Raises:
        ValueError: if ``progress`` falls outside ``[0, 1]``.

    Invariants:
        * ``progress >= 1.0`` must return ``canonical`` exactly -- training has to
          end on the distribution it is evaluated on.
        * At ``progress == 0.0`` the support must stay inside the easiest quarter of
          the pool, not spread across the whole of it; that is the entire point of
          annealing.
        * The support must not shrink as ``progress`` grows, and by late progress it
          must actually reach states the early support excluded.
        * An empty ``ranked_pool`` must return ``canonical`` rather than raise, so
          that a curriculum with nothing to offer degrades to ordinary training.
        * ``ranked_pool`` is never reordered or mutated.
    """
    if progress < 0.0 or progress > 1.0:
        raise ValueError("sample_start_state progress illegally lies outside the range [0, 1]")

    if len(ranked_pool) == 0 or progress == 1.0:
        return canonical

    # Determine 'k' states to select from
    k = max(1, int(len(ranked_pool) * progress))

    # Sample from the 'k' easiest states and return
    i = int(rng.integers(k))
    return ranked_pool[i]


def growing_window_bounds(pool_size: int, progress: float) -> tuple[int, int]:
    """The half-open rank band :attr:`CurriculumStrategy.GROWING` admits.

    Factored out of :func:`sample_start_state`, which computes the same band
    inline, so that :func:`sample_visit_weighted_start_state` can admit *exactly*
    the same states and differ from it in the weights alone. If the two admission
    rules ever drift apart the visit-weighted arm stops being a controlled
    comparison and starts being two changes at once.

    Returns:
        ``(0, k)`` with ``k`` at least ``1`` for a non-empty pool, and ``(0, 0)``
        for an empty one.
    """
    if pool_size <= 0:
        return (0, 0)
    return (0, max(1, int(pool_size * progress)))


def sliding_window_bounds(
    pool_size: int,
    progress: float,
    width_fraction: float = DEFAULT_WINDOW_FRACTION,
) -> tuple[int, int]:
    """The half-open rank band :attr:`CurriculumStrategy.SLIDING` admits.

    A band of ``round(pool_size * width_fraction)`` ranks whose left edge travels
    from ``0`` to ``pool_size - width`` as ``progress`` runs from ``0`` to ``1``.
    Width is held constant rather than the easy edge, which is the whole point:
    the number of states in play stays the same while *which* states they are moves
    on, so nothing is oversampled merely because it was admitted early.

    Raises:
        ValueError: if ``width_fraction`` is outside ``(0, 1]``.
    """
    if not 0.0 < width_fraction <= 1.0:
        raise ValueError(f"width_fraction must lie in (0, 1], got {width_fraction}")
    if pool_size <= 0:
        return (0, 0)
    width = min(pool_size, max(1, round(pool_size * width_fraction)))
    low = int((pool_size - width) * progress)
    return (low, low + width)


def sample_sliding_window_start_state(
    ranked_pool: Sequence[RoverState],
    canonical: RoverState,
    progress: float,
    rng: np.random.Generator,
    *,
    width_fraction: float = DEFAULT_WINDOW_FRACTION,
) -> RoverState:
    """Draw uniformly from the sliding band at ``progress``.

    Args:
        ranked_pool: admissible start states sorted easiest first. Read-only.
        canonical: the real mission start, returned once the anneal is over.
        progress: how far through the anneal this episode is, in ``[0, 1]``.
        rng: the injected generator.
        width_fraction: band width as a share of the pool.

    Raises:
        ValueError: if ``progress`` falls outside ``[0, 1]``, or ``width_fraction``
            outside ``(0, 1]``.
    """
    if progress < 0.0 or progress > 1.0:
        raise ValueError(f"progress must lie in [0, 1], got {progress}")
    if len(ranked_pool) == 0 or progress >= 1.0:
        return canonical
    low, high = sliding_window_bounds(len(ranked_pool), progress, width_fraction)
    return ranked_pool[low + int(rng.integers(high - low))]


def sample_visit_weighted_start_state(
    ranked_pool: Sequence[RoverState],
    canonical: RoverState,
    progress: float,
    rng: np.random.Generator,
    update_counts: Sequence[int] | NDArray[np.int64],
    *,
    exponent: float = DEFAULT_WEIGHT_EXPONENT,
) -> RoverState:
    """Draw from the growing window, tilted away from already-learned states.

    This is the sampler that answers the oversampling Connor measured: the growing
    window admits the easy end of the pool first and never retires it, so a uniform
    draw keeps spending episodes on states whose values converged thousands of
    episodes ago. Here the *same* admitted band is drawn from with a weight that
    falls as a state's Q-update count rises, so the budget follows the states that
    still have something to learn.

    Args:
        ranked_pool: admissible start states sorted easiest first -- index ``0`` is
            the cheapest state to finish from. Treat it as read-only.
        canonical: the real mission start, which the policy is ultimately evaluated
            from and which the anneal must end on.
        progress: how far through the anneal this episode is, in ``[0, 1]``.
        rng: the injected generator. Every draw must come from it.
        update_counts: how many Q-updates each pooled state has already received,
            index-aligned with ``ranked_pool`` and never negative. Zeros everywhere
            is the normal state of affairs on episode 0.
        exponent: how hard to tilt. ``0.0`` must reproduce the uniform draw
            exactly; larger values push harder toward the least-updated states.

    Returns:
        The state the episode should begin from: either ``canonical`` or a member
        of ``ranked_pool``.

    Raises:
        ValueError: if ``progress`` falls outside ``[0, 1]``, if ``exponent`` is
            negative, if ``update_counts`` is not the same length as
            ``ranked_pool``, or if any count is negative.

    Invariants:
        * The admitted band is exactly :func:`growing_window_bounds` -- do not
          invent a different one, or the arm stops being a controlled comparison
          against :func:`sample_start_state`.
        * ``progress >= 1.0`` returns ``canonical``, and an empty ``ranked_pool``
          returns ``canonical`` rather than raising.
        * Every admitted state keeps a strictly positive probability however large
          its count has grown. A state that can never be drawn again is a state
          whose value can never be corrected.
        * A state with more updates than another admitted state is never drawn more
          often than it in expectation.
        * ``ranked_pool`` and ``update_counts`` are neither reordered nor mutated.
    """
    if progress < 0.0 or progress > 1.0:
        raise ValueError(f"progress must lie in [0, 1], got {progress}")

    if exponent < 0.0:
        raise ValueError(f"exponent must be >= 0.0, got {exponent}")

    if len(update_counts) != len(ranked_pool):
        raise ValueError(
            "update_counts and ranked_pool must have the same "
            f"length, {len(update_counts)} != {len(ranked_pool)}"
        )

    if progress >= 1.0 or not ranked_pool:
        return canonical

    if min(update_counts) < 0.0:
        raise ValueError(f"update_counts must be >= 0.0, got negative value {min(update_counts)}")

    growing_bounds = growing_window_bounds(len(ranked_pool), progress)
    sample_set = ranked_pool[growing_bounds[0] : growing_bounds[1]]
    sampled_update_counts = np.asarray(update_counts[growing_bounds[0] : growing_bounds[1]])

    raw_weights = 1 / np.pow(1 + sampled_update_counts, exponent)
    normalized_weights = raw_weights / raw_weights.sum()

    sampled_i = rng.choice(len(sample_set), p=normalized_weights)
    return ranked_pool[sampled_i]


def curriculum_stub_status() -> tuple[str, ...]:
    """Probe the human-owned sampler and report whether it still looks unfinished.

    A behavioural smoke check in the shape of
    :func:`mars_rover_q.agent.teaching_stub_status`, not a grader: it runs the
    function on a synthetic eight-state pool and reports it as pending when the
    result plainly does not meet the documented contract -- when a mid-anneal draw
    never lands in the pool at all, or when a state with ten thousand updates is
    drawn as often as an untouched neighbour.
    """
    pool = [RoverState(row=0, col=0, battery=index + 1) for index in range(8)]
    canonical = RoverState(row=1, col=1, battery=20)
    zero_counts = np.zeros(len(pool), dtype=np.int64)

    try:
        # One generator across the 50 draws: a fresh `default_rng(0)` per call would
        # replay the same stream every time and collapse `drawn` to a single state
        # however correct the sampler is.
        spread_rng = np.random.default_rng(0)
        drawn = {
            sample_visit_weighted_start_state(pool, canonical, 0.5, spread_rng, zero_counts)
            for _ in range(50)
        }
        if not drawn <= set(pool) or len(drawn) < 2:
            return ("sample_visit_weighted_start_state",)

        # Ranks 0 and 1 are admitted at this progress; rank 0 is thoroughly learned.
        loaded = np.array([10_000, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        rng = np.random.default_rng(1)
        draws = [
            sample_visit_weighted_start_state(pool, canonical, 0.25, rng, loaded)
            for _ in range(200)
        ]
        if sum(1 for state in draws if state == pool[0]) >= sum(
            1 for state in draws if state == pool[1]
        ):
            return ("sample_visit_weighted_start_state",)
    except Exception:
        return ("sample_visit_weighted_start_state",)
    return ()


class StartStateCurriculum:
    """Builds the ranked start-state pool once and serves one state per episode.

    Construction is where the cost sits: the pool is enumerated and difficulty-sorted
    a single time, then indexed for the rest of training. A curriculum with
    ``anneal_fraction <= 0`` skips enumeration entirely and hands back the canonical
    start every episode, so an unconfigured run pays nothing.

    ``strategy`` selects which of the three schedules in :class:`CurriculumStrategy`
    draws the start; ``window_fraction`` and ``weight_exponent`` are the knobs of
    the sliding and visit-weighted ones respectively and are ignored by the others.
    """

    __slots__ = (
        "anneal_episodes",
        "anneal_fraction",
        "canonical",
        "pool_state_indices",
        "ranked_pool",
        "scenario",
        "strategy",
        "weight_exponent",
        "window_fraction",
    )

    def __init__(
        self,
        scenario: Scenario,
        *,
        total_episodes: int,
        anneal_fraction: float,
        strategy: CurriculumStrategy | str = CurriculumStrategy.GROWING,
        window_fraction: float = DEFAULT_WINDOW_FRACTION,
        weight_exponent: float = DEFAULT_WEIGHT_EXPONENT,
    ) -> None:
        if total_episodes <= 0:
            raise ValueError(f"total_episodes must be positive, got {total_episodes}")
        if not 0.0 <= anneal_fraction <= 1.0:
            raise ValueError(f"anneal_fraction must lie in [0, 1], got {anneal_fraction}")
        if not 0.0 < window_fraction <= 1.0:
            raise ValueError(f"window_fraction must lie in (0, 1], got {window_fraction}")
        if weight_exponent < 0.0:
            raise ValueError(f"weight_exponent must not be negative, got {weight_exponent}")
        self.scenario = scenario
        self.canonical = canonical_start_state(scenario)
        self.anneal_fraction = float(anneal_fraction)
        self.anneal_episodes = max(1, round(total_episodes * anneal_fraction))
        self.strategy = CurriculumStrategy(strategy)
        self.window_fraction = float(window_fraction)
        self.weight_exponent = float(weight_exponent)
        self.ranked_pool: tuple[RoverState, ...] = ()
        self.pool_state_indices: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        if self.enabled:
            encoder = StateEncoder(scenario.rows, scenario.cols, scenario.battery_capacity)
            self.ranked_pool = tuple(
                sorted(
                    enumerate_start_states(scenario),
                    # Ties are broken by state index so the ordering -- and therefore
                    # every sampled episode -- is reproducible from the seed alone.
                    key=lambda state: (
                        start_state_difficulty(scenario, state),
                        encoder.encode(state),
                    ),
                )
            )
            # Encoded once here rather than per episode: the visit-weighted strategy
            # needs to read a per-state quantity in pool order on every draw, and the
            # encoding never changes for a fixed map.
            self.pool_state_indices = np.array(
                [encoder.encode(state) for state in self.ranked_pool], dtype=np.int64
            )

    @property
    def enabled(self) -> bool:
        """Whether a curriculum was requested at all."""
        return self.anneal_fraction > 0.0

    @property
    def active(self) -> bool:
        """Whether a curriculum was requested *and* has a pool to draw from."""
        return self.enabled and bool(self.ranked_pool)

    def progress_at(self, episode: int) -> float:
        """How far through the anneal a zero-based episode index is, in ``[0, 1]``."""
        if not self.enabled:
            return 1.0
        return min(1.0, max(0.0, episode / self.anneal_episodes))

    @property
    def needs_update_counts(self) -> bool:
        """Whether this strategy reads learning progress back out of the table.

        Only the visit-weighted schedule does. The caller checks this before paying
        for the gather in :meth:`pool_update_counts`, which is ``O(pool size)`` per
        episode and pure waste for the two open-loop schedules.
        """
        return self.strategy is CurriculumStrategy.VISIT_WEIGHTED

    def pool_update_counts(self, visit_counts: NDArray[np.int64] | None) -> NDArray[np.int64]:
        """Per-state Q-update counts gathered into ranked-pool order.

        Args:
            visit_counts: the training loop's per-state counter, indexed by
                :class:`StateEncoder` row ID, or ``None`` before any exists.

        Returns:
            An array as long as :attr:`ranked_pool`, all zeros when there is nothing
            to report yet -- which is exactly the situation on episode 0 and is not
            an error.
        """
        if visit_counts is None or self.pool_state_indices.size == 0:
            return np.zeros(len(self.ranked_pool), dtype=np.int64)
        return np.asarray(visit_counts, dtype=np.int64)[self.pool_state_indices]

    def start_state_for(
        self,
        episode: int,
        rng: np.random.Generator,
        visit_counts: NDArray[np.int64] | None = None,
    ) -> RoverState:
        """The start state for a zero-based episode index.

        ``visit_counts`` is the training loop's per-state Q-update counter. It is
        read only by :attr:`CurriculumStrategy.VISIT_WEIGHTED`; the other two
        schedules are open-loop and ignore it entirely.
        """
        if not self.enabled:
            return self.canonical
        progress = self.progress_at(episode)
        if self.strategy is CurriculumStrategy.SLIDING:
            return sample_sliding_window_start_state(
                self.ranked_pool,
                self.canonical,
                progress,
                rng,
                width_fraction=self.window_fraction,
            )
        if self.strategy is CurriculumStrategy.VISIT_WEIGHTED:
            return sample_visit_weighted_start_state(
                self.ranked_pool,
                self.canonical,
                progress,
                rng,
                self.pool_update_counts(visit_counts),
                exponent=self.weight_exponent,
            )
        return sample_start_state(self.ranked_pool, self.canonical, progress, rng)

    def difficulty_quantiles(
        self, quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    ) -> dict[str, float]:
        """Difficulty at the given quantiles of the ranked pool, for reporting."""
        if not self.ranked_pool:
            return {}
        scores = [start_state_difficulty(self.scenario, s) for s in self.ranked_pool]
        return {
            f"q{int(q * 100)}": float(scores[min(len(scores) - 1, int(q * len(scores)))])
            for q in quantiles
        }

    def describe(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot recorded in every run manifest."""
        return {
            "requested": self.enabled,
            "active": self.active,
            "strategy": str(self.strategy),
            "anneal_fraction": self.anneal_fraction,
            "window_fraction": self.window_fraction,
            "weight_exponent": self.weight_exponent,
            "anneal_episodes": self.anneal_episodes if self.enabled else 0,
            "pool_size": len(self.ranked_pool),
            "canonical_difficulty": start_state_difficulty(self.scenario, self.canonical),
            "difficulty_quantiles": self.difficulty_quantiles(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"StartStateCurriculum(scenario={self.scenario.name!r}, "
            f"strategy={self.strategy.value!r}, anneal_fraction={self.anneal_fraction}, "
            f"pool_size={len(self.ranked_pool)})"
        )


@dataclass(frozen=True, slots=True)
class SimulatedAnneal:
    """A dry run of a start-state schedule: no environment, no learning, no reward.

    What it is for: the schedules differ in *which states they spend episodes on*,
    and that is a property of the sampler alone. Simulating it answers "is the easy
    end still being oversampled at three-quarters progress" in a second, without
    training anything.

    What it is not: a prediction. The feedback loop is stubbed -- a drawn start
    accrues one synthetic update, where a real episode updates every state along its
    trajectory. So the counts here are a lower bound on the real ones and the
    visit-weighted tilt is *weaker* in simulation than it will be in training.

    Attributes:
        ranks: the pool rank drawn for each simulated episode.
        difficulties: :func:`start_state_difficulty` of each drawn state.
        is_canonical: whether each draw was the lander start. Tracked separately
            from ``ranks`` because the canonical start is itself a pooled state
            with a rank of its own -- "the anneal is over" and "the sampler drew
            the state that happens to be canonical" are the same state and
            different events.
        update_counts: the synthetic per-rank counter as it stood at the end.
    """

    strategy: CurriculumStrategy
    pool_size: int
    episodes: NDArray[np.int64]
    ranks: NDArray[np.int64]
    difficulties: NDArray[np.float64]
    is_canonical: NDArray[np.bool_]
    update_counts: NDArray[np.int64]

    @property
    def canonical_share(self) -> float:
        """Share of simulated episodes that began at the lander."""
        if self.is_canonical.size == 0:
            return 0.0
        return float(np.count_nonzero(self.is_canonical) / self.is_canonical.size)


def simulate_start_distribution(
    curriculum: StartStateCurriculum,
    rng: np.random.Generator,
    *,
    episodes: int | None = None,
) -> SimulatedAnneal:
    """Replay a curriculum's schedule episode by episode without training.

    Args:
        curriculum: the schedule to exercise. Not mutated.
        rng: the injected generator; the draw sequence is reproducible from it.
        episodes: how many episodes to simulate. Defaults to the anneal length,
            which is the only part of training where the schedule does anything;
            pass the full budget to see the canonical tail as well.

    Returns:
        A :class:`SimulatedAnneal` over ``range(episodes)``.
    """
    total = curriculum.anneal_episodes if episodes is None else episodes
    if total <= 0:
        raise ValueError(f"episodes must be positive, got {total}")
    pool_size = len(curriculum.ranked_pool)
    rank_of = {state: rank for rank, state in enumerate(curriculum.ranked_pool)}
    encoder = StateEncoder(
        curriculum.scenario.rows,
        curriculum.scenario.cols,
        curriculum.scenario.battery_capacity,
    )
    counts = np.zeros(encoder.num_states, dtype=np.int64)

    ranks = np.empty(total, dtype=np.int64)
    difficulties = np.empty(total, dtype=np.float64)
    is_canonical = np.zeros(total, dtype=np.bool_)
    for episode in range(total):
        state = curriculum.start_state_for(episode, rng, counts)
        ranks[episode] = rank_of.get(state, pool_size)
        difficulties[episode] = start_state_difficulty(curriculum.scenario, state)
        is_canonical[episode] = state == curriculum.canonical
        counts[encoder.encode(state)] += 1

    return SimulatedAnneal(
        strategy=curriculum.strategy,
        pool_size=pool_size,
        episodes=np.arange(total, dtype=np.int64),
        ranks=ranks,
        difficulties=difficulties,
        is_canonical=is_canonical,
        update_counts=curriculum.pool_update_counts(counts),
    )


def rank_band_shares(
    ranks: Sequence[int] | NDArray[np.int64], pool_size: int, bands: int = 4
) -> tuple[float, ...]:
    """Share of draws landing in each equal-width band of the ranked pool.

    This is the number that makes oversampling visible: a schedule spending a third
    of its episodes in the easiest quarter of the pool says so here, where a mean
    difficulty rising monotonically does not. Ranks at or past ``pool_size`` belong
    to no band, so the shares sum to ``1`` only when every draw came from the pool.

    Raises:
        ValueError: if ``bands`` is not positive.
    """
    if bands <= 0:
        raise ValueError(f"bands must be positive, got {bands}")
    values = np.asarray(ranks, dtype=np.int64)
    if values.size == 0 or pool_size <= 0:
        return tuple(0.0 for _ in range(bands))
    inside = values[values < pool_size]
    edges = np.minimum((inside * bands) // pool_size, bands - 1)
    counts = np.bincount(edges, minlength=bands)
    return tuple(float(count / values.size) for count in counts)


__all__ = [
    "BASELINE_CURRICULUM",
    "CURRICULUM_STRATEGY_LABELS",
    "DEFAULT_WEIGHT_EXPONENT",
    "DEFAULT_WINDOW_FRACTION",
    "CurriculumStrategy",
    "SimulatedAnneal",
    "StartStateCurriculum",
    "canonical_start_state",
    "curriculum_arm_label",
    "curriculum_arm_slug",
    "curriculum_stub_status",
    "enumerate_start_states",
    "growing_window_bounds",
    "rank_band_shares",
    "sample_sliding_window_start_state",
    "sample_start_state",
    "sample_visit_weighted_start_state",
    "simulate_start_distribution",
    "sliding_window_bounds",
    "start_state_difficulty",
]
