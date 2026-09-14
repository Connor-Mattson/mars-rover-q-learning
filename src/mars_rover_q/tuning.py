"""Optuna hyper-parameter search: the best mission return in the fewest episodes.

Every study in this repository so far has *held* the hyper-parameters fixed -- learning
rate ``0.2``, ``gamma 0.99``, epsilon ``1.0 -> 0.05`` over 60% of the budget -- because
the question was about reward design or about exploration, and a factor you are not
studying has to be held constant. This module asks the other question: given the agent
we have, which hyper-parameters reach a good policy *soonest*?

Three design commitments make that question answerable rather than open-ended.

**A per-trial episode cap.** Every trial gets exactly the same episode budget
(:attr:`TuningConfig.trial_episode_cap`), and that cap is deliberately smaller than the
budget at which a well-tuned agent converges. Some hyper-parameter settings cannot reach
the map's best return inside it -- that is the point. A cap turns "fastest learner" into
a measurable quantity; without one, every setting that eventually converges looks equally
good and the search degenerates into a search for final performance.

**A bounded total budget.** The whole study is charged against one ledger
(:class:`EpisodeLedger`, :attr:`TuningConfig.study_episode_budget`, two million episodes
by default). The study stops when the ledger cannot fund another full trial, so the cost
of a search is known before it starts rather than discovered afterwards, and two searches
with different caps are directly comparable in total compute.

**One scalar, honestly scalarised.** "Best reward in the fewest episodes" is two
objectives, and :func:`score_learning_curve` is the single place where they are collapsed
into the number the sampler maximises. :func:`pareto_front` then reports the trade-off the
scalar resolved, so a reader can see what the score chose to weigh rather than having to
trust it. Both are human-owned; see ``.teacher/current.md``.

Pruning is what makes the bounded budget buy more than a fixed grid would: a trial whose
checkpoint curve is already hopeless is stopped where it stands, and the episodes it did
not run are returned to the ledger for another trial. The checkpoints it is judged on are
measured on a seed space disjoint from training, so pruning changes what a trial *costs*
and never what it *learns*.

Nothing here reports a result. The numbers a study writes are whatever the search found;
no expectation about which hyper-parameters win is recorded anywhere in this module.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .agent import teaching_stub_status
from .curriculum import DEFAULT_WEIGHT_EXPONENT, DEFAULT_WINDOW_FRACTION, CurriculumStrategy
from .evaluation import evaluate
from .experiment import environment_metadata, json_safe, save_run, write_json
from .metrics import CheckpointRecord
from .rewards import RewardMode
from .scenario import Scenario, resolve_scenario
from .state import BatteryEncoding
from .training import TrainConfig, TrainResult, train

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import optuna


class ParameterSuggester(Protocol):
    """The slice of ``optuna.trial.Trial`` that :func:`suggest_trial_params` uses.

    Declaring the dependency structurally rather than as the concrete ``Trial`` keeps
    the search space honest about what it needs -- two suggestion calls and nothing
    else, no study, storage, or sampler -- and lets it be tested against a recorder
    that reports which parameters were asked for, on what ranges, and under what
    condition. A real ``Trial`` satisfies it.
    """

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = ...) -> float: ...

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...


#: Total training episodes one study may spend, across every trial. The search stops
#: when what remains cannot fund another full trial.
DEFAULT_STUDY_EPISODE_BUDGET = 2_000_000

#: Episodes granted to one trial, per seed. Chosen to sit *below* convergence for a
#: well-tuned agent on ``safe_corridor`` (which reaches success 1.0 in roughly four
#: thousand episodes at the repository's fixed defaults), so that how fast a setting
#: learns still separates settings at the cap.
DEFAULT_TRIAL_EPISODE_CAP = 6_000

#: Checkpoints per trial: the resolution of the learning curve the objective reads, and
#: the number of decision points the pruner gets.
DEFAULT_CHECKPOINTS = 12

#: Greedy episodes per checkpoint. Small on purpose -- a checkpoint is a progress
#: measurement, not the evaluation a result is reported from.
DEFAULT_CHECKPOINT_EPISODES = 30

#: Score returned by the stubbed :func:`score_learning_curve`. A constant keeps a study
#: runnable while the objective is unimplemented: the sampler gets a flat landscape and
#: learns nothing, which is exactly what the teaching banner says is happening.
SCORE_WHEN_STUBBED = 0.0

#: The human-owned functions in this module, for the teaching banner.
HUMAN_OWNED_FUNCTIONS: tuple[str, ...] = ("score_learning_curve", "pareto_front")

TUNING_STUB_WARNING = """
=========================== HYPERPARAMETER SEARCH STUBBED =======================
The search objective in src/mars_rover_q/tuning.py is still a placeholder: {names}

Trials will train real Q-tables, but every one of them scores {score}, so the
sampler is drawing from its prior instead of optimising anything. This study is a
RANDOM SEARCH WITHOUT A CRITERION, not a hyper-parameter search. Its "best trial"
is whichever trial happened to be asked first.

Do not report these numbers. See .teacher/current.md.
================================================================================
"""


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """One hyper-parameter search: what to search, on what, and for how long.

    The scenario and reward mode are *held fixed*, not searched. Reward design is the
    subject of the other studies in ``docs/experiment-plan.md``, and a search that
    moved the reward mode would be optimising the measuring instrument along with the
    agent.
    """

    name: str = "hyperparameter_search"
    scenario: str = "safe_corridor"
    reward_mode: RewardMode = RewardMode.SPARSE
    #: Training seeds averaged within each trial. One seed is noisy but cheap; each
    #: extra seed multiplies what a trial costs the ledger and so divides how many
    #: trials the same total budget can fund.
    seeds: tuple[int, ...] = (1,)
    trial_episode_cap: int = DEFAULT_TRIAL_EPISODE_CAP
    study_episode_budget: int = DEFAULT_STUDY_EPISODE_BUDGET
    #: Hard cap on trial count, applied *in addition* to the episode ledger. ``None``
    #: lets the ledger alone decide when the study is over.
    max_trials: int | None = None
    checkpoints: int = DEFAULT_CHECKPOINTS
    checkpoint_episodes: int = DEFAULT_CHECKPOINT_EPISODES
    #: Greedy episodes used to evaluate the confirmation run of the best trial.
    eval_episodes: int = 200
    #: Seed for Optuna's sampler. The search itself is reproducible: same seed, same
    #: sequence of suggested configurations.
    sampler_seed: int = 0
    #: Trials drawn from the prior before the sampler starts modelling the objective.
    startup_trials: int = 10
    #: Checkpoints a trial is allowed before it may be pruned, and trials that must
    #: have completed before pruning begins at all.
    warmup_checkpoints: int = 3
    pruner_startup_trials: int = 5
    battery_encoding: BatteryEncoding = BatteryEncoding.AFFORDABILITY
    #: Whether the start-state curriculum and its knobs are part of the search space.
    search_curriculum: bool = True
    #: Whether the battery encoding is searched instead of held at ``battery_encoding``.
    #: Off by default, because a study that changes the table's resolution between
    #: trials is comparing two representations as well as two settings, and the other
    #: studies in ``docs/experiment-plan.md`` hold the representation fixed on purpose.
    search_battery_encoding: bool = False
    #: Overrides the y-axis scale the objective normalises against. ``None`` derives
    #: it from the map; see :func:`best_affordable_return`.
    reference_return: float | None = None
    #: Whether to re-train the best trial's configuration and save it as a full run
    #: directory. Charged to the same ledger as the trials.
    confirm_best: bool = True

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be distinct, got {self.seeds}")
        if self.trial_episode_cap <= 0:
            raise ValueError(f"trial_episode_cap must be positive, got {self.trial_episode_cap}")
        if self.study_episode_budget < self.episodes_per_trial:
            raise ValueError(
                "study_episode_budget cannot fund a single trial: "
                f"got {self.study_episode_budget} < {self.episodes_per_trial}"
            )
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError(f"max_trials must be positive when set, got {self.max_trials}")
        if self.checkpoints <= 1:
            raise ValueError(f"checkpoints must exceed 1, got {self.checkpoints}")
        if self.checkpoint_episodes <= 0:
            raise ValueError(
                f"checkpoint_episodes must be positive, got {self.checkpoint_episodes}"
            )
        if self.reference_return is not None and not (
            math.isfinite(self.reference_return) and self.reference_return > 0.0
        ):
            raise ValueError(
                f"reference_return must be positive and finite, got {self.reference_return}"
            )

    @property
    def episodes_per_trial(self) -> int:
        """What one full trial costs the ledger: the cap, once per seed."""
        return self.trial_episode_cap * len(self.seeds)

    @property
    def fundable_trials(self) -> int:
        """How many full trials the total budget can fund if none is ever pruned.

        The floor on trial count, not the expectation: pruning returns unspent
        episodes to the ledger, so a study that prunes runs more trials than this.
        """
        return self.study_episode_budget // self.episodes_per_trial

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot."""
        payload = asdict(self)
        payload["reward_mode"] = str(self.reward_mode)
        payload["battery_encoding"] = str(self.battery_encoding)
        payload["seeds"] = list(self.seeds)
        payload["episodes_per_trial"] = self.episodes_per_trial
        payload["fundable_trials"] = self.fundable_trials
        return payload

    @classmethod
    def from_file(cls, path: str | Path) -> TuningConfig:
        """Load a search definition from JSON, ignoring unknown keys."""
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        if "reward_mode" in known:
            known["reward_mode"] = RewardMode(known["reward_mode"])
        if "battery_encoding" in known:
            known["battery_encoding"] = BatteryEncoding(known["battery_encoding"])
        if "seeds" in known:
            known["seeds"] = tuple(int(s) for s in known["seeds"])
        return cls(**known)


@dataclass(slots=True)
class EpisodeLedger:
    """The study's episode budget, and the only thing allowed to spend it.

    Grants are all-or-nothing at ``per_trial``: a trial that ran on a short grant
    would have a shorter x-axis than its competitors, and :func:`score_learning_curve`
    normalises every trial against the same cap precisely so that scores are
    comparable. A partial grant would therefore be a trial that cannot be compared,
    which is worth less than the episodes it would cost.

    ``spent`` counts what was *run*, not what was granted, so a pruned trial returns
    its unused episodes to the pool.
    """

    total: int
    per_trial: int
    spent: int = 0

    def __post_init__(self) -> None:
        if self.per_trial <= 0:
            raise ValueError(f"per_trial must be positive, got {self.per_trial}")
        if self.total < self.per_trial:
            raise ValueError(
                f"total must fund at least one trial, got {self.total} < {self.per_trial}"
            )
        if self.spent < 0:
            raise ValueError(f"spent must not be negative, got {self.spent}")

    @property
    def remaining(self) -> int:
        """Episodes still available to grant."""
        return self.total - self.spent

    @property
    def exhausted(self) -> bool:
        """Whether what remains can no longer fund a full trial."""
        return self.remaining < self.per_trial

    def grant(self) -> int:
        """Reserve one full trial's episodes, or ``0`` when the study is over."""
        return 0 if self.exhausted else self.per_trial

    def record(self, episodes: int) -> None:
        """Charge the episodes a trial actually ran."""
        if episodes < 0:
            raise ValueError(f"episodes must not be negative, got {episodes}")
        if episodes > self.remaining:
            raise ValueError(
                f"charge of {episodes} exceeds the {self.remaining} episodes remaining"
            )
        self.spent += episodes

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot."""
        return {
            "total": self.total,
            "per_trial": self.per_trial,
            "spent": self.spent,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Everything one trial produced: what was tried, what it cost, how it did."""

    number: int
    params: dict[str, Any]
    episodes_granted: int
    episodes_run: int
    #: ``None`` for a pruned trial: it was stopped precisely because its curve was
    #: not going to be worth scoring, and a score computed from a truncated curve
    #: would be compared against full-budget ones as though it were the same
    #: measurement.
    score: float | None
    #: Best mean base return anywhere on the curve, and the earliest checkpoint that
    #: attained it. These two are the axes :func:`pareto_front` works on.
    best_return: float
    episodes_to_best: int
    final_return: float
    final_success_rate: float
    pruned: bool
    curve: tuple[tuple[int, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot."""
        payload = asdict(self)
        payload["curve"] = [list(point) for point in self.curve]
        return payload

    def as_row(self) -> dict[str, Any]:
        """A flat row for ``trials.csv``; the curve lives in ``study.json``."""
        row: dict[str, Any] = {
            "number": self.number,
            "score": self.score,
            "best_return": self.best_return,
            "episodes_to_best": self.episodes_to_best,
            "final_return": self.final_return,
            "final_success_rate": self.final_success_rate,
            "episodes_granted": self.episodes_granted,
            "episodes_run": self.episodes_run,
            "pruned": self.pruned,
        }
        row.update({f"param_{k}": v for k, v in sorted(self.params.items())})
        return row


# --------------------------------------------------------------------------------
# The two human-owned functions
# --------------------------------------------------------------------------------


def score_learning_curve(
    curve: Sequence[tuple[int, float]],
    episode_cap: int,
    reference_return: float,
) -> float:
    """Collapse one trial's learning curve into the number the search maximises.

    This *is* the objective. It decides what "the best reward in the fewest episodes"
    means, and therefore what the whole study is searching for. Every trial in a study
    is scored with the same ``episode_cap`` and the same ``reference_return``, so the
    two normalisers are shared and the scores are comparable across trials.

    Args:
        curve: the trial's checkpoint curve, as ``(episodes_trained, mean_base_return)``
            pairs ascending in ``episodes_trained`` and beginning at ``0`` -- the
            untrained table. ``mean_base_return`` is greedy mission return with shaping
            excluded, measured from the canonical lander start. It can be negative: a
            policy that strands the rover is charged the battery penalty.
        episode_cap: the per-trial episode budget the study grants, and the x-axis
            scale. No point on the curve lies beyond it.
        reference_return: the best mission return this map can pay, and the y-axis
            scale. See :func:`best_affordable_return`.

    Returns:
        A score in ``[0.0, 1.0]``, higher being better, satisfying all of:

        * **1.0** exactly when the curve is at or above ``reference_return`` at every
          point, episode ``0`` included. No real curve reaches this -- it would mean an
          untrained table already flying the best mission -- it is the ceiling the
          scale is measured against, not a target.
        * **0.0** when the curve never rises above ``0.0`` return.
        * A curve at least as good as another at every matching episode count never
          scores lower, and scores strictly higher when it is strictly better anywhere.
        * Of two curves that reach the same return, the one reaching it at a lower
          episode count scores strictly higher. This is the half that makes the search
          prefer a fast learner over a merely good one, and it is the half a plain
          "final performance" objective throws away.
        * Unchanged when every return and ``reference_return`` are multiplied by one
          positive factor, and when every episode count and ``episode_cap`` are
          multiplied by another. A score therefore means the same thing on two maps
          whose payoffs differ fourfold, and at two caps an order of magnitude apart.

        The study only ever scores a trial that ran its full budget, so the last point
        is at ``episode_cap`` in practice. A shorter curve must still be scored rather
        than rejected, but nothing in the study depends on how its unmeasured tail is
        treated.

    Raises:
        ValueError: if ``curve`` is empty; if its episode counts are not strictly
            ascending, or its first point is not episode ``0``, or any point lies
            beyond ``episode_cap``; if any return is not finite; if ``episode_cap`` is
            not positive; or if ``reference_return`` is not both positive and finite.
    """
    if not curve:
        raise ValueError("Curve must not be empty.")

    if episode_cap <= 0.0:
        raise ValueError(f"Episode cap must be positive, got {episode_cap}")

    if curve[0][0] != 0:
        raise ValueError(f"Curves first value is required to be at episode 0, got {curve[0][0]}.")

    # Check strictly ascending episodes
    episode_rate_of_change = [curve[i][0] - curve[i - 1][0] for i in range(1, len(curve))]
    if episode_rate_of_change and min(episode_rate_of_change) <= 0.0:
        raise ValueError("Curve episodes must be strictly ascending.")

    # Check episode cap
    if max(curve)[0] > episode_cap:
        raise ValueError(
            f"Curve episode number is required to be capped at {episode_cap}, got {max(curve)[0]}."
        )

    # Check finite
    if any(not math.isfinite(x[1]) for x in curve):
        raise ValueError("Curve returns must be finite.")

    if reference_return <= 0.0 or not math.isfinite(reference_return):
        raise ValueError(f"Reference return must be positive and finite, got {reference_return}.")

    normalized_x = [c[0] / episode_cap for c in curve]
    normalized_y = [max(min(c[1], reference_return), 0) / reference_return for c in curve]

    # Integrate a normalized curve to obtain score
    aoc = np.trapezoid(normalized_y, normalized_x)
    if aoc < 0.0:
        return 0.0

    return float(aoc)


def pareto_front(outcomes: Sequence[TrialOutcome]) -> list[TrialOutcome]:
    """The trials you cannot improve on without spending more episodes.

    :func:`score_learning_curve` answers "which hyper-parameters" with one number, and
    in doing so hides the trade-off it resolved. This restores it: the non-dominated
    set over the two axes the study actually cares about -- ``episodes_to_best``, where
    fewer is better, and ``best_return``, where more is better.

    One outcome *dominates* another when it is no worse on both axes and strictly
    better on at least one. The front is every outcome that no outcome in ``outcomes``
    dominates.

    Args:
        outcomes: the trials to consider. Whatever is passed is considered; it is the
            caller's decision whether a pruned trial belongs in the comparison.

    Returns:
        The non-dominated outcomes, ordered by ascending ``episodes_to_best``, then by
        descending ``best_return``, with ``number`` breaking any remaining tie so the
        output is deterministic. Two outcomes equal on both axes dominate each other
        in neither direction, so both are kept. An empty input returns an empty list.

    Raises:
        ValueError: if any outcome has a negative ``episodes_to_best``, or a
            ``best_return`` that is not finite.
    """
    if len(outcomes) == 0:
        return []

    x_points = [outcome.episodes_to_best for outcome in outcomes]
    y_points = [outcome.best_return for outcome in outcomes]
    ids = [outcome.number for outcome in outcomes]
    points = zip(x_points, y_points, ids, range(len(outcomes)), strict=True)

    # ValueError Edge cases
    if min(x_points) < 0.0:
        raise ValueError(
            "pareto_front expects every outcome to have a non-negative "
            f"'episodes_to_best', got {min(x_points)}"
        )
    if any(not math.isfinite(x) for x in y_points):
        raise ValueError(
            "pareto_front expects all outcomes to have a finite "
            "'best_return', value failed math.isfinite()"
        )

    # Sort ascending by x
    sorted_points = sorted(points, key=lambda p: (p[0], -p[1], p[2]))
    front = []
    max_y = float("-inf")
    last_x = float("-inf")
    for p in sorted_points:
        x, y, id, i = p
        if y > max_y or (y == max_y and x == last_x):
            front.append((x, y, id, i))
            max_y = y
            last_x = x
    return [outcomes[f[3]] for f in front]


def tuning_stub_status() -> tuple[str, ...]:
    """Probe the human-owned functions and report which still look unfinished.

    A behavioural smoke check in the shape of
    :func:`mars_rover_q.agent.teaching_stub_status`, not a grader. It asks each
    function for the one property the study cannot run without: that the objective
    separates a fast learner from a slow one, and that the front drops a dominated
    trial while keeping the two that dominate it.
    """
    pending: list[str] = []

    fast = ((0, 0.0), (50, 80.0), (100, 80.0))
    slow = ((0, 0.0), (50, 0.0), (100, 80.0))
    try:
        if not score_learning_curve(fast, 100, 100.0) > score_learning_curve(slow, 100, 100.0):
            pending.append("score_learning_curve")
    except Exception:
        pending.append("score_learning_curve")

    cheap = _probe_outcome(0, episodes_to_best=100, best_return=50.0)
    strong = _probe_outcome(1, episodes_to_best=900, best_return=160.0)
    dominated = _probe_outcome(2, episodes_to_best=900, best_return=40.0)
    try:
        front = pareto_front([cheap, strong, dominated])
        if {outcome.number for outcome in front} != {0, 1}:
            pending.append("pareto_front")
    except Exception:
        pending.append("pareto_front")

    return tuple(pending)


def _probe_outcome(number: int, *, episodes_to_best: int, best_return: float) -> TrialOutcome:
    """A minimal outcome carrying only the two axes the front reads."""
    return TrialOutcome(
        number=number,
        params={},
        episodes_granted=1000,
        episodes_run=1000,
        score=0.0,
        best_return=best_return,
        episodes_to_best=episodes_to_best,
        final_return=best_return,
        final_success_rate=0.0,
        pruned=False,
    )


# --------------------------------------------------------------------------------
# Scales, curves, and the search space
# --------------------------------------------------------------------------------


def best_affordable_return(scenario: Scenario) -> float:
    """The largest mission return this map can pay, from geometry alone.

    A successful delivery pays exactly the delivered sample's value, so the ceiling on
    ``mean_base_return`` is the most valuable sample whose lander round trip --
    including the ``COLLECT`` charge -- fits inside the battery. This is a planning
    bound computed from shortest paths, **not** a claim about what Q-learning finds and
    not a target any policy is expected to hit: slip makes the achievable mean strictly
    lower, and a discount below ``1.0`` can make a nearer, cheaper sample the optimal
    choice. It exists to give :func:`score_learning_curve` a y-axis that means the same
    thing on maps whose payoffs differ fourfold.

    Raises:
        ValueError: when no sample on the map is affordable, which would leave the
            objective with no scale to normalise against.
    """
    affordable = [
        float(scenario.samples[sample].value)
        for sample, cost in scenario.mission_costs().items()
        if cost <= scenario.battery_capacity
    ]
    if not affordable:
        raise ValueError(f"no sample on {scenario.name!r} is affordable; there is no return scale")
    return max(affordable)


def curve_peak(curve: Sequence[tuple[int, float]]) -> tuple[int, float]:
    """``(episodes, return)`` of the curve's best point, earliest first.

    The Pareto axes: the best return the trial ever showed, and how few episodes it
    needed to first show it. Reading the *earliest* attainment matters because a curve
    that peaks at 2k and then plateaus is a faster learner than one that first touches
    the same value at 6k, and both have the same maximum.
    """
    if not curve:
        raise ValueError("curve must not be empty")
    best_return = max(value for _, value in curve)
    episodes = min(episode for episode, value in curve if value >= best_return)
    return episodes, best_return


def average_curves(
    curves: Sequence[Sequence[tuple[int, float]]],
) -> tuple[tuple[int, float], ...]:
    """Average several seeds' checkpoint curves into one.

    Every seed of a trial runs the same budget and the same checkpoint schedule, so the
    curves are index-aligned by construction and the episode axis is shared. That is
    asserted rather than assumed: a mismatch means the seeds were not run under the
    same configuration, and averaging them would quietly invent a curve belonging to
    neither.
    """
    if not curves:
        raise ValueError("curves must not be empty")
    episodes = [episode for episode, _ in curves[0]]
    for index, curve in enumerate(curves[1:], start=1):
        if [episode for episode, _ in curve] != episodes:
            raise ValueError(
                f"curve {index} has episode axis {[e for e, _ in curve]}, expected {episodes}"
            )
    return tuple(
        (episode, math.fsum(curve[position][1] for curve in curves) / len(curves))
        for position, episode in enumerate(episodes)
    )


def suggest_trial_params(trial: ParameterSuggester, config: TuningConfig) -> dict[str, Any]:
    """Draw one candidate hyper-parameter setting from the search space.

    Three things about the space are deliberate.

    *Log scale where the parameter acts multiplicatively.* The interesting difference
    between learning rates ``0.01`` and ``0.02`` is the same as between ``0.4`` and
    ``0.8``; on a uniform axis the sampler would spend most of its draws in the top
    decade and barely probe the small end.

    *Epsilon's floor is sampled as a fraction of its ceiling* rather than as an
    absolute rate, because :class:`mars_rover_q.agent.EpsilonSchedule` requires
    ``end <= start``. Sampling the two independently would put a sizeable corner of
    the space out of bounds, and the sampler would keep proposing trials that raise
    instead of training.

    *The curriculum's knobs are conditional.* ``window_fraction`` means nothing to the
    growing window and ``weight_exponent`` means nothing to either open-loop schedule.
    Suggesting a parameter the run will ignore teaches the sampler's model that the
    value was tried and did nothing, which is how a dead dimension dilutes a search.
    """
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9995),
        # Optimistic initialisation is an exploration mechanism in its own right, and
        # the scale that matters is the map's own return ceiling.
        "initial_q": trial.suggest_float("initial_q", 0.0, 200.0),
        "epsilon_start": trial.suggest_float("epsilon_start", 0.2, 1.0),
        "epsilon_end_ratio": trial.suggest_float("epsilon_end_ratio", 0.001, 0.5, log=True),
        "epsilon_decay_fraction": trial.suggest_float("epsilon_decay_fraction", 0.05, 0.95),
    }
    if config.search_battery_encoding:
        # Not a knob on the learning rule but on the table the rule writes into, and
        # on these maps it moves the outcome further than any of the six above: the
        # coarse binning and the dense axis win on different maps and under different
        # curricula. A search that held it fixed would be fixing the deciding factor.
        params["battery_encoding"] = str(
            trial.suggest_categorical(
                "battery_encoding", [encoding.value for encoding in BatteryEncoding]
            )
        )

    if not config.search_curriculum:
        return params

    # A categorical switch rather than a fraction that may happen to land on zero: the
    # no-curriculum arm is a distinct hypothesis and deserves real probability mass.
    params["use_curriculum"] = trial.suggest_categorical("use_curriculum", [False, True])
    if not params["use_curriculum"]:
        return params

    params["curriculum_fraction"] = trial.suggest_float("curriculum_fraction", 0.1, 0.9)
    strategy = str(
        trial.suggest_categorical(
            "curriculum_strategy", [strategy.value for strategy in CurriculumStrategy]
        )
    )
    params["curriculum_strategy"] = strategy
    if strategy == CurriculumStrategy.SLIDING.value:
        params["curriculum_window_fraction"] = trial.suggest_float(
            "curriculum_window_fraction", 0.02, 0.5, log=True
        )
    elif strategy == CurriculumStrategy.VISIT_WEIGHTED.value:
        params["curriculum_weight_exponent"] = trial.suggest_float(
            "curriculum_weight_exponent", 0.0, 2.0
        )
    return params


def train_config_for(
    config: TuningConfig,
    params: dict[str, Any],
    seed: int,
    episodes: int,
) -> TrainConfig:
    """Turn one suggested parameter set into a runnable :class:`TrainConfig`.

    Absent keys fall back to :class:`TrainConfig`'s own defaults, which is what makes
    the conditional search space work: a trial that never suggested
    ``curriculum_window_fraction`` gets the repository default, and the run is
    unaffected because its strategy does not read it.
    """
    epsilon_start = float(params.get("epsilon_start", 1.0))
    return TrainConfig(
        scenario=config.scenario,
        reward_mode=config.reward_mode,
        seed=seed,
        episodes=episodes,
        learning_rate=float(params.get("learning_rate", 0.2)),
        gamma=float(params.get("gamma", 0.99)),
        initial_q=float(params.get("initial_q", 0.0)),
        battery_encoding=BatteryEncoding(params.get("battery_encoding", config.battery_encoding)),
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_start * float(params.get("epsilon_end_ratio", 0.05)),
        epsilon_decay_fraction=float(params.get("epsilon_decay_fraction", 0.6)),
        curriculum_fraction=(
            float(params.get("curriculum_fraction", 0.0)) if params.get("use_curriculum") else 0.0
        ),
        curriculum_strategy=CurriculumStrategy(
            str(params.get("curriculum_strategy", CurriculumStrategy.GROWING.value))
        ),
        curriculum_window_fraction=float(
            params.get("curriculum_window_fraction", DEFAULT_WINDOW_FRACTION)
        ),
        curriculum_weight_exponent=float(
            params.get("curriculum_weight_exponent", DEFAULT_WEIGHT_EXPONENT)
        ),
        eval_checkpoints=config.checkpoints,
        checkpoint_episodes=config.checkpoint_episodes,
        log_every=0,
    )


# --------------------------------------------------------------------------------
# Running the study
# --------------------------------------------------------------------------------


def _pruning_callback(trial: optuna.trial.Trial) -> Callable[[CheckpointRecord], bool]:
    """A checkpoint callback that reports to Optuna and obeys its pruner.

    The reported value is the checkpoint's raw mean base return, not the trial's
    score: a pruner compares trials *at the same step*, so it needs the quantity that
    is directly comparable at a fixed episode count. The score is for ranking finished
    trials against each other and is computed once, from the whole curve.
    """
    step = 0

    def report(record: CheckpointRecord) -> bool:
        nonlocal step
        trial.report(record.mean_base_return, step)
        step += 1
        return not trial.should_prune()

    return report


def run_trial(
    config: TuningConfig,
    scenario: Scenario,
    trial: optuna.trial.Trial,
    *,
    granted: int,
    reference_return: float,
) -> TrialOutcome:
    """Train, score, and summarise one trial.

    Pruning is decided on the *first* seed alone. A pruner compares trials step by
    step, and a trial that reported a step-0 value once per seed would be three
    incomparable sequences wearing one trial number; judging the first seed and
    abandoning the rest of the trial with it keeps the comparison well-formed and
    saves the most episodes a prune can save.
    """
    params = suggest_trial_params(trial, config)
    curves: list[tuple[tuple[int, float], ...]] = []
    episodes_run = 0
    pruned = False
    results: list[TrainResult] = []

    for index, seed in enumerate(config.seeds):
        train_config = train_config_for(config, params, seed, config.trial_episode_cap)
        result = train(
            scenario,
            train_config,
            warn_on_stubs=False,
            checkpoint_callback=_pruning_callback(trial) if index == 0 else None,
        )
        episodes_run += result.episodes_completed
        results.append(result)
        curves.append(tuple((c.episode, c.mean_base_return) for c in result.checkpoints))
        if result.stopped_early:
            pruned = True
            break

    curve = average_curves(curves) if len(curves) == len(config.seeds) else curves[0]
    episodes_to_best, best_return = curve_peak(curve)
    score = (
        None
        if pruned
        else float(score_learning_curve(curve, config.trial_episode_cap, reference_return))
    )
    final_success = math.fsum(
        result.checkpoints[-1].success_rate for result in results if result.checkpoints
    ) / max(1, sum(1 for result in results if result.checkpoints))

    return TrialOutcome(
        number=trial.number,
        params=params,
        episodes_granted=granted,
        episodes_run=episodes_run,
        score=score,
        best_return=best_return,
        episodes_to_best=episodes_to_best,
        final_return=curve[-1][1],
        final_success_rate=final_success,
        pruned=pruned,
        curve=curve,
    )


def _make_study(config: TuningConfig) -> optuna.study.Study:
    """A single-objective maximising study with a seeded sampler and a median pruner.

    The study is in-memory: a search is reproducible from its ``sampler_seed`` and its
    artefacts are ``study.json`` and ``trials.csv``, so there is nothing a SQLite
    storage file would add except a second, divergent record of the same run.

    Optuna's own logger is quietened here. It narrates every ask and tell at INFO, which
    would interleave three lines of its vocabulary with each of ours for a study that
    already prints one line per trial.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=config.name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=config.sampler_seed, n_startup_trials=config.startup_trials
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.pruner_startup_trials,
            n_warmup_steps=config.warmup_checkpoints,
        ),
    )


def _best_outcome(outcomes: Sequence[TrialOutcome]) -> TrialOutcome | None:
    """The highest-scoring completed trial, or ``None`` when none completed.

    Ties break toward the lower trial number, so the choice does not depend on how the
    pool happened to be ordered.
    """
    scored = [outcome for outcome in outcomes if outcome.score is not None]
    if not scored:
        return None
    return max(scored, key=lambda outcome: (outcome.score, -outcome.number))


def _confirm_best(
    config: TuningConfig,
    scenario: Scenario,
    best: TrialOutcome,
    output_root: Path,
    ledger: EpisodeLedger,
    out: Any,
) -> dict[str, Any] | None:
    """Re-train the winning configuration and save it as a full run directory.

    The trial itself kept no Q-table -- hundreds of trials' tables is gigabytes for
    artefacts nothing reads -- so the winner is trained once more and evaluated
    properly, on the same ``eval_episodes`` ruler every other study in the repository
    uses. Those episodes are charged to the same ledger: a confirmation run is part of
    the search's cost, and leaving it off the books would make the bound untrue.
    """
    if ledger.exhausted:
        print("budget exhausted: skipping the confirmation run", file=out)
        return None

    seed = config.seeds[0]
    train_config = train_config_for(config, best.params, seed, config.trial_episode_cap)
    result = train(scenario, train_config, warn_on_stubs=False, stream=out)
    ledger.record(result.episodes_completed)
    evaluation = evaluate(
        result.q_table,
        scenario,
        config.reward_mode,
        episodes=config.eval_episodes,
        seed=seed,
        gamma=train_config.gamma,
        battery_encoding=train_config.battery_encoding,
    )
    run_dir = output_root / "best_run"
    save_run(run_dir, scenario, result, evaluation, experiment=config.name)
    summary = evaluation.summary
    return {
        "trial_number": best.number,
        "run_dir": str(run_dir),
        "episodes": result.episodes_completed,
        "eval_episodes": summary.episodes,
        "eval_success_rate": summary.success_rate,
        "eval_mean_base_return": summary.mean_base_return,
        "eval_mean_delivered_value": summary.mean_delivered_value,
        "eval_mean_steps": summary.mean_steps,
    }


def run_study(
    config: TuningConfig,
    output_root: Path,
    *,
    stream: Any = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Run the search until the ledger is spent, then write every artefact.

    Args:
        config: what to search and for how long.
        output_root: where ``study.json``, ``trials.csv``, ``plots/`` and the
            confirmation run are written.
        stream: progress destination; defaults to ``sys.stderr``.
        make_plots: whether to render the figures.

    Returns:
        The payload written to ``study.json``. Its ``search_is_meaningful`` flag is
        ``False`` while any human-owned function is a placeholder -- in that state the
        trials are real training runs but the ranking over them is not a ranking.
    """
    out = stream if stream is not None else sys.stderr
    output_root.mkdir(parents=True, exist_ok=True)

    scenario = resolve_scenario(config.scenario)
    reference_return = (
        config.reference_return
        if config.reference_return is not None
        else best_affordable_return(scenario)
    )
    agent_pending = teaching_stub_status()
    tuning_pending = tuning_stub_status()
    if tuning_pending:
        print(
            TUNING_STUB_WARNING.format(names=", ".join(tuning_pending), score=SCORE_WHEN_STUBBED),
            file=out,
        )

    ledger = EpisodeLedger(total=config.study_episode_budget, per_trial=config.episodes_per_trial)
    study = _make_study(config)
    print(
        f"{config.name}: up to {config.fundable_trials} trials of "
        f"{config.episodes_per_trial:,} episodes, budget {ledger.total:,}, "
        f"reference return {reference_return:g}",
        file=out,
    )

    import optuna

    outcomes: list[TrialOutcome] = []
    while not ledger.exhausted:
        if config.max_trials is not None and len(outcomes) >= config.max_trials:
            break
        granted = ledger.grant()
        trial = study.ask()
        outcome = run_trial(
            config, scenario, trial, granted=granted, reference_return=reference_return
        )
        ledger.record(outcome.episodes_run)
        if outcome.pruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, outcome.score)
        outcomes.append(outcome)
        print(_trial_line(outcome, ledger), file=out, flush=True)

    front = pareto_front([outcome for outcome in outcomes if not outcome.pruned])
    best = _best_outcome(outcomes)
    confirmation = (
        _confirm_best(config, scenario, best, output_root, ledger, out)
        if best is not None and config.confirm_best
        else None
    )

    payload: dict[str, Any] = {
        "config": config.as_dict(),
        "environment": environment_metadata(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario": scenario.name,
        "reference_return": reference_return,
        "pending_human_functions": list(agent_pending) + list(tuning_pending),
        "learning_is_meaningful": not agent_pending,
        "search_is_meaningful": not (agent_pending or tuning_pending),
        "budget": ledger.as_dict(),
        "trials": [outcome.as_dict() for outcome in outcomes],
        "pareto_front": [outcome.number for outcome in front],
        "best_trial": best.as_dict() if best is not None else None,
        "confirmation_run": confirmation,
    }
    write_json(output_root / "study.json", payload)
    _write_trials_csv(output_root / "trials.csv", outcomes)

    if make_plots:
        from .plots import write_tuning_plots

        payload["plots"] = [str(path) for path in write_tuning_plots(output_root, outcomes, front)]
    return payload


def _trial_line(outcome: TrialOutcome, ledger: EpisodeLedger) -> str:
    """One progress line per trial."""
    score = "pruned" if outcome.score is None else f"score={outcome.score:0.4f}"
    return (
        f"trial {outcome.number:>3}  {score:<14} "
        f"best={outcome.best_return:7.2f} @ {outcome.episodes_to_best:>6} ep  "
        f"ran={outcome.episodes_run:>6}  spent={ledger.spent:,}/{ledger.total:,}"
    )


def _write_trials_csv(path: Path, outcomes: Sequence[TrialOutcome]) -> None:
    """Write one row per trial, with the union of every trial's parameter columns.

    A conditional search space means trials do not share a parameter set, so the header
    is the union and a trial that never suggested a parameter leaves its cell empty --
    which is the honest record of "not sampled", and is not the same as zero.
    """
    import csv

    rows = [outcome.as_row() for outcome in outcomes]
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(value) for key, value in row.items()})


__all__ = [
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_CHECKPOINT_EPISODES",
    "DEFAULT_STUDY_EPISODE_BUDGET",
    "DEFAULT_TRIAL_EPISODE_CAP",
    "HUMAN_OWNED_FUNCTIONS",
    "SCORE_WHEN_STUBBED",
    "EpisodeLedger",
    "ParameterSuggester",
    "TrialOutcome",
    "TuningConfig",
    "average_curves",
    "best_affordable_return",
    "curve_peak",
    "pareto_front",
    "run_study",
    "run_trial",
    "score_learning_curve",
    "suggest_trial_params",
    "train_config_for",
    "tuning_stub_status",
]
