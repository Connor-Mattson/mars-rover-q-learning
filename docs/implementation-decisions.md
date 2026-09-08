# Implementation decisions

Choices that the project brief left open, with the reasoning. Each entry is the kind of
thing an interviewer might poke at.

## MDP and dynamics

**Energy is charged for the tile the rover occupies *after* the transition.**
A successful move into rough terrain costs the rough tile's energy; a slip, a
deflection into a wall, or a wall collision leaves the rover in place and costs the
*current* tile's energy. One sentence covers every case, and it makes entering
expensive terrain expensive — which is what the risk/value trade-off needs. It also
makes the Dijkstra edge weight (`cost of entering the destination`) match the
environment exactly, so the shaping distances are the real thing and not an
approximation.

**Slip probabilities come from the tile the rover is departing from.**
The alternative — the destination tile — would let the rover's outcome depend on a cell
it may never enter, and would need a special case for out-of-bounds targets. Departure
terrain keeps the transition a function of the current state alone, which is what
"fully Markov" requires here.

**Delivery is checked before battery depletion.**
Arriving home on the last joule completes the mission. The opposite convention would
make a perfectly executed, perfectly budgeted mission read as a failure.

**Battery is stored as an exact integer, not binned.**
Measured table sizes are 24,400 / 40,896 / 104,256 states (× 5 actions) for the three
bundled scenarios — at most about 2.1 MB of `float64`. No binning is needed, so none was
introduced. If a future scenario forces a change, document it here *before*
implementing it.

**Every action is always legal.**
Driving into a wall and collecting on an empty tile are legal actions that waste a step
and energy. `action_mask` exists and is threaded through `select_action` because
masking is a real technique worth understanding, but it returns all-`True`: learning not
to waste energy is part of the task.

**Truncation is a non-bootstrapping boundary.**
Stated in the README. The step limit is not really an absorbing state, so treating it as
one is slightly pessimistic; it is applied identically in every condition, and it keeps
the update rule the human writes to a single case distinction.

## Rewards

**The subgoal heuristic is "highest-value sample whose round trip fits in 80% of the
battery", falling back to the cheapest reachable sample.**
Both shaping schemes need *some* fixed subgoal. A pure value-per-energy ratio picked the
basalt core on `risk_value_tradeoff`, which made shaping aim at the least interesting
mission on the map built to test ambition. Affordability-then-value is just as simple to
state, is computed once from static geometry, and resolves to the biosignature on all
three bundled maps. Its narrowness is recorded as a limitation in the README and the
experiment plan.

**Distances are energy-weighted, not step counts.**
"One shortest-path step closer" is measured in the same units the rover actually spends.
The naive scheme's asymmetry — and therefore the trap — is unaffected by the choice,
since a two-cell cycle changes the distance by the same magnitude in both directions.

**Naive shaping contributes nothing on the collection step.**
The subgoal changes from "the sample" to "the lander" at that moment, so a progress term
would be comparing distances to two different targets. Returning `0.0` for the one
transition where the stage flips is the honest option. Potential-based shaping has no
such problem: `Φ` is a function of the state, so the stage change is already priced in.

**`Φ = 0` in every episode-ending state, truncation included.**
This matches the non-bootstrapping convention and gives a clean, testable property: with
`gamma = 1`, the shaping accumulated over a finished episode telescopes exactly to
`−Φ(s₀)`, and a closed cycle contributes exactly zero. Both are asserted in
`tests/unit/test_rewards.py`. Zeroing only on true termination would leave a residual on
truncated episodes and no exact test.

## Structure

**Reward models live outside the agent.**
`MarsRoverEnv` owns a `RewardModel` and returns `base_reward` and `shaping_reward` as
separate `info` fields. The agent sees one scalar and cannot tell the conditions apart —
which is exactly what makes the comparison a comparison.

**Two modules beyond the brief's file list: `plots.py` and `experiment.py`.**
The brief's skeleton had no home for matplotlib figures or for the grid runner, and
folding either into `metrics.py` or `cli.py` would have made those files do two jobs.
`metrics.py` stays pure computation with no matplotlib import, so the test suite does
not pay for a plotting backend it does not use.

**`greedy_actions` lives in `metrics.py`, not `agent.py`.**
The policy overlay and the saved `policy.npy` need a plain per-state `argmax`. It is a
reporting utility for an already-learned table: no exploration, no tie-breaking, and
neither the training nor the evaluation loop calls it. Both loops go through the
human-owned `select_action`, and
`tests/integration/test_training_pipeline.py::test_training_calls_every_human_owned_function`
enforces that.

**`teaching_stub_status()` probes behaviour instead of reading a flag.**
It calls each human-owned function once on a trivial input and reports the ones whose
documented contract is obviously unmet. That means the teaching warning disappears on
its own the moment the functions work — no flag for Connor to remember to flip, and no
way for a stubbed run to quietly look like a real one.

## Tooling

**`pytest` deselects `human_todo` by default** (`addopts = -m 'not human_todo'`). The
default command is the quality gate and must be green; the teaching suite is run
explicitly. Both commands are documented in the README and `.teacher/how-to-test.md`.

**Confidence intervals use a Student-t critical value from a small built-in table.**
With five seeds a normal approximation is noticeably too narrow, and `scipy` is not
worth a runtime dependency for one number. A single sample reports `nan`, never a
zero-width interval.

**JSON output is strict.** Non-finite floats (an undefined interval, a mean over zero
successes) are written as `null` rather than the `NaN` literal that Python's `json`
emits by default, so the summaries can be read by anything.

**Ruff `ARG` is ignored in `agent.py` and `rewards.py`.** The teaching stubs deliberately
ignore most of their arguments, and reward models implement a fixed signature that not
every condition needs in full. The ignores are scoped to those two files with the reason
recorded in `pyproject.toml`.

## Measured runtime

Recorded on the development machine (Apple Silicon laptop, CPU only, Python 3.12.7):

- Environment throughput is roughly 10⁵ steps/second under a random policy.
- `configs/experiments/smoke.json` (1 scenario × 3 rewards × 2 seeds × 40 episodes)
  finishes in about **1 second**, including plots.
- The full `reward_comparison.json` grid (45 cells × 4000 training + 300 evaluation
  episodes) took **3 minutes 49 seconds** wall clock on one core, and wrote about
  **139 MB** of artifacts — most of it the 45 saved Q-tables, which is why `artifacts/`
  is gitignored.

**Caveat, stated because it matters:** these timings were taken while
`calculate_target`, `calculate_td_error`, and `update_q_value` are still no-op
placeholders. The real update math adds a `max` over five values and one array write per
step, so expect the full grid to get somewhat slower once the functions are implemented.
Re-measure and replace this section with the real figure before quoting it anywhere.
