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

## Exploration and the start-state curriculum

**The exploration problem is fixed with a start-state curriculum, not with reward
shaping or hyper-parameters.** Discovered by inspecting early runs: away from the
corridors the agent actually walked, the greedy policy flickered at random, which is the
signature of rows whose action values are all still at their initial value, so
`select_action` breaks a five-way tie by coin toss. Raising epsilon or lengthening its
decay spends more of the budget on random walks that still have to complete a long
round trip by accident; shaping would change what is being compared, which is the point
of the experiment. Changing where episodes *begin* leaves both the MDP and the three
reward definitions untouched, so the reward comparison stays valid.

**A start state must be physically reachable, and the check is exact rather than
heuristic.** The battery must account for at least the cheapest route from the lander —
via the carried sample's own cell, plus the `COLLECT` charge, when the bay is full —
using the same energy-weighted Dijkstra distances the shaping models use. Spending more
than the cheapest route is possible (the rover can wander), so the check is a lower
bound on energy spent, not an equality. States that cannot still complete a delivery are
excluded too: they can only teach failure, and they would otherwise crowd the easy end
of the ranking.

**Difficulty is the remaining energy cost to finish, not the distance from the lander.**
Ranking by "far from home" would call a doomed state with a flat battery easy. Ranking
by cost-to-go puts *carrying a sample one tile from the lander* at the easy end, which
is exactly the state whose value has to be learned first for the terminal reward to
propagate at all.

**The curriculum is off by default (`curriculum_fraction = 0.0`).** It is an
experimental condition that has to be requested, not a silent change to the baseline
every earlier run was measured against. `split_rngs` gained a third stream for
start-state sampling; `SeedSequence.spawn` is prefix-stable, so the environment and
agent streams are byte-identical to before and no existing run moved.

**Curriculum episodes are excluded from every learning curve and threshold metric.**
They are easier by construction, so a success rate that mixed them would describe a
different mission than the one being compared. `EpisodeRecord.from_canonical_start`
records the distinction per episode, and `env_steps_before` is left counting *every*
step the agent took, so sample-efficiency comparisons stay honest across conditions.
Evaluation always resets at the lander, and never uses the curriculum at all.

**Potential-based shaping still telescopes, but to a different constant per episode.**
A shaped episode's return telescopes to `-Phi(s_0)`, and `s_0` now varies. The optimal
policy is unaffected (that is the theorem's whole content), but it is another reason
shaped returns are only compared within the canonical-start episodes.

**A requested-but-inert curriculum is reported, never silently ignored.** If
`enumerate_start_states` returns nothing for a map, the pool is empty; `train` prints a
`CURRICULUM NOT ACTIVE` banner and the manifest records
`"curriculum": {"requested": true, "active": false}`, so a fallback run cannot be
mistaken for a curriculum result.

**Open thread: discovery is fixed, credit assignment to the lander is not.** A single-seed
diagnostic pair — `risk_value_tradeoff`, sparse, seed 1, 4000 episodes, with and without
`--curriculum-fraction 0.5` — showed the baseline delivering the 40-point basalt 3502
times and the 160-point biosignature zero times, against 138 biosignature and 102
hydrated-mineral deliveries under the curriculum. So the terminal reward is now being
seen. Greedy evaluation from the lander nonetheless returns basalt in both runs. Two
suspects, in order: `start_state_difficulty` ranks an empty-bay state by its *cheapest*
itinerary, which sorts the easy end of the pool toward whichever sample is nearest
irrespective of value, so the empty-handed states the curriculum practises most are the
ones next to the basalt; and the anneal finishes with epsilon already at its 0.05 floor,
leaving little exploration to carry the newly learned carrying-values back along the
fourteen-tile walk out to the biosignature. This is a diagnostic spot check on one seed,
not an experimental result, and it is deliberately not in the README's Results section.

**State coverage is measured, not eyeballed.** `metrics.tied_state_fraction` reports the
share of table rows whose action values are all equal — the number behind the flickering
overlay. It is printed by `train`, stored in every manifest, and aggregated across seeds
as `train_tied_state_fraction`.

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

## The budget sweep (Study 2)

**Each budget trains its own table rather than checkpointing one long run.** The
epsilon schedule (`epsilon_decay_fraction` of the budget) and the curriculum anneal
(`anneal_fraction` of the budget) are both defined as *fractions* of the episode
count, so a snapshot taken at episode 1000 of a 20000-episode run is an agent
mid-anneal, not a 1000-budget agent. Checkpointing would have cost about half as
much and answered a different question. Recorded because the cost difference is
visible and the reason for paying it is not.

**Budget and curriculum are plural factor fields, not new config classes.**
`ExperimentConfig` gained `episode_budgets` and `curriculum_fractions`, which are
empty by default and fall back to the existing `episodes` / `curriculum_fraction`
scalars via the `budgets` / `curricula` properties. Every config file written before
the sweep therefore still describes exactly the grid it always described, and
`aggregate_rows` groups on `CONDITION_KEYS` with `.get`, so rows from those older
grids still aggregate as a single condition instead of raising.

**Run-directory names only name the factors that vary.** `run_name` appends `cf…`
and `ep…` segments when there is more than one level of that factor. A grid that
does not sweep them produces the same directory names it did before the sweep
existed, and a swept grid still gets one unique directory per cell.

**`save_q_tables: "max_budget"` is the sweep's default.** The grid is 300 cells and
the Q-tables are 1.0–4.2 MB each; keeping them all would cost about 1 GB to store
tables that no part of the analysis reads. The largest budget keeps its table so the
final policy is still inspectable and replayable. Everything the analysis does read —
the summary row, the training CSV, the manifest — is written for every cell, and
`load_run` raises a message naming this setting rather than a bare `FileNotFoundError`
when a lean cell's table is asked for.

**`--jobs` uses a stdlib process pool and returns only the summary row.** Cells are
independent and every RNG is injected, so results do not depend on the worker count —
there is an integration test asserting exactly that. The worker returns the row rather
than the `TrainResult`, because pickling a Q-table and 20000 episode records back to
the parent costs more than the training did.

**The sweep figures use their own two-colour palette.** `CURRICULUM_BASELINE_COLOUR`
and `CURRICULUM_TREATMENT_COLOUR` are deliberately disjoint from `REWARD_COLOURS`:
these figures encode a different factor, and reusing a reward mode's hue for "no
curriculum" would make one colour mean two things across the artifact set. The pair
was checked for colour-vision separation (worst adjacent ΔE 21.6 protan / 18.1 tritan)
and ≥3:1 contrast on both light and dark surfaces.

**Budget axes tick at the measured budgets, labelled `1k`/`3k`/`20k`.** Matplotlib's
default log ticks read `2 x 10¹`, which is the wrong vocabulary for an episode count.
Ticking only where a cell was actually run is also the honest labelling: the line
between two ticks is drawn interpolation, not data.

**`cli analyze` re-reads `summary.json` instead of re-running.** The sweep takes tens
of minutes; changing the analysis metric, the target fraction, or a figure must not
cost another training run.

**`ruff` `ARG001` is ignored in `metrics.py` and `sweep.py` while the assignment is
open.** Same reason as the `agent.py` ignore: a `TODO(human):` placeholder cannot use
the arguments its contract documents. Scoped to those two files and removed when the
assignment closes.

**Visit counts are recorded during training rather than inferred from the table.**
Coverage is recoverable from a finished Q-table — a row that still equals `initial_q`
everywhere was never written — but *experience* is not: a state updated ten thousand
times and a state updated once are both simply "not the initial value". The trainer
therefore keeps an `int64` counter per state, incremented on the state each update was
applied to. It costs one array of `num_states` and one increment per environment step,
and it is what makes the experience heatmaps a measurement rather than a proxy.

**The counter increments on the updated state, not the state landed on.** The figure
is a record of where learning happened, so a terminal state the rover steps into and
never bootstraps from accrues nothing. This is why the lander cell reads "never
reached" in the three carrying panels: an episode ends the moment a sample is
delivered, so no update is ever applied from there with a full bay.

**`visit_counts.npy` is gated behind `save_q_tables` with the policy.** It is the same
size as `policy.npy` and, like the policy, nothing in the sweep analysis reads it. A
run saved without its table reports `visit_counts: null` in the manifest and loads as
`None` rather than as a zero array, so a reader can tell "no experience" apart from
"experience not recorded" — and the figures fall back to a coarser per-cell quantity,
labelled as such on the colour bar, rather than silently plotting something else under
the same legend.

**Per-run figures are written by the CLI, not by `save_run`.** The sweep calls
`save_run` 300+ times; drawing two matplotlib figures per cell would dominate its
runtime for artifacts nobody opens. `train` writes them because a single run is
exactly the case where they are read, `--no-figs` opts out, and `figures --run`
redraws them for any saved run without retraining.

**The per-battery value maps are a directory, not a page.** Battery is the axis the
mission turns on and the one both summary figures marginalise away; the only honest way
to keep it is one map per reading, so `figs/q_by_battery/` holds sixty-one frames on
`safe_corridor` and a hundred and eighty-one on `shaping_trap`. They are zero-padded
and named by battery so a viewer's arrow keys walk the drain in order, and
`--no-battery-figs` opts out for callers redrawing figures in bulk.

**The battery frames share one colour scale, computed once across the whole table.**
Per-frame normalisation would make each image internally legible and the set as a
sequence meaningless — a cell whose colour changed between two frames would carry no
information about whether its value moved. The scale is diverging and pinned to zero,
which is both the table's initial value and the boundary between a state worth
occupying and a liability, with the two arms scaled independently to their own extremes
and the colour bar ticked at both ends and at zero to say so.

**The policy arrow is drawn only where the best action is unique.** `argmax` returns
the lowest-numbered action of a tied set, so an arrow drawn from it unconditionally
would invent a preference the table does not hold — and under a sparse reward a large
share of updated states are still tied. The battery frames therefore gate the arrow on
a *unique* maximum, which is stricter than `tied_state_fraction`'s flat-row test: a row
can be split at the top and still be undecided about where to drive. The absent arrow
is the finding, and the panel title counts how many cells have one.

**The figures arrow the policy; the Pygame overlay letters it.** The renderer draws
`N`/`S`/`E`/`W`/`C` because it redraws every frame at speed and its glyphs sit in a
cell already carrying a rover sprite. A static map is read by scanning for the shape of
a route, which is what an arrow gives and a letter does not, so the figures use
`↑ ↓ ← →` with `O` for collect.

**Cells with no update are hatched rather than tinted.** The diverging ramp's midpoint
is nearly white, so a flat pale fill for "never updated" would sit one step away from a
genuine value of zero. Texture is not a colour and cannot be read as one.

**Both coverage denominators are reported.** A third of `safe_corridor`'s grid is
wall, so coverage against every encodable state understates what a run covered by
exactly that third — but the encodable total is what `num_states` means and what sizes
the table. The figure leads with the full denominator and states the traversable-ground
one beside it, with the reachable ceiling marked on every bar, rather than picking one
and leaving the reader to guess which.

**The run figures use a single-hue sequential ramp, and grey means "not data".** These
panels encode magnitude, so they take one blue ramp light→dark; a hue change across a
magnitude scale invites a reader to look for a category boundary that is not there.
Walls and never-reached cells are painted grey and legended, so no amount of blue is
ever read as either — and the ramp's lightest step is reserved for a count of one
rather than spent on a zero no cell is ever drawn with.

**A curriculum is a window plus a weight, and the two are separate knobs.** The
original schedule collapsed both into one number: the easiest `floor(N x progress)`
states, drawn uniformly. Splitting them is what makes the two new strategies
describable at all -- `sliding` changes the window and keeps the uniform draw,
`visit_weighted` changes the draw and keeps the window. `visit_weighted` deliberately
reuses `growing_window_bounds` rather than defining its own admission rule, so its
paired comparison against `growing` differs in exactly one respect. Two changes at once
would produce a number nobody could attribute.

**The visit-weighted schedule reads `visit_counts`, not a count of how often a state
was chosen as a start.** A state accrues Q-updates whenever a trajectory passes through
it, wherever that trajectory began; a state the agent drives through on every episode
is well estimated whether or not it was ever a start. The counter already existed for
the coverage figures, is indexed by `StateEncoder` row, and is counted on the state that
was *updated* -- so it is exactly "how much learning has this state had", which is the
quantity the tilt is supposed to invert.

**The gather is charged only to the strategy that needs it.** Reading counts in pool
order costs an `O(pool)` gather per episode -- 14298 elements on `risk_value_tradeoff`
-- so `StartStateCurriculum.needs_update_counts` gates it and the two open-loop
schedules pass `None`. `pool_state_indices` is encoded once at construction for the same
reason: the map is fixed, so the encoding never changes.

**The weight is a power law, not an exponential.** `(1 + n) ** -exponent` makes the
ratio between two states depend on their relative experience rather than on the absolute
count, so the tilt does not collapse into "only ever draw the newest state" once counts
reach five figures. It also keeps a heavily-updated state genuinely reachable: at a
hundred updates it is rare, not absent, and a state that can never be drawn again is a
state whose value can never be corrected. `exponent = 0` reproduces the uniform draw
exactly, which makes it the control for the weighting itself.

**The sliding band is a quarter of the pool by default.** That is the widest band that
still satisfies the "starts inside the easiest quarter" invariant the growing schedule
is held to, so both strategies begin from comparable support and diverge only in what
happens afterwards.

**A stubbed sampler is a labelled no-op, not an exception.** `curriculum_stub_status`
probes the human-owned sampler the way `agent.teaching_stub_status` probes the five
learning functions, and `train` prints a **CURRICULUM STRATEGY STUBBED** banner. The run
still produces a real Q-table -- every episode simply starts at the lander -- so the
failure worth guarding against is not a crash but a no-curriculum run being compared as
a curriculum one.

**The `curriculum` command dry-runs the schedule instead of probing it at fixed
progress.** The visit-weighted strategy is closed-loop: asking it for 200 draws at
`progress = 0.75` with no history is asking it a question it never faces. So the command
replays the anneal episode by episode with a synthetic counter, which is also why its
docstring says plainly that the feedback is a *lower bound* -- one synthetic update per
drawn start, where a real episode updates every state along its trajectory.

**The dry-run report is about the anneal, and the tail is its own row.** Post-anneal
episodes all start at the lander, and on `risk_value_tradeoff` the canonical start sits
in the easiest quarter of the pool by difficulty -- so folding those episodes into the
band shares reported 80% "easiest quarter" for a schedule whose anneal spends 60% there.
The summary line is computed over `progress < 1.0` and the tail is printed separately.

**Band shares, not mean difficulty, are the headline diagnostic.** Mean sampled
difficulty rises monotonically under the growing window and still hides the defect --
that was exactly the reading that looked healthy while the easy end was being resampled.
The share of episodes falling in each quarter of the ranked pool is the statistic that
shows it.

**One control arm, not one per strategy.** `curriculum_fraction = 0` starts every
episode at the lander whatever the sampler says, so `ExperimentConfig.cells` runs the
control once and `sweep.arm_of_row` normalises its strategy away. Running it per strategy
would train identical tables under three labels and then split one control three ways,
costing every paired comparison two-thirds of its seeds.

**Arm naming lives in `curriculum.py`.** The legend, the GIF filenames, the analysis
JSON, and the CLI table all name the same four arms, and they used to do it with four
copies of the same conditional. `curriculum_arm_label` and `curriculum_arm_slug` are now
the single source; the figures and the analysis cannot disagree about what an arm is
called.

**`--no-figs` in the CLI tests that are not about figures, and the `slow` marker
applied from measurement.** Five `train` smoke tests asserted on manifests, exit codes
and coverage output while silently rendering 61 battery frames each, which was 137s of
a 145s suite. They now pass `--no-figs`; the two tests that actually assert one frame
per battery level keep rendering and carry the `slow` marker. The marker had previously
been placed by intuition on seven sub-second `test_experiment.py` tests, so filtering on
it saved nothing -- it is now assigned from `pytest --durations`. Default `pytest` is
56s and still runs everything; `-m "not slow and not human_todo"` is the 12s inner loop.
The second clause is load-bearing: a command-line `-m` replaces `addopts` rather than
combining with it, so `-m "not slow"` alone re-enables the teaching suite.

