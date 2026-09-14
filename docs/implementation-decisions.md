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

**Battery is exact in the environment and binned in the table, at the samples'
mission costs.**
*Supersedes an earlier entry that read "battery is stored as an exact integer, not
binned", on the grounds that the tables were small enough in bytes that no binning was
needed. That was the wrong measure. Recorded after the change rather than before it,
which the workflow asks for the other way round.*

Memory was never the problem; **statistical** cost was. Under the dense encoding a
cell's 61 battery copies are 61 unrelated table rows that share no experience, and the
positional signal has to survive that many layers of chained bootstrapping. In the
`safe_corridor__sparse__seed1` run it did not: at battery 55 the learned value is
`79.775` at `(5,1)`, `(5,2)`, `(5,3)`, `(5,4)` and `(6,4)` alike — identical to five
decimals across the whole corridor, rising by exactly `1/gamma` per unit of charge. The
table had learned one scalar, "I will deliver the 90-point sample, discounted by how
much battery I have burned", with no positional structure left in it. At the canonical
start all five actions were equal to ten decimal places after 105,125 visits, so the
first action of the mission was a coin toss.

The bin edges are each sample's **mission cost** — its lander round trip plus the
`COLLECT` charge, the exact charge below which that sample stops being deliverable.
Between two adjacent edges the affordable set is constant and so is the decision the
rover faces; the readings in between differ only by a factor of `gamma`, which the
discount already accounts for. All three bundled maps have three distinct costs and
therefore four bins: 1,600 / 2,304 / 2,304 rows against 24,400 / 40,896 / 104,256.

Checked before it was made the default, by solving each scenario exactly (value
iteration on the full battery-aware MDP) and then computing the best policy available
to a coarser state, by policy iteration within that restricted class:

| Scenario | `V*(start)` | best 4-bin policy | best battery-free policy |
|---|---|---|---|
| `safe_corridor` | 117.567 | 117.567 | 79.775 |
| `risk_value_tradeoff` | 129.337 | 129.323 | 129.303 |
| `shaping_trap` | 128.805 | 128.805 | 128.805 |

Four bins cost nothing at the canonical start, and dropping battery *entirely* is what
does not work: over every admissible start state, mean loss against `V*` is 1.90 / 0.83
/ 0.17 for four bins against 5.84 / 7.66 / 1.01 with no battery axis at all. The
curriculum trains from states the mission never reaches — stranded mid-map on low
charge, where the right call is to abort for a nearer sample — and those are exactly
the states that need the axis.

These are **planning bounds, not learning results**: they say what a policy of each
shape *could* achieve, not what tabular Q-learning finds. No claim about learned
performance under either encoding belongs anywhere until a real experiment has been
run under both.

`--dense-battery` keeps the original encoding, and `ExperimentConfig.battery_encoding`
carries it through a whole grid, because the binning is a claim and the dense run is
the control it has to be measured against.

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
to keep it is one map per level of that axis, so `figs/q_by_battery/` holds four frames
per scenario by default and sixty-one / seventy-one / a hundred and eighty-one under
`--dense-battery`. They are zero-padded and named by *level index* rather than by
battery reading, because the index is what stays sortable under either encoding; the
range of readings a frame covers is on the frame itself, in its title and in a gauge
that fills solid to the bin's floor and lighter to its ceiling. `--no-battery-figs`
opts out for callers redrawing figures in bulk.

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

**Figures read a saved table through the encoding of the run that wrote it.** The
battery axis is a property of a run, not of the version drawing its figures, so
`run_figures` takes a `BatteryBinning` and defaults it to *dense* rather than to the
current training default: a run with nothing recorded about its encoding predates the
flag, and every one of those is dense. The CLI reads
`manifest["config"]["battery_encoding"]` and passes it to `figures`, `evaluate` and
`replay`; `evaluate` additionally refuses a table whose row count disagrees with the
scenario under the requested encoding, so a mismatch is an error rather than a silently
misfiled score.

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


**The reachable ceiling is per payload, not one traversable-cell count reused four
times.** `coverage_report` used to compute `traversable_cells * battery_levels` once and
hand the same number to all four payload slices. That is a statement about geometry and
the coverage figure reads it as a statement about physics: carrying a sample means
having already paid for the trip out to it, so the charge still aboard is capped by that
sample's distance from the lander. On `safe_corridor` the biosignature sits 16 energy
out, so a rover holding it can never have more than 43 of 60 charge, and 27 of the 172
states in that slice cannot exist -- which the figure was drawing as 27 states the run
failed to reach, right on the payload the curriculum exists to reach. `run_figures.
occupiable_state_mask` now applies the same three-part test `MarsRoverEnv._validated_start`
applies to an injected start -- non-wall, battery arithmetic payable, not already
terminal -- so the ceiling excludes impossible states, a flat battery, and a sample
carried onto the lander (that is the delivery, entered and never acted from). The
`safe_corridor__sparse__seed1` figure went from a biosignature slice reading 36.0% to
144 of 145 occupiable states, and the whole table from 94.5% of 688 to 99.5% of 653.
The direct bar labels now quote the against-reachable share; against-total was mostly a
count of how much of the map is wall.


## Hyper-parameter search (Optuna)

**Optuna is a dependency, and it is not an RL framework.** Non-negotiable #1 bans
Gymnasium, Stable-Baselines, RLlib and friends because they would hide the learning loop.
Optuna hides nothing: it never sees the environment, the agent, a reward, or a Q-table.
It proposes a dict of numbers and reads one float back, which is the same contract a
hand-written random search would have, with a better sampler behind it. It is declared in
the main dependency list rather than an extra because `cli tune` is not optional tooling.

**The per-trial episode cap is what makes the question measurable.** "Which
hyper-parameters learn fastest" has no answer if every trial trains to convergence: all
the converging settings tie, and the search quietly becomes a search for final
performance. The cap is set below convergence on purpose (6000 against roughly 4000 for
a tuned agent on `safe_corridor`), so that settings separate on *when* they get there.
The cost is that the cap is part of the objective's definition and a winner must name it.

**Grants are all-or-nothing.** `EpisodeLedger.grant` returns a full trial's episodes or
zero, never a partial remainder. `score_learning_curve` normalises every trial against
the same cap precisely so scores are comparable; a trial run on a short grant would have
a different x-axis from everything it is ranked against, so it would be worth less than
the episodes it cost. The ledger spends what trials *ran*, not what they were granted, so
a pruned trial returns the difference and the study funds more trials than
`fundable_trials`.

**Pruning reports the raw checkpoint return, not the trial's score.** A pruner compares
trials at a fixed step, so it needs the quantity that is comparable at a fixed episode
count; the score is a whole-curve summary and is computed once, at the end. This is also
why pruning is decided on the *first seed only*: reporting a step-0 value once per seed
would make one trial number carry three incomparable sequences.

**The prune decision travels through `train`'s new `checkpoint_callback`.** Returning
`False` stops the run where it stands. Checkpoints are measured on a seed space disjoint
from training and on their own environment, so a callback cannot change what a run
learns -- `test_not_stopping_leaves_the_run_identical_to_one_without_a_callback` asserts
the tables are bit-identical. `TrainResult.episodes_completed` and `stopped_early` exist
because the ledger has to charge what was run, and a stopped run does not append a final
checkpoint: it broke out immediately after one, and a second measurement at the same
episode index would be a duplicate point and a wasted evaluation.

**Epsilon's floor is sampled as a fraction of its ceiling.** `EpsilonSchedule` requires
`end <= start`. Sampling the two rates independently would put a triangular corner of the
space out of bounds, and the sampler would keep proposing trials that raise instead of
training. `epsilon_end_ratio` removes the invalid region entirely rather than rejecting
draws from it.

**The curriculum knobs are conditional, not always-sampled.** `window_fraction` means
nothing to the growing window; `weight_exponent` means nothing to either open-loop
schedule. Suggesting a parameter the run will ignore teaches TPE's model that the value
was tried and changed nothing, which is how a dead dimension dilutes a search. For the
same reason the no-curriculum arm is a categorical switch rather than a fraction that
might land on zero -- a continuous draw essentially never does, so the arm would never be
explored.

**The objective takes its own interface, not `optuna.trial.Trial`.**
`suggest_trial_params` is typed against the `ParameterSuggester` protocol: two suggestion
methods and nothing else. A real `Trial` satisfies it, and the search space can be tested
against a recorder that reports which parameters were asked for, on what ranges, and
under which condition -- which is exactly what a search space *is*, and what a real trial
would answer stochastically.

**The reference return is a planning bound, and the ceiling is unreachable on purpose.**
`best_affordable_return` is the most valuable sample whose lander round trip fits the
battery: 160 on all three bundled maps, from shortest paths alone. Slip lowers the
achievable mean and a discount below 1.0 can make a nearer sample optimal, so a score of
1.0 cannot be attained. That is the right shape for a *scale*: it is fixed by geometry
rather than by the best trial seen so far, so scores do not move as the study proceeds,
and two maps whose payoffs differ fourfold produce comparable numbers.

**Trials keep no Q-table; the winner is re-trained.** Hundreds of tables is gigabytes of
artefacts nothing reads. The best trial's configuration is trained once more into
`best_run/` and evaluated on the same `eval_episodes` ruler every other study uses, and
those episodes are charged to the same ledger -- confirming a winner is part of the cost
of the search, and leaving it off the books would make the stated bound untrue. The
confirmation number, not the winning trial's score, is the one to quote: the score was
measured on the data the winner was selected on.

**The study is in-memory, with no SQLite storage.** A search is reproducible from its
`sampler_seed`, and `study.json` plus `trials.csv` are the record. An Optuna storage file
would be a second, divergent copy of the same run, and would also make the artefact tree
depend on Optuna's schema version.

**`study.json` separates two kinds of meaningless.** `learning_is_meaningful` is false
when the agent's own functions are stubs -- the tables are garbage. `search_is_meaningful`
is false when the *objective* is stubbed -- the tables are real and only the ranking over
them is fiction. They are different failures with different banners, and a reader of an
old artefact needs to be able to tell which one they are holding.

**Underflow is excluded from the floating-point guard.** The entry points install
`divide`, `over` and `invalid` as raising (`src/mars_rover_q/numerics.py`) because each
one means a defect here and a `nan` in a Q-table is silent and unrecoverable. `under` is
left at numpy's default of ignore: underflow is not an error but the mechanism the
log-sum-exp identity depends on, and `np.seterr` applies to the whole process rather than
to this package -- Optuna's TPE sampler scores candidates that way, so raising on
underflow crashed the search at the first trial the sampler modelled.

**The battery encoding is searchable, but off by default (`search_battery_encoding`).**
It is not a knob on the learning rule but on the table the rule writes into, so a study
that varied it between trials would be comparing two representations as well as two
settings -- and every other study in `docs/experiment-plan.md` holds the representation
fixed for exactly that reason. It is searchable at all because on these maps it moves the
outcome further than any of the six continuous knobs, and the winning encoding depends on
the map and on the curriculum it is paired with, so there is no single value to fix it to.
When a trial suggests one it overrides `TuningConfig.battery_encoding`, and the
confirmation run evaluates the winner through the encoding that winner trained under.

**Curated README figures live in `docs/assets/`, not in `artifacts/`.** `artifacts/` is
generated output and is gitignored, so a figure referenced from the README has to be
copied somewhere tracked or the README is broken in a fresh clone. Keeping the copies
under `docs/` rather than force-adding into `artifacts/` leaves the ignore rule intact
and keeps the distinction clean: `artifacts/` is what a run produced, `docs/assets/` is
the small hand-picked subset the write-up actually refers to. The four files there total
436 KB.
