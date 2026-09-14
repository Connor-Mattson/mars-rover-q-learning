# Teaching history

A durable record of assignments Connor has completed on this repository. Entries are
appended **only after** an implementation passes review — never in advance, and never
for work that Claude supplied or substantially repaired.

Each entry records: the date, the assignment, the files touched, the review verdict, and
anything notable about the approach.

---

## 2026-09-08 — the tabular Q-learning core

- **Requested thrust:** a Mars sample-return rover trained by tabular Q-learning, with a
  hand-written Gymnasium-like environment, three reward schemes, a Pygame renderer, a
  CLI, and reproducible experiment plumbing.
- **Human-owned implementation:** `initialize_q_table`, `calculate_target`,
  `calculate_td_error`, `update_q_value`, and `select_action` in
  `src/mars_rover_q/agent.py` — the entire learning algorithm.
- **What it does:** allocates the state-by-action table at a requested fill value and
  dtype; builds the one-step Q-learning target, bootstrapping off the best successor
  action and never across a `terminated` or `truncated` boundary; computes the signed TD
  error; moves exactly one table entry along that error by the learning rate; and picks
  actions epsilon-greedily with uniform exploration, random tie-breaking among the
  greedy-best, and full respect for an action mask.
- **Skills practiced:** off-policy bootstrapping and the Bellman target, TD error sign
  conventions, in-place NumPy mutation with strict aliasing discipline, injected-RNG
  reproducibility, argmax tie-breaking as an exploration concern, and input validation
  at a module boundary.
- **Verification:** `pytest -m human_todo tests/human_todo` (34 passed), `pytest`
  (178 passed, 34 deselected), `ruff format --check .`, `ruff check .`, `mypy`
  (33 files, clean), and an end-to-end `train --scenario safe_corridor --reward sparse
  --seed 1 --episodes 4000` that printed no `TEACHING STATE` banner, climbed from
  `success=0.01` to `1.00`, and wrote `"learning_is_meaningful": true` with an
  evaluation `success_rate` of 1.0 over 200 episodes.
- **Review notes:** accepted on the second pass. The algorithm itself was correct on the
  first submission — notably the random tie-breaking in `select_action`, collected as
  all-argmax candidates rather than the usual `np.argmax` shortcut, and the single
  `terminated or truncated` case distinction in `calculate_target`. The revision round
  fixed three engineering defects rather than algorithmic ones: a hardcoded action count
  in `select_action` (now `q_table.shape[1]`), an empty-`action_mask` `ValueError` that
  was being raised incidentally by `max()` on an empty sequence rather than by an
  explicit guard, and `np.zeros(...) + initial_value` in `initialize_q_table` (now
  `np.full`), which left unreachable clamping behind and could silently promote the
  requested dtype. Remaining non-blocking polish: `max(next_state_values)` is the Python
  builtin where `np.max` is the idiom, `q_table[state_index][action_index]` chains two
  index operations where one tuple index would do, the `ValueError` messages are more
  verbose than the one-line `got {value}` style used elsewhere in the module, and the
  greedy candidate list is built even when the epsilon coin flip sends the call down the
  explore branch.

---

## 2026-09-09 — the annealed start-state curriculum

- **Requested thrust:** fix the exploration failure Connor diagnosed (see the Discovery
  log below) with an annealed curriculum over physically reachable start states, while
  evaluation continues to run from the canonical lander start.
- **Human-owned implementation:** `enumerate_start_states` and `sample_start_state` in
  `src/mars_rover_q/curriculum.py`.
- **What it does:** `enumerate_start_states` returns every state the rover could both
  have driven itself into and still finish the mission from — charging the cheapest
  itinerary that explains the state (via the carried sample's own cell plus the
  `COLLECT` cost when the bay is full) against the energy already missing, and requiring
  the remaining battery to cover `start_state_difficulty`. `sample_start_state` draws
  uniformly from the easiest `max(1, floor(len(pool) * progress))` states, widening the
  support as the anneal advances and returning the canonical start exactly at
  `progress == 1.0`.
- **Skills practiced:** reasoning about physical reachability as a two-sided constraint
  on a state's past and future, energy-weighted shortest-path composition over directed
  Dijkstra fields, exploiting a value's independence from a loop variable to turn a scan
  into an early-terminating band, curriculum design as a change to the state-visitation
  distribution rather than to the MDP, and injected-RNG determinism.
- **Verification:** `pytest -m human_todo tests/human_todo` (61 passed),
  `pytest` (223 passed, 61 deselected), `ruff format --check .`, `ruff check .`, `mypy`
  (36 files, clean), `cli curriculum --scenario risk_value_tradeoff` (pool 14298,
  difficulty quantiles 1.0 / 7.0 / 10.0 / 14.0 / 20.0, mean sampled difficulty rising
  monotonically across progress 0.00 → 0.75 and collapsing to the canonical start at
  1.00), and a matched pair of 4000-episode `risk_value_tradeoff` / sparse / seed 1
  training runs with and without `--curriculum-fraction 0.5`.
- **Result of that comparison:** the baseline delivered the 40-point basalt 3502 times
  and the 160-point biosignature **zero** times in 4000 episodes. The curriculum run
  delivered the biosignature 138 times and the 90-point hydrated mineral 102 times. The
  reward Connor identified as undiscoverable is now being discovered. Greedy evaluation
  from the lander still returns basalt in both runs (success rate 1.0, mean base return
  40.0 either way), so the curriculum has solved *discovery* without yet converting it
  into the canonical-start policy — the open thread recorded in
  `docs/implementation-decisions.md`.
- **Review notes:** accepted on the second pass; the first submission enumerated a single
  best-case battery per cell, priced the carrying case as if the rover had driven
  straight from the lander, and omitted the can-still-finish check, and Connor fixed all
  three from directional hints without being shown code. The accepted version is clean:
  the battery scan starts at the best case and stops at the first inadmissible level,
  which is correct precisely because `start_state_difficulty` does not depend on battery
  — a real observation, and cheaper than filtering the full band. Remaining non-blocking
  polish: `candidate_difficulty is not UNREACHABLE` compares floats by identity and is
  therefore always true (`Scenario.distance` returns a freshly boxed float), so the guard
  reads as live but is dead — the `battery_level >= candidate_difficulty` clause beside
  it is what actually rejects unreachable states; `candidate_difficulty > 0.0` excludes
  carrying-on-the-lander by side effect rather than by name; `scenario.distance(lander,
  curr_cell)` is computed inside the carried-sample loop and discarded on three of every
  four iterations; `[*list(COLLECTABLE_SAMPLES), SampleType.NONE]` has a redundant
  `list()` inside the splat; and the sampler's uniform draw over the easiest `k` keeps
  the trivial states at full weight late in the anneal, where a distribution tilted
  toward the newly admitted hard end would spend the budget better.

---

## 2026-09-09 — the budget-sweep analysis

- **Requested thrust:** validate the start-state curriculum scientifically — average
  base mission return against training budget, curriculum vs no curriculum, on all
  three maps, with the experiment design, run plumbing, and plotting to support it.
- **Human-owned implementation:** `paired_difference` in `src/mars_rover_q/metrics.py`
  and `budget_to_reach` in `src/mars_rover_q/sweep.py` — the two functions that turn
  300 rows of per-seed numbers into a defensible claim.
- **What it does:** `paired_difference` intersects the two arms on `pair_on`, drops
  pairs whose metric is `None` or non-finite on either side, differences within each
  surviving pair, and hands that sample to `mean_ci` — so the reported `n` is a count
  of pairs and the interval is the paired one. `budget_to_reach` scans for the first
  index at which a curve attains the target and interpolates between the bracketing
  pair on a `log10(budget)` axis, returning the measured budget where there is no
  usable predecessor to interpolate from, and `None` where the target is never reached.
- **Skills practiced:** paired versus independent comparison and why sharing an RNG
  stream makes the difference the right estimator; treating a `Raises:` clause as a
  specification rather than as whatever the tests sample; recognising an incidental
  raise; interpolation on a log-spaced axis; first-crossing semantics on a
  non-monotonic curve; and the discipline of never coercing a "did not reach" into a
  number.
- **Verification:** `pytest -m human_todo tests/human_todo/test_sweep_analysis.py`
  (28 passed), `pytest -m human_todo tests/human_todo` (89 passed), `pytest`
  (244 passed, 89 deselected), `ruff format --check .`, `ruff check .`, `mypy`
  (39 files, clean), a 12-cell smoke sweep end to end, and two randomised property
  checks against independently written references — 4000 budget curves (monotonic and
  noisy) and 3000 random arm pairs containing missing seeds, `None`s, and `nan`s, with
  zero mismatches in either.
- **Review notes:** accepted on the second pass. Both algorithms were correct on the
  first submission, including two decisions that were not forced by any test: returning
  the measured budget on an exact target match, which sidesteps the
  `10 ** log10(3000) == 2999.9999999999995` round trip that `pytest.approx` would have
  hidden; and handling `values[0] >= target` before the crossing scan, so the loop body
  never has to ask whether a predecessor exists and the nan-predecessor case drops in
  as one more branch. The revision round fixed two defects in the documented `Raises:`
  clauses, both of the same shape — a guard that covered what its test exercised and
  stopped there. `budgets[i] < 0` admitted a zero budget, which then raised
  `math domain error` from `math.log10` when it happened to be the crossing
  predecessor and returned `0` silently when it did not; this was the second appearance
  of the incidental-raise pattern first seen in the empty-`action_mask` guard, and it
  is now recorded in `.teacher/hints.md`. Duplicate-key detection sat inside the
  pairing loop and behind the usability flag, so it saw only keys common to both arms
  whose first occurrence was usable; hoisting it to a per-arm length comparison before
  pairing closed all three cases at once and removed the coupling that caused it.
  Remaining non-blocking polish: the baseline duplicate branch raises
  `"Treatment contains a duplicate key!"`, naming the wrong arm; the non-positive
  budget message reads `"contained a negative or value"` and the length message
  `"Values ant Budgets"`; error messages are prose sentences where the house style in
  `metrics.py` and `agent.py` is a one-line `f"... got {value}"`; `paired_difference`
  rescans both lists per key rather than building a lookup once, which is immaterial at
  ten seeds but is the same lookup that already fixed the duplicate check; the
  `differences` list is built in set-iteration order, which is deterministic for integer
  seeds but not for a string `pair_on`; `base_val` and `treat_val` are bound only inside
  the match branch and rely on the intersection invariant three lines away; and
  `math.pow(10, x)` is used where `10 ** x` is the module idiom.
## 2026-09-10 — the visit-weighted curriculum

- **Requested thrust:** two more ways for the start-state curriculum to draw from its
  ranked pool, because the growing window oversamples the easy end — a defect Connor
  found himself and diagnosed correctly (see the Discovery log entry for 2026-09-09).
- **Human-owned implementation:** `sample_visit_weighted_start_state` in
  `src/mars_rover_q/curriculum.py` — the closed-loop draw, and the first schedule in the
  repository that is a function of the table's own experience rather than of the episode
  counter.
- **What it does:** admits exactly the band `growing_window_bounds` returns, weights each
  admitted rank by the inverse power `1 / (1 + updates) ** exponent`, normalises, and
  draws one index from the injected generator with `rng.choice`. `exponent = 0` collapses
  every weight to one and recovers the uniform draw with no special case; the `+ 1` keeps
  an untouched state finite; the power law keeps a hundred-update state rare rather than
  absent, and keeps the tilt meaningful at both ends of a count vector that runs from
  zero to five figures within a single run.
- **Skills practiced:** choosing a weight family from the invariant it has to satisfy
  rather than from familiarity — the first attempt reached for softmax, and rejecting it
  required seeing that differences of unbounded counts have no fixed scale while ratios
  do; separating a per-element weight from the normaliser that turns weights into
  probabilities; reading a `Raises:` clause as a specification rather than as whatever
  the tests sample; recognising an incidental raise; and vectorising a function that sits
  on a per-episode critical path.
- **Verification:** `pytest -m human_todo tests/human_todo/test_visit_weighted_curriculum.py`
  (26 passed), `pytest -m human_todo tests/human_todo` (115 passed — every teaching test
  in the repository now green), `pytest` (370 passed, 115 deselected),
  `ruff format --check .`, `ruff check .`, `mypy` (46 files, clean), a 300-episode
  `visit_weighted` training run that no longer prints the **CURRICULUM STRATEGY STUBBED**
  banner, and the dry-run diagnostic on `risk_value_tradeoff`, which moved
  `visit_weighted` off `canonical=100%` to `easiest quarter=0.57 of anneal episodes` with
  `1895/14298` distinct starts, against `growing`'s `0.60` and `1767`. The margin is
  small because the dry run's synthetic counter increments only the sampled start and so
  understates the real feedback signal, which is documented in
  `docs/implementation-decisions.md`; Study 3 is what will actually measure this.
- **Review notes:** accepted on the second pass. The weight itself was right on the first
  submission and it was the part that needed judgement — inverse power rather than
  exponential, the `+ 1` in the denominator, the exponent placed where zero falls out as
  exactly uniform, weights built over the band rather than the pool, and
  `growing_window_bounds` called rather than reimplemented, which is what keeps the arm a
  controlled comparison. Three defects were fixed in the revision. The negative-count
  clause of the documented `Raises:` was missing entirely, and its test passed anyway on
  an incidental `ValueError` raised by `rng.choice` three frames below where the guard
  belonged; probing the space showed two of four representative inputs returning a state
  silently, including `exponent = 2.0`, where the sign squares away and a corrupt count
  is read as a *well-learned* one. This was the third appearance of the incidental-raise
  pattern, after the empty-`action_mask` guard and the zero-budget guard, and the third
  time the underlying cause was validating against the test rather than the sentence.
  The weights were built with a Python list comprehension over a band that reaches 14298
  elements once per episode, measured at 9.110 ms per call against 0.084 ms vectorised;
  the revision brought it to 0.461 ms. And the exponent guard admitted `0.0` while its
  message said `must be > 0.0`, naming the documented control for the whole feature as
  illegal. Remaining non-blocking polish: the builtin `min(update_counts)` iterates an
  ndarray element by element and is 0.336 ms of the remaining 0.461, where
  `np.min` is 0.001; that guard sits after the `progress >= 1.0` early return, so a
  terminal-progress call with a corrupt count vector returns `canonical` without raising,
  while the other three guards run before it; and the draw indexes `ranked_pool` with a
  band-relative index, which is correct only because `growing_window_bounds` anchors its
  low edge at zero — an invariant of a different function that nothing at the call site
  restates.
- **Correction to Claude's own code, found during this review:** `curriculum_stub_status`
  built a fresh `default_rng(0)` inside its draw comprehension, so all fifty draws
  replayed one stream and its `len(drawn) >= 2` check could never pass however correct
  the sampler was. Against the stub it reported pending for the right answer by the wrong
  route, which is why it went unnoticed. The generator is now hoisted, and
  `test_the_stub_probe_only_ever_names_the_human_owned_function` — which asserted a
  subset and so held in both states — is replaced by an equality assertion that a probe
  incapable of clearing would fail.



## 2026-09-12 — the search objective and the front it hides

- **Requested thrust:** Connor's own question — which hyper-parameters reach the best
  reward in the fewest episodes — together with the two constraints that make it
  answerable (see the Discovery log entry for 2026-09-10).
- **Human-owned implementation:** `score_learning_curve` and `pareto_front` in
  `src/mars_rover_q/tuning.py` — the scalarisation the Optuna study maximises, and the
  two-objective trade-off that scalarisation resolves and therefore conceals.
- **What they do:** the objective normalises the return axis by `reference_return` and
  the episode axis by `episode_cap`, clamps each normalised return into `[0, 1]`, and
  integrates with `numpy.trapezoid`. One area, and all six properties fall out of it:
  the ceiling integrates to exactly `1.0` over a unit-width axis, a curve at or below
  zero integrates to `0.0`, pointwise dominance is monotone because the integrand is,
  earliness is free because a plateau reached sooner is integrated over a wider
  interval, and both rescalings cancel in the normalisers before any arithmetic happens
  — so the invariances are structural rather than arranged. `pareto_front` validates
  every outcome up front, then sweeps the pool sorted by
  `(episodes_to_best, -best_return, number)`, keeping a point when its return beats the
  running best or ties it at the same cost, and returns the original objects by index.
- **Skills practiced:** finding the single quantity that satisfies six simultaneous
  constraints instead of bolting a speed penalty onto a performance term — a penalty
  would have introduced a third scale and broken both invariances; normalising units
  away before computing rather than correcting afterwards; reading two failing tests as
  one bug (the tie-order pair looked contradictory and was a single inverted tiebreak,
  invisible on the input that happened to arrive already in the wrong order); choosing
  the `O(n log n)` sweep over the quadratic definition and getting the case that sweep
  is known to fail — a tie on both axes — right; and placing validation before any
  comparison so a one-element input still raises.
- **Verification:** `pytest -m human_todo tests/human_todo/test_tuning_objective.py`
  (51 passed), `pytest -m human_todo tests/human_todo` (166 passed — every teaching test
  in the repository green again), `pytest` (512 passed, 166 deselected),
  `ruff format --check .`, `ruff check .`, `mypy` (50 files, clean). Beyond the suite: a
  randomised sweep of 20000 thirteen-point curves found no monotonicity or strictness
  violation and held both rescaling invariances to `1e-12` exactly. The real acceptance
  test was `tune --config configs/tuning/smoke.json`, which went from five unpruned
  trials all scoring `0.0000` to a ranking — `#3 score=0.2812` for the trial that
  reached `90.00` at 240 of 400 episodes, and `#4 0.1250` above `#1 0.0750` for two
  trials that both finished at `40.00` but arrived at 240 and 320 episodes, which is
  earliness doing exactly the job it exists for. The confirmation run of the winner
  reported `success=1.00 base_return=90.00`; before the fix it re-trained a stranded
  rover.
- **Review notes:** accepted on the second pass. `pareto_front` was complete and correct
  on the first submission, including the tie case and the validation placement, and was
  not touched in the revision. The objective had one blocking defect: the return axis
  was clamped at the ceiling but not at the floor, so negative returns kept subtracting
  area and a failure region large enough to cancel a later recovery sent a curve that
  *did* succeed to the same `0.0` as one that never left the battery penalty. Because
  every trial begins at `-100`, that region is the common shape rather than a corner: at
  the smoke config's 400-episode cap it flattened the objective to `0.0` on every
  scored trial in the study, and "best" fell through to the tie-break — trial `#0`,
  which never succeeded. What made it worse than a wrong number is that nothing said so.
  `tuning_stub_status` probes with curves starting at `0.0`, so it cleared, and
  `study.json` carried `search_is_meaningful: true` over a ranking that was uniformly
  zero — a fabricated result rather than a loud failure, which is the class of outcome
  non-negotiable 7 exists to prevent. The handoff test suite did not catch it either:
  every dominance test in it lived above zero. Two were added during the review,
  `test_dominance_survives_the_region_below_zero` and
  `test_the_floor_is_a_floor_and_not_a_ranking_of_failures`, the second pinning that
  `flat(-50)` and `flat(-100)` must both stay at `0.0` so the clause cannot be satisfied
  by removing the clamp altogether.
  Textual debris was cleaned up on acceptance rather than returned again: two
  commented-out `print(list(points))` lines — latent rather than inert, since `points` is
  a lazy `zip` and uncommenting the first would have silently emptied every front — a
  missing space across an implicitly concatenated error message, `"postive"`, and an
  empty-curve message that did not follow the house one-line shape. Remaining
  non-blocking polish, left as written: `max(curve)[0]` depends on tuples comparing
  lexicographically where strict ascent is already guaranteed three lines above and
  `curve[-1][0]` would say it directly; `id` shadows the builtin and
  `zip(..., range(len(outcomes)))` is `enumerate` spelled long; `aoc` names an area
  *under* a curve; and the `if aoc < 0.0: return 0.0` branch is now unreachable, which
  is itself the cleanest proof that the clamp ended up in the right place.
- **Correction to Claude's own code, found during this round:** four tests asserted the
  repository's *current* teaching state rather than the machinery — `tuning_stub_status()
  == HUMAN_OWNED_FUNCTIONS`, `search_is_meaningful is False`, and the two banner
  assertions — so all four failed the moment the assignment was finished, which is the
  one event they were supposed to survive. A stub detector whose tests require the stubs
  to exist can only be run once. They now install the placeholder behaviour through a
  `stubbed_tuning_objective` fixture in `tests/conftest.py`, and a new
  `test_the_tune_commands_teaching_line_tracks_the_payload_flag` asserts the coupling
  instead of the state: the warning must disappear exactly when `search_is_meaningful`
  becomes true, and must never print over a ranking that is real.
- **Second correction to Claude's own code, found when Connor first ran the study at
  scale:** the CLI entry point installed `np.seterr(all="raise")` process-wide. That
  guard is right for this codebase's own arithmetic — a `nan` reaching a Q-table is
  unrecoverable and silent — but `seterr` is not scoped to our code, and Optuna's TPE
  sampler evaluates the log-sum-exp identity, which computes `exp` of large negative
  numbers and *relies* on them flushing to zero: the terms it discards are the ones too
  small to matter. So the search died with `FloatingPointError: underflow encountered in
  exp` at trial 10 of 125 — the first trial after `startup_trials`, which is the first
  one the sampler models rather than draws from the prior. Every test was shorter than
  the startup window, so nothing in 512 tests could see it, and the smoke config's eight
  trials cannot reach it by construction. The policy now lives in
  `src/mars_rover_q/numerics.py`, raises on `divide`, `over` and `invalid`, and ignores
  `under` as numpy itself does. `tests/unit/test_numerics.py` pins each of the four
  conditions, and `test_the_sampler_survives_the_numeric_guard_past_its_startup_trials`
  drives 60 real suggestions over the real search space under the guard — no training, so
  it costs 0.2s and still fails on the old policy at the exact line from the traceback.
  The general lesson, which is the same one the `tuning_stub_status` probe taught: a
  check that cannot fire in any test the suite actually runs is not yet a check.



---

## Discovery log

Not assignment records. This section credits problems and designs that originated with
Connor rather than with an assignment brief, at the time they were raised.

### 2026-09-08 — the exploration failure and the start-state curriculum

**Found by Connor, from his own inspection of early results.** After the Q-learning core
went green he ran training and looked at the learned policy rather than only at the
success rate, and noticed that over most of the map the greedy action oscillated at
random between draws. He read that correctly as the signature of states whose action
values are all still identical — `select_action` breaking a five-way tie by coin toss —
and concluded that those states had never been visited: exploration under a decaying
epsilon was not discovering the 160-point biosignature at all, because reaching it *and*
returning before the battery dies is an exponentially unlikely accident on a random
walk.

**The fix was his idea too:** anneal a curriculum over start states that the rover could
physically have driven itself into, while continuing to evaluate from the canonical
lander start. Both halves matter — the reachability constraint keeps the agent from
learning values for a mission that cannot happen, and the fixed evaluation start keeps
the numbers comparable with every run trained without a curriculum.

Claude built the surrounding machinery (the `reset(start_state=...)` seam,
`StartStateCurriculum`, the difficulty metric, the `from_canonical_start` bookkeeping
that keeps curriculum episodes out of the learning curves, the `tied_state_fraction`
coverage diagnostic that now measures the symptom Connor spotted by eye, and the CLI,
manifest, and plot plumbing). The two functions that carry the actual reasoning --
`enumerate_start_states` and `sample_start_state` -- were handed back to Connor as the
open assignment; see `.teacher/current.md`.

### 2026-09-09 — the per-run coverage figures

**Requested by Connor, unprompted, and specified closely enough that the two figures
are his design and not a brief filled in.** He asked for visualisations written into a
`figs/` folder inside each run directory, and named both of them: how many states out
of the total hold a non-default Q approximation, and a heatmap of the map showing
where the agent gained experience — with the payload split spelled out (no carry plus
each of the three sample types) and battery explicitly marginalised away.

Both halves of that specification carry real judgement. Splitting by payload is what
turns a coverage number into a diagnosis: the same run that reads 20% covered overall
reads 13.3% with the biosignature in the bay, and the pre-curriculum run it replaced
had exactly **one** learned state in that slice. Marginalising battery is what makes
the figure a map at all — 61 battery levels per cell is not something a reader can
hold, and summing them away is what puts the experience back on the grid the mission
actually happens on.

It is also the direct successor to the exploration failure Connor found on 2026-09-08,
and it measures the same thing he originally spotted by eye. `tied_state_fraction` had
reduced that symptom to a single scalar in the manifest; this asks the question the
scalar cannot answer — not *how much* of the table is untouched, but *which parts*,
and where on the map the experience that filled the rest came from.

Claude built it: the per-state visit counter in the training loop and its manifest and
`.npy` serialisation, `learned_state_mask`, `StateEncoder.grid_view`, the
`run_figures.py` module, the `--no-figs` switch and the `figures --run` subcommand,
and the tests and docs. The secondary decisions were Claude's too and are recorded in
`docs/implementation-decisions.md` — recording visits during training rather than
inferring them from the table, the shared log colour scale across the four panels,
reporting both coverage denominators, and the fallback for runs saved before visit
counts existed. **No human-owned function was involved and this is not an assignment
record; it is credited here for the idea and the specification.**

### 2026-09-09 — the oversampled easy end, and two schedules that fix it

**Found by Connor, from the artefacts of his own previous assignment.** The annealed
curriculum he had just finished was working — the biosignature was being discovered —
but he kept reading its output rather than accepting it, and noticed the schedule was
spending its episodes badly. Two independent artefacts said so. The Q-value heatmaps he
had specified the day before showed the near-lander states saturating while the newly
admitted hard end stayed thin. And `cli curriculum` said it outright: at
`progress=0.75` the sampled difficulty ran `min 1.0 / mean 5.4 / max 10.0`. The window
had widened to ten, and the easiest state in the pool was still being drawn as often as
anything else.

**His diagnosis, in his words:** the window grows, but the draw inside it stays uniform,
so the easy states are oversampled. That is exactly right, and it is the same defect
recorded as non-blocking polish in his own review notes above — with the difference that
he found it in the *data* rather than being told, and arrived with two fixes.

**Both fixes are his:**

1. **A sliding window instead of a growing one.** Hold the band width constant and move
   both edges, so solved states are retired rather than accumulated.
2. **Weighted sampling against experience.** Keep the window, but downweight start
   states that have already had many Q updates.

The second is the more interesting idea, and the more subtle one: it makes the schedule
*closed-loop*. Every curriculum in the repository until now was a function of the
episode counter alone; this one reads the table's own experience back out and lets the
learning decide where the next episode starts. It also reuses a quantity that already
existed for a different purpose — the per-state visit counter behind the coverage
figures, itself a thing he asked for.

The framing that came out of the design conversation is worth recording because it is
what made both fixes describable at once: a start-state curriculum has two independent
knobs — which ranks are **admissible** at a given progress, and how the draw is
**spread** over them — and the original schedule collapsed them into one. Sliding
changes the first; visit-weighting changes the second. Connor chose to keep the
visit-weighted arm on the growing window's admission rule precisely so that the
comparison isolates one variable.

Claude built the surrounding machinery: the `CurriculumStrategy` enum and the window
helpers, `sample_sliding_window_start_state`, the `StartStateCurriculum` dispatch, the
`pool_state_indices` / `pool_update_counts` seam that feeds `visit_counts` back into the
sampler once per episode, the **CURRICULUM STRATEGY STUBBED** banner, the `curriculum`
command's dry-run diagnostic and the band-share statistic that makes oversampling
visible without training anything, the `--curriculum-strategy` flags, the strategy factor
in the experiment grid with its single shared control arm, arm-aware labelling across the
sweep analysis and the figures, Study 3's pre-registration, and the tests. The secondary
decisions are recorded in `docs/implementation-decisions.md`.

`sample_visit_weighted_start_state` — the closed-loop draw itself — was handed back to
Connor as the open assignment; see `.teacher/current.md`. **This is not an assignment
record; it is credited here for the observation and the design.**

### 2026-09-10 — battery as a clock, and the affordability-binned state

**Found by Connor, from one frame of a figure set he had asked for himself.** He opened
`artifacts/runs/safe_corridor__sparse__seed1/figs/q_by_battery/battery_50.png` and saw
that the greedy action at the lander, empty-handed, was `COLLECT` — an action that
collects nothing there and burns a joule doing it. He did not report it as a rendering
bug. He read it as evidence about the *representation*:

> Can't [battery] just be abstracted away as the horizon? Since the starting state is
> always going to be the lander, the optimal way to act from t=0 isn't going to change
> given that the battery life is long enough to go get the highest reward. Currently
> this is making the state space like WAY bigger than it needs to be and for a tabular
> method I think we are learning the wrong things.

**He was right, and the diagnosis was worse than he pitched it.** Reading the learned
table directly confirmed the symptom he inferred and then some. At battery 55, empty
bay, the learned value is `79.775` at `(5,1)`, `(5,2)`, `(5,3)`, `(5,4)` and `(6,4)` —
identical to five decimals along the whole corridor — and across the battery axis it
rises by exactly `1/gamma` per unit of charge. The table had not learned a value
function over the map at all; it had learned one scalar, *"I will deliver the 90-point
sample, discounted by how much battery I have burned"*, and smeared it across 61 battery
layers with no positional structure left in it. At the canonical start all five actions
were equal to ten decimal places after 105,125 visits, so the first action of the
mission was a coin toss. The `COLLECT` arrow he flagged was the tip of that.

**Where his argument holds exactly.** Solving each scenario by value iteration on the
full battery-aware MDP, then finding the best policy available to a battery-free state
by policy iteration inside that restricted class, gives `117.566966` against a true
`V*` of `117.566967` on `safe_corridor`, and the same story on the other two maps.
Dropping the battery axis costs **nothing** from the canonical start. Along the ribbon
of states reachable from the lander, battery is a deterministic function of the path
already taken — a clock duplicating work `gamma` is already doing.

**Where it does not, which is the more interesting half.** Over *every* admissible start
state — which is what the curriculum trains on — a battery-free policy loses badly:
mean shortfall against `V*` of 5.84 / 7.66 / 1.01 on the three maps, with the worst
cases stranded mid-map on low charge, where the right call is to abort for a nearer
sample and a battery-blind policy commits to the long route and dies. The two objectives
are in tension, and it shows up as a single number: optimise a battery-free policy for
the lander start and it is worth `117.567` there; optimise the same policy class
uniformly over all 10,140 curriculum starts and it is worth `79.775` at the lander —
almost exactly what the trained table had learned. The curriculum was asking one table
to hedge for marooned rovers and paying for the hedge with the mission.

**So the answer was not to drop battery but to coarsen it**, at the charge levels where
the decision actually changes: each sample's lander round trip plus the `COLLECT`
charge, below which that sample stops being deliverable. Three samples, three distinct
costs, four bins on all three bundled maps. That recovers the canonical-start optimum
(`117.567` / `129.323` / `128.805` against `V*` of `117.567` / `129.337` / `128.805`),
cuts the all-starts shortfall to 1.90 / 0.83 / 0.17, and takes the tables from
24,400 / 40,896 / 104,256 rows to 1,600 / 2,304 / 2,304.

**These are planning bounds, not learning results.** They say what a policy of each
*shape* can achieve, computed from exact solutions; they are not a measurement of what
tabular Q-learning finds under either encoding, and no such claim is made anywhere. The
`--dense-battery` flag exists precisely so that comparison can be run as an experiment
rather than argued from a bound.

One nuance worth recording against the frame that started this. In the true `Q*` at the
canonical start, `COLLECT` ranks **second of five** — above both `NORTH` and `SOUTH` —
because a wasted `COLLECT` costs one joule and a step in the wrong direction costs more.
It is the cheapest available mistake, so it will always sit near the top of that row and
any noise floats it to first. Connor's instinct that the arrow was wrong was right;
the reason it appears is not that the table scored `COLLECT` absurdly but that it had
stopped distinguishing the actions at all.

Claude built it: `BatteryBinning` and the encoder's binned axis, `Scenario.mission_costs`
and `Scenario.battery_binning`, the `--dense-battery` flag and its threading through
`TrainConfig`, the environment, the curriculum, evaluation and the experiment grid, the
`config.battery_encoding` manifest field and the dense fallback that keeps every run
saved before the flag readable, the figure relabelling (frames named by level, the range
in the title and in a gauge that fills solid to the bin's floor), and the tests. The
secondary decisions are in `docs/implementation-decisions.md`, where this also
supersedes a standing entry that said battery was stored exactly and needed no binning —
that entry justified itself on the table's size in bytes, which was the wrong measure.

**No human-owned function was involved and this is not an assignment record; it is
credited here for the observation and the argument.**

### 2026-09-10 — bounded hyper-parameter search, and the two constraints that make it a question

**Connor's question, unprompted, and specified closely enough that the design is his.**
Every study in the repository so far has held the hyper-parameters fixed at values nobody
tuned — learning rate `0.2`, `gamma 0.99`, epsilon `1.0 → 0.05` over 60% of the budget —
because Studies 1 to 3 were each asking about something else, and a factor you are not
studying has to be held constant. He asked the question none of them can: **which set of
hyper-parameters achieves the best reward in the fewest episodes?**

In his words: *"let's now do some great science by using optuna to find the set of
hyperparameters that achieves the best reward in the least number of episodes. Let's start
with the safe corridor env and keep an episode limit because some methods won't be able to
achieve max performance. Let's also limit the entire optimization to 2M episodes or
something so that the search is bounded."*

**Both constraints are his, and each one is load-bearing.** Neither is an implementation
detail he left to be filled in; together they are what turn a vague "tune it" into
something measurable and affordable.

1. **A per-trial episode limit, because some settings cannot reach max performance.**
   That clause is the whole reason the search has a well-posed objective. Remove the cap
   and every setting that eventually converges ties at the top, the differences between
   them are seed noise, and the search quietly becomes a search for *final performance* —
   the thing Study 1 already measures. With the cap below convergence, settings separate
   on **when** they arrive, which is the quantity he actually asked about. He also got the
   reason right in advance: the cap is not a handicap to be apologised for, it is what
   makes "fastest" a comparison rather than a tie.
2. **A bounded total budget, so the search is bounded.** 2M episodes, charged against one
   ledger, with the study stopping when what remains cannot fund another full trial. This
   is the difference between an experiment and an open-ended compute spend: the cost of
   the answer is known before it is asked, and two searches are comparable in total
   compute rather than in wall clock. It also made the pruning design fall out — a trial
   stopped early returns its unspent episodes to the ledger, so the bound buys more trials
   rather than merely capping them.

**Starting on `safe_corridor` was his call too**, and it is the right one for the same
reason it is Study 1's control: flat corridors, short routes, every sample comfortably
affordable, so learning speed is isolated from risk. A search run first on
`risk_value_tradeoff` would be measuring two things at once.

One consequence worth recording, because it is a constraint the framing creates rather
than one it inherits: "best reward in the fewest episodes" is **two objectives**, and they
trade off. The repository now resolves that in exactly one function and reports the
trade-off separately, rather than hiding a weighting inside a scalar and presenting the
verdict as if no choice had been made.

Claude built the surrounding machinery: the `tuning.py` module with `TuningConfig`, the
`EpisodeLedger` and its all-or-nothing grants, the conditional search space
(`suggest_trial_params` and the `ParameterSuggester` protocol it is typed against),
`best_affordable_return` as the planning bound the objective normalises by,
`average_curves` and `curve_peak`, the Optuna study with its seeded TPE sampler and median
pruner, the new `checkpoint_callback` seam in `train` together with
`TrainResult.episodes_completed` / `stopped_early` that lets a hopeless trial be abandoned
without changing what any run learns, `_confirm_best` re-training the winner into a full
run directory on the ledger's books, the `study.json` / `trials.csv` artefacts and their
two distinct "meaningless" flags, the three figures, the `tune` CLI command with its
config-plus-flag resolution, the **HYPERPARAMETER SEARCH STUBBED** banner, the
pre-registration as Study 4 in `docs/experiment-plan.md`, and the tests. The secondary
decisions are recorded in `docs/implementation-decisions.md`.

`score_learning_curve` — the scalarisation itself — and `pareto_front` — the trade-off it
resolved — were handed back to Connor as the open assignment; see `.teacher/current.md`.
**This is not an assignment record; it is credited here for the question, the two
constraints, and the choice of map.**

### 2026-09-14 — the sweep across all three maps, and a plateau that was a hyper-parameter failure after all

**Connor called for the suite, and the result overturned a conclusion Claude had
recorded two days earlier.** In his words: *"run a tuning sweep to find me the best
hypers for the 3 environments. Ensure that the candidate results are saved somewhere
and plot the evaluated candidates and pareto front of the search for use in my
report."* Every study before this one searched `safe_corridor` alone.

**Three pushbacks of his are what made the sweep find anything.** Each one redirected
work that was going the wrong way, and the last of them reversed a claim in writing.

1. **He refused the plateau as a brute fact.** Claude had reported ~100 trials stopping
   at 90 on `safe_corridor` and said plainly it had no hypothesis why. His answer —
   *"this is what we need to dig into. It's why I'm leveraging optuna. Maybe we are
   thinking of maximizing the wrong thing"* — treated a missing explanation as the
   thing to chase rather than a caveat to publish, which is what eventually located it.
2. **He caught a false claim by checking it against the maths.** Claude had asserted
   that the affordability binning *provably* caps `safe_corridor` at 90, on the strength
   of two single-seed runs. He pushed: *"Didn't you solve the MDP on the same state
   space? Don't you know that a true V leads to an optimal policy even under this
   encoding?"* He was right and the assertion was wrong. Three exact computations
   settled it: the best bin-measurable policy is worth `160.000`; greedy with respect to
   the bin-average of `Q*` is worth `159.984`; and the greedy policy of the aggregated
   Bellman fixed point is worth `160.000`. The binning caps nothing. Three claims were
   retracted, and the empirical effect that remained was labelled a hypothesis rather
   than a proof.
3. **He cut scope that had no consumer.** Claude proposed committing an exact-planner
   module to `src/`. *"What is the planner module for? I don't need V/V\* as a downstream
   reported metric. It was just for debugging."* Correct — both justifications for it had
   already evaporated. It stayed a scratchpad tool.

**What the sweep found.** Four studies, 26.2M training episodes, sparse reward
throughout, `battery_encoding` searched alongside the six agent knobs. Every value below
is a real run; the exact ones come from backward induction on the true MDP, which is
available because every traversable tile costs at least one energy, so the state graph is
a DAG ordered by battery.

| map | trials | best return (500 greedy ep) | success | exact `V^pi` | `V*` | ratio |
|---|---|---|---|---|---|---|
| `safe_corridor` (400k cap) | 20 (1 pruned) | 160.000 | 1.000 | 159.998 | 160.000 | 1.000 |
| `risk_value_tradeoff` | 32 (10 pruned) | 158.960 | 0.996 | 159.271 | 159.769 | 0.997 |
| `shaping_trap` | 25 (3 pruned) | 160.000 | 1.000 | 159.990 | 160.000 | 1.000 |
| `safe_corridor` (150k cap) | 25 (1 pruned) | 90.000 | 1.000 | 90.000 | 160.000 | 0.563 |

**`safe_corridor` reaches the optimum, and the earlier conclusion was wrong.** On
2026-09-12 Claude ran a three-seed test at a 400k cap and concluded that *"the study's
six tuned hyper-parameters are not what's binding"*, because `growing`, `sliding` and
`visit_weighted` returned exactly `90.000` on all nine runs while exploring starts
reached `158.6`. That test held everything else at `gamma 1.0`, learning rate `0.2`,
epsilon `1.0 -> 0.1` over 80%, `curriculum_fraction 0.9`. **They were binding.** The
winning trial is the *sliding window* — the arm that had looked dead — at
`gamma 0.9821`, learning rate `0.6114`, `initial_q 96.81`, epsilon
`0.627 -> 0.00106` over 64%, dense battery axis, `curriculum_fraction 0.5175` and a
window of `0.0217` of the pool, and it is worth `159.998` against `V* = 160.000`. On
seeds 2, 3 and 4, none of which the search saw: `159.998`, `159.993`, `159.998` — 3 of
3 at the optimum. So `EXPLORING` was a real fix for a real problem, and it was not the
only one; the narrow sliding window with a high learning rate solves the same map
slightly better.

**A discount factor low enough changes the problem rather than the difficulty.** The
undiscounted return of the `gamma`-optimal policy, by exact solution:

```
scenario                   0.9     0.95     0.98     0.99    0.995      1.0
safe_corridor            90.00    90.00   160.00   160.00   160.00   160.00
risk_value_tradeoff      40.00   159.76   159.76   159.77   159.77   159.77
shaping_trap            158.89   160.00   160.00   160.00   160.00   160.00
```

At `gamma 0.9` the optimal policy on `safe_corridor` **is** the 90-point sample, and on
`risk_value_tradeoff` it **is** the 40-point one. The search space runs from `0.9`, so
part of it optimises a different mission correctly rather than this one badly.

**That interacts with the episode cap, and it is what the 150k `safe_corridor` study
found.** Sixteen of its twenty-five trials landed in `gamma` between `0.90` and `0.97`.
The sampler was not failing to reach 160; it had found that choosing a discount which
makes the 90-sample genuinely optimal, then converging to it in 12,500 episodes, scores
`0.5391` — better than any slow climb toward a 160 that never appears inside the cap.
Its confirmation run is a *perfectly* solved MDP: 500 of 500 greedy deliveries in 13
steps, of the wrong sample. Raising the cap to 400k dissolved it; the winner moved to
`0.7142` and the plateau stopped being the best available answer. The objective is
behaving exactly as specified — the lesson is about the pair, not either half.

**Two settings that agree, and one that is the same draw twice.** The winners on
`risk_value_tradeoff` and `shaping_trap` both use the affordability binning, **no
curriculum**, and optimistic initialisation (`initial_q` of `97.3` and `188.9` against a
160-point return scale) paired with a low, fast-decaying epsilon — exploration coming
from the initial values rather than from the epsilon schedule. Only the
`risk_value_tradeoff` result shows the sampler converging on it, though: trials 19 to 24
cluster inside `gamma` `[0.982, 0.987]`, learning rate `[0.047, 0.103]`, `initial_q`
`[91, 97]`. `shaping_trap`'s winner is trial 2, a startup draw from the prior that
nothing later beat. Because every study shares `sampler_seed 0`, that trial 2 is the
*identical* parameter set as `safe_corridor`'s trial 2 — and the same setting scores
`0.9583` and solves one map exactly while scoring `0.5391` and trapping on the other.
That the no-curriculum arm wins outright on two of three maps is a real result and sits
against most of what this repository has spent its effort on.

Claude ran the studies and wrote the surrounding code: the `search_battery_encoding`
flag and the encoding's entry into `suggest_trial_params` (off by default, with the
confirmation run evaluating a winner through the encoding it trained under), the four
search configs, the cross-map figure, and the exact-evaluation and seed-check scripts,
which stayed in the scratchpad. One sizing error of its own is worth recording: all
three original budgets were exact multiples of the per-trial cost, so the ledger could
not fund the built-in confirmation run and all three skipped it; they were re-run
through the same code path afterwards and `study.json` records that they ran separately.
The secondary decision is in `docs/implementation-decisions.md`.

**No human-owned function was involved and this is not an assignment record; it is
credited here for the call to run the suite across all three maps, and for the three
pushbacks — one of which corrected a false claim Claude had already written down.**
