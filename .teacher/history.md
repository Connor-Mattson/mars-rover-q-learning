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
