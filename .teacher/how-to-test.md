# How to Test

No assignment is open, so **every suite in the repository is green**. Numbers measured
2026-09-12 on this machine.

## The gates

```bash
ruff format --check . && ruff check . && mypy && pytest
```

Expected: `60 files already formatted`, `All checks passed!`,
`Success: no issues found in 52 source files`, `516 passed, 166 deselected` in ~52s.

## The teaching suite

```bash
pytest -m human_todo tests/human_todo
```

`166 passed`. It is deselected from the default run by `addopts`, so this is the only
command that exercises it — and with no assignment open it must be **fully green**. A
failure here is now a regression in one of the five completed assignments, not an open
exercise.

## The inner loop

```bash
pytest -m "not slow and not human_todo"
```

`514 passed in 19s`. **Both clauses are needed.** A command-line `-m` replaces `addopts`
rather than combining with it, so a bare `-m "not slow"` silently pulls the teaching
suite back in.

Only two tests carry `slow`, and they are ~33s of the ~52s: the `--dense-battery` arms
that render a matplotlib frame per battery reading.

## The search, end to end

```bash
python -m mars_rover_q.cli tune --config configs/tuning/smoke.json --output /tmp/study
```

Eight trials of 400 episodes, three of them pruned, then a confirmation run — seconds.
What a healthy run looks like now that the objective is real:

- distinct scores, not a column of `0.0000`;
- the top-5 ordered so that two trials finishing at the same return are separated by
  *when* they got there;
- a front of a few trials under `pareto front`, not `unavailable`;
- no **HYPERPARAMETER SEARCH STUBBED** banner and no **TEACHING STATE** line;
- a confirmation run of the winner that actually succeeds.

```bash
python -c "import json; d=json.load(open('/tmp/study/study.json')); \
print(d['search_is_meaningful'], d['budget'], d['pareto_front'])"
```

`search_is_meaningful` must be `true` and `pareto_front` must be non-empty. Note that
`pareto_front` in the JSON holds trial *numbers*, not outcome objects.

Scores cluster low on this config and some trials legitimately sit at `0.0`: 400
episodes is far below what `safe_corridor` needs. That is the cap, not a defect.

## The one failure no short study can reach

Optuna's TPE sampler draws from the prior for its first `startup_trials` (10) and only
then builds a model — and the model's arithmetic is where the sampler first touches
`exp`. So any defect in that path is invisible to every study shorter than 10 trials,
which is every study in the test suite and the 8-trial smoke config as well. One lived
there: the CLI's `np.seterr(all="raise")` turned the log-sum-exp identity's deliberate
underflow into `FloatingPointError`, and the first real study died at trial 10.

`tests/unit/test_tuning.py::test_the_sampler_survives_the_numeric_guard_past_its_startup_trials`
now drives 60 real suggestions over the real search space under the real guard. It needs
no training, so it costs 0.2s — if you add anything to the sampler's configuration, that
is the cheap way to exercise it past the startup window.

## What a full study costs

`configs/tuning/safe_corridor_search.json` is the real Study 4. Connor raised it on
2026-09-12 to 5M episodes and 40000 per trial (20000 x two seeds) — 125 trials if nothing
is pruned and more in practice. Measured throughput is ~2900 episodes/second including
checkpoint evaluation, so the budget sets the clock: 5M is ~30 minutes. Do not reach for it
while iterating on anything else. A pilot at 30k episodes funded 42 trials against 25
nominal, one at 560k funded 16 against 14, and pruning returns roughly 60% of each cut
trial's budget.

What that run can and cannot show is in the README under *What the pilots say about this
map*: on `safe_corridor` every setting that learns reaches a 90-return policy by episode
200 and holds it, so the speed axis is nearly constant and a large tied front is the
expected result. Raising the cap from 3000 to 20000 changed nothing about that, and no
pilot has yet delivered the 160 sample.

## Checking the objective itself

Its properties are cheap to probe directly, which is worth doing after any change to it
— the contract tests pin invariants, and a randomised sweep finds violations they miss:

```bash
pytest -m human_todo tests/human_todo/test_tuning_objective.py -q   # 51 passed, <1s
```

Both rescaling invariances should hold to `1e-12` across random curves, and no
pointwise-better curve should ever score lower. The dominance clause holds **below**
zero return as well as above it; that is where the first implementation failed, and
`test_dominance_survives_the_region_below_zero` is the test that now pins it.

## Where the time goes

```
18.30s  test_run_figures.py::test_battery_set_holds_one_frame_...[dense]   [slow]
14.12s  test_cli.py::test_dense_battery_writes_a_frame_per_reading         [slow]
 1.47s  test_run_figures.py::test_battery_set_holds_one_frame_...[affordability]
 1.13s  test_cli.py::test_train_writes_a_battery_frame_per_battery_level
 1.07s  test_tuning_study.py::test_a_pruned_trial_costs_less_than_a_full_one
 0.64s  test_experiment.py::test_sweep_results_do_not_depend_on_the_worker_count
```
