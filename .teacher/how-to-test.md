# How to Test

No assignment is open. These are the standing gates.

## Everything

```bash
ruff format --check . && ruff check . && mypy && pytest
```

Expected: `56 files already formatted`, `All checks passed!`,
`Success: no issues found in 46 source files`, `370 passed, 115 deselected`.

## The teaching suite

Deselected from the default run by `addopts`, so it must be asked for by name:

```bash
pytest -m human_todo tests/human_todo
```

Expected: `115 passed`. It should stay green from here — a failure now is a
regression in a finished function, not an open assignment.

## The fast feedback loop

The dry run reports the start distribution of all three schedules without training
anything, in about a second:

```bash
python -m mars_rover_q.cli curriculum --scenario risk_value_tradeoff
```

Measured on 2026-09-10, over the anneal:

```
growing         distinct starts=1767/14298  easiest quarter=0.60 of episodes
sliding         distinct starts=1836/14298  easiest quarter=0.17
visit_weighted  distinct starts=1895/14298  easiest quarter=0.57
```

The dry run's synthetic counter increments only the sampled start, so it understates
the tilt a real run gets from `visit_counts`; treat these as a lower bound and let
Study 3 measure the rest.

## The inner loop

The full gate is 56s. For the edit-run-edit loop, deselect the two tests that render
a matplotlib frame per battery level:

```bash
pytest -m "not slow and not human_todo"
```

`368 passed in 11.6s`. **Both halves of that expression are needed.** A `-m` on the
command line replaces `addopts`'s `-m 'not human_todo'` rather than combining with it,
so a bare `-m "not slow"` silently pulls the teaching suite back in — it reports
`483 passed` instead of `368`, and the extra 115 are tests the default run deliberately
excludes.

The two `slow` tests are real coverage, not deadweight: they are the only assertions
that a battery frame is written per battery level. Run the unfiltered `pytest` before
calling anything done.

## Where the time goes

Measured 2026-09-10, `pytest --durations`:

```
22.39s  test_cli.py::test_train_writes_a_battery_frame_per_battery_level        [slow]
21.50s  test_run_figures.py::test_battery_set_holds_one_frame_per_battery_...   [slow]
 1.14s  test_experiment.py::test_sweep_results_do_not_depend_on_the_worker_count
 0.93s  test_experiment.py::test_sweep_writes_a_cell_per_budget_and_curriculum
 ...
```

Before 2026-09-10 the suite was 2m25s: five `train` CLI tests that assert on manifests,
coverage output and exit codes were rendering all 61 battery frames apiece because they
passed no figure flag, and the interactive replay test ran at `fps=1.0`, which the
renderer clamps at `max(1.0, speed)` and so really did sleep a second a frame. The
`slow` marker existed but had been applied by intuition to seven `test_experiment.py`
tests that take under a second each, so `-m "not slow"` saved nothing. It is now
applied from the durations report.
