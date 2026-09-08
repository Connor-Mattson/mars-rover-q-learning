# How to test

All commands run from the repository root with the virtual environment active.

```bash
source .venv/bin/activate     # or: uv run <command>
```

---

## The two commands that matter

```bash
# 1. The quality gate. Must be green, before and after your work.
pytest

# 2. Your assignment. EXPECTED TO FAIL until agent.py is implemented.
pytest -m human_todo tests/human_todo
```

`pytest` deselects the `human_todo` marker by default (`addopts` in `pyproject.toml`),
so the two suites never mix. Nothing in the first command should ever fail; everything
in the second should, until you are done.

## Focused loops while you work

One function at a time, in the order the assignment lists them:

```bash
pytest -m human_todo tests/human_todo -k q_table      # 1. initialize_q_table
pytest -m human_todo tests/human_todo -k target       # 2. calculate_target
pytest -m human_todo tests/human_todo -k td_error     # 3. calculate_td_error
pytest -m human_todo tests/human_todo -k update       # 4. update_q_value
pytest -m human_todo tests/human_todo -k "epsilon or tie or mask or selection"  # 5.
```

Useful flags: `-x` stops at the first failure, `-q` trims the output, `--tb=line` prints
one line per failure, and `-k <substring>` narrows to matching test names.

## Expected state **before** you implement anything

```
$ pytest
178 passed, 34 deselected

$ pytest -m human_todo tests/human_todo
25 failed, 9 passed
```

The nine that pass are the parts the placeholders happen to satisfy — correct table
shape, correct dtype, a zero initial value, and the two boundary cases of
`calculate_target` where "reward only" is genuinely the right answer. The twenty-five
failures are all plain assertion failures or `DID NOT RAISE`, never import errors or
crashes. If you see an error rather than a failure, something is wrong with the
environment, not with your code — say so rather than working around it.

## Expected state **after** you implement everything

```
$ pytest
178 passed, 34 deselected

$ pytest -m human_todo tests/human_todo
34 passed
```

Plus the broad gates:

```bash
ruff format --check .
ruff check .
mypy
```

## The end-to-end check

The tests prove the contracts. This proves the agent actually learns:

```bash
python -m mars_rover_q.cli train \
    --scenario safe_corridor --reward sparse --seed 1 \
    --episodes 4000 --eval-episodes 200 \
    --output artifacts/runs/handcheck
```

What to look for:

1. **No `TEACHING STATE` banner.** Its absence is the automatic signal that all five
   functions now satisfy their contracts.
2. The progress lines show `success=` climbing away from `0.00` as `eps` decays.
3. `artifacts/runs/handcheck/manifest.json` contains `"learning_is_meaningful": true`.
4. The printed evaluation summary shows a nonzero `success_rate`.

Then watch it:

```bash
python -m mars_rover_q.cli replay \
    --run artifacts/runs/handcheck --episode best --policy-overlay
```

`TAB` pauses, `.` single-steps, `[` / `]` change speed, `P` toggles the policy overlay,
`Q` quits. A rover that drives to a sample, collects, and returns is worth more
confidence than any single number.

## Rules

- **Do not edit anything under `tests/human_todo/`.** Not to skip, not to `xfail`, not
  to relax a tolerance. Changing a test to match an implementation is not a pass, and
  the review checks the diff.
- **Do not touch `training.py` or `evaluation.py`.** They already call your functions
  correctly. If the loop looks wrong to you, say so at review — do not silently patch it.
- If a test in the green suite starts failing, you have changed something outside the
  assignment. `git diff` will show what.

When all of the above holds, return to Claude and say: **check my work**
