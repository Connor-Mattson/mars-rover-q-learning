# Hints

Escalating, one stage at a time. Read stage 1, go back to the code, and only come back
here if you are still stuck. None of these stages contain a copyable implementation —
that is the point.

---

## Stage 1 — orientation

- Read `src/mars_rover_q/training.py` first. The loop is short and it shows exactly how
  your five functions are wired together. Everything you need to know about the shapes
  and the call order is there.
- `q_table` is 2-D: rows are states, columns are actions. `q_table[state_index]` is a
  1-D array of five values — one per action — for that state.
- The docstrings are the contract. When a test disagrees with your reading of a
  docstring, the docstring is probably clearer than you remember; re-read it.
- Start with `calculate_td_error`. It is one line and it forces you to decide the sign
  convention that the other functions inherit.

## Stage 2 — the shape of each function

- **`initialize_q_table`** — NumPy has a constructor that fills an array with a given
  scalar. You do not need to allocate and then assign. Validate the dimensions *before*
  allocating, and raise rather than clamp.
- **`calculate_target`** — there are two cases, not three. Ask "did the episode end?"
  once, and branch on that single boolean.
- **`calculate_td_error`** — "how far off am I, and in which direction?" Positive means
  *the entry is too low*.
- **`update_q_value`** — index one cell and add something to it. If you find yourself
  writing a slice or touching a whole row, step back.
- **`select_action`** — two branches. One draws uniformly from the legal actions; the
  other finds the best value and picks among the actions that achieve it.

## Stage 3 — the traps this repo tests for

1. **Bootstrapping past the end.** If a terminal transition's target includes
   `gamma * something`, a battery-depleted state inherits value from a successor that
   does not exist. Both `terminated` and `truncated` are boundaries here.
2. **`max` over values vs. `argmax` over actions.** `calculate_target` needs the best
   *value*. `select_action` needs the best *action*. They are different NumPy calls, and
   swapping them produces code that runs and learns nothing.
3. **First-index tie-breaking.** `np.argmax` returns the *first* maximum. On a
   zero-initialised table every action ties, so a plain `argmax` policy is "always go
   north". You need to find all the indices that achieve the maximum, then choose among
   them with `rng`.
4. **Sign flip.** If your TD error is `estimate - target`, then adding `lr * error`
   moves the estimate *away* from the target. Check with a concrete pair, e.g. estimate
   `1.0`, target `4.0`: after an update with `lr = 1.0`, the entry must be `4.0`.
5. **Module-level randomness.** `np.random.rand()` ignores your seed entirely. Use the
   `rng` argument for every draw, including the coin flip that decides explore-vs-exploit.

## Stage 4 — pinning down the pieces

- **Tie-breaking**, concretely: you have a 1-D array of values. Compute the maximum.
  Find every index whose value equals that maximum — `np.flatnonzero` over a boolean
  comparison gives you exactly that array. Then let `rng` choose one of those indices.
  `rng.integers(len(candidates))` and `rng.choice(candidates)` both work.
- **Masking**: the cleanest way to keep a masked action from ever being greedy is to
  compare values only among the legal indices, then map back to the original index. An
  alternative is to copy the row and set illegal entries to `-inf` before taking the
  maximum — just do not mutate `q_table` itself.
- **The explore coin flip**: `rng.random()` returns a float in `[0, 1)`. Compare it to
  `epsilon`. Check the two extremes by hand: with `epsilon = 0`, can the explore branch
  ever run? With `epsilon = 1`?
- **The update, as a sentence:** the new estimate is the old estimate plus the learning
  rate times the error. Written that way, the `lr = 1.0` property is obvious.

## Stage 5 — if the tests still fail

Read the failure, not the function. Each test name states the property it wants:

- `test_target_uses_the_best_successor_action_not_the_first` → you are indexing rather
  than maximising.
- `test_update_touches_exactly_one_entry` → you assigned to a row or a slice.
- `test_ties_are_broken_randomly_not_by_lowest_index` → plain `argmax`.
- `test_action_mask_restricts_exploration` → your explore branch ignores the mask.
- `test_intermediate_epsilon_is_reproducible_under_a_fixed_seed` → a draw is coming from
  somewhere other than `rng`, or you draw a different number of times per call depending
  on the branch.

If a *non*-`human_todo` test breaks, you have changed something outside your scope. Run
`git diff` and check.

## Stage 6 — where to read more

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., §6.5 (Q-learning)
  and §2.6 (optimistic initial values).
- Ng, Harada & Russell (1999), *Policy invariance under reward transformations* — the
  result behind `rewards.PotentialBasedShaping`. Worth reading once you have the agent
  running, because it explains why one of the three conditions is safe by construction.
