# Hints

No assignment is open, so there is nothing here to be stuck on. What follows is the
accumulated set of lessons that earned a revision round at least once — kept because the
next assignment will be written against them.

## Carried forward

- **Read a `Raises:` clause as a specification, not as a summary of what the tests
  sample.** Three revision rounds have come from the other reading: the empty
  `action_mask` guard, the zero-budget guard, and the negative-count guard.
- **Construct an input that misses the exploding line.** If a guard's test passes because
  something three frames down raised, the guard is not there. This was finally *not* the
  failure mode in the search objective, where all five guards were written from their own
  sentence and `episode_cap` was checked before anything divided by it.
- **Clamp both ends of a normalised axis, not just the one the anchors mention.** The
  search objective handled the ceiling and not the floor, which satisfied both anchors
  and still broke dominance: a negative region large enough to cancel a real recovery
  sent a curve that succeeded down to the same `0.0` as one that never did. The general
  shape — a transformation that is correct on the range the tests visit and lossy on the
  range the data visits.
- **A probe that cannot fail is worse than no probe.** Twice now a teaching stub detector
  has reported the right answer by the wrong route — `curriculum_stub_status` replaying
  one RNG stream, and `tuning_stub_status` probing only curves above zero, so it cleared
  over an objective that was uniformly `0.0`. When a check gates an artefact's
  "these numbers are real" flag, test that it can clear *and* that it can refuse.
- **Two failing tests that look contradictory are usually one bug.** The tie-ordering
  pair in the Pareto front looked like a conflict and was a single inverted tiebreak,
  invisible on the input that happened to arrive already sorted the wrong way.
- **`np.max` / `np.min` over an ndarray, not the Python builtins** — the builtins iterate
  element by element. House idiom, and occasionally a 300x difference on a per-episode
  path.
- **Error messages are one line, in the `f"... got {value}"` shape** used in `agent.py`
  and `metrics.py`.
- **Delete debugging scaffolding before submitting.** Commented-out `print` calls are not
  inert when what they would consume is a lazy iterator.
