# Solver Results Old Schema

Reference for every column in `data/solver_results.parquet`, produced by `src/solve_all_symbolic.py`.

---

## Identity / ground truth

| Column | Type | Meaning |
|--------|------|---------|
| `id` | string | Puzzle ID from the original dataset |
| `prompt` | string | Full puzzle text as given to the model |
| `answer` | string | Gold answer from the training data |

---

## Solver output

| Column | Type | Meaning |
|--------|------|---------|
| `solver_answer` | string | Answer returned by the solver (`null` = no solution found) |
| `solver_correct` | bool | Whether `solver_answer == answer` |

---

## Solver classification

| Column | Type | Meaning |
|--------|------|---------|
| `solver_type` | string | Top-level solver branch: `arithmetic`, `concat`, or `null` (failed) |
| `solver_category` | string | Finer sub-type (see table below) |
| `conditioned_on_answer` | bool | Whether the gold answer was fed as a hint to constrain the search — training/analysis only, not available at inference |

### `solver_category` values

| Value | Meaning |
|-------|---------|
| `arithmetic` | Pure digit arithmetic solved in standard mode |
| `little_endian` | Digit order reversed before arithmetic |
| `pure_concat` | Operator literally concatenates left + right strings |
| `mixed_concat` | Some operators do concat, others do arithmetic |
| `mixed_concat_little_endian` | Mixed concat, solved in little-endian mode |
| `query_unseen_concat` | Concat type, but the query operator did not appear in the training equations (required the gold answer hint) |

---

## Arithmetic search details

These columns are `null` for concat-type puzzles.

| Column | Type | Meaning |
|--------|------|---------|
| `solver_mode` | string | Search mode that produced the solution: `standard` or `little_endian` |
| `solver_ops` | JSON string | Map of operator symbol → arithmetic function found, e.g. `{"*": "mul", "-": "absdiff"}` |
| `solver_mapping` | JSON string | Map of symbol → assigned digit (the bijection found), e.g. `{"@": 5, "&": 8}` |
| `solver_radix` | float | Number base used (almost always `10`) |
| `solver_numeric_answer` | float | Numeric value of the answer under the found mapping |
| `solver_tier` | float | Size of the operator candidate set at the tier that found the solution (see [Tier vs Level](#solver_level-vs-solver_tier)) |
| `solver_level` | string | Whether the solver had to escalate its op universe: `normal` or `deep` (see [Tier vs Level](#solver_level-vs-solver_tier)) |

---

## Puzzle complexity

Derived from the prompt text, independent of the solver.

| Column | Type | Meaning |
|--------|------|---------|
| `op_counts` | JSON string | How many times each operator appears across all training equations in the puzzle, e.g. `{"*": 3, "-": 1}` |
| `num_ops` | int | Number of distinct operators in the puzzle |
| `min_op_count` | int | Appearance count of the least-seen operator |

### Why `min_op_count` matters

Each training equation that uses an operator is one constraint on which digit-mapping is valid. An operator that appears only once gives very weak constraints — the solver has fewer examples to pin down what that operator does, making spurious solutions more likely. A `min_op_count` of 1 means at least one operator is severely under-constrained.

---

## `solver_level` vs `solver_tier`

These are two orthogonal dimensions of the arithmetic search. Together they describe *how hard* the solver had to work to find a solution.

### `solver_level` — the op universe

Controls **which arithmetic functions are in the pool** that the solver is allowed to assign to operators.

| Value | Op count | What is included |
|-------|----------|-----------------|
| `normal` | ~22 | Core ops (`add`, `sub`, `mul`, `gcd`, `lcm`, `absdiff`, `concat`, `fdiv`, `mod`, `min`, `max`, ...) plus ±1/±2 offset variants |
| `deep` | ~38+ | Everything in `normal` plus bitwise and polynomial ops |

The solver starts at `normal` and **auto-escalates to `deep`** if no valid solution is found. A `solver_level` of `deep` therefore means the puzzle could not be explained by simple arithmetic alone.

### `solver_tier` — the search order within the universe

Even within a given level, the solver does not try all ops at once. It works through progressively larger subsets (tiers) to avoid spurious matches from overly-generic ops:

| Tier | Approx. size | Ops added vs previous tier |
|------|-------------|---------------------------|
| TIER0 | ~12 | Core: `add`, `sub`, `rsub`, `mul`, `gcd`, `lcm`, `absdiff`, `neg_absdiff`, `sub_signed`, `rsub_signed`, `concat_fwd`, `concat_rev` |
| TIER1 | ~18 | + `fdiv`, `rdiv`, `mod`, `rmod`, `min`, `max` |
| TIER2 | ~28 | + all ±1 offset variants (`add_p1`, `mul_m1`, `absdiff_p1`, ...) |
| full | all | Everything in the current level |

`solver_tier` records the **size of the tier set that produced the solution**. A small value (e.g. 12) means a simple TIER0 op sufficed; a large value (e.g. 47) means the solver needed the full `deep` set.

**Example:** `solver_level=deep, solver_tier=28` — the solver failed at `normal` level, escalated to `deep`, and found the answer using TIER2 of the deep op set.
