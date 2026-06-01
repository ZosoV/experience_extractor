# Solver Results Schema

Reference for every column in `data/solver_results*.parquet`, produced by `src/solver_eq_symbolic.py`.

> **Multi-row layout:** A puzzle may span **multiple rows** (one per solution found). All rows
> for the same puzzle share the same `id`, `solution_count`, and `search_reached` values.
> `eval_from_parquet.py` evaluates each row independently and handles multiple rows per `id`
> correctly. `src/post_processing/extend_ops_mappings.py` also handles the multi-row format.

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
| `solver_answer` | string \| null | Answer returned by the solver (`null` = no solution found) |
| `solver_correct` | bool | Whether `solver_answer == answer` |

---

## Solver classification

| Column | Type | Meaning |
|--------|------|---------|
| `category` | string \| null | High-level puzzle family: `arithmetic` or `concatenation` (null = no solution) |
| `solver_strategy` | string \| null | Sub-strategy within the category (see table below) |

### `category` values

| Value | Meaning |
|-------|---------|
| `arithmetic` | Puzzle solved by assigning digit values to symbols and evaluating arithmetic operators |
| `concatenation` | Operator literally concatenates the left and right operand strings |

### `solver_strategy` values

| Value | Category | Meaning |
|-------|----------|---------|
| `standard` | arithmetic | Digits read in normal (big-endian) order |
| `little_endian` | arithmetic | Digit order is reversed before arithmetic is applied |
| `pure_concat` | concatenation | All operators are simple string concatenation |

---

## Multi-solution metadata

| Column | Type | Meaning |
|--------|------|---------|
| `solution_count` | int | Number of distinct programs found for this puzzle (`0` = no solution, `1–k` = programs found, same value across all rows for a puzzle) |
| `search_reached` | string \| null | Furthest escalation stage the search advanced into (see table below); `null` for concat / no-solution rows |

### `search_reached` values

| Value | Meaning |
|-------|---------|
| `"digits"` | All solutions came from the first stage: `standard` mode + primary base. Search did not need to escalate further. |
| `"little_endian"` | Search advanced into `little_endian` mode (still the primary base) because fewer than `max_solutions` solutions were found in `standard` mode. |
| `"alt_base"` | Search advanced into a non-primary base (only possible when `force_base_10: false`). |

---

## Arithmetic search details

These columns are `null` for concatenation puzzles and no-solution rows.

| Column | Type | Meaning |
|--------|------|---------|
| `ops_mapping` | JSON string | Map of operator symbol → arithmetic function found, e.g. `{"*": "mul", "-": "absdiff"}` |
| `digits_mapping` | JSON string | Map of symbol → assigned digit (the bijection found), e.g. `{"@": 5, "&": 8}` |
| `solver_radix` | float | Number base used (almost always `10`) |
| `solver_tier` | string | Tier set that produced the solution: `tier0`, `tier1`, `tier2`, or `full` |
| `ops_candidates` | JSON string | Map of operator symbol → list of candidate operations that were tried |
| `equations` | JSON string | Parsed equation list (examples + query), including the gold `actual_rhs` |
