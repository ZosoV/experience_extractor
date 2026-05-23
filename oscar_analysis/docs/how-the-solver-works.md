# How the Equation-Symbolic Solver Works

This document explains the [Kaggle discussion post](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) by the author of this repo, who achieved **800/823 (97.2%) coverage with 0 wrong answers** on `equation_symbolic` training examples. It covers what the puzzles are, what "latent program" means, how the brute-force search is structured, and why the result is a research oracle rather than a submission.

---

## The puzzle

Each `equation_symbolic` example shows a few demonstration equations built from arbitrary Unicode symbols, then asks for the result of a query using the same hidden rules. Puzzle id `00457d26` (also pinned in `main.py`):

```
`!*[{ = '"[`
\'*'> = ![@
\'-!` = \\
`!*\& = '@'{
Determine the result for: [[-!'
```

Answer: `@&`

There is no stated rule. The solver's job is to infer the hidden rule purely from the demonstrations, then apply it to the query.

---

## What is a "latent program"?

The author's central claim is that for ~97% of puzzles there exists a **latent symbolic program** — a tuple of four things that simultaneously explains every demonstration and implies a unique query answer:

1. **Symbol → digit mapping.** Every unique non-operator character maps bijectively to a digit 0–9. Two adjacent characters form a two-digit integer.
2. **Operator assignment.** Each operator character maps to one of ~47 arithmetic functions (`add`, `mul`, `absdiff`, `gcd`, `concat_fwd`, etc.).
3. **Interpretation mode.** One of three digit-order conventions (`standard`, `little_endian`, `alice`) that controls how two-character operands are read and how results are written back.
4. **Numeric query answer.** The integer produced by applying the above to the query operands; re-encoded as symbols via the inverse mapping.

The program must be consistent with **all** demonstrations simultaneously — not just one or two of them.

---

## Two puzzle types

**Type 1 — Concatenation.** The operator just glues its left and right operands together (forward or reversed). This is detected structurally in `_try_concat()` (`src/solver_eq_symbolic.py:537`) with no search: if every demonstration's result equals `left + right` or `right + left` as symbol strings, the operator is a concatenator.

**Type 2 — Arithmetic.** Everything else. Symbols map to digits; operators map to arithmetic functions. Solved by brute-force search in `_try_arithmetic()` (`src/solver_eq_symbolic.py:771`).

The solver tries concatenation first; arithmetic only runs if concatenation fails.

---

## How the arithmetic search works

### State space

With `n` distinct content symbols, the search assigns each symbol a unique digit from 0–9. That's `P(10, n)` permutations — for the `00457d26` puzzle (n = 10), it is `10! = 3,628,800`. Layered on top, every operator character has an independent list of candidate functions. Naïve enumeration over permutations × op assignments is expensive; the solver prunes heavily.

### Step 1: structural pre-filtering of operator candidates

Before touching permutations, `_try_arithmetic()` narrows each operator's candidate list using structural features only (`src/solver_eq_symbolic.py:809-826`):

- **Sign-prefix pattern.** If a demonstration result starts with the operator character, the result is negative and the op must be a signed subtraction (`sub_signed`, `rsub_signed`) or `neg_absdiff`. If all results are unsigned, `neg_absdiff` is excluded. If results are *mixed* (some signed, some not), only `sub_signed` / `rsub_signed` survive.
- **Op-character heuristics.** `OP_PRIORITY` (`src/solver_eq_symbolic.py:213`) biases the candidate list by the operator's visual form: a `*` symbol is tried as `mul` first, a `+` symbol as `add`, a `-` symbol as `absdiff`, etc. This is just ordering — it does not eliminate candidates, but it means the DFS finds the right answer faster when the obvious mapping is correct.

### Step 2: tiered op pool

The solver restricts the operator pool in tiers and tries each tier in turn before expanding (`src/solver_eq_symbolic.py:838-869`):

| Tier | Ops added |
|------|-----------|
| TIER0 | `add`, `sub`, `rsub`, `absdiff`, `mul`, `gcd`, `lcm`, `concat_fwd`, `concat_rev`, signed variants |
| TIER1 | + `fdiv`, `rdiv`, `mod`, `rmod`, `min`, `max` |
| TIER2 | + `add_m1`, `add_p1`, `mul_m1`, `mul_p1`, `absdiff_m1`, `absdiff_p1`, `sub_m1`, `sub_p1`, `rsub_m1`, `rsub_p1` |
| FULL | everything (bitwise, polynomial) |

Why not always use the full pool? Offset ops like `mul_p1` (`a*b+1`) are generic enough that they fit many permutations spuriously, producing ambiguous solutions. Restricting to simpler ops first breaks the tie in favor of the canonical answer.

### Step 3: three interpretation modes

Within each tier, the solver tries three modes (`SEARCH_MODES`, `src/solver_eq_symbolic.py:210`):

**`standard`** — big-endian digit order (default):

- Operand `AB` → `mapping[A] × base + mapping[B]` (left character is most significant digit).
- Result integer encoded left-to-right: if the result is 45, the first output symbol maps to 4, the second to 5.

**`little_endian`** — digits are reversed for both operands and result:

- Operand `AB` → `mapping[B] × base + mapping[A]` (left character is **least** significant digit).
- Result integer encoded right-to-left: if the result is 45, the first output symbol maps to 5, the second to 4.
- Example: symbols `XY` where X→3, Y→7 represent the number **73** (not 37) in little-endian. A result of 73 is written back as `XY` (the ones digit 3 appears first as `X`, the tens digit 7 appears second as `Y`).
- Implementation (`rust/alice_sovler_helper/src/lib.rs:222-255`): `lv = (lv % base) * base + lv / base` swaps the two digits before the op; after the op, the result digits are reversed before comparing to expected symbols.

**`alice`** — computationally identical to `little_endian` (same digit-swap logic, `src/solver_eq_symbolic.py:384`):

- The solver uses "alice" as a separate label for historical reasons — it was the original name for reversed-digit mode (the Rust extension is `alice_sovler_helper`). The label distinction allows the output `category` field to differentiate `little_endian` (pure reversed-digit arithmetic) from `arithmetic` (standard) and `mixed_concat` (concat + reversed-digit) without ambiguity.
- `little_endian` is tried before `alice` in the search loop, so in practice `alice` is a fallback label rather than a distinct algorithm.

The outer loop is: `for tier in tiers: for mode in (standard, little_endian, alice): run DFS`.

### Step 4: DFS with forward-checking (the Rust core)

The actual permutation search runs in Rust (`rust/alice_sovler_helper/src/lib.rs`, entry point `arithmetic_search` line 544, DFS loop in `dfs()` line 298). The Python fallback `_search_python()` at `src/solver_eq_symbolic.py:1298` is identical in logic but slower.

The DFS works as follows:

1. **Variable ordering.** `variable_order()` (line 388) scores every symbol and sorts them so the most *useful-to-assign-early* symbol goes first. The score for symbol `s` is:

   ```
   score[s] = Σ (1 + candidates_in_group) × (appearances of s in that group's equations)
   ```

   Symbols are assigned in **descending score** order. A symbol scores high when it appears in many equations **and** those equations belong to operator groups with many surviving op candidates. The intuition is the inverse of classic MRV: assigning a symbol in a group that still has 30 candidate ops can immediately collapse those 30 down to a few (or to zero, triggering a fast prune); assigning a symbol in a group with only 1 candidate contributes almost nothing to pruning. So the heuristic maximises expected candidate elimination per assignment, rather than picking the variable with the smallest domain.

2. **Recursive assignment.** `dfs()` tries each unused digit `d` in `0..9` for the current symbol. After tentatively assigning `perm[sym_idx] = d`, it calls `group_has_candidate_for_assigned_eqs()` for every operator group. That function checks: is there at least one candidate op in this group that is consistent with the equations involving only already-assigned symbols? If any group has zero surviving candidates, backtrack immediately. This is forward-checking.

3. **Leaf verification.** When all `n` symbols are assigned (`depth == n`), verify every candidate op in every group against every equation. Keep only ops that pass all equations. If all groups have at least one surviving op, return the solution `(perm, ops_valid)`.

The result is a search whose pruning scales with constraint density — most branches are cut within the first 2–3 variable assignments on well-constrained puzzles.

### Step 5: encoding the answer

Once `(mapping, ops, mode)` is known, `_encode_answer()` (`src/solver_eq_symbolic.py:1374`):
1. Decode the query's two left-hand symbols and two right-hand symbols to integers using the mapping.
2. Apply the mode flip (if `alice` or `little_endian`).
3. Evaluate the operator function.
4. Convert the result integer back to digits, apply reverse mode flip, look up each digit in the inverse mapping to get output symbols.

---

## Worked example: `00457d26` end-to-end

### Parse

The puzzle has two operator characters: `*` (appears in 3 demonstrations) and `-` (appears in 1 demonstration and the query).

Content symbols — all non-operator, non-whitespace characters that appear anywhere in demonstrations or query, sorted by ASCII code:

| Symbol | Index | Assigned digit |
|--------|-------|---------------|
| `!`    | 0     | 6             |
| `"`    | 1     | 7             |
| `&`    | 2     | 8             |
| `'`    | 3     | 2             |
| `>`    | 4     | 0             |
| `@`    | 5     | 1             |
| `[`    | 6     | 4             |
| `\`    | 7     | 3             |
| `` ` ``| 8    | 5             |
| `{`    | 9     | 9             |

`n = 10`, `base = 10` — a full permutation of digits 0–9.

### Structural narrowing

- Operator `*`: all four results are unsigned (no sign prefix) and multi-digit (lengths 3 and 4). `neg_absdiff` is excluded. Result length 4 rules out `sub`, `rsub`, `absdiff`, `gcd`, `min`, `max` (max 2-digit results) and `add` (max 3-digit). Surviving candidates (TIER0 only): `mul`, `lcm`, `concat_fwd`, `concat_rev`.
- Operator `-`: single demonstration, unsigned 2-digit result. All non-sign ops are candidates: `absdiff`, `mul`, `add`, `sub`, `rsub`, `gcd`, `min`, `max`, `lcm`, …

### DFS finds the solution at TIER2, mode `standard`

TIER0 and TIER1 fail because the correct op for `*` is `mul_p1` (added only in TIER2). At TIER2, standard mode, the DFS finds the unique permutation:

```
mapping = {'>': 0, '@': 1, "'": 2, '\': 3, '[': 4, '`': 5, '!': 6, '"': 7, '&': 8, '{': 9}
```

Verification against the four demonstrations:

**Demo 1** — `` `!*[{ = '"[` ``
- Left operand `` `! `` → digits 5, 6 → **56**
- Right operand `[{` → digits 4, 9 → **49**
- `mul_p1(56, 49)` = 56 × 49 + 1 = **2745**
- Encode 2745 → digits 2, 7, 4, 5 → symbols `'`, `"`, `[`, `` ` `` → `` '"[` `` ✓

**Demo 2** — `\'*'> = ![@`
- Left `\'` → 3, 2 → **32**; Right `'>` → 2, 0 → **20**
- `mul_p1(32, 20)` = 32 × 20 + 1 = **641**
- Encode 641 → 6, 4, 1 → `!`, `[`, `@` → `![@` ✓

**Demo 3** — `\'-!`` = \\`
- Left `\'` → 3, 2 → **32**; Right `` !` `` → 6, 5 → **65**
- `absdiff(32, 65)` = |32 − 65| = **33**
- Encode 33 → 3, 3 → `\`, `\` → `\\` ✓

**Demo 4** — `` `!*\& = '@'{ ``
- Left `` `! `` → 5, 6 → **56**; Right `\&` → 3, 8 → **38**
- `mul_p1(56, 38)` = 56 × 38 + 1 = **2129**
- Encode 2129 → 2, 1, 2, 9 → `'`, `@`, `'`, `{` → `'@'{` ✓

### Apply to query

Query: `[[-!'`
- Left `[[` → 4, 4 → **44**
- Right `!'` → 6, 2 → **62**
- Operator `-` maps to `absdiff`: |44 − 62| = **18**
- Encode 18 → 1, 8 → `@`, `&` → **`@&`** ✓

---

## Gold-conditioning: why this is a research oracle

The 97.2% figure comes from running the solver with `answer_hint` set to the known gold answer (`src/solve_all_symbolic.py:70`):

```python
solver = AliceEquationSolver(row["prompt"], answer_hint=row["answer"])
```

In `_try_arithmetic_gold_conditioned()` (`src/solver_eq_symbolic.py:598`), the gold answer is encoded as **an extra equation** appended to the query operator's group (`src/solver_eq_symbolic.py:646`). The Rust DFS must then find a program consistent with both the demonstrations *and* the query+gold pair simultaneously.

Why this matters: a small set of demonstrations rarely pins down the latent program uniquely. Multiple `(mapping, ops, mode)` triples can satisfy three or four examples while disagreeing about the query. Adding the gold as a constraint eliminates the spurious solutions and recovers the canonical program.

**At inference time** you do not have the gold answer, so the gold-conditioned solver cannot be used directly in a submission. The author is explicit about this: the solver is an analysis tool that demonstrates the category *contains recoverable symbolic structure*, separate from whether a model can infer that structure from the prompt in a single forward pass. The gap between symbolic search (97.2%) and what a LoRA model can learn via SFT is described in the post as "still large."

---

## The numbers

| Result | Count |
|--------|-------|
| Solved correctly | 800 |
| Wrong answer | 0 |
| No solution found | 23 |
| **Total** | **823** |

Zero wrong answers means: when the solver returns an answer it has been verified against the full program. The 23 unsolved cases are puzzles where no `(mapping, ops, mode)` triple in the supported library is consistent with the demonstrations.

---

## Further reading

- `main.py` — single-puzzle invocation with full `details` output.
- `src/solver_eq_symbolic.py` lines 1–28 — docstring summarises the entire architecture.
- `rust/alice_sovler_helper/src/lib.rs` — the actual DFS, `arithmetic_search()` and `dfs()`.
- `docs/index.html` — interactive results viewer over the full 823-example dataset.
