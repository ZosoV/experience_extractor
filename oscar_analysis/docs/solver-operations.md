# Solver Operations Reference

Source: `src/solver_eq_symbolic.py`. Every operation takes two integers `(a, b)` and returns an integer or `None` (meaning "this pair is invalid for this op"). The index column is the stable numeric ID used across the Python↔Rust FFI (`OP_NAME_TO_IDX`, L159-207).

---

## Operations

### Core (idx 0–13)

| Idx | Name | Python expression | What it computes | Guard |
|-----|------|------------------|-----------------|-------|
| 0 | `add` | `a + b` | Sum | — |
| 1 | `sub` | `a - b if a >= b else None` | Left minus right | `a ≥ b`, else `None` |
| 2 | `rsub` | `b - a if b >= a else None` | Right minus left | `b ≥ a`, else `None` |
| 3 | `absdiff` | `abs(a - b)` | Absolute difference | — |
| 4 | `neg_absdiff` | `abs(a - b)` *(same math)* | Absolute difference with forced sign prefix | Encoder always emits op-symbol as prefix; only candidate when RHS carries that prefix |
| 5 | `mul` | `a * b` | Product | — |
| 6 | `gcd` | `gcd(a, b)` | Greatest common divisor | — |
| 7 | `lcm` | `a*b // gcd(a,b) if (a and b) else 0` | Least common multiple | Both non-zero (returns 0 if either is 0) |
| 8 | `fdiv` | `a // b if b else None` | Floor-divide left by right | `b ≠ 0` |
| 9 | `rdiv` | `b // a if a else None` | Floor-divide right by left | `a ≠ 0` |
| 10 | `mod` | `a % b if b else None` | Left modulo right | `b ≠ 0` |
| 11 | `rmod` | `b % a if a else None` | Right modulo left | `a ≠ 0` |
| 12 | `min` | `min(a, b)` | Minimum | — |
| 13 | `max` | `max(a, b)` | Maximum | — |

---

### Offsets ±1 / ±2 (idx 14–29)

All built via `_off(fn, delta)` (L83-93): compute `fn(a, b)`, add `delta`, return `None` if the base op already returned `None` **or** if the final result is negative.

| Idx | Name | Expression | Computes | Extra guard |
|-----|------|-----------|---------|-------------|
| 14 | `add_m1` | `a + b - 1` | Sum minus 1 | `a+b ≥ 1` |
| 15 | `add_p1` | `a + b + 1` | Sum plus 1 | — |
| 16 | `mul_m1` | `a * b - 1` | Product minus 1 | `a*b ≥ 1` |
| 17 | `mul_p1` | `a * b + 1` | Product plus 1 | — |
| 18 | `absdiff_m1` | `abs(a - b) - 1` | Abs-diff minus 1 | `abs(a-b) ≥ 1` |
| 19 | `absdiff_p1` | `abs(a - b) + 1` | Abs-diff plus 1 | — |
| 20 | `sub_m1` | `(a - b) - 1` | Sub minus 1 | `a ≥ b` AND `a-b ≥ 1` |
| 21 | `sub_p1` | `(a - b) + 1` | Sub plus 1 | `a ≥ b` |
| 22 | `rsub_m1` | `(b - a) - 1` | Rsub minus 1 | `b ≥ a` AND `b-a ≥ 1` |
| 23 | `rsub_p1` | `(b - a) + 1` | Rsub plus 1 | `b ≥ a` |
| 24 | `add_m2` | `a + b - 2` | Sum minus 2 | `a+b ≥ 2` |
| 25 | `add_p2` | `a + b + 2` | Sum plus 2 | — |
| 26 | `mul_m2` | `a * b - 2` | Product minus 2 | `a*b ≥ 2` |
| 27 | `mul_p2` | `a * b + 2` | Product plus 2 | — |
| 28 | `absdiff_m2` | `abs(a - b) - 2` | Abs-diff minus 2 | `abs(a-b) ≥ 2` |
| 29 | `absdiff_p2` | `abs(a - b) + 2` | Abs-diff plus 2 | — |

---

### Scaled / Polynomial (idx 30–39)

| Idx | Name | Python expression | Computes | Guard |
|-----|------|------------------|---------|-------|
| 30 | `mul_half` | `(a * b) // 2` | Half the product (floor) | — |
| 31 | `mul_double` | `a * b * 2` | Double the product | — |
| 32 | `sq_diff` | `(a - b) ** 2` | Squared difference | Always ≥ 0 |
| 33 | `sq_sum` | `(a + b) ** 2` | Squared sum | Always ≥ 0 |
| 34 | `mul_plus_a` | `a * b + a` | Product plus `a` | — |
| 35 | `mul_plus_b` | `a * b + b` | Product plus `b` | — |
| 36 | `mul_minus_a` | `_off(a*b - a, 0)` | Product minus `a` | Non-negative |
| 37 | `mul_minus_b` | `_off(a*b - b, 0)` | Product minus `b` | Non-negative |
| 38 | `a2_plus_b` | `a * a + b` | `a²` plus `b` | — |
| 39 | `a_plus_b2` | `a + b * b` | `a` plus `b²` | — |

---

### Bitwise (idx 40–42)

| Idx | Name | Python expression | Computes | Guard |
|-----|------|------------------|---------|-------|
| 40 | `xor` | `a ^ b` | Bitwise XOR | — |
| 41 | `band` | `a & b` | Bitwise AND | — |
| 42 | `bor` | `a \| b` | Bitwise OR | — |

---

### Signed subtractions (idx 43–44)

These return a **signed** integer. The sign of the result must match whether the RHS in that example actually carries the op-character as a leading prefix. The magnitude `|result|` is what gets encoded; the prefix character is the op symbol itself. Collected in `SIGNED_OPS` (L156) and hardcoded as `{43, 44}` in Rust (`lib.rs:141-144`).

| Idx | Name | Python expression | Computes | Constraint |
|-----|------|------------------|---------|-----------|
| 43 | `sub_signed` | `a - b` | Signed left-minus-right | Raw sign must match RHS prefix presence |
| 44 | `rsub_signed` | `b - a` | Signed right-minus-left | Raw sign must match RHS prefix presence |

---

### Sequence ops (idx 45–46)

These have no numeric function (`OPERATIONS` entry is `None`). They compare symbol sequences directly and produce no `numeric_answer`. Only added as candidates when every example has `has_sign == False` and `result_len == 4`.

| Idx | Name | Produces | Notes |
|-----|------|---------|-------|
| 45 | `concat_fwd` | `l0 l1 r0 r1` | Forward concat of left and right operand symbols |
| 46 | `concat_rev` | `r0 r1 l0 l1` | Reversed concat (right then left) |

---

## Tiers

The solver tries ops in escalating tiers to prefer parsimonious explanations. From the source (`solver_eq_symbolic.py:834-838`):

> "Tiered search: prefer simple ops. If a (mapping, ops) solution exists using only simple ops, use it. Offset/polynomial ops are generic enough that they can spuriously fit multiple permutations; restricting the op pool first breaks ambiguity in favour of the canonical answer."

Tiers are defined at `solver_eq_symbolic.py:684-710` and `838-864`.

| Tier | Ops added in this tier | Cumulative total |
|------|----------------------|-----------------|
| **TIER0** | `add`, `sub`, `rsub`, `absdiff`, `mul`, `gcd`, `lcm`, `neg_absdiff`, `sub_signed`, `rsub_signed`, `concat_fwd`, `concat_rev` | 12 |
| **TIER1** | + `fdiv`, `rdiv`, `mod`, `rmod`, `min`, `max` | 18 |
| **TIER2** | + `add_m1`, `add_p1`, `mul_m1`, `mul_p1`, `absdiff_m1`, `absdiff_p1`, `sub_m1`, `sub_p1`, `rsub_m1`, `rsub_p1` | 28 |
| **FULL** | Whatever `OP_LEVELS[search_level]` resolves to (see below) | up to 47 |

Iteration order: **TIER0 → TIER1 → TIER2 → FULL** (`solver_eq_symbolic.py:711-715`). A tier is skipped if any op-character has zero allowed candidates in it (L722-723 / L876-877). Tiers are deduplicated, so if e.g. `search_level='fast'` makes FULL == TIER1, only one pass runs.

### `OP_LEVELS` — the FULL set (`solver_eq_symbolic.py:328-364`)

`search_level` caps what counts as "FULL":

| `search_level` | Ops included | Count |
|---------------|-------------|-------|
| `fast` | Core (0–13) + `sub_signed`, `rsub_signed`, `neg_absdiff`, `concat_fwd`, `concat_rev` | 18 |
| `normal` *(default)* | `fast` + ±1 offsets (`add_m1/p1`, `mul_m1/p1`, `absdiff_m1/p1`, `sub_m1/p1`, `rsub_m1/p1`) | 28 |
| `deep` | Everything — all 47 ops including ±2 offsets, polynomial, and bitwise | 47 |

The solver auto-escalates from `normal` to `deep` on failure (`solver_eq_symbolic.py:526-531`; gold-conditioned path: `L514-519`).

### Search modes (`solver_eq_symbolic.py:209-210`)

Applied per tier, in order **standard → little_endian → alice** (`L729`, `L883`).

| Mode | Effect |
|------|--------|
| `standard` | MSB-first: `L = perm[l0]*base + perm[l1]`. Result digits compared in declaration order. |
| `little_endian` | Operand digits flipped before the op (`L1333-1335`); result digits reversed after zfill before comparison (`L1354-1356`). |
| `alice` | *Same engine as little_endian* (Rust shares the branch). Distinguished only by downstream labelling in `_solver_category` (`L388-396`) and by the fact that gold-conditioned little_endian solutions skip the deduction trace (`L746-754`). |

---

## Special Cases and Pruning

### 1. Sign-prefix candidate filtering (`solver_eq_symbolic.py:809-822`)

Before search starts, the solver checks whether each example's RHS begins with the op-character (a "sign prefix"):

| Observation | Candidates reduced to |
|------------|----------------------|
| Every example is signed (has prefix) | `{neg_absdiff, sub_signed, rsub_signed}` |
| Every example is unsigned (no prefix) | All ops **except** `neg_absdiff` |
| Mixed (some signed, some not) | `{sub_signed, rsub_signed}` only |

### 2. Concat eligibility (`solver_eq_symbolic.py:823-826`, `L1310-1325`)

`concat_fwd` / `concat_rev` are added to the candidate set **only when**:
- Every training example has `has_sign == False`, **and**
- `result_len == 4` for every example.

When a concat op is chosen, the answer is the four query letters re-arranged — no `numeric_answer` is produced (`_encode_answer`, `L1388-1395`).

### 3. Encoder rules — answer rejection (`_encode_answer`, `solver_eq_symbolic.py:1374-1438`)

After a candidate `(permutation, ops)` pair passes the search, the encoder applies these checks before accepting the answer:

- Non-signed ops reject negative results.
- `neg_absdiff` requires `result ≥ 0` AND always emits the op-character as a prefix.
- Signed ops emit a prefix if and only if `result < 0`.
- If any result digit lacks a content-symbol mapping (underdetermined puzzle), the answer is **rejected** — the solver never guesses.

### 4. Base / radix gating (`solver_eq_symbolic.py:606`, `L772`, `L725-728`, `L879-882`)

- If `self.base > 12` (more than 12 distinct content symbols), arithmetic search exits immediately — the search space is too large.
- When `self.base ≠ 10`, a second pass is run at `radix=10, digit_count=10`. This handles puzzles where the underlying numeric base is 10 even though fewer distinct symbols appear in the examples.

### 5. Deduction-trace skip conditions (`solver_eq_symbolic.py:745-749`, `L909-915`)

The CSP-style deduction trace (used for generating human-readable reasoning) is skipped when any of the following hold:
- `radix ≠ 10`
- `mode == "little_endian"` (`alice` is intentionally not skipped — propagation handles the digit flip at `L1010-1012`)
- Any assigned op is a concat op

`solve_with_trace` also never explores the `little_endian` mode at all (`L1649`).

### 6. Op-character priority (`OP_PRIORITY`, `solver_eq_symbolic.py:213-275`)

Each operator character has a preferred candidate ordering. `_priority` (`L488-493`) filters the list by `op_types`, then appends any remaining allowed ops at the end. Within a tier, this ordering determines which ops are evaluated first — it is a tie-break, not a hard pruning rule.

| Character | Top candidates (in priority order) |
|-----------|----------------------------------|
| `*` | `mul`, `mul_m1`, `mul_p1`, `mul_m2`, `mul_p2`, `absdiff`, `add`, `gcd`, … |
| `+` | `add`, `add_m1`, `add_p1`, `add_m2`, `add_p2`, `mul`, `absdiff`, `gcd`, … |
| `-` | `absdiff`, `sub`, `rsub`, `sub_signed`, `rsub_signed`, `absdiff_m1`, … |
| `/` | `fdiv`, `rdiv`, `mul`, `add`, `absdiff`, `concat_fwd`, `concat_rev` |
| *(other)* | `DEFAULT_PRIORITY` — same list but starting from `add`, `absdiff`, `mul` |

### 7. Rust parity and the signed-op index contract

The Rust accelerator (`rust/alice_sovler_helper/src/lib.rs`) implements the same 47-op vocabulary with the same indices, same guards, same signed/concat semantics, and the same digit-flip mechanics for standard / alice / little_endian. One hard constraint: `is_signed_op` is hardcoded to indices `{43, 44}` in Rust (`lib.rs:141-144`). Any future signed op added on the Python side **must** update this Rust list to stay in sync.
