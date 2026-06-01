# Deduction Order and Trace

This document explains what `deduction_order` and `deduction_trace` are, how
they are built by `_derive_order` (`src/solver_eq_symbolic.py:938`), and what
every field in a trace entry means.

---

## What problem does `_derive_order` solve?

By the time `_derive_order` is called, the brute-force search is **already
done**. The solver has found the correct `mapping` (symbol → digit) and `ops`
(op-char → function name). `_derive_order` does not help solve the puzzle. Its
job is to answer a different question:

> _If someone were reasoning through this puzzle step by step — starting from
> "every symbol could be any digit" and narrowing using the equations — in what
> order would they pin down each symbol, and what would they see at each step?_

The answer is a **chain-of-thought trace** used downstream to generate
human-readable rationales.

**Inputs:**

| parameter | type | meaning |
|---|---|---|
| `mapping_dict` | `{str: int}` | The solved symbol→digit bijection |
| `ops` | `{str: str}` | The solved op-char→op-type dict |
| `mode` | `str` | `"standard"` or `"alice"` |

**Outputs:**

| output | type | meaning |
|---|---|---|
| `deduction_order` | `list[str]` | Content symbols in the order they are committed |
| `deduction_trace` | `list[dict]` | One entry per commit step (see schema below) |

**When are they skipped?** If `radix != 10`, `mode == "little_endian"`, or
any operator uses a `concat_*` op type, both outputs are `[]`. References:
`src/solver_eq_symbolic.py:745-754` and `:909-920`.

---

## The mental model: MRV with max-propagation tie-break

Think of each content symbol as a variable whose value is unknown. It starts
with a **domain** — the set of digits it could plausibly be — initially
`{0, 1, …, 9}`. The algorithm narrows domains by repeatedly asking: "given the
current partial assignment, which digit values for this symbol are consistent
with every equation?"

At each step the algorithm commits to one symbol using two rules:

1. **Minimum Remaining Values (MRV).** Pick the symbol with the fewest
   possible values left in its domain. A smaller domain means fewer choices
   and a more constrained decision.

2. **Max-propagation tie-break.** If multiple symbols share the smallest
   domain size, simulate committing each one to its gold digit, re-run
   propagation, and pick whichever commit shrinks the _others'_ domains the
   most — i.e. "opens the most doors" for subsequent steps.

This is a CSP (constraint-satisfaction) heuristic applied post-hoc to the
already-known solution, not a search. The gold digit is always known, so
there is no backtracking.

---

## The three helper closures

```
_derive_order
├── compute_v          (apply op, check sign/magnitude)
├── update_feasible    (enumerate legal digit assignments for one equation)
├── propagate          (forward-check all equations to fixed point)
└── per_example_pass   (record what a learner sees, equation by equation)
```

### `compute_v` (lines 971–988)

Applies the resolved operator to two two-digit integers and validates the
result's sign:

```python
def compute_v(L, R, ot, has_sign):
    fn = OPERATIONS.get(ot)
    v = fn(L, R)
    if ot in SIGNED_OPS:
        if (v < 0) != has_sign: return None, False
        return (-v if v < 0 else v), True
    if has_sign or v < 0: return None, False
    return v, True
```

Returns `(magnitude, True)` on success or `(None, False)` if the assignment
is inconsistent with the sign prefix in the example.

### `update_feasible` (lines 990–1056)

For one equation, enumerate every legal way to fill its still-unknown symbols:

1. Collect all unknown symbols that appear in the equation's LHS and RHS.
2. Try every permutation of available digits (those not yet assigned globally)
   for the unknown LHS symbols.
3. For each permutation, evaluate `compute_v` and check whether the resulting
   digits for the RHS symbols are consistent with their current domains and the
   AllDifferent constraint.
4. Accumulate, for each symbol, the set of digit values that appeared in any
   consistent assignment → `feasible[sym]`.

`update_feasible` does not mutate domains; it just reports what is feasible.

### `propagate` (lines 1058–1084)

Runs forward-checking to a fixed point. It repeatedly:

1. Removes already-determined digits from every undetermined symbol's domain
   (AllDifferent).
2. For every equation, calls `update_feasible` to collect feasible digits, then
   intersects each involved symbol's domain with its feasible set.

Keeps looping until no domain changes (fixed point). This is what actually
narrows the working domains.

---

## `per_example_pass` — what the learner sees

`propagate` runs all examples simultaneously and loops globally. That is
efficient but opaque. `per_example_pass` (lines 1086–1171) does the same
traversal **visibly**, in document order, one example at a time:

- Start from the pre-propagation domains.
- Process each example: call `update_feasible`, intersect into a running copy
  of the domains, record which symbols narrowed and to what.
- Repeat until a full pass produces no changes (capped at 5 passes).

The result is a structured record of every narrowing step, attached to the
trace entry as `per_example_feasibility`. It is used by CoT generators to
show the learner's reasoning ("looking at equation 2, symbol `>` must be one
of…").

Importantly, `per_example_pass` is called **before** `propagate` each step,
on the pristine pre-propagation domains. This means the per-example trace
always shows the view from the start of the step, not after the full fixed
point.

---

## The main loop (lines 1178–1254)

```
domains = {s: {0..9} for each symbol}
determined = {}

while undetermined symbols remain:
    1. Snapshot pristine domains → run per_example_pass → record per_ex
    2. propagate(determined, domains)         # narrows domains to fixed point
    3. Find MRV: symbols with smallest domain
    4. If one symbol: commit it.
       Else: tie-break via cascade simulation.
    5. Record trace entry.
    6. Commit chosen symbol → update domains.
```

### Cascade simulation (lines 1196–1224)

For each tied candidate:

```python
sim_det = dict(determined)
sim_det[cand] = mapping_dict[cand]           # commit gold digit
sim_dom = {s: set(domains[s]) for s in symbols}
propagate(sim_det, sim_dom)                  # re-propagate
score = sum(len(domains[s]) - len(sim_dom[s]) for s not in sim_det)
forced = count(s not in sim_det where len(sim_dom[s]) == 1)
```

Priority key: `(score, forced, -ord(cand))`. The candidate with the highest
score (most total domain reduction) wins; ties broken by most newly-forced
symbols; further ties broken by character code.

---

## Schema of a `deduction_trace` entry

| key | type | meaning |
|---|---|---|
| `sym` | `str` | The symbol committed in this step |
| `digit` | `int` | Its gold digit (from `mapping_dict`) |
| `domain` | `list[int]` | Symbol's domain at the moment of commit (post-propagation) |
| `kind` | `str` | `"forced"` if `len(domain) == 1`, else `"guess"` |
| `tied` | `list[str]` | All symbols that tied for smallest domain |
| `cascade` | `list[dict]` | Per-tied-candidate `{sym, score, forced}` from the simulation |
| `domains_before` | `dict[str, list[int]]` | Snapshot of all domains after propagation, before commit |
| `domains_after` | `dict[str, list[int]]` | Snapshot after committing this symbol and re-propagating |
| `determined_before` | `dict[str, int]` | Symbols already committed before this step |
| `per_example_feasibility` | `dict` | Structured trace from `per_example_pass` — see below |

### `per_example_feasibility` sub-structure

```python
{
  "initial":  {sym: [digits]},          # domains after AllDifferent, before any example
  "passes": [
    {
      "iter": int,                       # 1-indexed pass number
      "any_change": bool,
      "examples": [
        {
          "ex_idx": int,                 # 1-indexed example number
          "label": "ex1",
          "lhs": str, "rhs": str,        # raw equation strings
          "op_char": str, "op_type": str,
          "syms_unknown": [str],         # undetermined symbols in this equation
          "feasible_per_sym": {sym: [digits]},   # what update_feasible found
          "running_before_this_ex": {sym: [digits]},
          "intersected_after": {sym: [digits]},  # running domains after intersection
          "narrowed": bool,              # did any domain shrink?
        }
      ]
    }
  ],
  "final": {sym: [digits]}              # domains after all passes converge
}
```

---

## Worked example: puzzle `d23f4166`

### The puzzle

```
{]-{{ = )
}@*>> = })]|
)>->] = {@
Determine the result for: '}*{@
```

The solver finds: `-` = `absdiff`, `*` = `mul`, and this mapping:

| symbol | digit |
|---|---|
| `'` | 0 |
| `)` | 2 |
| `>` | 6 |
| `@` | 9 |
| `]` | 5 |
| `{` | 3 |
| `\|` | 4 |
| `}` | 1 |

### Step 0 — first commit (`')'` → 2, kind: `guess`)

**Starting domains** (all `{0..9}` since nothing is determined yet, and
`'` has been excluded from 0 by AllDifferent via the Rust solver's prior
assignment — wait, actually all symbols start at `{0..9}` before
`per_example_pass` runs).

`per_example_pass` walks the three equations in order:

- **ex1 `{]-{{ = )`** uses `absdiff`. The equation simplifies to
  `|mapping[]] − mapping[{]| = mapping[)]`, so `)` can't be 0 (two distinct
  symbols always differ by at least 1). After ex1: `)` narrows from `{0..9}`
  to `{1..8}`.

- **ex2 `}@*>> = })]|`** uses `mul`. `}@` × `>>` = `})]|` (four result
  symbols). The LHS symbols are `}`, `@`, `>` and the RHS symbols `}`, `)`,
  `]`, `|`. Trying all legal digit permutations, any assignment where `>>` = 0
  (i.e. `>` = 0) is rejected because `}×0 = 0` but the 4-symbol RHS can't all
  be 0. After ex2: `>` is removed from `{0, 1, 5}` (values that can't
  produce a valid 4-digit product); `)` is further narrowed.

- **ex3 `)>->] = {@`** uses `absdiff`. LHS left = `)>`, right = `>]`. So
  `|)×10+> − >×10+]|` = a 2-digit number `{@`. After ex3, `{` and `@` gain
  constraints.

After `propagate` runs to fixed point, the domains are:

| symbol | domain (size) |
|---|---|
| `)` | `{1,2,3,4,5,6,7}` (7) ← MRV |
| `>` | `{2,3,4,6,7,8,9}` (7) ← MRV |
| `{` | `{1,2,3,4,5,6,7}` (7) ← MRV |
| `\|` | `{0,2,3,4,6,7,8}` (7) ← MRV |
| `@` | `{2,3,4,5,6,7,8,9}` (8) |
| `]` | `{1..9}` (9) |
| `}` | `{0..8}` (9) |
| `'` | `{0..9}` (10) |

MRV picks the 4-way tie: `{)`, `>`, `{`, `|}`. The cascade simulation commits
each candidate to its gold digit and re-propagates:

| candidate | gold | score | forced |
|---|---|---|---|
| `)` | 2 | 42 | 6 |
| `>` | 6 | 42 | 6 |
| `{` | 3 | 42 | 6 |
| `\|` | 4 | 18 | 0 |

`)`, `>`, and `{` all score 42 (each forces 6 other symbols to singletons).
`|` scores only 18 and forces nobody. Among the three tied top scorers,
the tie-break is `-ord(cand)`: `ord(')')=41`, `ord('>')=62`, `ord('{')=123`.
Highest `-ord` wins, so `)` (−41 > −62 > −123) is chosen.

The trace entry for step 0:

```json
{
  "sym": ")",
  "digit": 2,
  "domain": [1, 2, 3, 4, 5, 6, 7],
  "kind": "guess",
  "tied": [")", ">", "{", "|"],
  "cascade": [
    {"sym": ")", "score": 42, "forced": 6},
    {"sym": ">", "score": 42, "forced": 6},
    {"sym": "{", "score": 42, "forced": 6},
    {"sym": "|", "score": 18, "forced": 0}
  ],
  "domains_before": {"'": [0..9], ")": [1..7], ">": [2,3,4,6,7,8,9], ...},
  "domains_after":  {
    "'": [0, 1, 3, 4, 5, 6, 7, 8, 9],   // still unconstrained — only in query
    ")": [2], ">": [6], "@": [9], "]": [5], "{": [3], "|": [4], "}": [1]
  }
}
```

### Steps 1–7 — all forced

Committing `)` → 2 and propagating collapses the six equation-constrained
symbols to singletons. Symbol `'` is only used in the query, not in any
example equation, so it stays unconstrained until AlDifferent finally pins it
to the only remaining digit at the last step.

Steps 1–7 are all `kind: "forced"`, in this commit order:

```
'>' → 6, '@' → 9, ']' → 5, '{' → 3, '|' → 4, '}' → 1, "'" → 0
```

---

## Downstream consumers

`_derive_order` is called from two solve paths:

- **`_try_arithmetic`** (`src/solver_eq_symbolic.py:918`) — the standard path
  after the Rust search finds a solution.
- **`_try_arithmetic_with_gold`** (`src/solver_eq_symbolic.py:752`) — the
  gold-conditioned path that includes the known answer as an extra constraint.

The trace functions that consume `deduction_order` / `deduction_trace` to
produce learner-facing CoT rationales are monkey-patched onto
`AliceEquationSolver` at module load time:

| function | lines | purpose |
|---|---|---|
| `solve_with_trace` | 1479–1534 | Top-level wrapper that records the search branch before calling `_derive_order` |
| `_try_arithmetic_traced` | 1537–1727 | Mirror of `_try_arithmetic` that logs per-tier branch info and calls `_derive_order` |
| `apply_query_trace` | 1746–1825 | Generates a structured trace of how the final answer is encoded for the query |
| `narrow_op_candidates_structural` | 1831–1958 | Returns per-op structural filter info (sign pattern, result lengths, surviving candidates) |
