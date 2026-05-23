# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup (run once after cloning):**
```bash
uv sync
uv run maturin develop --release
```

**Test a single solve:**
```bash
uv run python main.py
```

**Run solver over the full dataset:**
```bash
uv run python src/solve_all_symbolic.py
# output: data/solver_results.parquet

# Quick smoke test (3 rows, single worker):
uv run python src/solve_all_symbolic.py --limit 3 --workers 1
```

**Serve the visualization UI:**
```bash
uv run python -m http.server 8000
# then open http://localhost:8000/docs/
```

**Rebuild the Rust extension after changes to `rust/`:**
```bash
uv run maturin develop --release
```

## Architecture

This repo solves `equation_symbolic` puzzles from the [Kaggle NeMoTron competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview). Each puzzle presents a few symbolic "transformation" equations (e.g., `` `!*[{ = '"[` ``) and asks for the result of a query equation using the same hidden rules.

### Solver (`src/solver_eq_symbolic.py`)

`AliceEquationSolver` is the main entry point. It classifies puzzles into two types and solves them:

**Type 1 — Concatenation:** The operator simply concatenates left/right operands (forward or reversed). Detected by structural pattern matching with no search needed.

**Type 2 — Arithmetic:** Each unique symbol maps bijectively to a digit (0–9). Each operator maps to one of ~47 arithmetic functions (add, mul, gcd, absdiff, signed sub, concat, offsets, bitwise, etc.). The solver brute-forces symbol→digit permutations and operator assignments.

The arithmetic search uses a **tiered strategy** (TIER0 → TIER1 → TIER2 → full), trying simpler op sets first to avoid spurious matches from overly-generic ops like `add_m1`.

Three **search modes** are tried per tier: `standard`, `little_endian` (digit order reversed), and `alice` (operands and result digits reversed before/after op).

When `answer_hint` is provided (gold answer from training data), `_try_arithmetic_gold_conditioned()` adds the query+answer as an extra constraint to the search.

### Rust acceleration (`rust/alice_sovler_helper/`)

The brute-force permutation search is implemented in Rust (`lib.rs`) via PyO3 and compiled into a Python extension with maturin. The Python solver falls back to `_search_python()` if the Rust module is unavailable. The Rust module exposes a single function: `arithmetic_search(n, is_alice, groups, radix, digit_count, mode_idx)`.

### Batch runner (`src/solve_all_symbolic.py`)

Reads `data/train.csv`, filters to `equation_symbolic` rows, runs `AliceEquationSolver` in parallel via `ThreadPoolExecutor`, and writes results to `data/solver_results.parquet`.

### Visualization (`docs/index.html`)

A static HTML/JS page that loads `solver_results.parquet` (via a JS parquet reader) and displays solve results in a filterable table. Served locally with `python -m http.server`.

## Key data flow

```
data/train.csv
    → solve_all_symbolic.py (batch + parallel)
        → AliceEquationSolver.solve()
            → _try_concat()          # fast structural check
            → _try_arithmetic()      # tiered brute-force (Rust if available)
    → data/solver_results.parquet
        → docs/index.html            # visualization
```

## Notes

- The `answer_hint` parameter is for training/analysis only — it conditions the search on the known gold answer. Normal inference uses `solve()` without it.
- `_derive_order()` computes a CSP-style deduction trace (MRV + max-propagation tie-break) for generating human-readable reasoning chains; it is not part of the solve path.
- `solve_with_trace`, `apply_query_trace`, `narrow_op_candidates_structural`, and `_try_arithmetic_traced` are monkey-patched onto `AliceEquationSolver` at module load time for use in data generation pipelines.
