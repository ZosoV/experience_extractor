"""Strategy B: backward-encoded synthetic problems for n=3, 4, 5.

Generates brand-new synthetic cryptarithm problems using the statistics of the
823 training set.  The backward encoding process:

  1. Pick strategy (arithmetic/little_endian/pure_concat) ~ empirical distribution.
  2. Pick op_name ~ P(op_name).
  3. Sample operands L, R ∈ [0, 99].
  4. Apply the op to get the result (no solver needed — result is known by construction).
  5. Assign symbols bijectively to digits using P(symbol | position) for new digits.
  6. Build the structured equation dict directly.
  7. Validate with verify_program.

Each synthetic problem produces N = n+1 equations, the last being to_predict.
The digit2sym mapping is shared across all equations in a problem (bijective, global).

Documented limitations:
  - Position-conditioned symbol draw fires only on a digit's first encounter
    (mapping is global; position is used only when choosing a fresh symbol).
  - Puzzles may be non-unique (ambiguous); solution_count=1 means "a known solution."
  - pure_concat problems have no digit mapping.
"""

from __future__ import annotations

import random
import string
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_generation.build_example_datasets import lhs_str, make_prompt, rhs_str
from src.data_generation_v2.schema import make_solver_row
from src.data_generation_v2.stats import DataStats
from src.encoding import _int_to_base_digits, _is_reversed_digit_mode, verify_program
from src.op_index import CONCAT_OPS, SIGNED_OPS
from src.operations import OPERATIONS

# ---------------------------------------------------------------------------
# Symbol alphabet: printable non-whitespace, non-'=', non-digit, non-letter
# (stays consistent with the "Alice's Wonderland" puzzle flavour)
# ---------------------------------------------------------------------------
_SYMBOL_POOL: list[str] = sorted(
    c
    for c in string.printable
    if c not in string.whitespace
    and c != "="
    and c not in string.digits
    and c not in string.ascii_letters
)
# _SYMBOL_POOL ≈ 30 chars:
# ! " # $ % & ' ( ) * + , - . / : ; < > ? @ [ \ ] ^ _ ` { | } ~

_MAX_RETRIES = 200   # per equation; reject whole problem and restart if exceeded
_MAX_PROBLEM_RETRIES = 50  # per synthetic problem


def build_synthetic(
    stats: DataStats,
    target_counts: dict[int, int],   # {n: how_many_to_generate}
    seed: int = 42,
    include_pure_concat: bool = True,
) -> list[dict]:
    """Return synthetic samples for the given bucket→count mapping.

    target_counts example: {3: 694, 4: 504, 5: 322}
    """
    rng = random.Random(seed)
    results: list[dict] = []
    total_idx = 0

    for n, target in sorted(target_counts.items()):
        if target <= 0:
            continue
        generated = 0
        n_equations = n + 1  # n examples + 1 to_predict

        while generated < target:
            row = _try_generate_problem(n_equations, stats, rng, include_pure_concat)
            if row is None:
                continue

            problem_id = f"synth_n{n}_{total_idx:06d}"
            total_idx += 1

            results.append(
                make_solver_row(
                    problem_id=problem_id,
                    prompt=row["prompt"],
                    answer=row["answer"],
                    equations=row["equations"],
                    digits_mapping=row.get("digits_mapping"),
                    ops_mapping=row.get("ops_mapping"),
                    radix=row["radix"],
                    strategy=row["strategy"],
                    dataset_category=f"examples_{n}",
                )
            )
            generated += 1

        print(f"  [synth] n={n}: generated {generated}")

    return results


# ---------------------------------------------------------------------------
# Problem generator
# ---------------------------------------------------------------------------

def _try_generate_problem(
    n_equations: int,
    stats: DataStats,
    rng: random.Random,
    include_pure_concat: bool = True,
) -> dict | None:
    """Attempt to generate a single valid synthetic problem. Returns None on failure."""

    # Pick strategy
    strategy = _sample_strategy(stats, rng, include_pure_concat=include_pure_concat)
    is_little_endian = strategy == "little_endian"
    is_concat = strategy == "pure_concat"
    radix = 10
    mode = "little_endian" if is_little_endian else "standard"

    # Problem-level bijections
    digit2sym: dict[int, str] = {}
    sym2digit: dict[str, int] = {}
    op_name2sym: dict[str, str] = {}
    op_sym2name: dict[str, str] = {}
    used_syms: set[str] = set()

    equations: list[dict] = []

    for eq_idx in range(n_equations):
        eq_type = "to_predict" if eq_idx == n_equations - 1 else "example"
        eq = _try_generate_equation(
            strategy, mode, radix, stats, rng,
            digit2sym, sym2digit, op_name2sym, op_sym2name, used_syms,
            is_concat=is_concat,
        )
        if eq is None:
            return None
        eq["type"] = eq_type
        equations.append(eq)

    # Final verification
    if not is_concat and digit2sym:
        ok = verify_program(equations, sym2digit, op_sym2name, mode, radix)
        if not ok:
            return None

    # Build prompt + answer from the equations
    ex_eqs = equations[:-1]
    tp_eq = equations[-1]
    example_lines = [f"{lhs_str(e['lhs'])} = {rhs_str(e['rhs'])}" for e in ex_eqs]
    prompt = make_prompt(example_lines, lhs_str(tp_eq["lhs"]))
    answer = rhs_str(tp_eq["rhs"])

    return {
        "prompt": prompt,
        "answer": answer,
        "equations": equations,
        "digits_mapping": sym2digit if not is_concat else None,
        "ops_mapping": op_sym2name if not is_concat else None,
        "radix": radix,
        "strategy": strategy,
    }


def _try_generate_equation(
    strategy: str,
    mode: str,
    radix: int,
    stats: DataStats,
    rng: random.Random,
    digit2sym: dict[int, str],
    sym2digit: dict[str, int],
    op_name2sym: dict[str, str],
    op_sym2name: dict[str, str],
    used_syms: set[str],
    *,
    is_concat: bool,
) -> dict | None:
    """Try up to _MAX_RETRIES times to generate one valid equation."""

    for _ in range(_MAX_RETRIES):
        if is_concat:
            eq = _generate_concat_equation(stats, rng, op_sym2name, used_syms)
        else:
            eq = _generate_arithmetic_equation(
                strategy, mode, radix, stats, rng,
                digit2sym, sym2digit, op_name2sym, op_sym2name, used_syms,
            )
        if eq is not None:
            return eq
    return None


# ---------------------------------------------------------------------------
# Arithmetic equation builder
# ---------------------------------------------------------------------------

def _generate_arithmetic_equation(
    strategy: str,
    mode: str,
    radix: int,
    stats: DataStats,
    rng: random.Random,
    digit2sym: dict[int, str],
    sym2digit: dict[str, int],
    op_name2sym: dict[str, str],
    op_sym2name: dict[str, str],
    used_syms: set[str],
) -> dict | None:
    """Generate one arithmetic equation or return None on failure."""
    # Sample op
    op_name = _sample_op(stats, rng, concat=False)

    # Sample operands L, R ∈ [0, 99]
    L_val = rng.randint(0, 99)
    R_val = rng.randint(0, 99)

    # Decompose operand values to (l0_d, l1_d, r0_d, r1_d) based on mode
    # Standard:      l0 = tens (MSD), l1 = units (LSD)
    # Little_endian: l0 = units (LSD), l1 = tens (MSD)
    if _is_reversed_digit_mode(mode):
        l0_d, l1_d = L_val % radix, L_val // radix
        r0_d, r1_d = R_val % radix, R_val // radix
    else:
        l0_d, l1_d = L_val // radix, L_val % radix
        r0_d, r1_d = R_val // radix, R_val % radix

    # Compute the result using the numeric operand values
    fn = OPERATIONS.get(op_name)
    if fn is None:
        return None
    result = fn(L_val, R_val)
    if result is None:
        return None

    # Sign handling
    has_sign = False
    sign_pos = None
    sign_sym: str | None = None

    if op_name in SIGNED_OPS:
        if result < 0:
            has_sign = True
        mag = abs(result)
    elif op_name == "neg_absdiff":
        has_sign = True
        mag = abs(result)
    else:
        if result < 0:
            return None
        mag = result

    # Magnitude → digit list
    mag_digits = _int_to_base_digits(mag, radix)

    # For little_endian: result digits are written LSB-first
    if _is_reversed_digit_mode(mode):
        res_digits = list(reversed(mag_digits))
    else:
        res_digits = mag_digits

    # Assign symbols to all required digits
    # Positions for symbol assignment: l0/l1/r0/r1 and result digits
    positions = {
        "l0": l0_d, "l1": l1_d, "r0": r0_d, "r1": r1_d,
    }
    for pos, d in positions.items():
        if not _assign_digit_symbol(d, pos, stats, rng, digit2sym, sym2digit, used_syms):
            return None  # ran out of symbols

    # Assign symbols to result digits (use global distribution as fallback)
    for d in res_digits:
        if not _assign_digit_symbol(d, None, stats, rng, digit2sym, sym2digit, used_syms):
            return None

    # Assign operator symbol
    op_sym = _assign_op_symbol(op_name, stats, rng, op_name2sym, op_sym2name, used_syms)
    if op_sym is None:
        return None

    # If has_sign, the op_sym serves as the sign glyph in the RHS prefix
    if has_sign:
        sign_pos = "prefix"
        sign_sym = op_sym

    # Build symbol lists
    l0_sym = digit2sym[l0_d]
    l1_sym = digit2sym[l1_d]
    r0_sym = digit2sym[r0_d]
    r1_sym = digit2sym[r1_d]
    res_syms = [digit2sym[d] for d in res_digits]

    lhs = {"l0": l0_sym, "l1": l1_sym, "op": op_sym, "r0": r0_sym, "r1": r1_sym}
    rhs = {"op": sign_sym, "sign_pos": sign_pos, "actual_rhs": res_syms}

    return {"lhs": lhs, "rhs": rhs}


# ---------------------------------------------------------------------------
# Concat equation builder
# ---------------------------------------------------------------------------

def _generate_concat_equation(
    stats: DataStats,
    rng: random.Random,
    op_sym2name: dict[str, str],
    used_syms: set[str],
) -> dict | None:
    """Generate one pure_concat equation. No digit mapping needed."""
    # Pick concat op
    if stats.concat_ops:
        op_name = rng.choice(stats.concat_ops)
    else:
        op_name = "concat_fwd"

    # Assign operator symbol (must not clash with content symbols)
    # For concat, draw from the global unused pool
    avail_op = [s for s in _SYMBOL_POOL if s not in used_syms]
    if not avail_op:
        return None
    op_sym = rng.choice(avail_op)
    op_sym2name[op_sym] = op_name
    used_syms.add(op_sym)

    # Draw 4 distinct content symbols (l0, l1, r0, r1)
    avail = [s for s in _SYMBOL_POOL if s not in used_syms]
    if len(avail) < 4:
        return None
    sym_sample = rng.sample(avail, 4)
    l0_sym, l1_sym, r0_sym, r1_sym = sym_sample
    # Don't permanently add to used_syms — content symbols can repeat across equations
    # in a pure_concat problem (the same symbol just means itself).

    lhs = {"l0": l0_sym, "l1": l1_sym, "op": op_sym, "r0": r0_sym, "r1": r1_sym}
    if op_name == "concat_fwd":
        actual_rhs = [l0_sym, l1_sym, r0_sym, r1_sym]
    else:  # concat_rev
        actual_rhs = [r0_sym, r1_sym, l0_sym, l1_sym]

    rhs = {"op": None, "sign_pos": None, "actual_rhs": actual_rhs}
    return {"lhs": lhs, "rhs": rhs}


# ---------------------------------------------------------------------------
# Symbol assignment helpers
# ---------------------------------------------------------------------------

def _assign_digit_symbol(
    digit: int,
    position: str | None,
    stats: DataStats,
    rng: random.Random,
    digit2sym: dict[int, str],
    sym2digit: dict[str, int],
    used_syms: set[str],
) -> bool:
    """Assign a symbol to `digit` if not already mapped. Returns True on success."""
    if digit in digit2sym:
        return True  # already assigned

    # Draw a fresh symbol
    if position and position in stats.pos_probs:
        dist = stats.pos_probs[position]
        sym = _weighted_sample_unused(dist, used_syms, rng)
    elif stats.global_sym_probs:
        sym = _weighted_sample_unused(stats.global_sym_probs, used_syms, rng)
    else:
        sym = None

    if sym is None:
        # Fallback: uniform from _SYMBOL_POOL
        avail = [s for s in _SYMBOL_POOL if s not in used_syms]
        if not avail:
            return False
        sym = rng.choice(avail)

    digit2sym[digit] = sym
    sym2digit[sym] = digit
    used_syms.add(sym)
    return True


def _assign_op_symbol(
    op_name: str,
    stats: DataStats,
    rng: random.Random,
    op_name2sym: dict[str, str],
    op_sym2name: dict[str, str],
    used_syms: set[str],
) -> str | None:
    """Assign an operator symbol to `op_name` if not already mapped."""
    if op_name in op_name2sym:
        return op_name2sym[op_name]

    # Sample from P(op_char | op_name)
    dist = stats.op_char_probs.get(op_name, {})
    sym = _weighted_sample_unused(dist, used_syms, rng)

    if sym is None:
        # Fallback: uniform from _SYMBOL_POOL
        avail = [s for s in _SYMBOL_POOL if s not in used_syms]
        if not avail:
            return None
        sym = rng.choice(avail)

    op_name2sym[op_name] = sym
    op_sym2name[sym] = op_name
    used_syms.add(sym)
    return sym


def _weighted_sample_unused(
    dist: dict[str, float],
    used: set[str],
    rng: random.Random,
) -> str | None:
    """Sample from `dist` skipping already-used symbols. Returns None if all used."""
    candidates = [s for s in dist if s not in used]
    if not candidates:
        return None
    weights = [dist[s] for s in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Strategy and op sampling
# ---------------------------------------------------------------------------

def _sample_strategy(
    stats: DataStats,
    rng: random.Random,
    include_pure_concat: bool = True,
) -> str:
    strats = [s for s in stats.strategy_probs if include_pure_concat or s != "pure_concat"]
    weights = [stats.strategy_probs[s] for s in strats]
    return rng.choices(strats, weights=weights, k=1)[0]


def _sample_op(stats: DataStats, rng: random.Random, *, concat: bool) -> str:
    """Sample an op_name from P(op_name), restricted to concat or arithmetic ops."""
    ops = stats.concat_ops if concat else stats.arith_ops
    if not ops:
        return "mul"
    weights = [stats.op_probs.get(op, 1e-6) for op in ops]
    return rng.choices(ops, weights=weights, k=1)[0]
