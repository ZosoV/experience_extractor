"""Equation-Symbolic Solver.

Entry point:
    uv run python src/solver_eq_symbolic.py [config.yaml]

Reads a .parquet with columns (id, prompt, answer), infers a latent program
for each puzzle, and writes solver_results.parquet.

Public re-exports (for eval_from_parquet.py and other consumers):
    OPERATIONS, SIGNED_OPS, OP_NAME_TO_IDX,
    _int_to_base_digits, _is_reversed_digit_mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import SolverConfig, load_config
from src.encoding import (
    _int_to_base_digits,
    _is_reversed_digit_mode,
    encode_answer,
    verify_program,
)
from src.heuristics import (
    concat_optin,
    op_glyph_priority,
    order_ops,
    sign_pattern_filter,
    tier_skip,
)
from src.op_index import OP_IDX_TO_NAME, OP_NAME_TO_IDX, SIGNED_OPS
from src.operations import (
    OPERATIONS,
    TIER0,
    TIER1,
    TIER2,
    TIER_FULL,
)
from src.parsing import (
    build_op_groups,
    content_symbols,
    extract_all_equations,
)

# ---------------------------------------------------------------------------
# Tier name → set mapping
# ---------------------------------------------------------------------------

_TIER_SETS: dict[str, set[str]] = {
    "tier0": TIER0,
    "tier1": TIER1,
    "tier2": TIER2,
    "full": TIER_FULL,
}

_TIER_LABEL: dict[str, str] = {
    "tier0": "tier0",
    "tier1": "tier1",
    "tier2": "tier2",
    "full": "full",
}


# ---------------------------------------------------------------------------
# Rust bridge
# ---------------------------------------------------------------------------

def _search(
    op_eqs: dict[str, list[tuple]],
    tier_cands: dict[str, list[str]],
    sorted_ops: list[str],
    n: int,
    mode: str,
    radix: int,
    digit_count: int,
    max_solutions: int = 1,
) -> list[tuple[list, list]]:
    """Call the Rust arithmetic_search core and return a list of (perm, ops_valid) tuples.

    Returns up to *max_solutions* distinct digit assignments. Returns an empty list when
    no assignment satisfies the constraints.

    Args:
        op_eqs:         {op_char: [encoded equation tuples]}
        tier_cands:     {op_char: [op names filtered to current tier]}
        sorted_ops:     op chars in MRV-first order (Heuristic #5)
        n:              number of content symbols
        mode:           'standard' | 'little_endian'
        radix:          positional base for arithmetic
        digit_count:    digit pool size (== radix for normal case)
        max_solutions:  stop after collecting this many digit assignments
    """
    from alice_sovler_helper import arithmetic_search  # type: ignore[import]

    mode_int = 0 if mode == "standard" else 2  # standard=0, little_endian=2

    groups = []
    for op_char in sorted_ops:
        cand_indices = [OP_NAME_TO_IDX[name] for name in tier_cands[op_char]]
        eqs_packed = []
        for l0, l1, r0, r1, res_syms, has_sign, res_len in op_eqs[op_char]:
            packed = [l0, l1, r0, r1, res_len, int(has_sign)] + list(res_syms)
            eqs_packed.append(packed)
        groups.append((cand_indices, eqs_packed))

    return arithmetic_search(n, False, groups, radix, digit_count, mode_int, max_solutions)


# ---------------------------------------------------------------------------
# Concat fast-path (spec §2.2)
# ---------------------------------------------------------------------------

def _try_concat(equations: list[dict]) -> str | None:
    """Check whether every equation with the query's operator is pure concat.

    Returns the answer string if the puzzle is solved by concatenation,
    or None to fall through to arithmetic search.
    """
    to_predict = equations[-1]
    op_of_interest = to_predict["lhs"]["op"]

    # Condition: no sign in query rhs AND exactly 4 rhs symbols.
    if to_predict["rhs"]["op"] is not None:
        return None
    if len(to_predict["rhs"]["actual_rhs"]) != 4:
        return None

    relevant = [eq for eq in equations if eq["lhs"]["op"] == op_of_interest]

    # Try forward concat: rhs == [l0, l1, r0, r1]
    fwd_ok = True
    for eq in relevant:
        lhs = eq["lhs"]
        expected = [lhs["l0"], lhs["l1"], lhs["r0"], lhs["r1"]]
        if eq["rhs"]["actual_rhs"] != expected:
            fwd_ok = False
            break
    if fwd_ok:
        q = to_predict["lhs"]
        return q["l0"] + q["l1"] + q["r0"] + q["r1"]

    # Try reverse concat: rhs == [r0, r1, l0, l1]
    rev_ok = True
    for eq in relevant:
        lhs = eq["lhs"]
        expected = [lhs["r0"], lhs["r1"], lhs["l0"], lhs["l1"]]
        if eq["rhs"]["actual_rhs"] != expected:
            rev_ok = False
            break
    if rev_ok:
        q = to_predict["lhs"]
        return q["r0"] + q["r1"] + q["l0"] + q["l1"]

    return None


# ---------------------------------------------------------------------------
# Arithmetic search (spec §2.4)
# ---------------------------------------------------------------------------

def _try_arithmetic(
    equations: list[dict],
    content_syms: list[str],
    sym2i: dict[str, int],
    gold_answer: str,
    config: SolverConfig,
) -> tuple[list[tuple[str, dict]], str | None]:
    """Run the tiered arithmetic search and return (solutions, search_reached).

    The to_predict equation (last in equations) is included as a constraint
    so the search is gold-conditioned.

    Collects up to config.max_solutions fully-verified programs. Each program is a
    distinct (digit mapping, mode, radix) combination. The search escalates in order:
      1. distinct digit mappings under 'standard' mode and the primary base
      2. 'little_endian' mode (same base)
      3. alternative base (when not force_base_10)

    search_reached reflects the furthest stage the loop advanced into:
      'digits'       — stayed within standard + primary base
      'little_endian'— advanced into little_endian (same base)
      'alt_base'     — advanced into a non-primary base

    Returns ([], None) when no solution is found, falling back to
    ([first_partial], None) when a partial candidate exists (encode_answer
    succeeded but verify failed), so the caller can surface it as wrong_answer.
    """
    base = len(content_syms)

    # Step A: Build operator groups (includes to_predict as a constraint).
    op_eqs = build_op_groups(equations, sym2i)

    # Step B: Per-op candidate lists.
    op_candidates: dict[str, list[str]] = {}
    for op_char, eqs in op_eqs.items():
        if "op_glyph_priority" not in config.disabled_heuristics:
            cands = op_glyph_priority(op_char)        # #3
        else:
            cands = list(OPERATIONS.keys())

        if "sign_pattern_filter" not in config.disabled_heuristics:
            cands = sign_pattern_filter(cands, eqs)   # #2
        if "concat_optin" not in config.disabled_heuristics:
            cands = concat_optin(cands, eqs)           # #4

        op_candidates[op_char] = cands

    # Step C: Op ordering.
    if "order_ops" not in config.disabled_heuristics:
        sorted_ops = order_ops(op_eqs, op_candidates)  # #5
    else:
        sorted_ops = list(op_eqs.keys())

    # Steps D/E: Tiered search loop.
    tiers_to_try = [(name, _TIER_SETS[name]) for name in config.tiers]

    # Collect verified solutions; dedup key = (frozen digits, mode, radix).
    solutions: list[tuple[str, dict]] = []
    seen: set[tuple] = set()

    # First partial candidate found (encode_answer succeeded but verify failed).
    # Returned as wrong_answer only after all options are exhausted.
    first_partial: tuple[str, dict] | None = None

    # Track the furthest escalation stage the loop advances into.
    search_reached: str | None = None

    for tier_name, tier_set in tiers_to_try:
        if len(solutions) >= config.max_solutions:
            break

        tier_cands = {
            op: [c for c in cands if c in tier_set]
            for op, cands in op_candidates.items()
        }

        if "tier_skip" not in config.disabled_heuristics and tier_skip(tier_cands):  # #9
            continue

        if config.force_base_10:
            radix_candidates = [(10, 10)]
        else:
            radix_candidates = [(base, base)]
            if base != 10:
                radix_candidates.append((10, 10))

        primary_radix = radix_candidates[0][0]

        for radix, digit_count in radix_candidates:
            if len(solutions) >= config.max_solutions:
                break

            for mode in ("standard", "little_endian"):
                if len(solutions) >= config.max_solutions:
                    break

                # Update search_reached to the furthest stage entered.
                if radix != primary_radix:
                    stage = "alt_base"
                elif mode == "little_endian":
                    stage = "little_endian"
                else:
                    stage = "digits"
                if search_reached != "alt_base":
                    if stage == "alt_base" or (
                        stage == "little_endian" and search_reached == "digits"
                    ) or search_reached is None:
                        search_reached = stage

                remaining = config.max_solutions - len(solutions)
                results = _search(
                    op_eqs, tier_cands, sorted_ops, base, mode, radix, digit_count,
                    max_solutions=remaining,
                )

                for perm, ops_valid_indices in results:
                    # Decode mapping: symbol -> digit.
                    mapping: dict[str, int] = {
                        sym: int(perm[sym2i[sym]]) for sym in content_syms
                    }

                    # Dedup: skip if we've already recorded this (mapping, mode, radix).
                    dedup_key = (frozenset(mapping.items()), mode, radix)
                    if dedup_key in seen:
                        continue

                    # Decode surviving op candidates per group.
                    ops_candidates_found: dict[str, list[str]] = {
                        op_char: [OP_IDX_TO_NAME[int(idx)] for idx in ops_valid_indices[i]]
                        for i, op_char in enumerate(sorted_ops)
                    }
                    ops_mapping: dict[str, str] = {
                        op_char: ops_candidates_found[op_char][0]
                        for op_char in sorted_ops
                    }

                    # Step F: Encode the query answer.
                    to_predict = equations[-1]
                    encoded = encode_answer(
                        mapping=mapping,
                        ops=ops_mapping,
                        op_char=to_predict["lhs"]["op"],
                        query_lhs=to_predict["lhs"],
                        mode=mode,
                        radix=radix,
                        answer_hint=gold_answer,
                    )
                    if encoded is None:
                        continue

                    answer_str, _ = encoded
                    strategy = mode  # "standard" | "little_endian"

                    details = {
                        "solver_strategy": strategy,
                        "ops_mapping": ops_mapping,
                        "digits_mapping": mapping,
                        "solver_tier": tier_name,
                        "solver_radix": radix,
                        "ops_candidates": ops_candidates_found,
                    }

                    # Full-program verification: every equation (examples + to_predict)
                    # must hold under the found program.  When verification fails, keep
                    # searching in case a later combination gives a fully consistent
                    # solution.  Only fall back to first partial when all options are
                    # exhausted.
                    if not verify_program(equations, mapping, ops_mapping, mode, radix):
                        if first_partial is None:
                            first_partial = (answer_str, {**details, "_verify_failed": True})
                        continue

                    seen.add(dedup_key)
                    solutions.append((answer_str, details))
                    if len(solutions) >= config.max_solutions:
                        break

    if solutions:
        return solutions, search_reached

    # No fully-verified solution found; return the first partial as wrong_answer.
    if first_partial is not None:
        return [first_partial], None
    return [], None


# ---------------------------------------------------------------------------
# Per-sample solver
# ---------------------------------------------------------------------------

def solve_sample(row: dict, config: SolverConfig) -> list[dict]:
    """Solve one puzzle row and return a list of result dicts (one per solution found).

    Each dict has a 'solution_count' key (same value across all dicts for this puzzle)
    and a 'search_reached' key indicating the furthest escalation stage entered.

    Concat puzzles → 1 dict (solution_count=1, search_reached=None).
    Arithmetic with N verified solutions → N dicts (solution_count=N).
    Wrong-answer / no-solution → 1 dict (solution_count=0, search_reached=None).
    """
    gold_answer = str(row["answer"]).strip()

    try:
        equations = extract_all_equations(row)
    except Exception as exc:
        return [_no_solution(equations=[], error=str(exc))]

    syms = content_symbols(equations)
    sym2i = {s: i for i, s in enumerate(syms)}

    # Concat fast-path.
    concat_ans = _try_concat(equations)
    if concat_ans is not None:
        return [{
            "solver_answer": concat_ans,
            "solver_correct": concat_ans == gold_answer,
            "solver_strategy": "pure_concat",
            "category": "concatenation",
            "ops_mapping": None,
            "digits_mapping": None,
            "solver_tier": None,
            "solver_radix": None,
            "ops_candidates": None,
            "equations": equations,
            "solution_count": 1,
            "search_reached": None,
        }]

    # Arithmetic search.
    solutions, search_reached = _try_arithmetic(equations, syms, sym2i, gold_answer, config)

    if not solutions:
        return [_no_solution(equations=equations)]

    solution_count = len(solutions)
    result_dicts: list[dict] = []
    for answer_str, details in solutions:
        verify_failed = details.pop("_verify_failed", False)
        result_dicts.append({
            "solver_answer": answer_str,
            # solver_correct=False for verify failures (wrong_answer labeling).
            "solver_correct": (answer_str == gold_answer) and not verify_failed,
            **details,
            "category": "arithmetic",
            "equations": equations,
            "solution_count": solution_count if not verify_failed else 0,
            "search_reached": search_reached if not verify_failed else None,
        })
    return result_dicts


def _no_solution(equations: list, error: str | None = None) -> dict:
    return {
        "solver_answer": None,
        "solver_correct": False,
        "solver_strategy": None,
        "category": None,
        "ops_mapping": None,
        "digits_mapping": None,
        "solver_tier": None,
        "solver_radix": None,
        "ops_candidates": None,
        "equations": equations,
        "solution_count": 0,
        "search_reached": None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _to_json(obj) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Equation-Symbolic Solver")
    parser.add_argument("config", nargs="?", default=None, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    input_path = _ROOT / config.input_parquet
    output_path = _ROOT / config.output_parquet

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    records = []
    solved = wrong = no_sol = 0

    for _, row in df.iterrows():
        result_list = solve_sample(row.to_dict(), config)
        gold = str(row["answer"]).strip()

        # Count per-puzzle outcome using the first result dict.
        first = result_list[0]
        solver_ans = first["solver_answer"]
        if solver_ans is None:
            no_sol += 1
        elif first["solver_correct"]:
            solved += 1
        else:
            wrong += 1

        # Emit one record per solution (multiple rows per puzzle when solutions > 1).
        for res in result_list:
            records.append({
                "id": row.get("id"),
                "prompt": row.get("prompt"),
                "answer": gold,
                "solver_answer": res["solver_answer"],
                "solver_correct": res["solver_correct"],
                "solver_strategy": res["solver_strategy"],
                "category": res["category"],
                "ops_mapping": _to_json(res["ops_mapping"]),
                "digits_mapping": _to_json(res["digits_mapping"]),
                "solver_tier": res["solver_tier"],
                "solver_radix": res["solver_radix"],
                "ops_candidates": _to_json(res["ops_candidates"]),
                "equations": _to_json(res["equations"]),
                "solution_count": res["solution_count"],
                "search_reached": res["search_reached"],
            })

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    total = solved + wrong + no_sol
    print(f"\n=== solver results ===")
    print(f"solved correctly : {solved}")
    print(f"wrong answer     : {wrong}")
    print(f"no solution      : {no_sol}")
    print(f"total            : {total}")
    print(f"\nwritten to {output_path}")


if __name__ == "__main__":
    main()
