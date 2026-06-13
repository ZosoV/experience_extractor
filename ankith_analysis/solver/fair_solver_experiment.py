from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import signal
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class RowTimeoutError(TimeoutError):
    pass


def _raise_row_timeout(signum: int, frame: object) -> None:
    raise RowTimeoutError


ALL_FUNCS = (
    "add",
    "add_m1",
    "add_p1",
    "add_p2",
    "mul",
    "mul_m1",
    "mul_p1",
    "absdiff",
    "absdiff_m1",
    "absdiff_m2",
    "absdiff_p1",
    "neg_absdiff",
    "sub_signed",
    "rsub_signed",
    "mod",
    "rmod",
    "fdiv",
    "gcd",
    "lcm",
    "concat_fwd",
    "concat_rev",
    "a2_plus_b",
)

CORE_FUNCS = (
    "add",
    "mul",
    "absdiff",
    "sub_signed",
    "rsub_signed",
    "concat_fwd",
    "concat_rev",
)


@dataclass(frozen=True)
class Equation:
    lhs: str
    rhs: str

    @property
    def op(self) -> str:
        return self.lhs[2]

    @property
    def left(self) -> str:
        return self.lhs[:2]

    @property
    def right(self) -> str:
        return self.lhs[3:]

    @property
    def operand_symbols(self) -> str:
        return self.left + self.right


@dataclass(frozen=True)
class LocalOption:
    op: str
    func: str
    assignment: tuple[tuple[str, int], ...]
    sign: str | None


def parse_prompt(prompt: str) -> tuple[list[Equation], str]:
    examples: list[Equation] = []
    for line in prompt.splitlines():
        if " = " in line:
            lhs, rhs = line.split(" = ", 1)
            examples.append(Equation(lhs, rhs))
    match = re.search(r"Now, determine the result for: (.*)$", prompt)
    if not match:
        raise ValueError("could not parse query")
    return examples, match.group(1)


def decode_digits(digits: Iterable[int], base: int, mode: str) -> int:
    ds = list(digits)
    if mode == "little_endian":
        return sum(digit * (base**i) for i, digit in enumerate(ds))
    return sum(digit * (base ** (len(ds) - 1 - i)) for i, digit in enumerate(ds))


def fixed_width_digits(value: int, width: int, base: int, mode: str) -> list[int] | None:
    if value < 0 or value >= base**width:
        return None
    ds = []
    n = value
    for _ in range(width):
        ds.append(n % base)
        n //= base
    if n:
        return None
    if mode == "standard":
        ds.reverse()
    return ds


def minimal_width(value: int, base: int) -> int:
    value = abs(value)
    if value == 0:
        return 1
    width = 0
    while value:
        width += 1
        value //= base
    return width


def eval_numeric(func: str, a: int, b: int) -> int | None:
    if func == "add":
        return a + b
    if func == "add_m1":
        return a + b - 1
    if func == "add_p1":
        return a + b + 1
    if func == "add_p2":
        return a + b + 2
    if func == "mul":
        return a * b
    if func == "mul_m1":
        return a * b - 1
    if func == "mul_p1":
        return a * b + 1
    if func == "absdiff":
        return abs(a - b)
    if func == "absdiff_m1":
        return abs(a - b) - 1
    if func == "absdiff_m2":
        return abs(a - b) - 2
    if func == "absdiff_p1":
        return abs(a - b) + 1
    if func == "neg_absdiff":
        return -abs(a - b)
    if func == "sub_signed":
        return a - b
    if func == "rsub_signed":
        return b - a
    if func == "mod":
        return None if b == 0 else a % b
    if func == "rmod":
        return None if a == 0 else b % a
    if func == "fdiv":
        return None if b == 0 else a // b
    if func == "gcd":
        return math.gcd(a, b)
    if func == "lcm":
        return 0 if a == 0 or b == 0 else abs(a * b) // math.gcd(a, b)
    if func == "a2_plus_b":
        return a * a + b
    raise KeyError(func)


def merge_digit_assignment(
    base_assignment: dict[str, int],
    chars: str,
    digits: Iterable[int],
) -> dict[str, int] | None:
    assignment = dict(base_assignment)
    used = {digit: char for char, digit in assignment.items()}
    for char, digit in zip(chars, digits, strict=True):
        old = assignment.get(char)
        if old is not None:
            if old != digit:
                return None
            continue
        other = used.get(digit)
        if other is not None and other != char:
            return None
        assignment[char] = digit
        used[digit] = char
    return assignment


def local_options(
    eq: Equation,
    funcs: tuple[str, ...],
    base: int,
    mode: str,
    canonical_examples: bool,
    no_operand_leading_zero: bool,
    option_cap: int,
) -> list[LocalOption]:
    symbols = sorted(set(eq.operand_symbols))
    options: list[LocalOption] = []
    for digits in itertools.permutations(range(base), len(symbols)):
        lhs_assignment = dict(zip(symbols, digits, strict=True))
        left_digits = [lhs_assignment[char] for char in eq.left]
        right_digits = [lhs_assignment[char] for char in eq.right]
        if no_operand_leading_zero:
            if mode == "standard" and (left_digits[0] == 0 or right_digits[0] == 0):
                continue
            if mode == "little_endian" and (left_digits[-1] == 0 or right_digits[-1] == 0):
                continue
        a = decode_digits(left_digits, base, mode)
        b = decode_digits(right_digits, base, mode)
        for func in funcs:
            sign_interpretations = [(None, eq.rhs)]
            if len(eq.rhs) > 1:
                sign_interpretations.append((eq.rhs[0], eq.rhs[1:]))
            if func == "concat_fwd":
                result_digits = left_digits + right_digits
                for sign, rhs_digits in sign_interpretations[:1]:
                    if len(rhs_digits) != len(result_digits):
                        continue
                    merged = merge_digit_assignment(lhs_assignment, rhs_digits, result_digits)
                    if merged is not None:
                        options.append(LocalOption(eq.op, func, tuple(sorted(merged.items())), sign))
                        if len(options) >= option_cap:
                            return options
                continue
            if func == "concat_rev":
                result_digits = right_digits + left_digits
                for sign, rhs_digits in sign_interpretations[:1]:
                    if len(rhs_digits) != len(result_digits):
                        continue
                    merged = merge_digit_assignment(lhs_assignment, rhs_digits, result_digits)
                    if merged is not None:
                        options.append(LocalOption(eq.op, func, tuple(sorted(merged.items())), sign))
                        if len(options) >= option_cap:
                            return options
                continue

            value = eval_numeric(func, a, b)
            if value is None:
                continue
            for sign, rhs_digits in sign_interpretations:
                if sign is not None and sign in lhs_assignment:
                    continue
                if sign is None and value < 0:
                    continue
                if sign is not None and value >= 0:
                    continue
                width = len(rhs_digits)
                if canonical_examples and width != minimal_width(value, base):
                    continue
                result_digits = fixed_width_digits(abs(value), width, base, mode)
                if result_digits is None:
                    continue
                merged = merge_digit_assignment(lhs_assignment, rhs_digits, result_digits)
                if merged is None:
                    continue
                if sign is not None and sign in merged:
                    continue
                options.append(LocalOption(eq.op, func, tuple(sorted(merged.items())), sign))
                if len(options) >= option_cap:
                    return options
    return options


def assignment_consistent(
    current: dict[str, int],
    option_assignment: tuple[tuple[str, int], ...],
    base: int,
) -> dict[str, int] | None:
    merged = dict(current)
    used = {digit: char for char, digit in merged.items()}
    for char, digit in option_assignment:
        old = merged.get(char)
        if old is not None:
            if old != digit:
                return None
            continue
        if digit >= base:
            return None
        other = used.get(digit)
        if other is not None and other != char:
            return None
        merged[char] = digit
        used[digit] = char
    return merged


def digit_strings(
    digit_options: list[int],
    assignment: dict[str, int],
    alphabet: tuple[str, ...],
    use_palette: bool,
    cap: int,
) -> set[str]:
    inv: dict[int, str] = {}
    for char, digit in assignment.items():
        inv[digit] = char
    unknown_digits = sorted({digit for digit in digit_options if digit not in inv})
    if not unknown_digits:
        return {"".join(inv[digit] for digit in digit_options)}
    if not use_palette:
        return set()
    available = [char for char in alphabet if char not in assignment]
    out: set[str] = set()
    for chars in itertools.permutations(available, len(unknown_digits)):
        local = dict(inv)
        local.update(zip(unknown_digits, chars, strict=True))
        out.add("".join(local[digit] for digit in digit_options))
        if len(out) >= cap:
            break
    return out


def query_candidates_for_model(
    query_lhs: str,
    funcs: tuple[str, ...],
    base: int,
    mode: str,
    assignment: dict[str, int],
    op_assignment: dict[str, str],
    sign: str | None,
    alphabet: tuple[str, ...],
    use_palette: bool,
    padded_outputs: bool,
    no_operand_leading_zero: bool,
    cap: int,
) -> set[str]:
    qop = query_lhs[2]
    qsymbols = sorted(set(query_lhs[:2] + query_lhs[3:]) - set(assignment))
    remaining_digits = [digit for digit in range(base) if digit not in set(assignment.values())]
    out: set[str] = set()
    for extra_digits in itertools.permutations(remaining_digits, len(qsymbols)):
        local_assignment = dict(assignment)
        local_assignment.update(zip(qsymbols, extra_digits, strict=True))
        left_digits = [local_assignment[char] for char in query_lhs[:2]]
        right_digits = [local_assignment[char] for char in query_lhs[3:]]
        if no_operand_leading_zero:
            if mode == "standard" and (left_digits[0] == 0 or right_digits[0] == 0):
                continue
            if mode == "little_endian" and (left_digits[-1] == 0 or right_digits[-1] == 0):
                continue
        a = decode_digits(left_digits, base, mode)
        b = decode_digits(right_digits, base, mode)
        qfuncs = (op_assignment[qop],) if qop in op_assignment else funcs
        for func in qfuncs:
            if func == "concat_fwd":
                for s in digit_strings(left_digits + right_digits, local_assignment, alphabet, use_palette, cap):
                    out.add(s)
                    if len(out) >= cap:
                        return out
                continue
            if func == "concat_rev":
                for s in digit_strings(right_digits + left_digits, local_assignment, alphabet, use_palette, cap):
                    out.add(s)
                    if len(out) >= cap:
                        return out
                continue
            value = eval_numeric(func, a, b)
            if value is None:
                continue
            widths = range(minimal_width(value, base), 5) if padded_outputs else (minimal_width(value, base),)
            for width in widths:
                result_digits = fixed_width_digits(abs(value), width, base, mode)
                if result_digits is None:
                    continue
                bodies = digit_strings(result_digits, local_assignment, alphabet, use_palette, cap)
                if value < 0:
                    if sign is not None:
                        sign_candidates = [sign]
                    elif use_palette:
                        sign_candidates = [char for char in alphabet if char not in local_assignment]
                    else:
                        sign_candidates = ["-"] if "-" not in local_assignment else []
                    for sign_char in sign_candidates:
                        for body in bodies:
                            out.add(sign_char + body)
                            if len(out) >= cap:
                                return out
                else:
                    out.update(bodies)
                    if len(out) >= cap:
                        return out
    return out


def solve_prompt(
    examples: list[Equation],
    query_lhs: str,
    funcs: tuple[str, ...],
    bases: tuple[int, ...],
    modes: tuple[str, ...],
    alphabet: tuple[str, ...],
    use_palette: bool,
    padded_outputs: bool,
    canonical_examples: bool,
    no_operand_leading_zero: bool,
    local_option_cap: int,
    model_cap: int,
    candidate_cap: int,
) -> tuple[set[str], dict[str, int | bool]]:
    candidates: set[str] = set()
    total_models = 0
    search_capped = False
    local_capped = False
    states_tried = 0

    for base in bases:
        for mode in modes:
            option_lists = [
                local_options(
                    eq,
                    funcs,
                    base,
                    mode,
                    canonical_examples,
                    no_operand_leading_zero,
                    local_option_cap,
                )
                for eq in examples
            ]
            if any(len(options) >= local_option_cap for options in option_lists):
                local_capped = True
            if any(not options for options in option_lists):
                continue
            order = sorted(range(len(examples)), key=lambda idx: len(option_lists[idx]))
            states_tried += sum(len(options) for options in option_lists)

            def backtrack(
                pos: int,
                assignment: dict[str, int],
                op_assignment: dict[str, str],
                sign: str | None,
            ) -> None:
                nonlocal total_models, search_capped, candidates
                if search_capped or len(candidates) >= candidate_cap:
                    search_capped = True
                    return
                if total_models >= model_cap:
                    search_capped = True
                    return
                if pos == len(order):
                    total_models += 1
                    new_candidates = query_candidates_for_model(
                        query_lhs,
                        funcs,
                        base,
                        mode,
                        assignment,
                        op_assignment,
                        sign,
                        alphabet,
                        use_palette,
                        padded_outputs,
                        no_operand_leading_zero,
                        candidate_cap - len(candidates),
                    )
                    candidates.update(new_candidates)
                    if len(candidates) >= candidate_cap:
                        search_capped = True
                    return

                eq_idx = order[pos]
                for option in option_lists[eq_idx]:
                    assigned_func = op_assignment.get(option.op)
                    if assigned_func is not None and assigned_func != option.func:
                        continue
                    next_sign = sign
                    if option.sign is not None:
                        if option.sign in assignment:
                            continue
                        if sign is not None and sign != option.sign:
                            continue
                        next_sign = option.sign
                    merged = assignment_consistent(assignment, option.assignment, base)
                    if merged is None:
                        continue
                    if next_sign is not None and next_sign in merged:
                        continue
                    next_ops = op_assignment if assigned_func is not None else {**op_assignment, option.op: option.func}
                    backtrack(pos + 1, merged, next_ops, next_sign)
                    if search_capped:
                        return

            backtrack(0, {}, {}, None)
    diagnostics = {
        "models": total_models,
        "capped": search_capped or local_capped,
        "candidate_count": len(candidates),
        "states_tried": states_tried,
        "local_capped": local_capped,
    }
    return candidates, diagnostics


def global_alphabet(df: pd.DataFrame) -> tuple[str, ...]:
    chars: set[str] = set()
    for _, row in df.iterrows():
        examples, query = parse_prompt(row.prompt)
        for eq in examples:
            chars.update(eq.lhs)
            chars.update(eq.rhs)
        chars.update(query)
        chars.update(row.answer)
    return tuple(sorted(chars))


def true_answer_visible(prompt: str, answer: str) -> bool:
    examples, query = parse_prompt(prompt)
    chars: set[str] = set(query)
    for eq in examples:
        chars.update(eq.lhs)
        chars.update(eq.rhs)
    return set(answer).issubset(chars)


def direct_prompt(row: pd.Series) -> bool:
    examples, query_lhs = parse_prompt(row.prompt)
    example_ops = {eq.op for eq in examples}
    example_digits = set("".join(eq.operand_symbols for eq in examples))
    query_digits = set(query_lhs[:2] + query_lhs[3:])
    return (
        query_lhs[2] in example_ops
        and query_digits.issubset(example_digits)
        and true_answer_visible(row.prompt, row.answer)
    )


def run_variant(df: pd.DataFrame, args: argparse.Namespace, funcs: tuple[str, ...], name: str) -> pd.DataFrame:
    alphabet = global_alphabet(df)
    rows = []
    for index, row in df.iterrows():
        if args.limit is not None and index >= args.limit:
            break
        if args.filter_direct and not direct_prompt(row):
            continue
        examples, query_lhs = parse_prompt(row.prompt)
        timed_out = False
        old_handler = signal.signal(signal.SIGALRM, _raise_row_timeout)
        try:
            if args.row_timeout > 0:
                signal.setitimer(signal.ITIMER_REAL, args.row_timeout)
            candidates, diag = solve_prompt(
                examples,
                query_lhs,
                funcs,
                args.bases,
                args.modes,
                alphabet,
                args.palette,
                args.padded_outputs,
                args.canonical_examples,
                args.no_operand_leading_zero,
                args.local_option_cap,
                args.model_cap,
                args.candidate_cap,
            )
        except RowTimeoutError:
            timed_out = True
            candidates = set()
            diag = {"capped": True, "local_capped": True, "models": 0, "states_tried": 0}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        answer = row.answer
        rows.append(
            {
                "id": row.id,
                "variant": name,
                "category": row.solver_category if not pd.isna(row.solver_category) else "None",
                "solver_correct": bool(row.solver_correct),
                "answer": answer,
                "candidate_count": len(candidates),
                "contains_answer": answer in candidates,
                "unique": len(candidates) == 1,
                "unique_correct": len(candidates) == 1 and answer in candidates,
                "capped": bool(diag["capped"]),
                "timed_out": timed_out,
                "models": diag["models"],
                "states_tried": diag["states_tried"],
                "local_capped": bool(diag["local_capped"]),
                "answer_visible": true_answer_visible(row.prompt, answer),
            }
        )
        if args.progress and (index + 1) % args.progress == 0:
            print(f"{name}: processed {index + 1} rows", flush=True)
    return pd.DataFrame(rows)


def summarize(result: pd.DataFrame) -> str:
    lines = []
    lines.append(f"rows: {len(result)}")
    for col in [
        "unique_correct",
        "contains_answer",
        "unique",
        "capped",
        "local_capped",
        "timed_out",
        "answer_visible",
    ]:
        lines.append(f"{col}: {int(result[col].sum())} ({result[col].mean():.1%})")
    lines.append("candidate_count:")
    lines.append(result["candidate_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    lines.append("")
    lines.append("by category:")
    by_cat = result.groupby("category", dropna=False).agg(
        rows=("id", "count"),
        unique_correct=("unique_correct", "sum"),
        contains=("contains_answer", "sum"),
        unique=("unique", "sum"),
        capped=("capped", "sum"),
        local_capped=("local_capped", "sum"),
        timed_out=("timed_out", "sum"),
        visible=("answer_visible", "sum"),
        median_candidates=("candidate_count", "median"),
    )
    lines.append(by_cat.to_string())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.parquet")
    parser.add_argument("--outdir", default="research/outputs")
    parser.add_argument("--func-set", choices=["core", "all", "both"], default="core")
    parser.add_argument("--bases", type=int, nargs="+", default=[10])
    parser.add_argument("--modes", nargs="+", default=["standard", "little_endian"])
    parser.add_argument("--palette", action="store_true")
    parser.add_argument("--padded-outputs", action="store_true")
    parser.add_argument("--canonical-examples", action="store_true")
    parser.add_argument("--no-operand-leading-zero", action="store_true")
    parser.add_argument("--local-option-cap", type=int, default=1000)
    parser.add_argument("--model-cap", type=int, default=2000)
    parser.add_argument("--candidate-cap", type=int, default=2000)
    parser.add_argument("--row-timeout", type=float, default=3.0)
    parser.add_argument("--filter-direct", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress", type=int, default=100)
    args = parser.parse_args()
    args.bases = tuple(args.bases)
    args.modes = tuple(args.modes)

    df = pd.read_parquet(args.dataset)
    variants = []
    if args.func_set in {"core", "both"}:
        variants.append(("core", CORE_FUNCS))
    if args.func_set in {"all", "both"}:
        variants.append(("all", ALL_FUNCS))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for name, funcs in variants:
        result = run_variant(df, args, funcs, name)
        all_results.append(result)
        suffix = [
            name,
            "direct" if args.filter_direct else "allrows",
            "palette" if args.palette else "observed",
            "padded" if args.padded_outputs else "canonical",
            "canonical_examples" if args.canonical_examples else "flex_examples",
            "noleadz" if args.no_operand_leading_zero else "leadok",
            "b" + "-".join(map(str, args.bases)),
            "m" + "-".join(args.modes),
        ]
        stem = "_".join(suffix)
        result.to_csv(outdir / f"{stem}.csv", index=False)
        (outdir / f"{stem}.summary.txt").write_text(summarize(result) + "\n", encoding="utf-8")
        print(f"\n=== {stem} ===")
        print(summarize(result))
    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(outdir / "combined.csv", index=False)


if __name__ == "__main__":
    main()
