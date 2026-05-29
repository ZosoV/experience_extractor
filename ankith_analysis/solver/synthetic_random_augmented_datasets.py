from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import fair_solver_experiment as fair
except ImportError:  # pragma: no cover - supports `python -m research...`
    from research import fair_solver_experiment as fair


QUERY_PREFIX = "Now, determine the result for: "
DEFAULT_AUGMENTED_OUTDIR = Path("research/outputs/synthetic_random_augmented")
DEFAULT_CANDIDATE_OUTDIR = Path("research/outputs/synthetic_random_candidates")
DEFAULT_ORIGINAL_VALIDATION_DIR = Path("research/outputs/original_candidate_validation")
DEFAULT_PREFIX_AUGMENTED_DIR = Path("research/outputs/prefix_example_augmented")
CANDIDATE_MODES = ("standard", "little_endian")
VERSIONS = ("v1_original_distribution", "v2_uniform")

CANDIDATE_COLUMNS = [
    "row_index",
    "id",
    "source_id",
    "example_count",
    "target_example_index",
    "candidate_rank",
    "function_candidate_rank",
    "candidate_answer",
    "matches_answer",
    "mode",
    "digit_mapping",
    "operator_mapping",
    "sign_symbol",
    "query_function",
    "query_left_value",
    "query_right_value",
    "query_numeric_answer",
    "output_width",
    "padded_output",
    "query_operator_seen",
    "used_metadata_seed",
    "candidate_capped",
    "local_capped",
    "state_capped",
    "timed_out",
    "selection_version",
    "selection_policy",
    "selection_example_count",
    "selection_target_per_prompt",
    "selection_rank",
    "selection_requested_function",
    "selection_random_seed",
    "selection_random_value",
]


def stable_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_random_value(seed: int, *parts: object) -> float:
    payload = "|".join(str(part) for part in (seed, *parts)).encode("utf-8")
    raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return raw / float(2**64)


def json_dumps_mapping(mapping: dict[str, object]) -> str:
    return json.dumps(mapping, sort_keys=True)


def encode_numeric_value(value: int, digit_to_char: dict[int, str], mode: str) -> str:
    digits = fair.fixed_width_digits(abs(value), fair.minimal_width(value, 10), 10, mode)
    if digits is None:
        raise ValueError(f"could not encode value: {value}")
    body = "".join(digit_to_char[digit] for digit in digits)
    return f"-{body}" if value < 0 else body


def decode_operand(chars: str, digit_mapping: dict[str, int], mode: str) -> int:
    return fair.decode_digits([digit_mapping[char] for char in chars], 10, mode)


def render_answer(
    lhs: str,
    func: str,
    digit_mapping: dict[str, int],
    digit_to_char: dict[int, str],
    mode: str,
) -> tuple[str | None, int | None, int, int]:
    left_value = decode_operand(lhs[:2], digit_mapping, mode)
    right_value = decode_operand(lhs[3:], digit_mapping, mode)
    if func == "concat_fwd":
        return lhs[:2] + lhs[3:], None, left_value, right_value
    if func == "concat_rev":
        return lhs[3:] + lhs[:2], None, left_value, right_value
    value = fair.eval_numeric(func, left_value, right_value)
    if value is None:
        return None, None, left_value, right_value
    return encode_numeric_value(value, digit_to_char, mode), value, left_value, right_value


def prompt_digit_operator_counts(prompt: str) -> tuple[int, int]:
    examples, query_lhs = fair.parse_prompt(prompt)
    digit_chars = set(query_lhs[:2] + query_lhs[3:])
    operator_chars = {query_lhs[2]}
    for eq in examples:
        digit_chars.update(eq.left + eq.right)
        operator_chars.add(eq.op)
    return len(digit_chars), len(operator_chars)


def reference_counts(prefix_augmented_dir: Path, example_count: int) -> list[dict[str, object]]:
    frame = pd.read_parquet(
        prefix_augmented_dir / f"dataset_prefix_examples_{example_count}.parquet",
        columns=["id", "prompt"],
    )
    rows = []
    for _, row in frame.iterrows():
        digit_count, operator_count = prompt_digit_operator_counts(str(row.prompt))
        rows.append(
            {
                "reference_id": str(row.id),
                "digit_char_count": digit_count,
                "operator_char_count": operator_count,
            }
        )
    return rows


def answer_supported_operator_weights(
    original_validation_dir: Path,
    operators: tuple[str, ...],
    smoothing: float,
) -> dict[str, float]:
    candidates = pd.read_parquet(
        original_validation_dir / "original_dataset_for_candidates_candidate_solutions.parquet",
        columns=["source_id", "query_function", "matches_answer"],
    )
    matches = candidates[candidates["matches_answer"]].drop_duplicates(
        ["source_id", "query_function"]
    )
    by_source = matches.groupby("source_id")["query_function"].apply(list)
    counts: Counter[str] = Counter({func: smoothing for func in operators})
    for funcs in by_source:
        valid = [func for func in funcs if func in counts]
        if not valid:
            continue
        share = 1.0 / len(valid)
        for func in valid:
            counts[func] += share
    return {func: float(counts[func]) for func in operators}


def weighted_slots_by_capacity(
    available_counts: dict[str, int],
    k: int,
    weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    remaining = {func: count for func, count in available_counts.items() if count > 0}
    slots: list[str] = []
    while len(slots) < k and remaining:
        available = sorted(remaining)
        local_weights = [max(weights.get(func, 0.0), 0.0) for func in available]
        if not any(local_weights):
            local_weights = [1.0] * len(available)
        chosen = rng.choices(available, weights=local_weights, k=1)[0]
        slots.append(chosen)
        remaining[chosen] -= 1
        if remaining[chosen] <= 0:
            del remaining[chosen]
    return slots


def distribute_references(
    refs: list[dict[str, object]],
    rows_per_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    return [dict(rng.choice(refs)) for _ in range(rows_per_count)]


def sample_mapping(
    alphabet: tuple[str, ...],
    rng: random.Random,
) -> tuple[dict[str, int], dict[int, str], dict[str, str]]:
    digit_chars = rng.sample(list(alphabet), 10)
    digit_values = list(range(10))
    rng.shuffle(digit_values)
    digit_mapping = dict(zip(digit_chars, digit_values, strict=True))
    digit_to_char = {digit: char for char, digit in digit_mapping.items()}

    op_chars = rng.sample(list(alphabet), len(fair.ALL_FUNCS))
    funcs = list(fair.ALL_FUNCS)
    rng.shuffle(funcs)
    operator_mapping = dict(zip(op_chars, funcs, strict=True))
    return digit_mapping, digit_to_char, operator_mapping


def leading_zero_ok(chars: str, digit_mapping: dict[str, int], mode: str) -> bool:
    digits = [digit_mapping[char] for char in chars]
    if mode == "little_endian":
        return digits[-1] != 0
    return digits[0] != 0


def make_operand(
    digit_chars: tuple[str, ...],
    digit_mapping: dict[str, int],
    mode: str,
    rng: random.Random,
) -> str:
    for _ in range(100):
        operand = "".join(rng.choice(digit_chars) for _ in range(2))
        if leading_zero_ok(operand, digit_mapping, mode):
            return operand
    raise ValueError("could not sample non-leading-zero operand")


def make_lhs(
    op_char: str,
    digit_chars: tuple[str, ...],
    digit_mapping: dict[str, int],
    mode: str,
    rng: random.Random,
) -> str:
    left = make_operand(digit_chars, digit_mapping, mode, rng)
    right = make_operand(digit_chars, digit_mapping, mode, rng)
    return f"{left}{op_char}{right}"


def visible_digit_body(answer: str) -> str:
    return answer[1:] if answer.startswith("-") else answer


def make_equations(
    example_count: int,
    digit_chars: tuple[str, ...],
    op_chars: tuple[str, ...],
    digit_mapping: dict[str, int],
    digit_to_char: dict[int, str],
    operator_mapping: dict[str, str],
    true_mode: str,
    rng: random.Random,
) -> tuple[list[fair.Equation], str, str, str]:
    total_equations = example_count + 1
    op_sequence = list(op_chars)
    while len(op_sequence) < total_equations:
        op_sequence.append(rng.choice(op_chars))
    rng.shuffle(op_sequence)

    required_digits = set(digit_chars)
    for _ in range(2000):
        examples: list[fair.Equation] = []
        lhs_values: list[str] = []
        used_digits: set[str] = set()
        ok = True
        for eq_index, op_char in enumerate(op_sequence):
            lhs = make_lhs(op_char, digit_chars, digit_mapping, true_mode, rng)
            used_digits.update(lhs[:2] + lhs[3:])
            answer, _, _, _ = render_answer(
                lhs,
                operator_mapping[op_char],
                digit_mapping,
                digit_to_char,
                true_mode,
            )
            if answer is None:
                ok = False
                break
            lhs_values.append(lhs)
            if eq_index < example_count:
                if not set(visible_digit_body(answer)).issubset(required_digits):
                    ok = False
                    break
                examples.append(fair.Equation(lhs, answer))
        if ok and used_digits == required_digits:
            query_lhs = lhs_values[-1]
            query_answer, _, _, _ = render_answer(
                query_lhs,
                operator_mapping[query_lhs[2]],
                digit_mapping,
                digit_to_char,
                true_mode,
            )
            if query_answer is not None:
                return examples, query_lhs, query_answer, operator_mapping[query_lhs[2]]

    raise ValueError("could not construct equations with requested character counts")


def render_prompt(examples: list[fair.Equation], query_lhs: str) -> str:
    if not examples:
        return "\n".join(
            [
                "In Alice's Wonderland, a secret set of transformation rules is applied to an equation:",
                "",
                f"{QUERY_PREFIX}{query_lhs}",
            ]
        )

    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:",
    ]
    lines.extend(f"{eq.lhs} = {eq.rhs}" for eq in examples)
    lines.append(f"{QUERY_PREFIX}{query_lhs}")
    return "\n".join(lines)


def make_synthetic_row(
    row_index: int,
    example_count: int,
    reference: dict[str, object],
    alphabet: tuple[str, ...],
    original_prompts: set[str],
    existing_prompts: set[str],
    base_seed: int,
) -> dict[str, object]:
    row_seed = stable_seed("synthetic-row", example_count, row_index, base_seed=base_seed)
    rng = random.Random(row_seed)
    digit_target = int(reference["digit_char_count"])
    operator_target = int(reference["operator_char_count"])
    if operator_target > example_count + 1:
        raise ValueError(f"operator target {operator_target} is impossible for {example_count} examples")

    for attempt in range(5000):
        local_seed = stable_seed("attempt", attempt, base_seed=row_seed)
        local_rng = random.Random(local_seed)
        digit_mapping, digit_to_char, operator_mapping = sample_mapping(alphabet, local_rng)
        digit_chars = tuple(local_rng.sample(list(digit_mapping), digit_target))
        if digit_target == 1 and digit_mapping[digit_chars[0]] == 0:
            continue
        op_chars = tuple(local_rng.sample(list(operator_mapping), operator_target))
        true_mode = local_rng.choice(CANDIDATE_MODES)
        try:
            examples, query_lhs, answer, true_query_function = make_equations(
                example_count,
                digit_chars,
                op_chars,
                digit_mapping,
                digit_to_char,
                operator_mapping,
                true_mode,
                local_rng,
            )
        except ValueError:
            continue
        prompt = render_prompt(examples, query_lhs)
        if prompt in existing_prompts or prompt in original_prompts:
            continue
        actual_digit_count, actual_operator_count = prompt_digit_operator_counts(prompt)
        if actual_digit_count != digit_target or actual_operator_count != operator_target:
            continue
        existing_prompts.add(prompt)
        source_id = f"synthetic_random_{example_count}_{row_index:04d}"
        return {
            "id": source_id,
            "source_id": source_id,
            "source_row_index": row_index,
            "prompt": prompt,
            "prompt_text": prompt,
            "answer": answer,
            "example_count": example_count,
            "target_example_index": example_count,
            "target_example_lhs": query_lhs,
            "target_example_rhs": answer,
            "query_lhs": query_lhs,
            "true_query_function": true_query_function,
            "true_mode": true_mode,
            "digit_mapping": json_dumps_mapping(digit_mapping),
            "operator_mapping": json_dumps_mapping(operator_mapping),
            "digit_chars": "".join(sorted(digit_chars)),
            "operator_chars": "".join(sorted(op_chars)),
            "digit_char_count": actual_digit_count,
            "operator_char_count": actual_operator_count,
            "reference_id": str(reference["reference_id"]),
            "reference_digit_char_count": digit_target,
            "reference_operator_char_count": operator_target,
            "augmentation_mode": "synthetic_random_mapping",
            "seed": row_seed,
        }

    raise RuntimeError(
        f"could not build synthetic row {row_index} for example_count={example_count} "
        f"with digit_count={digit_target}, operator_count={operator_target}"
    )


def make_candidate_pool(row_index: int, row: pd.Series) -> list[dict[str, object]]:
    digit_mapping = json.loads(str(row.digit_mapping))
    operator_mapping = json.loads(str(row.operator_mapping))
    digit_to_char = {int(digit): char for char, digit in digit_mapping.items()}
    query_lhs = str(row.query_lhs)
    answer = str(row.answer)
    example_ops = {eq.op for eq in fair.parse_prompt(str(row.prompt_text))[0]}
    true_visible_ops = {op: operator_mapping[op] for op in sorted(example_ops) if op in operator_mapping}

    rows: list[dict[str, object]] = []
    candidate_rank = 1
    per_function_rank: Counter[str] = Counter()
    for func in fair.ALL_FUNCS:
        for mode in CANDIDATE_MODES:
            candidate_answer, numeric_answer, left_value, right_value = render_answer(
                query_lhs,
                func,
                digit_mapping,
                digit_to_char,
                mode,
            )
            if candidate_answer is None:
                continue
            per_function_rank[func] += 1
            op_map = dict(true_visible_ops)
            op_map[query_lhs[2]] = func
            rows.append(
                {
                    "row_index": row_index,
                    "id": str(row.id),
                    "source_id": str(row.source_id),
                    "example_count": int(row.example_count),
                    "target_example_index": int(row.target_example_index),
                    "candidate_rank": candidate_rank,
                    "function_candidate_rank": per_function_rank[func],
                    "candidate_answer": candidate_answer,
                    "matches_answer": candidate_answer == answer,
                    "mode": mode,
                    "digit_mapping": json_dumps_mapping(digit_mapping),
                    "operator_mapping": json_dumps_mapping(op_map),
                    "sign_symbol": "-" if candidate_answer.startswith("-") else None,
                    "query_function": func,
                    "query_left_value": left_value,
                    "query_right_value": right_value,
                    "query_numeric_answer": float(numeric_answer) if numeric_answer is not None else None,
                    "output_width": len(visible_digit_body(candidate_answer)),
                    "padded_output": False,
                    "query_operator_seen": query_lhs[2] in example_ops,
                    "used_metadata_seed": False,
                    "candidate_capped": False,
                    "local_capped": False,
                    "state_capped": False,
                    "timed_out": False,
                }
            )
            candidate_rank += 1
    return rows


def selection_target(example_count: int) -> int:
    targets = {
        0: len(fair.ALL_FUNCS) * len(CANDIDATE_MODES),
        1: 19,
        2: 9,
        3: 5,
    }
    try:
        return targets[example_count]
    except KeyError as exc:
        raise ValueError(example_count) from exc


def select_candidates(
    pool: list[dict[str, object]],
    example_count: int,
    version: str,
    operator_weights: dict[str, float],
    seed: int,
) -> list[dict[str, object]]:
    target = selection_target(example_count)
    rng = random.Random(seed)
    if example_count == 0:
        slots = [func for func in fair.ALL_FUNCS for _ in CANDIDATE_MODES]
        policy = "all_operators_two_modes"
    else:
        counts = Counter(str(row["query_function"]) for row in pool)
        if version == "v1_original_distribution":
            slots = weighted_slots_by_capacity(counts, target, operator_weights, rng)
            policy = "answer_supported_original_operator_distribution_random"
        elif version == "v2_uniform":
            uniform = {func: 1.0 for func in fair.ALL_FUNCS}
            slots = weighted_slots_by_capacity(counts, target, uniform, rng)
            policy = "uniform_operator_distribution_random"
        else:
            raise ValueError(version)

    pool_by_func: dict[str, list[dict[str, object]]] = {}
    for candidate in pool:
        pool_by_func.setdefault(str(candidate["query_function"]), []).append(candidate)

    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[object, ...]] = set()
    for slot_index, func in enumerate(slots, start=1):
        available = [
            candidate
            for candidate in pool_by_func.get(func, [])
            if (
                candidate["candidate_rank"],
                candidate["function_candidate_rank"],
                candidate["mode"],
            )
            not in selected_keys
        ]
        if not available:
            continue
        available.sort(
            key=lambda candidate: stable_random_value(
                seed,
                slot_index,
                candidate["candidate_rank"],
                candidate["function_candidate_rank"],
                candidate["candidate_answer"],
            )
        )
        picked = dict(available[0])
        selected_keys.add(
            (
                picked["candidate_rank"],
                picked["function_candidate_rank"],
                picked["mode"],
            )
        )
        picked["selection_version"] = version
        picked["selection_policy"] = policy
        picked["selection_example_count"] = example_count
        picked["selection_target_per_prompt"] = target
        picked["selection_rank"] = len(selected) + 1
        picked["selection_requested_function"] = func
        picked["selection_random_seed"] = seed
        picked["selection_random_value"] = stable_random_value(
            seed,
            "selected",
            slot_index,
            picked["candidate_rank"],
            picked["function_candidate_rank"],
            picked["candidate_answer"],
        )
        selected.append(picked)

    return selected


def write_frame(frame: pd.DataFrame, path: Path, write_csv: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    if write_csv:
        frame.to_csv(path.with_suffix(".csv"), index=False)


def generate(args: argparse.Namespace) -> dict[str, object]:
    source_df = pd.read_parquet(args.dataset)
    alphabet = fair.global_alphabet(source_df)
    original_prompts = set(source_df["prompt"].astype(str))
    augmented_outdir = Path(args.augmented_outdir)
    candidate_outdir = Path(args.candidate_outdir)
    prefix_augmented_dir = Path(args.prefix_augmented_dir)
    original_validation_dir = Path(args.original_validation_dir)
    operator_weights = answer_supported_operator_weights(
        original_validation_dir,
        tuple(fair.ALL_FUNCS),
        args.distribution_smoothing,
    )

    report: dict[str, object] = {
        "config": {
            "rows_per_example_count": args.rows_per_example_count,
            "seed": args.seed,
            "versions": list(VERSIONS),
            "operators": list(fair.ALL_FUNCS),
            "candidate_modes": list(CANDIDATE_MODES),
            "selection_targets": {str(count): selection_target(count) for count in range(4)},
            "distribution_smoothing": args.distribution_smoothing,
            "digit_operator_count_reference": "prefix augmented prompt LHS operand/operator counts",
        },
        "operator_weights": operator_weights,
        "augmented": {},
        "candidates": {},
    }

    augmented_frames: list[pd.DataFrame] = []
    for example_count in range(4):
        refs = reference_counts(prefix_augmented_dir, example_count)
        ref_rng = random.Random(stable_seed("refs", example_count, base_seed=args.seed))
        selected_refs = distribute_references(refs, args.rows_per_example_count, ref_rng)
        existing_prompts: set[str] = set()
        rows = [
            make_synthetic_row(
                row_index,
                example_count,
                reference,
                alphabet,
                original_prompts,
                existing_prompts,
                args.seed,
            )
            for row_index, reference in enumerate(selected_refs)
        ]
        frame = pd.DataFrame(rows)
        augmented_frames.append(frame)
        path = augmented_outdir / f"synthetic_random_examples_{example_count}.parquet"
        write_frame(frame, path, args.csv)
        report["augmented"][str(example_count)] = {
            "path": str(path),
            "rows": int(len(frame)),
            "unique_prompts": int(frame["prompt"].nunique()),
            "original_prompt_overlap": int(len(set(frame["prompt"].astype(str)) & original_prompts)),
            "digit_char_count_distribution": frame["digit_char_count"].value_counts().sort_index().to_dict(),
            "operator_char_count_distribution": frame["operator_char_count"].value_counts().sort_index().to_dict(),
            "all_counts_match_reference": bool(
                (
                    (frame["digit_char_count"] == frame["reference_digit_char_count"])
                    & (frame["operator_char_count"] == frame["reference_operator_char_count"])
                ).all()
            ),
        }
        print(f"synthetic examples={example_count}: wrote {len(frame)} rows to {path}", flush=True)

    combined_augmented = pd.concat(augmented_frames, ignore_index=True)
    combined_augmented_path = augmented_outdir / "synthetic_random_examples_0_to_3.parquet"
    write_frame(combined_augmented, combined_augmented_path, args.csv)
    report["augmented"]["combined"] = {
        "path": str(combined_augmented_path),
        "rows": int(len(combined_augmented)),
    }

    for version in VERSIONS:
        version_frames: list[pd.DataFrame] = []
        version_report: dict[str, object] = {}
        for example_count, frame in enumerate(augmented_frames):
            candidate_rows: list[dict[str, object]] = []
            for row_index, row in frame.iterrows():
                pool = make_candidate_pool(int(row_index), row)
                seed = stable_seed(version, example_count, row.id, base_seed=args.seed)
                candidate_rows.extend(
                    select_candidates(pool, example_count, version, operator_weights, seed)
                )
            candidate_frame = pd.DataFrame(candidate_rows)
            candidate_frame = candidate_frame[
                CANDIDATE_COLUMNS
                + [column for column in candidate_frame.columns if column not in CANDIDATE_COLUMNS]
            ]
            version_frames.append(candidate_frame)
            path = candidate_outdir / version / f"synthetic_random_examples_{example_count}_candidate_solutions.parquet"
            write_frame(candidate_frame, path, args.csv)
            per_prompt = candidate_frame.groupby("id").size()
            per_operator_exact = None
            funcs_per_prompt = candidate_frame.groupby("id")["query_function"].nunique()
            if example_count == 0:
                per_operator = candidate_frame.groupby(["id", "query_function"]).size()
                per_operator_exact = bool(
                    len(per_operator) == len(frame) * len(fair.ALL_FUNCS)
                    and (per_operator == len(CANDIDATE_MODES)).all()
                    and (funcs_per_prompt == len(fair.ALL_FUNCS)).all()
                )
            version_report[str(example_count)] = {
                "path": str(path),
                "rows": int(len(candidate_frame)),
                "prompt_rows": int(frame["id"].nunique()),
                "unique_prompts": int(frame["prompt"].nunique()),
                "target_per_prompt": selection_target(example_count),
                "min_per_prompt": int(per_prompt.min()) if len(per_prompt) else 0,
                "max_per_prompt": int(per_prompt.max()) if len(per_prompt) else 0,
                "all_prompts_match_target": bool((per_prompt == selection_target(example_count)).all()),
                "per_operator_two_modes": per_operator_exact,
                "functions_per_prompt_min": int(funcs_per_prompt.min()) if funcs_per_prompt is not None else None,
                "functions_per_prompt_max": int(funcs_per_prompt.max()) if funcs_per_prompt is not None else None,
                "matches_answer": int(candidate_frame["matches_answer"].eq(True).sum()),
                "function_counts": candidate_frame["query_function"].value_counts().to_dict(),
            }
            print(
                f"{version} synthetic examples={example_count}: "
                f"wrote {len(candidate_frame)} rows to {path}",
                flush=True,
            )

        combined_candidates = pd.concat(version_frames, ignore_index=True)
        combined_path = candidate_outdir / version / "synthetic_random_examples_0_to_3_candidate_solutions.parquet"
        write_frame(combined_candidates, combined_path, args.csv)
        report["candidates"][version] = {
            "combined_path": str(combined_path),
            "combined_rows": int(len(combined_candidates)),
            "summaries": version_report,
        }

    report_path = candidate_outdir / "synthetic_random_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (augmented_outdir / "synthetic_random_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {report_path}", flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic random mapping datasets and selected candidate solutions."
    )
    parser.add_argument("--dataset", default="dataset.parquet")
    parser.add_argument("--prefix-augmented-dir", default=str(DEFAULT_PREFIX_AUGMENTED_DIR))
    parser.add_argument("--original-validation-dir", default=str(DEFAULT_ORIGINAL_VALIDATION_DIR))
    parser.add_argument("--augmented-outdir", default=str(DEFAULT_AUGMENTED_OUTDIR))
    parser.add_argument("--candidate-outdir", default=str(DEFAULT_CANDIDATE_OUTDIR))
    parser.add_argument("--rows-per-example-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--distribution-smoothing", type=float, default=0.25)
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    generate(build_arg_parser().parse_args())
