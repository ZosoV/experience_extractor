from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import fair_solver_experiment as fair
    from synthetic_random_augmented_datasets import CANDIDATE_MODES, stable_random_value, visible_digit_body
except ImportError:  # pragma: no cover - supports `python -m research...`
    from research import fair_solver_experiment as fair
    from research.synthetic_random_augmented_datasets import (
        CANDIDATE_MODES,
        stable_random_value,
        visible_digit_body,
    )


DEFAULT_AUGMENTED_DIR = Path("research/outputs/synthetic_random_augmented")
DEFAULT_OUTDIR = Path("research/outputs/synthetic_random_constrained_training_format")
TARGETS = {0: 128, 1: 128, 2: 128, 3: 128}


def train_columns(train_csv: Path) -> list[str]:
    frame = pd.read_csv(train_csv, nrows=0)
    columns = list(frame.columns)
    if columns != ["id", "prompt", "answer"]:
        raise ValueError(f"expected train.csv columns ['id', 'prompt', 'answer'], got {columns}")
    return columns


def visible_prompt_digit_chars(prompt: str) -> set[str]:
    examples, query_lhs = fair.parse_prompt(prompt)
    chars = set()
    for eq in examples:
        chars.update(eq.left + eq.right)
        chars.update(visible_digit_body(eq.rhs))
    return chars


def decode_operand(chars: str, digit_mapping: dict[str, int], mode: str) -> int:
    return fair.decode_digits([digit_mapping[char] for char in chars], 10, mode)


def output_widths(value: int, max_output_width: int) -> range:
    min_width = fair.minimal_width(value, 10)
    return range(min_width, max(max_output_width, min_width) + 1)


def sign_choices(
    value: int,
    visible_assignment: dict[str, int],
    alphabet: tuple[str, ...],
    seed: int,
    cap: int,
) -> list[str]:
    if value >= 0:
        return [""]
    choices = ["-"] if "-" not in visible_assignment else []
    available = [char for char in alphabet if char not in visible_assignment and char != "-"]
    available.sort(key=lambda char: stable_random_value(seed, "sign", char))
    choices.extend(available[: max(0, cap - len(choices))])
    return choices[:cap]


def output_choices_for_digits(
    digits: list[int],
    visible_assignment: dict[str, int],
    alphabet: tuple[str, ...],
    seed: int,
    symbol_choices: int,
    cap: int,
) -> list[tuple[str, tuple[tuple[str, int], ...]]]:
    inv = {digit: char for char, digit in visible_assignment.items()}
    unknown_digits = sorted({digit for digit in digits if digit not in inv})
    if not unknown_digits:
        return [("".join(inv[digit] for digit in digits), tuple(sorted(visible_assignment.items())))]

    available = [char for char in alphabet if char not in visible_assignment]
    ranked_by_digit: dict[int, list[str]] = {}
    for digit in unknown_digits:
        ranked = sorted(available, key=lambda char: stable_random_value(seed, "digit", digit, char))
        ranked_by_digit[digit] = ranked[: max(symbol_choices, len(unknown_digits))]

    choices: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for chars in itertools.product(*(ranked_by_digit[digit] for digit in unknown_digits)):
        if len(chars) != len(set(chars)):
            continue
        local_inv = dict(inv)
        local_assignment = dict(visible_assignment)
        for digit, char in zip(unknown_digits, chars, strict=True):
            local_inv[digit] = char
            local_assignment[char] = digit
        choices.append(
            (
                "".join(local_inv[digit] for digit in digits),
                tuple(sorted(local_assignment.items())),
            )
        )
        if len(choices) >= cap:
            break
    return choices


def candidate_pool_for_row(
    row_index: int,
    row: pd.Series,
    alphabet: tuple[str, ...],
    max_output_width: int,
    output_symbol_choices: int,
    output_choice_cap: int,
    sign_choice_cap: int,
    base_seed: int,
) -> list[dict[str, object]]:
    prompt = str(row.prompt)
    query_lhs = str(row.query_lhs)
    answer = str(row.answer)
    digit_mapping = {char: int(digit) for char, digit in json.loads(str(row.digit_mapping)).items()}
    operator_mapping = json.loads(str(row.operator_mapping))

    visible_assignment = {
        char: digit_mapping[char]
        for char in visible_prompt_digit_chars(prompt)
        if char in digit_mapping
    }
    visible_ops = {eq.op for eq in fair.parse_prompt(prompt)[0]}
    
    query_op = query_lhs[2]
    
    if query_op in visible_ops:
        allowed_funcs = [operator_mapping[query_op]]
    else:
        visible_funcs = {operator_mapping[op] for op in visible_ops if op in operator_mapping}
        allowed_funcs = [func for func in fair.ALL_FUNCS if func not in visible_funcs]

    import itertools
    import random
    query_unmapped_chars = list(set(query_lhs[:2] + query_lhs[3:]) - set(visible_assignment.keys()))
    available_digits = set(range(10)) - set(visible_assignment.values())
    if not query_unmapped_chars:
        query_permutations = [()]
    else:
        query_permutations = list(itertools.permutations(available_digits, len(query_unmapped_chars)))

    rng = random.Random(base_seed + row_index)
    rng.shuffle(allowed_funcs)
    rng.shuffle(query_permutations)

    rows: list[dict[str, object]] = []
    unique_answers = set()
    rank = 1
    
    for func in allowed_funcs:
        for mode in [row.true_mode]:
            for perm in query_permutations:
                if len(unique_answers) >= 500:
                    break
                    
                temp_mapping = dict(visible_assignment)
                for i, char in enumerate(query_unmapped_chars):
                    temp_mapping[char] = perm[i]
                    
                left_value = decode_operand(query_lhs[:2], temp_mapping, mode)
                right_value = decode_operand(query_lhs[3:], temp_mapping, mode)
                
                variant_seed = int(row.seed) if "seed" in row else base_seed
                variant_seed = int(
                    stable_random_value(
                        base_seed,
                        row.id,
                        func,
                        mode,
                        variant_seed,
                    )
                    * (2**64)
                )

                if func == "concat_fwd":
                    cand = query_lhs[:2] + query_lhs[3:]
                    rows.append(
                        {
                            "source_prompt_id": str(row.id),
                            "candidate_rank": rank,
                            "candidate_answer": cand,
                            "matches_answer": cand == answer,
                            "query_function": func,
                            "mode": mode,
                            "query_left_value": left_value,
                            "query_right_value": right_value,
                            "query_numeric_answer": None,
                            "output_width": 4,
                            "padded_output": False,
                            "query_operator_seen": query_op in visible_ops,
                            "pool_source": "concat",
                        }
                    )
                    unique_answers.add(cand)
                    rank += 1
                    continue

                if func == "concat_rev":
                    cand = query_lhs[3:] + query_lhs[:2]
                    rows.append(
                        {
                            "source_prompt_id": str(row.id),
                            "candidate_rank": rank,
                            "candidate_answer": cand,
                            "matches_answer": cand == answer,
                            "query_function": func,
                            "mode": mode,
                            "query_left_value": left_value,
                            "query_right_value": right_value,
                            "query_numeric_answer": None,
                            "output_width": 4,
                            "padded_output": False,
                            "query_operator_seen": query_op in visible_ops,
                            "pool_source": "concat",
                        }
                    )
                    unique_answers.add(cand)
                    rank += 1
                    continue

                value = fair.eval_numeric(func, left_value, right_value)
                if value is None:
                    continue

                for width in output_widths(value, max_output_width):
                    digits = fair.fixed_width_digits(abs(value), width, 10, mode)
                    if digits is None:
                        continue
                    choices = output_choices_for_digits(
                        digits,
                        temp_mapping,
                        alphabet,
                        variant_seed,
                        output_symbol_choices,
                        output_choice_cap,
                    )
                    for sign in sign_choices(value, visible_assignment, alphabet, variant_seed, sign_choice_cap):
                        for body, _digit_assignment in choices:
                            candidate_answer = sign + body
                            rows.append(
                                {
                                    "source_prompt_id": str(row.id),
                                    "candidate_rank": rank,
                                    "candidate_answer": candidate_answer,
                                    "matches_answer": candidate_answer == answer,
                                    "query_function": func,
                                    "mode": mode,
                                    "query_left_value": left_value,
                                    "query_right_value": right_value,
                                    "query_numeric_answer": float(value),
                                    "output_width": width,
                                    "padded_output": width != fair.minimal_width(value, 10),
                                    "query_operator_seen": query_op in visible_ops,
                                    "pool_source": "numeric_width_symbol_choice",
                                }
                            )
                            unique_answers.add(candidate_answer)
                            rank += 1
                            
            if len(unique_answers) >= 500:
                break
        if len(unique_answers) >= 500:
            break
            
    return rows


def select_candidates(
    pool: list[dict[str, object]],
    target: int,
    seed: int,
) -> list[dict[str, object]]:
    randomized = sorted(
        pool,
        key=lambda row: stable_random_value(
            seed,
            row["candidate_rank"],
            row["candidate_answer"],
            row["query_function"],
            row["mode"],
            row["output_width"],
            row["pool_source"],
        ),
    )

    selected: list[dict[str, object]] = []
    selected_ranks: set[int] = set()
    seen_answers: set[str] = set()
    for candidate in randomized:
        answer = str(candidate["candidate_answer"])
        if answer in seen_answers:
            continue
        selected.append(candidate)
        selected_ranks.add(int(candidate["candidate_rank"]))
        seen_answers.add(answer)
        if len(selected) >= target:
            return selected

    for candidate in randomized:
        rank = int(candidate["candidate_rank"])
        if rank in selected_ranks:
            continue
        selected.append(candidate)
        selected_ranks.add(rank)
        if len(selected) >= target:
            break
    return selected


def process_single_row(args):
    row_index, row, alphabet, max_output_width, output_symbol_choices, output_choice_cap, sign_choice_cap, base_seed, target, example_count = args
    
    pool = candidate_pool_for_row(
        int(row_index),
        row,
        alphabet,
        max_output_width,
        output_symbol_choices,
        output_choice_cap,
        sign_choice_cap,
        base_seed,
    )
    seed = int(stable_random_value(base_seed, example_count, row.id, "constrained-select") * (2**64))
    selected = select_candidates(pool, target, seed)
    
    return {
        "row_index": int(row_index),
        "row_id": str(row.id),
        "prompt": str(row.prompt),
        "answer": str(row.answer),
        "pool_count": len(pool),
        "selected_count": len(selected),
        "unique_answer_count": len({str(c["candidate_answer"]) for c in pool}),
        "selected": selected,
        "seed": seed,
    }

def build_dataset(
    example_count: int,
    augmented: pd.DataFrame,
    alphabet: tuple[str, ...],
    columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target = TARGETS[example_count]
    train_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    generated_counts: list[int] = []
    selected_counts: list[int] = []
    unique_answer_counts: list[int] = []

    import multiprocessing
    
    mp_args = [
        (
            row_index, row, alphabet,
            args.max_output_width, args.output_symbol_choices, args.output_choice_cap, args.sign_choice_cap, args.seed,
            target, example_count
        )
        for row_index, row in augmented.iterrows()
    ]
    
    with multiprocessing.Pool() as mp_pool:
        # unordered is faster as it yields as soon as one finishes
        results = mp_pool.imap_unordered(process_single_row, mp_args)
        
        for res in results:
            generated_counts.append(res["pool_count"])
            selected_counts.append(res["selected_count"])
            unique_answer_counts.append(res["unique_answer_count"])
            
            for selection_rank, candidate in enumerate(res["selected"], start=1):
                out_id = f"{res['row_id']}_constrained_cand{selection_rank:03d}"
                train_rows.append(
                    {
                        "id": out_id,
                        "prompt": res["prompt"],
                        "answer": str(candidate["candidate_answer"]),
                    }
                )
                detail = dict(candidate)
                detail.update(
                    {
                        "id": out_id,
                        "prompt": res["prompt"],
                        "example_count": example_count,
                        "selection_rank": selection_rank,
                        "selection_target_per_prompt": target,
                        "selection_seed": res["seed"],
                        "source_gold_answer": res["answer"],
                    }
                )
                detail_rows.append(detail)

    train_frame = pd.DataFrame(train_rows, columns=columns)
    detail_frame = pd.DataFrame(detail_rows)
    per_prompt = Counter(selected_counts)
    summary = {
        "example_count": example_count,
        "target_per_prompt": target,
        "prompt_rows": int(augmented["id"].nunique()),
        "rows": int(len(train_frame)),
        "unique_ids": int(train_frame["id"].nunique()),
        "unique_prompts": int(train_frame["prompt"].nunique()),
        "candidate_rows_per_prompt": {str(count): int(freq) for count, freq in sorted(per_prompt.items())},
        "prompts_below_target": int(sum(count < target for count in selected_counts)),
        "min_selected_per_prompt": int(min(selected_counts) if selected_counts else 0),
        "max_selected_per_prompt": int(max(selected_counts) if selected_counts else 0),
        "min_generated_pool": int(min(generated_counts) if generated_counts else 0),
        "median_generated_pool": float(pd.Series(generated_counts).median()) if generated_counts else 0.0,
        "min_unique_pool_answers": int(min(unique_answer_counts) if unique_answer_counts else 0),
        "median_unique_pool_answers": float(pd.Series(unique_answer_counts).median())
        if unique_answer_counts
        else 0.0,
        "selected_matches_gold": int(detail_frame["matches_answer"].eq(True).sum()) if len(detail_frame) else 0,
    }
    return train_frame, detail_frame, summary


def write_constrained_datasets(args: argparse.Namespace) -> dict[str, object]:
    train_csv = Path(args.train_csv)
    columns = train_columns(train_csv)
    source_df = pd.read_parquet(args.dataset)
    alphabet = fair.global_alphabet(source_df)
    augmented_dir = Path(args.augmented_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "train_csv_reference": str(train_csv),
        "columns": columns,
        "dataset": args.dataset,
        "augmented_dir": str(augmented_dir),
        "outdir": str(outdir),
        "targets": {str(key): value for key, value in TARGETS.items()},
        "config": {
            "seed": args.seed,
            "max_output_width": args.max_output_width,
            "output_symbol_choices": args.output_symbol_choices,
            "output_choice_cap": args.output_choice_cap,
            "sign_choice_cap": args.sign_choice_cap,
            "candidate_modes": list(CANDIDATE_MODES),
            "operators": list(fair.ALL_FUNCS),
            "selection": "stable random; constrained by operator mappings from examples",
        },
        "datasets": {},
    }

    for example_count in range(4):
        augmented_path = augmented_dir / f"synthetic_random_examples_{example_count}.parquet"
        augmented = pd.read_parquet(augmented_path)
        train_frame, detail_frame, summary = build_dataset(
            example_count,
            augmented,
            alphabet,
            columns,
            args,
        )
        if list(train_frame.columns) != columns:
            raise ValueError(f"output columns do not match train.csv for examples={example_count}")
        if train_frame["id"].nunique() != len(train_frame):
            raise ValueError(f"duplicate output ids for examples={example_count}")

        csv_path = outdir / f"synthetic_random_constrained_examples_{example_count}_train.csv"
        parquet_path = outdir / f"synthetic_random_constrained_examples_{example_count}_train.parquet"
        detail_path = outdir / f"synthetic_random_constrained_examples_{example_count}_selected_candidates.parquet"
        train_frame.to_csv(csv_path, index=False)
        train_frame.to_parquet(parquet_path, index=False)
        detail_frame.to_parquet(detail_path, index=False)

        summary.update(
            {
                "augmented_path": str(augmented_path),
                "csv_path": str(csv_path),
                "parquet_path": str(parquet_path),
                "selected_candidate_detail_path": str(detail_path),
            }
        )
        report["datasets"][str(example_count)] = summary
        print(
            f"examples={example_count}: wrote {len(train_frame)} rows to {csv_path} "
            f"with distribution {summary['candidate_rows_per_prompt']}",
            flush=True,
        )

    report_path = outdir / "synthetic_random_constrained_training_format_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}", flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build constrained train.csv-shaped synthetic datasets."
    )
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--dataset", default="dataset.parquet")
    parser.add_argument("--augmented-dir", default=str(DEFAULT_AUGMENTED_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--max-output-width", type=int, default=0)
    parser.add_argument("--output-symbol-choices", type=int, default=1)
    parser.add_argument("--output-choice-cap", type=int, default=1)
    parser.add_argument("--sign-choice-cap", type=int, default=4)
    return parser


if __name__ == "__main__":
    write_constrained_datasets(build_arg_parser().parse_args())
