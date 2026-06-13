from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from research import fair_solver_experiment as fair
from research.synthetic_random_augmented_datasets import (
    CANDIDATE_MODES,
    DEFAULT_AUGMENTED_OUTDIR,
    QUERY_PREFIX,
    json_dumps_mapping,
    leading_zero_ok,
    make_operand,
    prompt_digit_operator_counts,
    render_answer,
    render_prompt,
    sample_mapping,
    stable_seed,
    visible_digit_body,
    write_frame,
)


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


def make_custom_equations(
    example_count: int,
    digit_chars: tuple[str, ...],
    operator_mapping: dict[str, str],
    digit_mapping: dict[str, int],
    digit_to_char: dict[int, str],
    true_mode: str,
    rng: random.Random,
):
    available_ops = list(operator_mapping.keys())
    
    for _ in range(5000):
        if example_count == 0:
            query_op = rng.choice(available_ops)
            op_sequence = [query_op]
        else:
            in_examples = rng.random() < 0.95
            if in_examples:
                k = rng.randint(1, min(example_count, 3))
                selected_ops = rng.sample(available_ops, k)
                example_ops = random.choices(selected_ops, k=example_count)
                query_op = rng.choice(list(set(example_ops)))
                op_sequence = example_ops + [query_op]
            else:
                k = rng.randint(1, min(example_count, 2))
                selected_ops = rng.sample(available_ops, k)
                example_ops = random.choices(selected_ops, k=example_count)
                remaining_ops = list(set(available_ops) - set(example_ops))
                query_op = rng.choice(remaining_ops)
                op_sequence = example_ops + [query_op]
                
        examples: list[fair.Equation] = []
        lhs_values: list[str] = []
        used_digits: set[str] = set()
        ok = True
        
        required_digits = set(digit_chars)
        
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
            if answer is None or len(visible_digit_body(answer)) > 4:
                ok = False
                break
                
            lhs_values.append(lhs)
            if eq_index < example_count:
                if not set(visible_digit_body(answer)).issubset(required_digits):
                    ok = False
                    break
                examples.append(fair.Equation(lhs, answer))
                
        if ok:
            query_lhs = lhs_values[-1]
            query_answer, _, _, _ = render_answer(
                query_lhs,
                operator_mapping[query_lhs[2]],
                digit_mapping,
                digit_to_char,
                true_mode,
            )
            if query_answer is not None and len(visible_digit_body(query_answer)) <= 4:
                # Check character overlap constraint ONLY for example_count == 3
                if example_count == 3:
                    example_chars = set()
                    for eq in examples:
                        example_chars.update(eq.lhs[:2] + eq.lhs[3:] + eq.rhs.lstrip('-'))
                    query_chars = set(query_lhs[:2] + query_lhs[3:])
                    is_full_overlap = query_chars.issubset(example_chars)
                    
                    # 61.02% chance of full overlap for 3 examples
                    full_overlap_target = (rng.random() < 0.6102)
                    if is_full_overlap != full_overlap_target:
                        continue
                        
                return examples, query_lhs, query_answer, operator_mapping[query_lhs[2]], op_sequence
                
    raise ValueError("could not construct custom equations")


def make_custom_synthetic_row(
    row_index: int,
    example_count: int,
    alphabet: tuple[str, ...],
    original_prompts: set[str],
    existing_prompts: set[str],
    base_seed: int,
) -> dict[str, object]:
    row_seed = stable_seed("synthetic-row-custom", example_count, row_index, base_seed=base_seed)
    rng = random.Random(row_seed)
    
    for attempt in range(5000):
        local_seed = stable_seed("attempt", attempt, base_seed=row_seed)
        local_rng = random.Random(local_seed)
        digit_mapping, digit_to_char, operator_mapping = sample_mapping(alphabet, local_rng)
        
        # We always allow the generator to pick from the full 10-digit alphabet!
        digit_chars = tuple(digit_mapping.keys())
            
        true_mode = local_rng.choice(CANDIDATE_MODES)
        
        try:
            examples, query_lhs, answer, true_query_function, op_seq = make_custom_equations(
                example_count,
                digit_chars,
                operator_mapping,
                digit_mapping,
                digit_to_char,
                true_mode,
                local_rng,
            )
        except ValueError:
            continue
            
        prompt = render_prompt(examples, query_lhs)
        if prompt in existing_prompts or prompt in original_prompts:
            continue
            
        actual_digit_count, actual_operator_count = prompt_digit_operator_counts(prompt)
        
        existing_prompts.add(prompt)
        source_id = f"synthetic_custom_{example_count}_{row_index:04d}"
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
            "operator_chars": "".join(sorted(set(op_seq))),
            "digit_char_count": actual_digit_count,
            "operator_char_count": actual_operator_count,
            "reference_id": "custom",
            "reference_digit_char_count": actual_digit_count,
            "reference_operator_char_count": actual_operator_count,
            "augmentation_mode": "synthetic_custom_mapping",
            "seed": row_seed,
        }

    raise RuntimeError(f"could not build custom synthetic row {row_index}")


def generate_custom_augmented(args: argparse.Namespace):
    source_df = pd.read_parquet(args.dataset)
    alphabet = fair.global_alphabet(source_df)
    original_prompts = set(source_df["prompt"].astype(str))
    
    # We will overwrite the old augmented directory or write to a new one
    augmented_outdir = Path(args.augmented_outdir)
    augmented_outdir.mkdir(parents=True, exist_ok=True)
    
    augmented_frames: list[pd.DataFrame] = []
    
    for example_count in range(4):
        existing_prompts: set[str] = set()
        rows = []
        for row_index in range(args.rows_per_example_count):
            row = make_custom_synthetic_row(
                row_index,
                example_count,
                alphabet,
                original_prompts,
                existing_prompts,
                args.seed,
            )
            rows.append(row)
            
        frame = pd.DataFrame(rows)
        augmented_frames.append(frame)
        path = augmented_outdir / f"synthetic_random_examples_{example_count}.parquet"
        write_frame(frame, path, args.csv)
        print(f"synthetic examples={example_count}: wrote {len(frame)} rows to {path}", flush=True)

    combined_augmented = pd.concat(augmented_frames, ignore_index=True)
    combined_augmented_path = augmented_outdir / "synthetic_random_examples_0_to_3.parquet"
    write_frame(combined_augmented, combined_augmented_path, args.csv)
    print(f"Combined wrote {len(combined_augmented)} rows to {combined_augmented_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.parquet")
    parser.add_argument("--augmented-outdir", default="research/outputs/synthetic_random_augmented")
    parser.add_argument("--rows-per-example-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=True)
    
    generate_custom_augmented(parser.parse_args())
