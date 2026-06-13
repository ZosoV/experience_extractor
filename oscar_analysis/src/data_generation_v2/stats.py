"""Compute empirical distributions from the 823 solved dataset.

Three distributions are extracted and used by synth.py:
  op_probs        P(op_name)              — for sampling arithmetic operations
  op_char_probs   P(op_char | op_name)    — for assigning operator symbols
  pos_probs       P(symbol | position)    — for assigning digit symbols (positions l0/l1/r0/r1)
  strategy_probs  P(strategy)             — for picking standard/le/pure_concat
  global_sym_probs P(symbol) global       — fallback for result-digit symbol assignment
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataStats:
    op_probs: dict[str, float]               # op_name -> probability
    op_char_probs: dict[str, dict[str, float]]  # op_name -> {op_char -> prob}
    pos_probs: dict[str, dict[str, float]]   # position -> {symbol -> prob}
    global_sym_probs: dict[str, float]       # symbol -> prob (union of positions)
    strategy_probs: dict[str, float]         # strategy -> prob
    arith_ops: list[str]                     # ops sorted by P(op) descending (non-concat)
    concat_ops: list[str]                    # concat op names present


def compute_stats(parquet_path: str | Path) -> DataStats:
    """Load solver_results.parquet and compute the empirical distributions."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    # Use only solver_correct=True rows
    df = df[df["solver_correct"] == True].copy()

    op_counter: Counter[str] = Counter()
    op_char_counter: dict[str, Counter[str]] = defaultdict(Counter)
    pos_counter: dict[str, Counter[str]] = {
        "l0": Counter(), "l1": Counter(), "r0": Counter(), "r1": Counter()
    }
    global_sym_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()

    for _, row in df.iterrows():
        strategy = str(row.get("solver_strategy") or "")
        strategy_counter[strategy] += 1

        # op_char -> op_name distribution from ops_mapping
        om_raw = row.get("ops_mapping")
        if isinstance(om_raw, str):
            try:
                om: dict[str, str] = json.loads(om_raw)
            except (json.JSONDecodeError, TypeError):
                om = {}
            for op_char, op_name in om.items():
                op_char_counter[op_name][op_char] += 1

        # Parse equations for op_name frequency and position symbol frequency
        eq_raw = row.get("equations")
        if not isinstance(eq_raw, str):
            continue
        try:
            equations = json.loads(eq_raw)
        except (json.JSONDecodeError, TypeError):
            continue

        for eq in equations:
            lhs = eq.get("lhs", {})
            op_char = lhs.get("op", "")
            if isinstance(om_raw, str) and op_char and op_char in om:
                op_name = om[op_char]
                op_counter[op_name] += 1

            # Position symbol frequencies (content symbols only)
            for pos in ("l0", "l1", "r0", "r1"):
                sym = lhs.get(pos, "")
                if sym:
                    pos_counter[pos][sym] += 1
                    global_sym_counter[sym] += 1

            # Also count result symbols in global pool
            for sym in eq.get("rhs", {}).get("actual_rhs", []):
                if sym:
                    global_sym_counter[sym] += 1

    def _normalise(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.most_common()}

    op_probs = _normalise(op_counter)
    op_char_probs = {op: _normalise(ch) for op, ch in op_char_counter.items()}
    pos_probs = {pos: _normalise(cnt) for pos, cnt in pos_counter.items()}
    global_sym_probs = _normalise(global_sym_counter)
    strategy_probs = _normalise(strategy_counter)

    from src.op_index import CONCAT_OPS
    arith_ops = [op for op in op_probs if op not in CONCAT_OPS]
    concat_ops = [op for op in op_probs if op in CONCAT_OPS]

    return DataStats(
        op_probs=op_probs,
        op_char_probs=op_char_probs,
        pos_probs=pos_probs,
        global_sym_probs=global_sym_probs,
        strategy_probs=strategy_probs,
        arith_ops=arith_ops,
        concat_ops=concat_ops,
    )
