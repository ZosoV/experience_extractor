"""SolverConfig dataclass and YAML loader.

YAML keys:
  force_base_10: bool          — only try (10, 10) radix/digit_count pair
  disabled_heuristics: list    — Python heuristic function names to bypass;
                                 choose from: op_glyph_priority, sign_pattern_filter,
                                 concat_optin, order_ops, tier_skip
  tiers: list[str]             — subset/order of tiers to search;
                                 choices: tier0, tier1, tier2, full
  input_parquet: str           — path to input .parquet (columns: id, prompt, answer)
  output_parquet: str          — path to write solver_results.parquet
  max_solutions: int           — collect up to this many distinct programs per puzzle;
                                 search escalates: digit mappings → little_endian → alt base
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverConfig:
    force_base_10: bool = False
    disabled_heuristics: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=lambda: ["tier0", "tier1", "tier2", "full"])
    input_parquet: str = "data/solver_results_base.parquet"
    output_parquet: str = "data/solver_results.parquet"
    max_solutions: int = 5


def load_config(path: str | Path | None = None) -> SolverConfig:
    """Load a SolverConfig from a YAML file, or return defaults if path is None."""
    if path is None:
        return SolverConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import yaml  # pyyaml; imported lazily so the module loads without it

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return SolverConfig(
        force_base_10=bool(data.get("force_base_10", False)),
        disabled_heuristics=list(data.get("disabled_heuristics", [])),
        tiers=[t.lower() for t in data.get("tiers", ["tier0", "tier1", "tier2", "full"])],
        input_parquet=str(data.get("input_parquet", "data/solver_results_base.parquet")),
        output_parquet=str(data.get("output_parquet", "data/solver_results.parquet")),
        max_solutions=int(data.get("max_solutions", 5)),
    )
