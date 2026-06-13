# CLAUDE.md — Operating Protocols

### To run pythong code

- To run python code, use the .venv environment with uv run

---

## 1. Token-Efficiency Protocol

Minimize unnecessary token use in every interaction.

- **Read targeted, not whole.** Use line-range reads and search tools; reference code as `path:line` in responses instead of pasting file contents.
- **No re-reads after edits.** `Edit`/`Write` error on failure — a silent success means the file is correct. Skip the redundant read-back.
- **Parallel tool calls.** Run independent tool calls in a single message. Do not serialize work that can be parallelized.
- **No padding.** No preamble, no postamble, no restating the question. Answer directly.
- **Defer mechanical transforms.** Deterministic file-to-file transforms (reformatting, bulk renames, schema migrations) are scripted by the user, not done token-by-token in-flow.
- **Fan-out via subagents.** For broad codebase searches, use Explore agents so only conclusions return — not raw file dumps.

---

## 2. Specs & Memory Protocol

Two knowledge stores live in the repo. **Always read their index first** before opening any individual file.

### Specs — `specs/SPECS.md`

Full implementation documents (design, algorithms, schema). Use when:
- Starting work on any subsystem — check the index to find the relevant spec before reading code.
- A spec exists for the area being changed — keep it in sync (see §3).

### Memory — `memory/MEMORY.md`

Small, targeted findings: rules, constraints, and conclusions discovered through analysis that would be expensive to re-derive. Use when:
- Beginning any task — scan the index for entries relevant to the area of work.
- A non-obvious condition or invariant is confirmed — add an entry so it isn't re-analysed later.

### Keeping both stores updated

- After any code or spec change, check whether an existing memory entry is now stale; update or remove it.
- After discovering a non-obvious rule through analysis, add a memory entry before closing the task.
- After adding or renaming a spec file, update `specs/SPECS.md`.

---

## 3. Spec ↔ Code Sync Protocol

**Non-negotiable:** code and its spec must agree by the end of every change, regardless of which came first.

### Direction

Both orderings are valid:

| Mode | Workflow |
|---|---|
| **Code-first** | Change the code → update the matching spec in the same change |
| **Spec-first** | Update the spec as the design → implement to match |

Never leave them out of sync at the end of a change.

### Drift rules

- **Flag, don't silently fix.** When drift is spotted (stale `file:line` citation, renamed function, changed config key, schema change not reflected in the spec), surface it to the user before or during the change — do not silently patch unrelated areas.
- **Mark Deferred.** Intentionally-unimplemented spec items must be marked `Deferred` in the spec. Example:
  ```
  ### Feature X  *(Deferred)*
  ```
- **Drift-prone couplings in this repo** — pay extra attention to:
  - `lib.rs:NNN` line citations inside specs — these go stale whenever `lib.rs` changes (currently modified).
  - `src/op_index.py` integer indices **must** match the `apply()` match arms in `lib.rs` exactly.
  - `config.yaml` keys ↔ `src/config.py` (`SolverConfig` fields); heuristic labels #2–#9 ↔ `src/heuristics.py`.

### Other standing rules

- Generated CoT text is a deliberate mix of mathematical expressions drawn from the solver's code functions (op names, formulas, encodings) and natural-language reasoning that explains each step. Pure natural-language-only is **not** required.
- `src/operations.py` is the authoritative op registry (22 ops + 2 concat). The spec's tier tables are illustrative, not normative.
- `specs/ideas.md` is a non-authoritative scratchpad ("Don't execute anything from here") — it is **not** under the sync protocol.
