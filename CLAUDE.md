# CLAUDE.md — operating rules for this repo

Project: reproducible GPT pretraining pipeline on FineWeb-Edu. Portfolio-grade
ML engineering artifact, **not** a tutorial clone. Quality, reproducibility,
and clean abstractions over speed-to-results.

## Hardware & environment (defaults must fit this box)

- GPU: NVIDIA RTX 2050 (mobile), **4 GB VRAM**, sm_86 (Ampere), driver 566.24.
  Effective budget after torch/CUDA load is ~3.3 GB. This is *very* tight for
  pretraining — every Phase 1+ config default must account for it:
  - small model (target ~10–25 M params, NOT the 124 M nanoGPT baseline);
  - AMP (bf16 if supported, else fp16) is mandatory, not optional;
  - aggressive gradient accumulation to reach useful effective batch size;
  - short context length to start (e.g., 256–512), grow only after profiling;
  - consider gradient checkpointing once architecture stabilizes.
- System RAM: 16 GB (tight — careful with workers, prefetch, in-memory shards).
- OS: Windows 11 native (**not** WSL). Shell: PowerShell 7.
- Python: 3.11 managed via `uv`. Project root: `C:\dev\fineweb-edu-pretraining-lab`.
- W&B is the only experiment tracker. Do not suggest others.

## Windows-native constraints (bake into code)

- DataLoader `num_workers` default = `0`. Make it configurable but document the
  Windows spawn-vs-fork gotcha in the config comment.
- Every multi-process entry point needs `if __name__ == "__main__":`.
- Paths: `pathlib.Path` only, never raw strings with backslashes.
- Honor `HF_HOME` (user redirects HF cache to `D:\hf_cache`). Never hardcode.
- Avoid symlinks (require admin on Windows).
- Source files use LF line endings (`.gitattributes` enforces this).
- Tool checks: `Get-Command`, not `which`.

## Collaboration rules

- Ask clarifying questions before non-trivial code. Don't guess layout, scope,
  or naming.
- Build file by file. Explain key design decisions in 2–3 lines per new file.
  No giant code dumps.
- No features the user didn't ask for. No premature abstractions.
- Modern PyTorch APIs only (`torch.amp`, not `torch.cuda.amp`, etc.).
- Type hints + Google-style docstrings. Inline comments only for non-obvious "why".
- Tests as we go — target >70 % coverage on core modules.
- Conventional commits, one per phase.
- Never run a long training job without confirming the config first.

## Phased workflow

Work is organized in phases. Each phase is tightly scoped — finish it, commit,
then plan the next one. Don't bleed scope across phases. Phase 0 = repo
scaffolding only (no app logic, no model, no training code).

## Tooling quick reference

- `uv sync` to resolve deps. Torch comes from the explicit cu121 index pinned
  in `pyproject.toml`; if it ever drops back to a CPU wheel run
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --reinstall`.
- `uv run pytest -v` for tests, `uv run ruff check .` + `uv run ruff format .` for lint.
- `uv run pre-commit install` once per clone.
