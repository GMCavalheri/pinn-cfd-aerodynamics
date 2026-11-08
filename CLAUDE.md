# CLAUDE.md

Physics-informed neural networks (PINNs) for aerodynamics: potential flow around a cylinder →
a parameterized NACA airfoil family → neuroevolutionary architecture/loss-weight search →
surrogate speed against the AERO-01 panel method → exploratory steady 2D Euler.
Plan: `AERO-06-pinn-cfd-aerodynamics-PLAN.md` (tick the phase checkboxes as phases land).

## Commands
- `uv sync --extra cu132` — install with CUDA PyTorch (local RTX 2060); `uv sync --extra cpu`
  for CPU-only (what CI runs). Plain `uv sync` installs **no** torch.
- `uv run pytest` — tests (CPU, small budgets; the whole suite stays within a few minutes)
- `uv run ruff check . && uv run ruff format --check .` — lint/format (CI runs both). Ruff also
  formats the ```python blocks inside Markdown, so run it after editing docs too
- `uv run python validation/<script>.py` — train/evaluate one phase, write figures to
  `docs/figures/`, tables to `validation/results/`, and append rows to
  `validation/results/compute_budget.csv`

## Layout
- `src/pinncfd/` — single package. Shared core: `nn.py` (MLPs, Fourier features),
  `autodiff.py` (derivatives through `torch.autograd.grad`), `sampling.py` (collocation
  points), `training.py` (Adam → L-BFGS loop with convergence tracking), `io.py` (checkpoints
  and the compute-budget record), `device.py`, `seed.py`, `plotting.py`. Subpackages mirror the
  plan: `pinn_laplace_2d`, `pinn_airfoil_flowfield`, `pinn_euler_2d` (exploratory),
  `neuroevolution_search`.
- `src/pinncfd/external/` — the **only** place `aero` (AERO-01, uv git dependency pinned in
  `pyproject.toml`) is imported. Bump its `rev` deliberately and rerun everything.
- `tests/` — pytest, one file per module. `validation/` — one script per phase,
  `reference_data/` (cited), `results/` (tables, small checkpoints, compute-budget CSV).
  `validation/runs/` is scratch (git-ignored).
- `docs/` — theory note per module, reports, figures; `docs/lessons/` is the course (keep its
  numbers in sync with `validation/results/`). `notebooks/` — one executed demo per lesson.

## Conventions
- Nondimensional: cylinder radius = 1 or chord = 1, freestream speed = 1.
- Angles are **radians** internally; helpers that take degrees say so (`*_deg`).
- Potential flow is solved for the **stream function** ψ (u = ∂ψ/∂y, v = −∂ψ/∂x): it stays
  single-valued with circulation, the velocity potential does not.
- float32 for training; float64 for exact solutions and error metrics.
- Every training run goes through `pinncfd.io.record_run`: error, wall time per optimizer
  stage, steps to the target error and to the plateau, device, seed, git commit. The user's
  paper reads `validation/results/compute_budget.csv`; never edit rows by hand.

## Docs (GitHub-rendered math)
- Display equations go in ```` ```math ```` fences (no Markdown processing inside).
- Inline `$...$` is processed as Markdown first: no `\,` or `\;`, no `\\` row breaks, and no
  `_` right after `}` `)` `]`. `tests/test_docs.py` guards this.

## Testing rules
- Every test asserts against a closed-form result or a cited published value; name the source
  in a comment. Never loosen a tolerance to make a failing model pass without explaining why.
- Training tests use fixed seeds, CPU and short budgets; their bounds are stated with the
  reason (what a short run is expected to reach), not tuned to the last run.
- Reference data in `validation/reference_data/` must carry its citation. Do not invent data.
- Report failures honestly (Phase 5 is exploratory): a case that does not converge is
  documented with evidence, not hidden.

## Workflow
- Commit small logical changes directly to `main`, push after
  `uv run ruff check . && uv run ruff format --check . && uv run pytest` pass (no pipes: a pipe
  hides the exit code). Stage explicit paths rather than `git add -A`.
- Each phase lands as code + tests, then validation results, then its lesson in
  `docs/lessons/` as a separate commit.
- GPU training (≈ 1 h budget per phase) runs in the background; the GPU is clock-capped at
  1000 MHz, so wall times are for that setting.
