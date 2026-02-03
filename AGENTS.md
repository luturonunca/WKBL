# Repository Guidelines

## Project Structure & Module Organization
- `wkbl/`: core Python package for simulation analysis. Key entry point is `wkbl/astro/galaxy_peeker.py` (`Galaxy_Hound`).
- `tutorials/`: Jupyter notebooks demonstrating loading, centering, rotating, and DM profile workflows.
- `README.md`: high-level placeholder; expand if you add new features or usage steps.

## Build, Test, and Development Commands
- `bash wkbl/get_started`: adds this repo to `PYTHONPATH` via `~/.bashrc` and drops `.installed`.
- Example import check: `python3 -c "import wkbl; print(wkbl.__file__)"`
- No build system or package manager is defined; keep usage script-driven.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8-style naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Keep modules small and domain-focused (e.g., `wkbl/astro/`, `wkbl/particle/`).
- Prefer explicit parameter names for scientific routines (e.g., `r_max`, `bins`, `halo_vel`).

## Testing Guidelines
- No automated tests are present. If you add tests, place them in `tests/` and use `pytest`.
- Suggested naming: `tests/test_*.py`, and keep fixtures small (avoid large snapshot data).

## Commit & Pull Request Guidelines
- Current history uses short, informal messages. Prefer concise, descriptive subjects (e.g., `fix halo center shift`).
- PRs should include:
  - A brief summary of the change and affected modules.
  - Steps to reproduce (simulation path, snapshot format, assumptions).
  - Any notebook output changes (plots or derived values).

## Data & Environment Notes
- Simulations and snapshot files are not stored in the repo; document expected paths in scripts or notebooks.
- External scientific dependencies (e.g., `unsiotools`, `emcee`) should be listed when introduced.

## Agent Instructions (AI Assistance)
- Mission: maintain a Python library for analyzing numerical galaxy simulations (RAMSES or similar snapshots), focused on dark matter halos, galaxy components, structure finding, and physical/statistical diagnostics.
- API goals: keep a modular design with classes like `SnapshotLoader`, `HaloAnalyzer`, `Profiles`, plus utilities for plotting, filtering, and comparisons.
- Dependencies: use `numpy`, `scipy`, `astropy`, and `h5py` where appropriate; mention and integrate third‑party readers/halo finders when helpful.
- Quality: add unit tests (`pytest`) and examples, document code with docstrings, and provide a CLI via `argparse`.
- Performance: design for large datasets; parallelize when feasible.
- Packaging: follow Python packaging best practices (`pyproject.toml`/`setup.py`).
- Response format: start with a high‑level description, list required functions/classes, include example usage. Use Markdown and code blocks; add a TOC for large replies.
