# AGENTS Guide
## Purpose
- `mesospim_utils` is a Python toolkit for MesoSPIM metadata, deconvolution, OME-Zarr conversion, BigStitcher/Fiji alignment, Imaris workflows, and SLURM orchestration.
- Most modules are used as script-style entry points from inside the repo, not as a polished installed-package API.
- Many modules rely on config-derived globals loaded at import time from `mesospim_utils/constants.py`.
- Before changing behavior, read `PROJECT_CONTEXT.md`, `README.md`, `mesospim_utils/constants.py`, and the closest pipeline module.
## Key Repository Facts
- Python target is `3.12` per `setup.cfg`.
- Packaging uses `setuptools_scm` via `setup.py`.
- `requirements.txt` only contains `-e .`.
- Optional PSF support exists as the `psfmodels` extra in `setup.cfg`.
- There is no `pyproject.toml` in the repo root.
- `mesospim_utils/main.py` appears stale; do not treat it as the canonical entry point.
- Imports are often bare, for example `from metadata import ...` and `from constants import ...`.
- Because of those bare imports, script execution from within the package tree is the safest default.
- `mesospim_utils/config/main.yaml` is machine-specific; use `mesospim_utils/config/example.yaml` for portable guidance.
- `fiji-linux/` is a vendored runtime tree and should usually be ignored during code exploration.
- `mesospim_utils/dev/` is experimental and not the main production path.
## Build And Environment Commands
### Create or refresh a dev environment
```bash
conda create -n mesospim_utils python=3.12 -y
conda activate mesospim_utils
pip install -e .
```
### Install with optional PSF support
```bash
pip install -e ".[psfmodels]"
```
### Basic packaging sanity check
```bash
python setup.py --version
```
- `setup.py --version` currently works, but emits setuptools deprecation warnings.
- Treat those warnings as existing packaging debt unless your change touches packaging.
## Test Commands
- `python -m pytest` currently collects `0` tests in the repo root.
- There is no established automated unit test suite checked into this repository right now.
- Validation is mainly targeted script execution and workflow smoke tests.
### Run all tests
```bash
python -m pytest
```
### Run a single test file
```bash
python -m pytest path/to/test_file.py -q
```
### Run a single test function
```bash
python -m pytest path/to/test_file.py::test_name -q
```
### Run a single parametrized case
```bash
python -m pytest path/to/test_file.py::test_name[param_id] -q
```
### Useful smoke checks for this repo
```bash
python mesospim_utils/metadata.py --help
python mesospim_utils/automated.py --help
python mesospim_utils/omezarr.py --help
python mesospim_utils/bigstitcher.py --help
python mesospim_utils/slurm.py --help
```
- Prefer these script-style invocations over `python -m ...` because many modules use bare local imports.
- After editing a workflow, run the narrowest relevant CLI help or command path first.
- If you add tests, use `pytest` and document the exact node path you exercised.
## Linting And Formatting Status
- No repo-level configuration was found for `ruff`, `black`, `flake8`, `isort`, `mypy`, `pyright`, or `pylint`.
- No pre-commit config was found in the repo root during this review.
- Do not assume auto-formatting is part of CI.
- If you use local tooling, keep it narrowly scoped and avoid large style-only churn.
- Match the surrounding file instead of imposing a new project-wide style.
## Cursor And Copilot Rules
- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.
- This file is therefore the main agent-specific instruction source in the repo.
## Code Style Guidelines
### Imports
- Follow the existing import style of the file you are editing.
- In legacy modules, preserve bare local imports such as `from utils import ...` and `from constants import ...` unless you are intentionally refactoring the execution model.
- In newer modules, group imports as standard library, third-party, then local imports.
- Prefer one import per line when the list is long.
- Avoid introducing package-qualified imports into a legacy file unless script execution still works.
### Formatting
- Use 4-space indentation.
- Keep functions and blocks readable rather than aggressively compressed.
- Match existing quote style in the file; both single and double quotes are already present.
- Keep long command assembly readable, usually by incremental string building or multiline calls.
- Avoid wholesale reformatting of legacy files.
### Types
- Type hints are mixed across the codebase.
- Add type hints when touching newer or already-typed code, especially for public helpers, dataclasses, and new functions.
- Do not force a full typing pass on older files with minimal annotations.
- Prefer concrete types like `Path`, `list[Path]`, `tuple[int, int, int]`, and `Optional[...]` when useful.
- Preserve runtime compatibility with Python 3.12.
### Naming
- Follow local conventions instead of normalizing the whole repository.
- Module-level config constants are uppercase and imported from `constants.py`.
- Functions are generally `snake_case`.
- Typer command functions are also `snake_case`; CLI names are derived from them.
- Clarity is preferred over brevity for workflow variables.
### Paths And Filesystem Work
- Prefer `pathlib.Path` for new code and preserve existing `ensure_path(...)` usage where established.
- Many workflows manipulate Linux, Windows, and Wine path mappings; be careful not to break translation logic.
- When writing JSON or metadata artifacts, preserve the current path-conversion behavior.
- Do not hardcode machine-specific paths outside config handling.
### Configuration
- Importing `mesospim_utils/constants.py` loads YAML immediately.
- Changes to config keys can have wide repo impact because many modules import config-derived globals directly.
- Prefer additive config changes and keep `mesospim_utils/config/example.yaml` in sync.
- Do not treat `mesospim_utils/config/main.yaml` as authoritative project truth.
### Error Handling
- Raise specific built-in exceptions when practical, for example `FileNotFoundError`, `ValueError`, `RuntimeError`, `TimeoutError`, or `KeyError`.
- Include the relevant path, parameter, or dataset identifier in the message.
- Avoid broad `except Exception` unless the surrounding code already uses it for narrow recovery.
- Prefer failing loudly over silently swallowing errors in orchestration code.
- For subprocess logic, use `check=True` unless there is a documented reason not to.
### CLI And Pipeline Changes
- Many user-facing entry points are Typer apps; preserve help text and option behavior when extending them.
- Keep CLI option names descriptive and compatible with existing scripts.
- For automation code, think through downstream SLURM dependencies and generated command strings.
- For BigStitcher and Fiji changes, inspect both Python orchestration and generated macro or template content.
### Comments And Docs
- Add comments only when the code is not obvious from the implementation.
- Prefer short docstrings for public helpers and command functions that benefit from explanation.
- Do not add noisy comments that restate straightforward assignments.
## Practical Editing Guidance
- Read the full function before editing; many modules rely on side effects and import-time constants.
- Search for sibling code paths before changing one workflow branch.
- Keep changes minimal and targeted in legacy modules.
- If you add a new dependency, update `setup.cfg` instead of `requirements.txt` alone.
- If you add tests, place them in a conventional `tests/` tree and use `pytest`.
## Good Entry Points By Task
- Metadata issues: `mesospim_utils/metadata.py`
- SLURM orchestration: `mesospim_utils/automated.py` and `mesospim_utils/slurm.py`
- OME-Zarr conversion: `mesospim_utils/omezarr.py`
- BigStitcher/Fiji flow: `mesospim_utils/bigstitcher.py` and `mesospim_utils/bigstitcher_macro_templates.py`
- Deconvolution: `mesospim_utils/rl.py` and `mesospim_utils/psf.py`
- Imaris workflows: `mesospim_utils/imaris.py`, `mesospim_utils/stitch.py`, and `mesospim_utils/resample_ims.py`
## Preferred Agent Behavior
- Make the smallest safe change that matches local conventions.
- Verify with the narrowest relevant command first.
- Report clearly when a command is only a smoke test because the repo lacks formal tests.
- Call out stale entry points or packaging debt if they affect your change.
## Project-Specific Workflow Notes
- Use `PROJECT_CONTEXT.md` only for session-handoff notes: recent changes, current priorities, validation gaps, and open questions.
- Keep stable repo guidance in `AGENTS.md`; do not duplicate build commands, coding conventions, or evergreen architecture notes in `PROJECT_CONTEXT.md`.
- Read `PROJECT_CONTEXT.md` before starting work, and update it at the end of each work session so the next agent can get up to speed quickly.
