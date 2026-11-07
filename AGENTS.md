# Repository Guidelines

## Project Structure & Module Organization

This repository contains football match data and Python analysis notebooks, all at the repository root. There is no application package or dedicated source, test, or asset directory.

- `starter.ipynb`: dataset introduction, pitch plots, and pass networks.
- `demo_goal_moment.ipynb`: worked analysis connecting a goal to tracking data.
- `player_summary.csv`, `events.csv`, `players.parquet`, `ball.csv`: source datasets; preserve these inputs.
- `DATA_DICTIONARY.md`: schemas, units, joining rules, and interpretation caveats. Read it before changing analyses.
- `pyproject.toml`, `uv.lock`, `.python-version`: dependencies and Python 3.12 environment.

## Build, Test, and Development Commands

Run commands from the repository root so notebook-relative data paths resolve.

- `uv sync --locked`: install the locked project dependencies into `.venv`.
- `uv run --with notebook jupyter notebook starter.ipynb`: launch the introductory notebook with Jupyter supplied temporarily.
- `uv run --with notebook jupyter notebook demo_goal_moment.ipynb`: launch the goal walkthrough.

Select the project environment as the notebook kernel. No build step, automated test command, or lint command is configured. When adding dependencies, update both `pyproject.toml` and `uv.lock`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` variables and functions, and `UPPER_SNAKE_CASE` constants. Follow the existing pandas, NumPy, and Matplotlib idioms. Give new notebooks descriptive `snake_case.ipynb` names, introduce analyses with Markdown cells, and label plots with units. No formatter or linter is configured.

## Testing Guidelines

Restart the kernel and run every cell in order after notebook edits; verify tables, figures, and interactive outputs. No testing framework or coverage threshold exists. Check dataset baselines: 33 summary rows, 1,289 events, and six `event_name == "goal"` rows. Only four shots have `is_goal` set.

Use nearest-frame timestamp matching, interpret `timestamp_ms` as video time, respect `attack_dir` across halves, and handle missing ball coordinates.

## Commit & Pull Request Guidelines

The repository has no commits yet, so no message convention is established. Use concise imperative subjects, such as `Add passing network analysis`. Keep changes focused. PRs should describe the analysis, assumptions, and validation performed; link relevant issues and include screenshots for changed visualizations. Review notebook diffs for unrelated output churn.

## Dataset Usage

Preserve anonymized player identifiers. The dataset is restricted to sprint participation and personal learning; do not redistribute it. Obtain permission for other uses as described in `README.md`.
