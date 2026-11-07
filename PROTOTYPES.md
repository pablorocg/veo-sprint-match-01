# Hackathon Prototypes

Six independent local Streamlit mockups, each assigned to an Orca Antigravity worker with requested model `gemini3.8flash`. All six prototypes are complete and committed on independent branches. Workers reported passing analytical and headless Streamlit UI checks for every prototype. All six working trees are clean; the coordinator verified original datasets, documentation, contributor guide, and notebooks are unchanged from the baseline.

All branches start from baseline commit `852655c`. No branch is merged or pushed. Original datasets and notebooks are preserved.

| Idea | Branch | Port | Status / commit |
| --- | --- | --- | --- |
| Goal buildup and match story | `pablorocg/mockup-match-story` | 8501 | Complete; `dd9b27a`; 6 tests and startup check passed |
| Off-ball space creators | `pablorocg/mockup-space-creators` | 8502 | Complete; `35bc9e2`; 5 tests passed |
| Defensive shape explorer | `pablorocg/mockup-defensive-shape` | 8503 | Complete; `e76a110`; 6 tests and startup check passed |
| Second-half intensity | `pablorocg/mockup-second-half` | 8504 | Complete; `41f9452`; analytical/UI/startup checks passed |
| Accessible coach dashboard | `pablorocg/mockup-coach-dashboard` | 8505 | Complete; `8fd9208`; 6 tests passed after wording review |
| Passing options explorer | `pablorocg/mockup-passing-options` | 8506 | Complete; `ca42999`; 6 tests passed |

## Open a prototype

Worktrees live under `/home/prg/orca/workspaces/veo-sprint-match-01/<name>`, where `<name>` is the branch suffix above. Each worktree contains a `PROTOTYPE.md` with its exact launch command, features, validation evidence, and limitations.

Launch any prototype from its worktree:

```bash
cd /home/prg/orca/workspaces/veo-sprint-match-01/mockup-match-story
uv sync --locked
uv run streamlit run app.py --server.port 8501
```

Use the corresponding directory and port for the other prototypes. Separate worktrees let you run several prototypes side by side without switching the primary checkout.

## Coordination

Orca run: `run_e36c290d9c28`.

Each worker completed its assigned branch and reported a commit plus real-data/headless validation. Heuristic insights are labeled; source data stays local under the usage terms in `README.md`. The coach dashboard received an additional wording review distinguishing event share from possession and inferred time from official match time.

| Worktree name | Worker dispatch |
| --- | --- |
| `mockup-match-story` | `ctx_f99c2292c8c3` |
| `mockup-space-creators` | `ctx_40016ab5b1b3` |
| `mockup-defensive-shape` | `ctx_9aa4ed8b3ca3` |
| `mockup-second-half` | `ctx_11e1f5440e2e` |
| `mockup-coach-dashboard` | `ctx_fd8fe001df34` (review follow-up; original `ctx_1da451f4b72f`) |
| `mockup-passing-options` | `ctx_d16734d0720e` |

## Validation and limitations

Worker checks cover data baselines, same-half timestamp alignment, missing data, metric-specific invariants, and dashboard controls. These are exploratory single-match mockups; heuristic rankings do not establish tactical causation or physiological fatigue. AppTest checks are headless functional checks, not a full visual design review.
