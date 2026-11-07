# Prototype: Second-Half Intensity & Movement Comparison (`mockup-second-half`)

## Purpose

In **Veo Sprint Match 01**, `away` defeated `home` **4–2**, with **all six goals scored in the second half** despite nearly identical full-match process totals (pass completion 64.4% vs 64.5%, shots 14 vs 15, outfield distance 103.6 km vs 101.0 km).

This local Streamlit prototype (`app.py` + `analytics.py`) helps coaches and analysts compare **first-half vs second-half player movement intensity and spatial positioning** without overclaiming physiological fatigue or confounding the comparison with substitutions.

Specifically, it computes and visualizes:
- **High-Speed Frame Share (`speed_zone >= 3`, $\ge 20\text{ km/h}$)** and **Sprint Frame Share (`speed_zone == 4`, $\ge 25\text{ km/h}$)** as **sampled frame shares** (% of a player's observed frames in that half).
- **Per-Half Tracking Coverage & Equal-Observation Eligibility**: compares only players observed in both halves who meet user-configurable minimum coverage (`min_coverage_pct`, default $\ge 50\%$) and maximum coverage imbalance (`|cov_H2 - cov_H1|`, default $\le 30\text{ pp}$).
- **Percentage Points (`pp`) vs Relative Percent (`%`)**: explicitly separates additive percentage-point differences (`H2 - H1`) from relative percent changes (`100 * (H2 - H1) / H1`), safely flagging zero first-half denominators (e.g., `away_12` sprint share rising from `0.00%` in H1 to `0.18%` in H2) and single-half appearances.
- **Within-Half Intensity & Tactical Shape Trends**: 5/10/15-minute binned movement shares and attack-normalized team depth (`x_att = x * attack_dir`) split strictly by half, overlaid with the 6 second-half goal timestamps (`event_name == "goal"` matched to the nearest frame in the same half).
- **Player Half-by-Half Pitch Footprint**: side-by-side 105.0 m × 67.7 m pitch maps (`x_att`, `y_att`) showing where any selected player operated and executed Zone 3 / Zone 4 efforts in each half.

---

## Exact Launch Command

From the repository root (`/home/prg/orca/workspaces/veo-sprint-match-01/mockup-second-half`):

```bash
uv run streamlit run app.py --server.port 8504
```

Then open **`http://localhost:8504`** in your browser.

---

## What to Try

1. **Inspect the Eligible Full-Match Cohort (Default View: Min Coverage $\ge 50\%$, Max Imbalance $\le 30\text{ pp}$)**
   - Observe the **11 eligible outfield players** (5 `away`: `away_2`, `away_6`, `away_8`, `away_12`, `away_14`; 6 `home`: `home_2`, `home_7`, `home_8`, `home_10`, `home_12`, `home_14`).
   - Compare pooled High-Speed Frame Share: `away`'s eligible starters dropped from **1.52% in H1 to 1.07% in H2 (`-0.45 pp`, `-29.6%` relative)**, while `home`'s eligible starters held steady (**1.61% in H1 vs 1.63% in H2, `+0.01 pp`, `+0.9%` relative**).
2. **Toggle Primary Intensity Metric to Sprint Frame Share (`Zone == 4, >= 25 km/h`)**
   - Check the **Percentage-Point (`pp`) vs Relative Percent (`%`)** chart and table: `away_12` recorded `0` sprint frames in H1 (`0.00%`) and `13` sprint frames in H2 (`0.18%`), showing **`+0.18 pp`** while explicitly reporting **`Undefined (0% in H1)`** for relative `%`.
3. **Explore Within-Half Movement & Compactness Trends (Tab 2)**
   - Switch between **Selected Eligible Players Only** (controlling for fresh substitutes) and **All Active Outfield Players on Pitch**.
   - See how `away` scored three goals in the opening 10 minutes of the second half (`away_9` at video `61.67'`, `away_8` at `66.17'`, `away_9` at `68.96'`) and subsequently dropped their mean attacking depth `x_att` deeper (`-2.7 m` in H1 $\rightarrow$ `-5.0 m` in H2), whereas `home` pushed higher (`-2.4 m` $\rightarrow$ `-1.3 m`) and narrowed width (`47.3 m` $\rightarrow$ `38.1 m`) while chasing the scoreline.
4. **Compare Player Pitch Footprints Across Halves (Tab 3)**
   - Select `away_8` (100% coverage in both halves, H1 high-speed share `2.42%` $\rightarrow$ H2 `1.14%`) or `home_10` (`1.87%` $\rightarrow$ `2.39%`) to view their attack-normalized (`x_att`, `y_att`) Zone 3 and Zone 4 locations and half centroids on the 105.0 m × 67.7 m pitch.
5. **Adjust Eligibility Sliders & Check Edge Cases (Tab 4)**
   - Lower **Min Per-Half Tracking Coverage (%)** to `40%` and raise **Max Coverage Imbalance** to `60 pp` to bring partial-half players (`away_10` with 100% H1 / 43.3% H2; `away_11` with 84.7% H1 / 46.2% H2) into the comparison.
   - Verify how single-half players (e.g., `away_16` in H1 only; `away_9` with 12 frames / `0.2%` in H1 and `50.9%` in H2) are transparently labeled in the table.

---

## Limitations & Analytical Caveats

- **Observed Decline $\neq$ Physiological Fatigue**: A lower second-half high-speed or sprint frame share cannot establish physical fatigue. Game state (`away` leading 3–0 early in the second half), tactical low-block pacing, and match stoppages strongly influence running demands.
- **Sampled Frame Share vs Elapsed Time**: Tracking is sampled at 2.5 Hz (400 ms steps during in-play time), with occasional per-player tracking gaps when a player leaves the camera track or is substituted. All intensity ratios are computed strictly as **sampled frame shares** (`target_frames / observed_player_frames` within each half).
- **Substitutions & Single-Half Players**: Out of 31 outfield players in `player_summary.csv`, 17 appeared in only one half (or for 12 frames in H1 for `away_9`). Unfiltered team averages confound starter pacing with fresh second-half substitutes; the dashboard's eligibility controls isolate equally observed players.
- **Single-Match Scope**: All findings apply to this single Veo match (`Match 01`) and are descriptive rather than predictive or causal.

---

## Validation

Run the automated validation suite from the repository root:

```bash
uv run python validate_prototype.py
```

What `validate_prototype.py` checks:
- Python syntax compilation (`py_compile`) of `analytics.py` and `app.py`.
- Dataset baseline invariants (`33` summary rows, `1,289` events, `6` `event_name == "goal"` rows, `4` `is_goal` shot rows, `7,148` H1 frames, `7,253` H2 frames, half-time gap separation).
- Attack-direction normalization (`x_att`, `y_att`), same-half nearest-frame event matching with tolerance handling, and trajectory gap segmentation (`segment_id`).
- Sampled frame share identities (`z0..z4` sum to 100%, `hs_share == z3 + z4`, `sprint_share == z4`), zero-denominator handling (`away_12`), absent-half handling (`away_16`), and empty cohort handling.
- Headless Streamlit `AppTest` execution across default load, metric toggle (`hs` $\rightarrow$ `sprint`), team filtering (`away`), and empty player selection state.
