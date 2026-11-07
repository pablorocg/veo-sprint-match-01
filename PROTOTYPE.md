# Prototype: Off-Ball Space Creators (`mockup-space-creators`)

## Purpose

Standard match reports (`player_summary.csv` and `events.csv`) record who took a shot or completed a pass, but they cannot show **who created the space without the ball** so that the attack could succeed.

**Off-Ball Space Creators** is an interactive coaching dashboard built on Veo's player and ball tracking data (`players.parquet` and `events.csv`) for **Veo Football Analytics Sprint Match 01 (`away` 4 – 2 `home`)**. It allows coaches and analysts to:
1. Select an attacking team (`away` or `home`) and any real **shot (29 rows)** or **goal (6 complete rows)** moment from the match.
2. Inspect a configurable lead-up tracking window (`4–20 s`) strictly within the **same half** and split on any tracking gap.
3. Visualize full player motion trails or before/after vectors on a 2D pitch (`105.0 m × 67.7 m`, origin at centre spot) alongside ball trajectories and nearest outfield defender movements.
4. Rank candidate off-ball runs using a transparent, explainable **4-component heuristic score (0–100)** and inspect the exact player, nearest defender, and metric components that produced the rank.

---

## Exact Launch Command (Default Local Port: `8502`)

Run from the worktree root (`/home/prg/orca/workspaces/veo-sprint-match-01/mockup-space-creators`):

```bash
uv run streamlit run app.py --server.port 8502
```

*(Note: `.streamlit/config.toml` also sets `port = 8502` and `headless = true` by default, so `uv run streamlit run app.py` launches on port `8502` automatically.)*

---

## What to Try in the Dashboard

1. **Default View — Away Goal at 83:02.3 (`Event #1012`, `away_9`):**
   - Observe how `away_9` is automatically identified as the on-ball scorer and excluded from off-ball ranking, while outfield teammates (`away_13`, `away_12`, `away_6`, `away_10`, `away_8`) are ranked by their forward progression (`Δx_att`), nearest-defender separation change (`ΔDefSep`), ball separation, and running tempo.
   - Select different players in **"4. Inspect Attacking Outfield Player"** to highlight their run in bright cyan (`#00f5d4`) and their nearest outfield defender in dashed red/blue on the pitch, and inspect their frame-by-frame timeline chart below the pitch.
2. **Toggle Pitch Visualization Mode:**
   - Switch between **"Full Motion Trails"** (complete continuous frame-by-frame trajectories) and **"Before / After Vectors"** (clean displacement arrows from window start to shot/goal climax).
3. **Inspect Unattached Goal Rows & Missing Ball Coordinates (`Event #792` and `Event #1157`):**
   - Switch **Attacking Team** to `HOME`, set **Filter Attack Type** to `"Goals Only (All 6 Goal Rows)"`, and select **`[94:33.0 | H2] HOME - Goal Event (No Shot Row Recorded) by home_18 [Event #1157]`**.
   - In this late corner-kick goal window, ball tracking (`ball_x`, `ball_y`) is missing in 18 of 26 frames (`31%` coverage). Observe how the dashboard displays an explicit **Data Availability Notice** and marks off-ball rankings as **`Unavailable`** rather than fabricating `0` or `NaN` rankings.
4. **Adjust Heuristic Weights & Ball-Proximity Radius:**
   - Open **"⚙️ Heuristic Weights & Off-Ball Eligibility Rules"** in the sidebar to tune the weights for Forward Displacement, Defender Separation Gain, Distance Away from Ball, and Tempo/Effort, or adjust the ball-proximity exclusion radius (`1.5–6.0 m`).

---

## Methodology, Data Correctness & Limitations

- **Coordinate System & Halves:** Pitch coordinates are centred metres (`x ∈ [-52.5, +52.5]`, `y ∈ [-33.85, +33.85]`). Because teams swap ends at half-time, forward displacement is computed using `x_att = x * attack_dir` (`+1` attacking `+x`, `-1` attacking `-x`).
- **Same-Half Nearest Frame & Continuous Segments:** Event timestamps (`timestamp_ms`, video clock) are matched to the nearest tracking frame strictly within the same half (tolerance `≤ 1000 ms`). Both global windows and per-player trajectories are split at any tracking gap `> 600 ms` (normal cadence is `400 ms` / `2.5 Hz`), reporting the **actual available duration** in seconds.
- **Role Filtering:** Goalkeepers (`away_1`, `home_1`) and the `referee` are excluded from outfield runner rankings and nearest-outfield-defender distance calculations.
- **Honest Off-Ball Eligibility:** `events.csv` is a sample of 1,289 on-ball actions and **cannot prove absence of touches or possession**. Off-ball eligibility combines recorded events with continuous ball-proximity tracking (`distance_to_ball_m > radius` for at least `70%` of valid frames).
- **Observational Defender Co-Movement (No Causal Claims):** While the inspector reports how far the initial nearest defender moved and how separation changed (`d_end - d_start`), tracking co-movement is purely observational and **does not prove the attacker caused the defender's displacement** (defenders also react to the ball, teammates, and zonal structure). No predictive model is trained on the 29 shots.

---

## Validation

Run the automated test suite (covering dataset baselines, same-half nearest-frame matching, per-player gap splitting, missing ball handling, and headless Streamlit `AppTest` interaction checks):

```bash
uv run python -m py_compile analytics.py app.py test_prototype.py
uv run pytest -v test_prototype.py
```
