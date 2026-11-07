# Veo Amateur Coach Match-Review Dashboard (`mockup-coach-dashboard`)

## Purpose
This prototype provides an accessible, local Streamlit match-review dashboard built for amateur football coaches reviewing **Veo Match 01 (`away 4 – 2 home`)**. Rather than overwhelming coaches with raw data tables or unverified predictive models, it answers the practical post-match question: **"What should I review next?"**

It connects:
1. **Trustworthy Team & Squad Summary Cards** from `player_summary.csv` and `events.csv` (explaining why Away won 4–2 despite near-identical possession, pass completion, and total shots).
2. **Searchable & Selectable Key Moments (All 6 Goals + All 29 Shots)** with exact **Video Recording Timestamps** (`MM:SS` on `match_01_full.mp4`), estimated in-play match clocks, 22-player pitch snapshots, lead-up pass/ball trajectories, and defensive shape spatial heuristics.
3. **Interactive Player Pitch & Stats Inspector** where selecting any of the 33 players updates their summary metrics, per-half tracking coverage (`H1` vs `H2` tracked frames/minutes), and a customizable 2D pitch heatmap / speed-zone & on-ball events visualization.

---

## Exact Launch Command (Default Port: `8505`)
Run from the repository worktree root (`/home/prg/orca/workspaces/veo-sprint-match-01/mockup-coach-dashboard`):

```bash
uv run streamlit run app.py --server.port 8505
```

---

## What to Try in the Dashboard
1. **Filter & Select Key Moments ("What should I review next?"):**
   - In the sidebar, switch **"What should I review next?"** between `All Goals (6)`, `All Goals & Shots (35)`, `Shots Only (29)`, `Stretched Defense Moments (Heuristic)`, and `Close-Range Shots (<= 15 m)`.
   - Type a query in **Search Moment** (e.g., `away_9`, `83:02`, or `No Attached Shot Row`) to locate specific clips immediately.
   - Select any moment from the dropdown to inspect the **105.0 m × 67.7 m pitch snapshot** (nearest tracking frame within the same half), adjust the **Lead-up Window (5–25 s)** slider, toggle **Player Trails** or **Deepest-4 Defender Line**, and read the plain-English **Coach-Friendly Observation**.
2. **Inspect the Two Unattached Goals vs. Four Shot-Linked Goals:**
   - Select `[GOAL #2] Video 66:10 | AWAY - away_8` or `[GOAL #6] Video 94:32 | HOME - home_18` to see how the dashboard handles the two `event_name == "goal"` rows that have no corresponding `is_goal == True` shot row in `events.csv`.
3. **Compare Player Spatial Profiles & Attacking Direction Normalization:**
   - In the sidebar **Select Player for Pitch & Stats**, pick `away_9` (hat-trick super-sub in 24.7 min), `away_12` (highest distance: 11.33 km), or `home_10` (4 shots, 1 goal, +28% high-intensity rate in H2).
   - Toggle **Normalize Attacking Direction (`attack_dir` → +x)** on and off in Section 3 to see why raw coordinates cancel out across halves when teams swap ends, whereas `attack_dir`-normalized coordinates (`x_att = x * attack_dir`, `y_att = y * attack_dir`) show true attacking roles.
   - Switch **Pitch Visualization Layer** between `Heatmap + On-Ball Events`, `Speed Zones + On-Ball Events`, and `Heatmap Only`.

---

## Data Correctness & Limitations
- **Video Clock vs. Match Clock:** `timestamp_ms` and `minute` represent elapsed time in the video file `match_01_full.mp4` (`seconds = timestamp_ms // 1000`), which starts before kick-off (`05:51` / `351.2 s`) and includes half-time (`H2` starts at `59:45` / `3585.2 s`). Both clocks are displayed side-by-side.
- **Summary Minutes vs. Per-Half Tracking Coverage:** `player_summary.csv` provides a single `minutes_played` number per player. To prevent misleading H1 vs. H2 fatigue comparisons when players were substituted, the dashboard explicitly computes per-half tracked frame counts (`1,500 frames = 10 min` at 2.5 Hz) and flags players without ≥10 minutes in both halves.
- **Nearest-Frame Matching & Trajectory Gaps:** Events are matched to the nearest tracking frame **strictly within the same half** (`max_tolerance_ms = 1000`, typical offset `0–200 ms`). Player and ball trajectories are segmented on missing coordinates (`ball_x`/`ball_y` are null in ~12% of frames) and never interpolate across tracking gaps or the half-time boundary.
- **Spatial Heuristics vs. Causation:** Defensive block width/depth and "Stretched Defense" tags (`outfield_depth >= 50 m`, `outfield_width >= 42 m`, or `back4_width >= 28 m`, excluding goalkeeper and referee) are descriptive spatial heuristics at a single frame, not causal proof or predictive xG models (no model is trained on 29 shots).
- **Incomplete Event Coverage:** `events.csv` is a detected sample of 1,289 on-ball actions; missing event rows in a time window do not establish absence of touches or possession.

---

## Validation
Run syntax compilation and the automated validation suite (covering dataset invariants, goal/shot schema rules, same-half frame matching, gap-safe trajectory splitting, and headless Streamlit `AppTest` interaction checks):

```bash
uv run python -m py_compile analytics.py app.py test_app.py
uv run pytest -v test_app.py
```
