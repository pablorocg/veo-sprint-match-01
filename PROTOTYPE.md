# Prototype: Match Story & Goal Buildup Explorer (`mockup-match-story`)

## Purpose
On headline match totals, **Away** and **Home** looked nearly identical (14 vs 15 shots, 64.4% vs 64.5% pass completion, 103.6 km vs 101.0 km outfield distance covered, 48.7% vs 51.3% on-ball event share), yet **Away won 4–2** with all six goals scored in the second half.

This local Streamlit prototype helps coaches and analysts explore **what may have decided the match** and inspect the **15–20 second buildup** to every goal alongside a transparently matched unsuccessful attack:
1. **All Six Goals Explorer (`event_name == "goal"`):** Choose any of the six goals, adjust the 15–20 second lead-in window, scrub frame-by-frame via the 2.5 Hz tracking slider, or use the interactive **Play / Pause** pitch animation.
2. **Transparent Goal Location Provenance:** Because `goal` rows in `events.csv` carry no coordinates and only four shots have `is_goal == True`, the explorer explicitly labels the source and video timestamp for each goal's location reference (either the linked `is_goal` shot event for Goals `#725`, `#829`, `#914`, `#1012`, or the nearest tracked ball coordinates prior to the goal for Goal `#792` by `away_8` and Goal `#1157` by `home_18`—never invented coordinates).
3. **Comparable Unsuccessful Attack Comparison:** Select a transparent matching rule (same team & half + closest normalized shot location, same team across either half, closest normalized location across either team, or closest distance to goal mouth) to compare the selected goal side-by-side against a real unsuccessful shot (`is_goal == False` from a different possession sequence).
4. **Reproducible Descriptive Spatial Metrics & Factual Coach Observation:** At every frame and across the buildup timeline, the dashboard computes outfield-only (`role == "outfield"`) spatial metrics with units:
   - **Nearest Outfield Defender to Ball (`m`)**
   - **Attacking Outfield Width (`m`)** (`max(y) - min(y)`)
   - **Numbers Near Ball (`≤ 10 m`)** and **Penalty Box Occupation (`Attackers vs Defenders`)**
   - A plain-language factual summary comparing the goal frame and the unsuccessful shot frame without claiming causal proof.
5. **Match Story Overview:** Compares process parity against normalized attacking shot locations (`x_att`, `y_att`), showing that Away took 8 of its 14 shots from within 15 m (median 13.4 m, mean goal angle 25.4°) whereas Home took 12 of its 15 shots from outside 15 m (median 20.6 m, mean goal angle 17.2°).

---

## Exact Launch Command

From the worktree root (`/home/prg/orca/workspaces/veo-sprint-match-01/mockup-match-story`):

```bash
uv sync --locked
uv run streamlit run app.py --server.port 8501
```

Then open `http://localhost:8501` in your browser.

---

## What to Try

1. **Inspect Goal 5 (`#1012` at `83:02` video clock, `away_9`):**
   - Observe the 18-second counter-attack starting from `away_13`'s interception in Away's defending half (`x = +36.4 m`), progressing through `away_4` and `away_9` (`-x` attacking direction in Half 2).
   - Scrub the **Buildup Frame Timeline** slider or open the **Play / Pause Animation** expander to watch the 22 players and ball move across the pitch.
2. **Inspect the Two Goals Without Shot Rows (`#792` at `66:10` and `#1157` at `94:33`):**
   - Switch **Select Goal to Inspect** to **Goal 2 (`#792`, `away_8`)** or **Goal 6 (`#1157`, `home_18`)**.
   - Check the blue provenance banner: it explicitly notes that no shot row is attached in `events.csv` and reports the exact timestamp and coordinates of the tracked ball used as the reference location.
3. **Compare Against Unsuccessful Attacks:**
   - Change **Comparable Unsuccessful Shot Rule** in the sidebar or pick different ranked shots from the dropdown to see how defender proximity (`m`), attacking width (`m`), and penalty-box numbers (`Att vs Def`) evolve over the buildup window.
4. **Filter the Match Story Shot & Goal Map by Half:**
   - In the bottom section, toggle between **Full Match**, **First Half (0–0)**, and **Second Half (Away 4–2 Home)** on the `attack_dir`-normalized half-pitch map.

---

## Limitations & Data Caveats

- **Descriptive, Not Causal or Predictive:** With 29 shots and 6 goals from a single match, spatial metrics and comparisons describe what happened geometrically in the tracked frames; they do not establish causation or train a predictive xG model.
- **Video Clock (`timestamp_ms`):** All timestamps are video time (`timestamp_ms / 1000`), which starts before kick-off (`05:51`) and includes half-time.
- **Missing Ball Coordinates & Event Sampling:** Ball coordinates (`ball_x`, `ball_y`) are null in ~12% of tracking frames (e.g., around the corner kick on Goal `#1157` or Shot `#777`). Trajectories split cleanly across gaps and never interpolate across half-time. Furthermore, `events.csv` is a detected sample of on-ball actions—missing event rows do not establish absence of touches or possession.
- **Computer Vision Occlusions:** Occasional player tracking drop-outs occur in crowded penalty boxes (for example, `home_18` is not tracked in `players.parquet` during the corner-kick scramble on Goal `#1157`, though `events.csv` records `home_18` as the scorer).

---

## Validation

Run the automated compilation, analytical smoke tests, and headless Streamlit `AppTest` suite from the repository root:

```bash
uv run python -m py_compile analytics.py app.py test_prototype.py
uv run pytest -v test_prototype.py
```

What `test_prototype.py` verifies:
- Dataset baselines: 33 summary rows, 1,289 events, 6 `event_name == "goal"` rows (all lacking coordinates), 29 shots, and 4 shots with `is_goal == True`.
- All 6 goals resolve with explicit location/time provenance (4 linked `is_goal` shots, 2 tracked-ball references).
- Nearest-frame matching strictly within the same half and within `tolerance_ms`, rejecting cross-half timestamps.
- Trajectory segmentation splitting on missing ball coordinates, tracking gaps, and half-time boundaries.
- Outfield-only (`role == "outfield"`) spatial metrics and comparable shot ranking across all criteria.
- Headless Streamlit `AppTest` execution of `app.py` across multiple goal selections, slider changes, comparable shot rules, and half filters without errors.
