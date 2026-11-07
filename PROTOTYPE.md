# Defensive Shape Explorer (`mockup-defensive-shape`)

## Purpose

The **Defensive Shape Explorer** is a local Streamlit prototype built for the Veo Football Analytics Sprint dataset (`veo-sprint-match-01`). It helps coaches and analysts inspect a team's outfield spatial organization—specifically **team width**, **team depth**, **compactness (mean distance to centroid & convex hull area)**, and the **largest longitudinal gap between adjacent outfield teammates**—across both halves of the match.

Instead of manually scrubbing through 90 minutes of video to find moments where a defensive block became disconnected, coaches can rank de-clustered peak stretched snapshots within a half, filter by tactical context (e.g., ball or team centroid in the defending half), inspect an interactive 105 × 67.7 m pitch overlay with exact video timestamps, and review nearby on-ball events.

---

## Launch Command

From the repository root (`/home/prg/orca/workspaces/veo-sprint-match-01/mockup-defensive-shape`):

```bash
uv run streamlit run app.py --server.port 8503
```

*(Note: `.streamlit/config.toml` also sets the default port to `8503` and `headless = true`.)*

---

## What to Try in the Dashboard

1. **Choose Defending Team & Half (Sidebar — Section 1)**:
   - Switch between **Away (Red)** and **Home (Blue)** and toggle between **Half 1** and **Half 2**.
   - Notice how attacking direction (`attack_dir`) automatically updates on the pitch banner and how each half is analyzed independently without interpolating across the half-time gap.
2. **Switch Shape Metric, Alert Threshold & Tactical Context Filter (Sidebar — Section 2)**:
   - Change **Shape Metric** between:
     - *Composite Stretch Index (z-score)*: weighted within-half z-score (`0.35 × z(Width) + 0.35 × z(Depth) + 0.30 × z(Max Longitudinal Gap)`).
     - *Outfield Team Width (y-span, m)*
     - *Outfield Team Depth (x-span, m)*
     - *Largest Longitudinal Gap (adjacent x-gap, m)*
     - *Mean Centroid Distance (dispersion, m)*
   - Adjust the **Stretch Alert Threshold** slider and watch the orange threshold line and highlighted exceedances update on the Half Timeline chart.
   - Change the **Tactical Context Filter** from *All adequate-coverage frames* to *Ball in defending half (`ball x_att < 0 m`)* or *Team centroid in own half (`centroid x_att < 0 m`)* to focus on out-of-possession / deeper defensive structures rather than attacking build-up.
   - Adjust **Minimum Observed Outfield Players** (default `8` out of `10` expected outfielders) and inspect the **Outfield Coverage** KPI card showing both current-frame player count and the percentage of half frames meeting the coverage threshold.
3. **Inspect Ranked Stretched Moments or Scrub Video Time (Main View)**:
   - Use **Top Ranked Stretched Moments** to step through the top 10 de-clustered stretched snapshots (separated by at least 15 seconds so a single 4-second spell does not flood the list).
   - Or switch to **Scrub Video Clock (All Half Frames)** to inspect any timestamp in the half.
   - Toggle pitch overlays (**Convex Hull**, **Width × Depth Bounding Box**, **Largest Longitudinal Gap**, **Opponent Players**) and review the **Nearby Match Events (±15s)** table and video clock timestamp (`MM:SS` and exact seconds) for video review.

---

## Methodological & Dataset Limitations

- **Stretched Shape ≠ Defensive Mistake**: A wide or deep outfield shape is a geometric observation, not proof of a tactical error. Teams deliberately expand width and depth during attacking possession, goal kicks, and counter-attacking transitions. The composite stretch index is explicitly labeled as a transparent within-half heuristic.
- **Camera Tracking Coverage**: Because tracking is derived from a broadcast/panoramic camera, out-of-frame players may temporarily drop out of `players.parquet` (or briefly overlap to 11 rows during a substitution window in Half 2). Outfield shape metrics strictly filter to `role == "outfield"` (excluding goalkeepers and the referee) and require `n_outfield >= min_outfield` (default 8) for ranking and timeline plotting. When scrubbing manually to a low-coverage frame, the app displays an explicit warning.
- **Event Coverage & Goal Accounting**: Missing event rows in `events.csv` do not establish absence of possession or touches—only logged on-ball actions appear. All 6 match goals occur in Half 2 and are identified via `event_name == "goal"` (which has no `x, y` coordinates); only 4 `shot` rows have `is_goal == True`.
- **Missing Ball Coordinates**: `ball_x` and `ball_y` are `NaN` in ~12% of tracking frames. The dashboard reports ball status transparently and handles missing coordinates without error.

---

## Validation Performed

Run the automated validation suite from the repository root:

```bash
uv run python -m py_compile analytics.py app.py test_prototype.py
uv run pytest -v test_prototype.py
```

The validation suite (`test_prototype.py`) verifies:
1. **Dataset baselines**: 33 `player_summary.csv` rows, 1,289 `events.csv` rows, 6 `event_name == "goal"` rows, 4 `shot` rows with `is_goal == True`, and pitch coordinate bounds.
2. **Nearest-frame timestamp matching**: Strictly matches within the same `half` and returns empty/unmatched when outside `max_tolerance_ms` or across the half boundary.
3. **Outfield filtering & shape invariants**: Confirms goalkeepers and the referee are excluded from outfield width, depth, compactness, and longitudinal gap calculations, and verifies `0 <= max_x_gap_m <= depth_m`.
4. **Attacking direction flip**: Confirms `attack_dir` reverses sign between Half 1 and Half 2 for both teams.
5. **De-clustered ranking & missing ball handling**: Verifies ranked moments are separated by at least 15s (`15,000 ms`) and frames with `NaN` ball coordinates render cleanly.
6. **Headless Streamlit `AppTest` & Server Check**: Executes `AppTest.from_file("app.py")` on default load and after changing team, half, metric selector, and moment selection mode, plus a live headless server check on `http://localhost:8503/_stcore/health`.
