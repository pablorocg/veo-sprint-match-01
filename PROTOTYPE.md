# Veo Football Analytics Sprint: Passing Options Explorer (`mockup-passing-options`)

## Purpose

The **Passing Options Explorer** is a local Streamlit prototype built for the Veo Football Analytics Sprint (Match 01). At any real on-ball pass event in `events.csv`, it retrieves the nearest same-half player tracking frame from `players.parquet` within an explicit millisecond tolerance, locates the tracked passer by `player_id`, and compares the **factual event pass** against **plausible snapshot-based teammate alternatives**.

Each candidate teammate lane is scored using three explainable 2D geometric components:
1. **Clamped Lane Clearance (`m`):** Minimum distance from any tracked opponent defender to the line segment connecting the tracked passer $A$ and candidate teammate $B$, with the projection parameter $t$ clamped strictly to $[0, 1]$ between the endpoints.
2. **Target Separation / Receiver Pressure (`m`):** Euclidean distance from the candidate teammate to their nearest tracked opponent defender.
3. **Normalized Forward Progression (`m`):** Forward yardage gained toward the opponent's goal, normalized across halves via $\Delta x_{\text{att}} = (x_{\text{target}} - x_{\text{passer}}) \times \text{attack\_dir}$.

## Exact Launch Command

From the repository root of this worktree (`/home/prg/orca/workspaces/veo-sprint-match-01/mockup-passing-options`), run:

```bash
uv run streamlit run app.py --server.port 8506
```

*(Default local port: **8506**)*

## What to Try

1. **Compare Factual Passes vs. Snapshot Alternatives across Teams & Halves:**
   - Use the sidebar **Attacking Team** (`Home (Blue)` / `Away (Red)`) and **Match Half** (`Half 1` / `Half 2`) toggles to filter the 832 pass events. Notice how the pitch banner and forward progression automatically adjust when teams swap ends at half-time (`attack_dir = +1` vs `-1`).
2. **Inspect Incomplete Passes & Missing End Coordinates:**
   - Filter **Pass Outcome Filter** to `Incomplete Only (pass_success=False)`. Observe how passes where `receiver_x`/`receiver_y` are present draw the full orange actual pass line, while passes with null `receiver_x`/`receiver_y` draw only the open orange start marker without fabricating an end location.
3. **Inspect Clamped Defender Projection on Specific Candidate Targets:**
   - Use the **Inspect Specific Candidate Target** dropdown above the pitch to highlight any teammate option in cyan (`#00f5d4`) and draw the dotted red perpendicular line from the nearest lane defender to their clamped projection point ($0 \le t \le 1$) on the passing segment.
4. **Adjust Risk vs. Progression Weighting:**
   - Move the sidebar sliders for **Lane Clearance Weight**, **Target Separation Weight**, and **Forward Progression Weight**. Watch the candidate rankings, composite scores, and top-N dashed lanes update immediately.
5. **Test Edge Cases (Unattributed Passers, Goalkeeper Option, Tight Frame Tolerance):**
   - Uncheck **"Only passes with identified passer (player_id)"** to inspect the 19 pass events with null `player_id` and see the explicit non-fabrication warning.
   - Reduce **Same-Half Frame Tolerance (± ms)** to `50 ms` on events that sit `~150 ms` from a tracking frame to verify out-of-tolerance handling.

## Limitations & Analytical Caveats

- **Heuristic Snapshot Lanes Only (Not `xPass` or Ball Physics):** Scores reflect static 2D positions at a single tracking frame. They do not model player velocity vectors, body orientation, offside rules, aerial chipped passes over defenders, or expected pass completion probabilities.
- **Event vs. Tracking Coordinate Source Distinction:** In `events.csv`, pass rows store their reception coordinates in `(receiver_x, receiver_y)` while `x_end`/`y_end` are null on all 832 pass rows. Furthermore, event coordinates (`x_start`, `y_start`) can differ by `~0.2–2.4 m` from the tracked passer's `(x, y)` in `players.parquet` at the nearest timestamp. Both are shown and explicitly labeled.
- **Unattributed Tracking & Event Rows:** 19 pass rows in `events.csv` have null `player_id`, and some tracking rows in `players.parquet` lack a `player_id`. Unattributed rows are never guessed or fabricated into candidate targets.

## Validation

Run syntax compilation, unit/invariant tests, and headless Streamlit `AppTest` verification with:

```bash
uv run python -m py_compile analytics.py app.py test_prototype.py
uv run pytest -v test_prototype.py
```
