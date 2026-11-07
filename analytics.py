"""Analytical engine for the Veo Off-Ball Space Creators prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 67.7
HALF_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_WIDTH_M = PITCH_WIDTH_M / 2.0

TEAM_COLORS = {
    "home": "#1f77b4",
    "away": "#d62728",
    "referee": "#888888",
}

SPEED_ZONE_LABELS = {
    0: "0–10 km/h (Walk/Jog)",
    1: "10–15 km/h (Low-speed run)",
    2: "15–20 km/h (Moderate run)",
    3: "20–25 km/h (High-speed run)",
    4: "25+ km/h (Sprint)",
}

MAX_FRAME_GAP_MS = 600
NEAREST_FRAME_TOLERANCE_MS = 1000
MIN_TRACK_DURATION_S = 1.6


def format_video_time(timestamp_ms: int | float) -> str:
    """Format video timestamp in milliseconds as MM:SS.s."""
    total_sec = float(timestamp_ms) / 1000.0
    minutes = int(total_sec // 60)
    seconds = total_sec - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def load_datasets(data_dir: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Load and validate core Veo match datasets."""
    root = Path(data_dir)
    summary = pd.read_csv(root / "player_summary.csv")
    events = pd.read_csv(root / "events.csv")
    players = pd.read_parquet(root / "players.parquet")

    # Precompute attacking coordinates as recommended in DATA_DICTIONARY.md
    players = players.copy()
    players["x_att"] = players["x"] * players["attack_dir"]
    players["y_att"] = players["y"] * players["attack_dir"]
    players["distance_to_ball_m"] = np.hypot(
        players["x"] - players["ball_x"],
        players["y"] - players["ball_y"],
    )

    return {
        "summary": summary,
        "events": events,
        "players": players,
    }


def build_attack_moments(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build the catalog of real shot and goal attacking moments.

    Notes from DATA_DICTIONARY.md:
    - Six `event_name == "goal"` rows are the complete record of goals (no coordinates).
    - Four `event_name == "shot"` rows have `is_goal == True` (with shot coordinates).
    - Two goals have no shot row attached (events 792 and 1157).
    We expose all 29 shots and all 6 goals clearly, linking the 4 shot+goal pairs so
    coaches can filter by All Attacks, Goals Only (all 6), or Shots Only (all 29)
    without losing the two unattached goals or double-counting blindly.
    """
    shots = events[events["event_name"] == "shot"].copy()
    goals = events[events["event_name"] == "goal"].copy()

    # Map goal rows to preceding is_goal shot if within 2 seconds in the same half/team
    linked_shot_to_goal: dict[int, int] = {}
    linked_goal_to_shot: dict[int, int] = {}

    for g in goals.itertuples():
        cand = shots[
            (shots["half"] == g.half)
            & (shots["team"] == g.team)
            & (shots["is_goal"] == True)  # noqa: E712
            & (shots["timestamp_ms"] <= g.timestamp_ms)
            & (shots["timestamp_ms"] >= g.timestamp_ms - 2500)
        ]
        if not cand.empty:
            s_id = int(cand.iloc[-1]["event_id"])
            g_id = int(g.event_id)
            linked_shot_to_goal[s_id] = g_id
            linked_goal_to_shot[g_id] = s_id

    records: list[dict[str, Any]] = []

    for r in events[events["event_name"].isin(["shot", "goal"])].itertuples():
        eid = int(r.event_id)
        ename = str(r.event_name)
        is_goal_shot = bool(r.is_goal) if pd.notna(r.is_goal) else False
        is_goal_moment = (ename == "goal") or is_goal_shot

        linked_goal_id = linked_shot_to_goal.get(eid)
        linked_shot_id = linked_goal_to_shot.get(eid)

        if ename == "shot" and is_goal_shot:
            moment_type = "Shot (Scored Goal)"
        elif ename == "shot":
            moment_type = "Shot (No Goal)"
        elif ename == "goal" and linked_shot_id is not None:
            moment_type = f"Goal Event (from Shot #{linked_shot_id})"
        else:
            moment_type = "Goal Event (No Shot Row Recorded)"

        dist_str = (
            f"{r.shot_distance_m:.1f} m"
            if pd.notna(r.shot_distance_m)
            else "Coords N/A on goal row"
        )
        vid_time = format_video_time(r.timestamp_ms)
        label = (
            f"[{vid_time} | H{r.half}] {r.team.upper()} - {moment_type} "
            f"by {r.player_id} ({dist_str}) [Event #{eid}]"
        )

        records.append(
            {
                "event_id": eid,
                "timestamp_ms": int(r.timestamp_ms),
                "minute": float(r.minute),
                "video_time": vid_time,
                "half": int(r.half),
                "team": str(r.team),
                "player_id": str(r.player_id),
                "attack_dir": int(r.attack_dir),
                "event_name": ename,
                "is_goal_shot": is_goal_shot,
                "is_goal_moment": is_goal_moment,
                "linked_goal_event_id": linked_goal_id,
                "linked_shot_event_id": linked_shot_id,
                "moment_type": moment_type,
                "x_start": float(r.x_start) if pd.notna(r.x_start) else np.nan,
                "y_start": float(r.y_start) if pd.notna(r.y_start) else np.nan,
                "shot_distance_m": (
                    float(r.shot_distance_m) if pd.notna(r.shot_distance_m) else np.nan
                ),
                "shot_angle_deg": (
                    float(r.shot_angle_deg) if pd.notna(r.shot_angle_deg) else np.nan
                ),
                "sequence_id": int(r.sequence_id) if pd.notna(r.sequence_id) else -1,
                "label": label,
            }
        )

    return pd.DataFrame(records)


def find_nearest_frame_in_half(
    players: pd.DataFrame,
    timestamp_ms: int,
    half: int,
    tolerance_ms: int = NEAREST_FRAME_TOLERANCE_MS,
) -> dict[str, Any]:
    """
    Match an event timestamp to the nearest player tracking frame WITHIN the same half.
    Explicitly checks tolerance and missing frames.
    """
    half_df = players[players["half"] == half]
    if half_df.empty:
        return {
            "matched": False,
            "frame_timestamp_ms": None,
            "offset_ms": None,
            "frame_df": pd.DataFrame(),
            "reason": f"No tracking frames available in half {half}.",
        }

    unique_ts = half_df["timestamp_ms"].unique()
    diffs = np.abs(unique_ts - int(timestamp_ms))
    min_idx = int(diffs.argmin())
    nearest_ts = int(unique_ts[min_idx])
    offset_ms = int(nearest_ts - int(timestamp_ms))

    if abs(offset_ms) > tolerance_ms:
        return {
            "matched": False,
            "frame_timestamp_ms": nearest_ts,
            "offset_ms": offset_ms,
            "frame_df": pd.DataFrame(),
            "reason": (
                f"Nearest frame in half {half} is {abs(offset_ms)} ms away "
                f"(exceeds {tolerance_ms} ms tolerance)."
            ),
        }

    frame_df = half_df[half_df["timestamp_ms"] == nearest_ts].copy()
    return {
        "matched": True,
        "frame_timestamp_ms": nearest_ts,
        "offset_ms": offset_ms,
        "frame_df": frame_df,
        "reason": "OK",
    }


def extract_continuous_same_half_window(
    players: pd.DataFrame,
    anchor_timestamp_ms: int,
    half: int,
    window_seconds: float,
    max_frame_gap_ms: int = MAX_FRAME_GAP_MS,
) -> dict[str, Any]:
    """
    Extract tracking frames in [anchor_timestamp_ms - window_seconds*1000, anchor_timestamp_ms]
    strictly within the same half, and trim to the latest continuous segment ending at
    the anchor frame if any global tracking gap > max_frame_gap_ms exists.
    """
    match_res = find_nearest_frame_in_half(players, anchor_timestamp_ms, half)
    if not match_res["matched"]:
        return {
            "valid": False,
            "reason": match_res["reason"],
            "requested_duration_s": float(window_seconds),
            "actual_duration_s": 0.0,
            "n_frames": 0,
            "ball_valid_frames": 0,
            "ball_coverage_pct": 0.0,
            "anchor_frame_ms": None,
            "start_frame_ms": None,
            "anchor_offset_ms": match_res["offset_ms"],
            "window_df": pd.DataFrame(),
            "was_gap_trimmed": False,
        }

    end_ms = int(match_res["frame_timestamp_ms"])
    start_bound_ms = int(round(end_ms - float(window_seconds) * 1000.0))

    half_df = players[
        (players["half"] == half)
        & (players["timestamp_ms"] >= start_bound_ms)
        & (players["timestamp_ms"] <= end_ms)
    ].copy()

    if half_df.empty:
        return {
            "valid": False,
            "reason": "Empty tracking window.",
            "requested_duration_s": float(window_seconds),
            "actual_duration_s": 0.0,
            "n_frames": 0,
            "ball_valid_frames": 0,
            "ball_coverage_pct": 0.0,
            "anchor_frame_ms": end_ms,
            "start_frame_ms": None,
            "anchor_offset_ms": match_res["offset_ms"],
            "window_df": pd.DataFrame(),
            "was_gap_trimmed": False,
        }

    sorted_ts = np.sort(half_df["timestamp_ms"].unique())
    was_gap_trimmed = False
    if len(sorted_ts) > 1:
        diffs = np.diff(sorted_ts)
        gap_indices = np.where(diffs > max_frame_gap_ms)[0]
        if len(gap_indices) > 0:
            # Keep only the continuous segment leading up to end_ms
            last_gap_idx = int(gap_indices[-1])
            valid_start_ts = int(sorted_ts[last_gap_idx + 1])
            half_df = half_df[half_df["timestamp_ms"] >= valid_start_ts].copy()
            sorted_ts = np.sort(half_df["timestamp_ms"].unique())
            was_gap_trimmed = True

    start_ms = int(sorted_ts[0])
    actual_duration_s = float(end_ms - start_ms) / 1000.0
    n_frames = len(sorted_ts)

    half_df["t_rel_s"] = (half_df["timestamp_ms"] - end_ms) / 1000.0

    # Ball coverage across frames in this window
    frame_balls = half_df.drop_duplicates("timestamp_ms")[["timestamp_ms", "ball_x", "ball_y"]]
    ball_valid_frames = int(frame_balls["ball_x"].notna().sum())
    ball_coverage_pct = (ball_valid_frames / n_frames * 100.0) if n_frames > 0 else 0.0

    return {
        "valid": True,
        "reason": "OK",
        "requested_duration_s": float(window_seconds),
        "actual_duration_s": actual_duration_s,
        "n_frames": n_frames,
        "ball_valid_frames": ball_valid_frames,
        "ball_coverage_pct": ball_coverage_pct,
        "anchor_frame_ms": end_ms,
        "start_frame_ms": start_ms,
        "anchor_offset_ms": int(match_res["offset_ms"]),
        "window_df": half_df,
        "was_gap_trimmed": was_gap_trimmed,
    }


def split_player_continuous_segments(
    player_df: pd.DataFrame,
    max_frame_gap_ms: int = MAX_FRAME_GAP_MS,
) -> list[pd.DataFrame]:
    """
    Split a single player's tracking rows into continuous same-half segments whenever
    a frame gap exceeds max_frame_gap_ms or a half boundary occurs.
    """
    if player_df.empty:
        return []

    df = player_df.sort_values("timestamp_ms").copy()
    ts = df["timestamp_ms"].to_numpy()
    halves = df["half"].to_numpy()

    split_points: list[int] = [0]
    for i in range(1, len(df)):
        if halves[i] != halves[i - 1] or (ts[i] - ts[i - 1]) > max_frame_gap_ms:
            split_points.append(i)
    split_points.append(len(df))

    segments: list[pd.DataFrame] = []
    for start_idx, end_idx in zip(split_points[:-1], split_points[1:]):
        seg = df.iloc[start_idx:end_idx].copy()
        if not seg.empty:
            segments.append(seg)
    return segments


def compute_off_ball_runs(
    window_info: dict[str, Any],
    events: pd.DataFrame,
    attacking_team: str,
    focal_player_id: str | None = None,
    ball_proximity_radius_m: float = 3.0,
    min_off_ball_ratio: float = 0.70,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate and rank candidate off-ball space-creating runs by outfield players on
    `attacking_team` within a continuous same-half tracking window.

    Heuristic components (all normalized to [0, 100] and combined with explainable weights):
    1. Forward Attacking Gain (`forward_disp_m`):
       Increase in attacking coordinate `x_att = x * attack_dir` from start to end of run.
       Scaled so +20 m forward gain = 100 pts (negative/backward gain = 0 pts).
    2. Nearest-Defender Separation Change (`def_sep_change_m`):
       Change in distance to the nearest outfield defender (`d_end - d_start`), plus
       credit for maintaining or creating open receiving space (`d_end`).
       Scaled so +6 m separation gain = 100 pts (0 m change = 40 pts baseline, <= -4 m = 0 pts).
    3. Off-Ball Width / Ball Separation (`mean_ball_dist_m`):
       Mean distance away from the ball across frames where ball coordinates exist,
       rewarding runs that stretch the defense away from the ball carrier (optimal 8–28 m).
    4. Sprinting / High-Speed Effort (`peak_speed_zone`, `total_dist_m`):
       Rewards purposeful high-tempo movement off the ball.

    Strict Data Integrity Rules:
    - Filter out `referee` and `goalkeeper` when evaluating outfield space creators and outfield defenders.
    - Split per-player tracking gaps; compute motion on the player's latest continuous segment in the window.
    - Define ball-proximity and off-ball eligibility honestly:
      * Incomplete event coverage in `events.csv` CANNOT prove absence of touches or possession.
      * We combine recorded on-ball events in the window with tracking ball-proximity (`distance_to_ball_m <= radius`).
      * If ball coordinates are missing for >50% of a player's frames (e.g., Goal #1157 at 94:33),
        ball-proximity metrics and composite off-ball rankings are explicitly marked `Unavailable (Missing Ball Coords)`
        with `np.nan` score rather than fabricating 0 or NaN rankings silently.
      * Similarly, if no outfield defenders are tracked at the start or end frame, defender separation
        and composite score are marked `Unavailable (Missing Defenders)`.
    """
    if weights is None:
        weights = {
            "forward": 0.40,
            "defender_sep": 0.30,
            "ball_dist": 0.15,
            "tempo": 0.15,
        }

    if not window_info.get("valid", False) or window_info["window_df"].empty:
        return pd.DataFrame()

    w_df: pd.DataFrame = window_info["window_df"]
    start_ms = int(window_info["start_frame_ms"])
    end_ms = int(window_info["anchor_frame_ms"])
    half = int(w_df["half"].iloc[0])

    # Outfield attackers and outfield defenders in the window
    attackers_df = w_df[
        (w_df["team"] == attacking_team) & (w_df["role"] == "outfield")
    ].copy()
    defending_team = "home" if attacking_team == "away" else "away"
    defenders_df = w_df[
        (w_df["team"] == defending_team) & (w_df["role"] == "outfield")
    ].copy()

    # On-ball events recorded during this window for the attacking team
    win_events = events[
        (events["half"] == half)
        & (events["timestamp_ms"] >= start_ms - 400)
        & (events["timestamp_ms"] <= end_ms + 400)
        & (events["team"] == attacking_team)
    ]

    # Pre-index defenders by timestamp_ms for fast nearest-defender lookup
    def_by_ts: dict[int, pd.DataFrame] = {
        int(ts): grp for ts, grp in defenders_df.groupby("timestamp_ms")
    }

    rows: list[dict[str, Any]] = []

    for pid, p_raw in attackers_df.groupby("player_id"):
        segments = split_player_continuous_segments(p_raw)
        if not segments:
            continue

        # Use the latest continuous segment in the window (closest to the shot/goal climax)
        seg = segments[-1].sort_values("timestamp_ms")
        n_player_frames = len(seg)
        had_tracking_gap = len(segments) > 1

        t_start = int(seg["timestamp_ms"].iloc[0])
        t_end = int(seg["timestamp_ms"].iloc[-1])
        avail_dur_s = float(t_end - t_start) / 1000.0

        # Start and end coordinates (real pitch + attacking frame)
        x_start = float(seg["x"].iloc[0])
        y_start = float(seg["y"].iloc[0])
        x_end = float(seg["x"].iloc[-1])
        y_end = float(seg["y"].iloc[-1])

        x_att_start = float(seg["x_att"].iloc[0])
        y_att_start = float(seg["y_att"].iloc[0])
        x_att_end = float(seg["x_att"].iloc[-1])
        y_att_end = float(seg["y_att"].iloc[-1])

        # Displacement and path distance within continuous segment
        forward_disp_m = x_att_end - x_att_start
        lateral_disp_m = abs(y_att_end - y_att_start)
        net_disp_m = float(np.hypot(x_end - x_start, y_end - y_start))

        dx = np.diff(seg["x"].to_numpy())
        dy = np.diff(seg["y"].to_numpy())
        path_dist_m = float(np.sum(np.hypot(dx, dy))) if len(dx) > 0 else 0.0

        peak_speed_zone = int(seg["speed_zone"].max())
        high_speed_frames = int((seg["speed_zone"] >= 3).sum())

        # Event involvement in window (note: incomplete event coverage cannot prove no touch)
        p_ev = win_events[
            (win_events["player_id"] == pid)
            | (win_events["receiver_player_id"] == pid)
        ]
        n_recorded_events = len(p_ev)
        is_focal_actor = (pid == focal_player_id)

        # Ball-proximity evaluation across frames where ball is located
        valid_ball_seg = seg[seg["ball_x"].notna() & seg["ball_y"].notna()]
        n_ball_frames = len(valid_ball_seg)
        ball_frame_ratio = (
            float(n_ball_frames) / float(n_player_frames) if n_player_frames > 0 else 0.0
        )

        if n_ball_frames > 0:
            ball_dists = valid_ball_seg["distance_to_ball_m"].to_numpy()
            min_ball_dist_m = float(np.min(ball_dists))
            mean_ball_dist_m = float(np.mean(ball_dists))
            near_ball_frames = int(np.sum(ball_dists <= ball_proximity_radius_m))
            off_ball_frame_ratio = float(
                np.sum(ball_dists > ball_proximity_radius_m)
            ) / float(n_ball_frames)
        else:
            min_ball_dist_m = np.nan
            mean_ball_dist_m = np.nan
            near_ball_frames = 0
            off_ball_frame_ratio = np.nan

        # Nearest outfield defender at start and end of player's continuous segment
        def_start_df = def_by_ts.get(t_start, pd.DataFrame())
        def_end_df = def_by_ts.get(t_end, pd.DataFrame())

        if not def_start_df.empty and not def_end_df.empty:
            dists_start = np.hypot(
                def_start_df["x"].to_numpy() - x_start,
                def_start_df["y"].to_numpy() - y_start,
            )
            min_i_start = int(np.argmin(dists_start))
            nearest_def_start_id = str(def_start_df["player_id"].iloc[min_i_start])
            nearest_def_start_m = float(dists_start[min_i_start])

            dists_end = np.hypot(
                def_end_df["x"].to_numpy() - x_end,
                def_end_df["y"].to_numpy() - y_end,
            )
            min_i_end = int(np.argmin(dists_end))
            nearest_def_end_id = str(def_end_df["player_id"].iloc[min_i_end])
            nearest_def_end_m = float(dists_end[min_i_end])

            def_sep_change_m = nearest_def_end_m - nearest_def_start_m

            # Also track how much the initial marking defender moved during the run
            # (observational only: never claim the attacker caused the defender's displacement)
            marker_at_end = def_end_df[
                def_end_df["player_id"] == nearest_def_start_id
            ]
            if not marker_at_end.empty:
                mx_start = float(def_start_df["x"].iloc[min_i_start])
                my_start = float(def_start_df["y"].iloc[min_i_start])
                mx_end = float(marker_at_end["x"].iloc[0])
                my_end = float(marker_at_end["y"].iloc[0])
                initial_marker_disp_m = float(
                    np.hypot(mx_end - mx_start, my_end - my_start)
                )
            else:
                initial_marker_disp_m = np.nan
        else:
            nearest_def_start_id = "Unavailable"
            nearest_def_start_m = np.nan
            nearest_def_end_id = "Unavailable"
            nearest_def_end_m = np.nan
            def_sep_change_m = np.nan
            initial_marker_disp_m = np.nan

        # Determine off-ball eligibility & metric availability honestly
        missing_reasons: list[str] = []
        win_ball_cov = float(window_info.get("ball_coverage_pct", 0.0)) / 100.0
        if avail_dur_s < MIN_TRACK_DURATION_S:
            missing_reasons.append(
                f"Short tracking segment ({avail_dur_s:.1f}s < {MIN_TRACK_DURATION_S:.1f}s)"
            )
        if win_ball_cov < 0.50 or ball_frame_ratio < 0.50 or np.isnan(mean_ball_dist_m):
            missing_reasons.append(
                f"Insufficient ball tracking ({n_ball_frames}/{n_player_frames} player frames; "
                f"{win_ball_cov * 100:.0f}% window ball coverage)"
            )
        if np.isnan(def_sep_change_m):
            missing_reasons.append("Missing outfield defender coordinates")

        metrics_available = len(missing_reasons) == 0

        if not metrics_available:
            eligibility_status = f"Metrics Unavailable ({'; '.join(missing_reasons)})"
            is_eligible_off_ball = False
        elif is_focal_actor:
            eligibility_status = "On-Ball Shooter / Scorer (Excluded from Off-Ball Rank)"
            is_eligible_off_ball = False
        elif off_ball_frame_ratio < min_off_ball_ratio:
            eligibility_status = (
                f"Primary Ball Carrier / Near Ball "
                f"({off_ball_frame_ratio * 100:.0f}% frames > {ball_proximity_radius_m:.1f}m from ball)"
            )
            is_eligible_off_ball = False
        else:
            if n_recorded_events > 0 or near_ball_frames > 0:
                eligibility_status = (
                    f"Eligible Off-Ball Runner (Mostly off-ball: "
                    f"{off_ball_frame_ratio * 100:.0f}% frames > {ball_proximity_radius_m:.1f}m; "
                    f"{n_recorded_events} event(s))"
                )
            else:
                eligibility_status = (
                    f"Eligible Off-Ball Runner (100% tracked frames > {ball_proximity_radius_m:.1f}m "
                    f"from ball; 0 recorded events*)"
                )
            is_eligible_off_ball = True

        # Compute heuristic sub-scores ONLY when underlying metrics are genuinely available
        if metrics_available:
            # 1. Forward displacement score (0 to 100, where +18m forward = 100)
            forward_score = float(np.clip(forward_disp_m / 18.0, 0.0, 1.0) * 100.0)

            # 2. Defender separation change score (0 to 100, where -3m -> 0, 0m -> 37.5, +5m -> 100)
            def_sep_score = float(
                np.clip((def_sep_change_m + 3.0) / 8.0, 0.0, 1.0) * 100.0
            )

            # 3. Distance away from ball score (rewards spacing 6m..25m away from ball, tapering >35m)
            if mean_ball_dist_m <= 3.0:
                ball_dist_score = 0.0
            elif mean_ball_dist_m <= 22.0:
                ball_dist_score = float((mean_ball_dist_m - 3.0) / 19.0 * 100.0)
            else:
                ball_dist_score = float(
                    np.clip(100.0 - (mean_ball_dist_m - 22.0) * 2.5, 30.0, 100.0)
                )

            # 4. Tempo / effort score (combines path distance and speed zone)
            dist_component = float(np.clip(path_dist_m / 22.0, 0.0, 1.0) * 60.0)
            speed_component = float((peak_speed_zone / 4.0) * 40.0)
            tempo_score = dist_component + speed_component

            w_sum = (
                weights["forward"]
                + weights["defender_sep"]
                + weights["ball_dist"]
                + weights["tempo"]
            )
            if w_sum <= 0:
                w_sum = 1.0

            heuristic_score = (
                weights["forward"] * forward_score
                + weights["defender_sep"] * def_sep_score
                + weights["ball_dist"] * ball_dist_score
                + weights["tempo"] * tempo_score
            ) / w_sum
        else:
            forward_score = np.nan
            def_sep_score = np.nan
            ball_dist_score = np.nan
            tempo_score = np.nan
            heuristic_score = np.nan

        rows.append(
            {
                "player_id": str(pid),
                "team": attacking_team,
                "is_eligible_off_ball": is_eligible_off_ball,
                "metrics_available": metrics_available,
                "eligibility_status": eligibility_status,
                "had_tracking_gap": had_tracking_gap,
                "n_frames": n_player_frames,
                "available_duration_s": avail_dur_s,
                "start_timestamp_ms": t_start,
                "end_timestamp_ms": t_end,
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end,
                "x_att_start": x_att_start,
                "y_att_start": y_att_start,
                "x_att_end": x_att_end,
                "y_att_end": y_att_end,
                "forward_disp_m": forward_disp_m,
                "lateral_disp_m": lateral_disp_m,
                "net_disp_m": net_disp_m,
                "path_dist_m": path_dist_m,
                "peak_speed_zone": peak_speed_zone,
                "peak_speed_label": SPEED_ZONE_LABELS.get(peak_speed_zone, "Unknown"),
                "high_speed_frames": high_speed_frames,
                "n_ball_frames": n_ball_frames,
                "ball_frame_ratio": ball_frame_ratio,
                "min_ball_dist_m": min_ball_dist_m,
                "mean_ball_dist_m": mean_ball_dist_m,
                "off_ball_frame_ratio": off_ball_frame_ratio,
                "n_recorded_events": n_recorded_events,
                "nearest_def_start_id": nearest_def_start_id,
                "nearest_def_start_m": nearest_def_start_m,
                "nearest_def_end_id": nearest_def_end_id,
                "nearest_def_end_m": nearest_def_end_m,
                "def_sep_change_m": def_sep_change_m,
                "initial_marker_disp_m": initial_marker_disp_m,
                "forward_score": forward_score,
                "def_sep_score": def_sep_score,
                "ball_dist_score": ball_dist_score,
                "tempo_score": tempo_score,
                "heuristic_score": heuristic_score,
            }
        )

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        return res_df

    # Sort eligible off-ball runners first by heuristic_score descending, then ineligible/unavailable
    res_df = res_df.sort_values(
        by=["is_eligible_off_ball", "metrics_available", "heuristic_score", "forward_disp_m"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Assign integer rank ONLY to eligible off-ball players with valid metrics
    ranks: list[int | None] = []
    current_rank = 1
    for r in res_df.itertuples():
        if r.is_eligible_off_ball and r.metrics_available and pd.notna(r.heuristic_score):
            ranks.append(current_rank)
            current_rank += 1
        else:
            ranks.append(None)
    res_df["off_ball_rank"] = pd.Series(ranks, dtype="Int64")

    return res_df


def compute_player_defender_timeline(
    window_info: dict[str, Any],
    attacker_id: str,
    attacking_team: str,
) -> pd.DataFrame:
    """
    Build a frame-by-frame time series for the selected attacking player over their
    continuous segment in the window, including:
    - Forward progression (`x_att`, `y_att`)
    - Distance to ball (`distance_to_ball_m`)
    - Nearest outfield defender ID and separation (`nearest_def_dist_m`)
    - Speed zone
    """
    if not window_info.get("valid", False) or window_info["window_df"].empty:
        return pd.DataFrame()

    w_df: pd.DataFrame = window_info["window_df"]
    defending_team = "home" if attacking_team == "away" else "away"

    p_raw = w_df[w_df["player_id"] == attacker_id].copy()
    segments = split_player_continuous_segments(p_raw)
    if not segments:
        return pd.DataFrame()

    seg = segments[-1].sort_values("timestamp_ms").copy()
    defenders_df = w_df[
        (w_df["team"] == defending_team) & (w_df["role"] == "outfield")
    ].copy()
    def_by_ts: dict[int, pd.DataFrame] = {
        int(ts): grp for ts, grp in defenders_df.groupby("timestamp_ms")
    }

    records: list[dict[str, Any]] = []
    x_att_0 = float(seg["x_att"].iloc[0])

    for r in seg.itertuples():
        ts = int(r.timestamp_ms)
        d_df = def_by_ts.get(ts, pd.DataFrame())
        if not d_df.empty:
            dists = np.hypot(
                d_df["x"].to_numpy() - float(r.x),
                d_df["y"].to_numpy() - float(r.y),
            )
            min_idx = int(np.argmin(dists))
            ndef_id = str(d_df["player_id"].iloc[min_idx])
            ndef_dist = float(dists[min_idx])
            ndef_x = float(d_df["x"].iloc[min_idx])
            ndef_y = float(d_df["y"].iloc[min_idx])
        else:
            ndef_id = "Unavailable"
            ndef_dist = np.nan
            ndef_x = np.nan
            ndef_y = np.nan

        records.append(
            {
                "timestamp_ms": ts,
                "t_rel_s": float(r.t_rel_s),
                "x": float(r.x),
                "y": float(r.y),
                "x_att": float(r.x_att),
                "y_att": float(r.y_att),
                "forward_gain_from_start_m": float(r.x_att) - x_att_0,
                "speed_zone": int(r.speed_zone),
                "speed_label": SPEED_ZONE_LABELS.get(int(r.speed_zone), "Unknown"),
                "ball_x": float(r.ball_x) if pd.notna(r.ball_x) else np.nan,
                "ball_y": float(r.ball_y) if pd.notna(r.ball_y) else np.nan,
                "distance_to_ball_m": (
                    float(r.distance_to_ball_m)
                    if pd.notna(r.distance_to_ball_m)
                    else np.nan
                ),
                "nearest_def_id": ndef_id,
                "nearest_def_dist_m": ndef_dist,
                "nearest_def_x": ndef_x,
                "nearest_def_y": ndef_y,
            }
        )

    return pd.DataFrame(records)
