"""Core data loading, validation, and coach-friendly analytics for the Veo match dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PITCH_L = 105.0
PITCH_W = 67.7

TEAM_COLOURS = {
    "home": "#1f77b4",
    "away": "#d62728",
    "referee": "#888888",
}

TEAM_LABELS = {
    "home": "Home (Blue)",
    "away": "Away (Red)",
}

HALF_START_MS = {
    1: 351200,
    2: 3585200,
}

HALF_END_MS = {
    1: 3210000,
    2: 6486000,
}

FRAME_DT_SEC = 0.4


def load_datasets(root_dir: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load player_summary.csv, events.csv, and players.parquet from root_dir."""
    root = Path(root_dir)
    summary = pd.read_csv(root / "player_summary.csv")
    events = pd.read_csv(root / "events.csv")
    players = pd.read_parquet(root / "players.parquet")

    # Normalise boolean columns on events
    if "is_goal" in events.columns:
        events["is_goal"] = events["is_goal"].fillna(False).astype(bool)
    if "pass_success" in events.columns:
        events["pass_success"] = events["pass_success"].where(events["pass_success"].notna(), None)

    return summary, events, players


def format_video_clock(timestamp_ms: int | float) -> str:
    """Format video milliseconds as MM:SS video clock using integer seconds (timestamp_ms // 1000)."""
    if pd.isna(timestamp_ms):
        return "--:--"
    total_sec = int(float(timestamp_ms) // 1000)
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"


def format_inplay_clock(timestamp_ms: int | float, half: int) -> str:
    """Compute elapsed time since half kick-off (inferred from tracking half start, not an official referee match clock)."""
    if pd.isna(timestamp_ms) or half not in HALF_START_MS:
        return "Unknown"
    elapsed_ms = max(0, int(round(float(timestamp_ms))) - HALF_START_MS[int(half)])
    elapsed_sec = elapsed_ms // 1000
    mins = elapsed_sec // 60
    secs = elapsed_sec % 60
    return f"+{mins:02d}:{secs:02d} into H{int(half)} (inferred)"


def get_nearest_frame(
    players: pd.DataFrame,
    timestamp_ms: int | float,
    half: int,
    max_tolerance_ms: int = 1000,
) -> tuple[pd.DataFrame, int | None]:
    """
    Find the tracking rows for the frame nearest to timestamp_ms WITHIN the same half.
    Returns (frame_df, time_diff_ms). If no frame is within max_tolerance_ms, returns (empty_df, time_diff_ms).
    """
    half_df = players[players["half"] == int(half)]
    if half_df.empty:
        return pd.DataFrame(columns=players.columns), None

    unique_ts = half_df["timestamp_ms"].unique()
    diffs = np.abs(unique_ts - int(round(float(timestamp_ms))))
    min_idx = int(diffs.argmin())
    nearest_ts = int(unique_ts[min_idx])
    diff_ms = int(diffs[min_idx])

    if diff_ms > max_tolerance_ms:
        return pd.DataFrame(columns=players.columns), diff_ms

    frame_df = half_df[half_df["timestamp_ms"] == nearest_ts].copy()
    return frame_df, diff_ms


def split_trajectory_segments(
    track_df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    max_gap_ms: int = 1000,
) -> list[pd.DataFrame]:
    """
    Split a single-entity tracking DataFrame into continuous segments so plots never
    connect across half boundaries, missing coordinates, or tracking time gaps > max_gap_ms.
    """
    if track_df.empty:
        return []

    valid = track_df.dropna(subset=["timestamp_ms", "half", x_col, y_col]).sort_values("timestamp_ms").copy()
    if valid.empty:
        return []

    half_changed = valid["half"].diff().fillna(0) != 0
    time_gap = valid["timestamp_ms"].diff().fillna(0) > max_gap_ms
    segment_id = (half_changed | time_gap).cumsum()

    segments: list[pd.DataFrame] = []
    for _, seg in valid.groupby(segment_id):
        if not seg.empty:
            segments.append(seg)
    return segments


def compute_defensive_shape_heuristic(frame_df: pd.DataFrame, defending_team: str) -> dict[str, Any]:
    """
    Compute spatial heuristic metrics for the defending team's outfield players at a single frame.
    Explicitly filters out goalkeeper and referee.
    """
    outfield_def = frame_df[
        (frame_df["team"] == defending_team) & (frame_df["role"] == "outfield")
    ].dropna(subset=["x", "y"]).copy()

    if len(outfield_def) < 4:
        return {
            "n_outfield_defenders": len(outfield_def),
            "outfield_width_m": np.nan,
            "outfield_depth_m": np.nan,
            "back4_width_m": np.nan,
            "back4_depth_spread_m": np.nan,
            "is_stretched_heuristic": False,
            "stretch_reason": "Insufficient tracked outfield defenders (<4)",
        }

    # Normalize x relative to defending team's attacking direction:
    # Most negative x_att means closest to the goal they are defending.
    outfield_def["x_att"] = outfield_def["x"] * outfield_def["attack_dir"]
    outfield_width = float(outfield_def["y"].max() - outfield_def["y"].min())
    outfield_depth = float(outfield_def["x"].max() - outfield_def["x"].min())

    back4 = outfield_def.nsmallest(4, "x_att")
    back4_width = float(back4["y"].max() - back4["y"].min())
    back4_depth_spread = float(back4["x_att"].max() - back4["x_att"].min())

    reasons: list[str] = []
    if outfield_depth >= 50.0:
        reasons.append(f"Outfield length stretched ({outfield_depth:.1f} m >= 50 m)")
    if outfield_width >= 42.0:
        reasons.append(f"Outfield width stretched ({outfield_width:.1f} m >= 42 m)")
    if back4_width >= 28.0:
        reasons.append(f"Deepest 4 defenders wide ({back4_width:.1f} m >= 28 m)")

    is_stretched = len(reasons) > 0
    return {
        "n_outfield_defenders": int(len(outfield_def)),
        "outfield_width_m": round(outfield_width, 1),
        "outfield_depth_m": round(outfield_depth, 1),
        "back4_width_m": round(back4_width, 1),
        "back4_depth_spread_m": round(back4_depth_spread, 1),
        "is_stretched_heuristic": is_stretched,
        "stretch_reason": "; ".join(reasons) if reasons else "Compact / standard block",
    }


def compute_team_summary_metrics(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    players: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Compute trustworthy team-level metrics from source datasets.
    - Goals come strictly from events.csv where event_name == 'goal' (6 total: 4 away, 2 home).
    - Shots come from events.csv where event_name == 'shot' (29 total: 14 away, 15 home).
    - Outfield distance filters role == 'outfield'.
    """
    result: dict[str, dict[str, Any]] = {}
    total_team_events = len(events[events["team"].isin(["home", "away"])])

    for team in ["away", "home"]:
        t_sum = summary[summary["team"] == team]
        t_out_sum = t_sum[t_sum["role"] == "outfield"]
        t_ev = events[events["team"] == team]

        goals_df = t_ev[t_ev["event_name"] == "goal"]
        shots_df = t_ev[t_ev["event_name"] == "shot"]
        passes_df = t_ev[t_ev["event_name"] == "pass"]
        completed_passes = int(passes_df["pass_success"].fillna(False).sum())
        pass_pct = (100.0 * completed_passes / len(passes_df)) if len(passes_df) > 0 else 0.0

        shots_with_goal_flag = int(shots_df["is_goal"].fillna(False).sum())

        # Half splits for shots and goals
        h1_shots = int((shots_df["half"] == 1).sum())
        h2_shots = int((shots_df["half"] == 2).sum())
        h1_goals = int((goals_df["half"] == 1).sum())
        h2_goals = int((goals_df["half"] == 2).sum())

        # Outfield high intensity rate from players.parquet (speed_zone >= 3)
        t_p_out = players[(players["team"] == team) & (players["role"] == "outfield")]
        h1_p = t_p_out[t_p_out["half"] == 1]
        h2_p = t_p_out[t_p_out["half"] == 2]
        h1_hi_rate = 100.0 * float((h1_p["speed_zone"] >= 3).mean()) if len(h1_p) > 0 else 0.0
        h2_hi_rate = 100.0 * float((h2_p["speed_zone"] >= 3).mean()) if len(h2_p) > 0 else 0.0

        result[team] = {
            "team": team,
            "goals": int(len(goals_df)),
            "h1_goals": h1_goals,
            "h2_goals": h2_goals,
            "shots": int(len(shots_df)),
            "h1_shots": h1_shots,
            "h2_shots": h2_shots,
            "shots_with_is_goal": shots_with_goal_flag,
            "goals_without_shot_row": int(len(goals_df) - shots_with_goal_flag),
            "mean_shot_distance_m": round(float(shots_df["shot_distance_m"].mean()), 1) if not shots_df.empty else np.nan,
            "close_range_shots_le_15m": int((shots_df["shot_distance_m"] <= 15.0).sum()),
            "passes": int(len(passes_df)),
            "passes_completed": completed_passes,
            "pass_completion_pct": round(pass_pct, 1),
            "tackles": int((t_ev["event_name"] == "tackle").sum()),
            "interceptions": int((t_ev["event_name"] == "interception").sum()),
            "duels": int((t_ev["event_name"] == "duel").sum()),
            "outfield_distance_km": round(float(t_out_sum["distance_km"].sum()), 1),
            "total_distance_km": round(float(t_sum["distance_km"].sum()), 1),
            "total_sprints": int(t_sum["sprints"].sum()),
            "event_share_pct": round(100.0 * len(t_ev) / total_team_events, 1) if total_team_events else 0.0,
            "h1_hi_rate_pct": round(h1_hi_rate, 2),
            "h2_hi_rate_pct": round(h2_hi_rate, 2),
        }

    return result


def build_player_catalog(summary: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """
    Join player_summary.csv with per-half tracking frame counts and spatial averages from players.parquet.
    Explicitly separates summary minutes_played from H1/H2 tracked frames and rates.
    """
    non_ref = players[players["role"] != "referee"].copy()
    non_ref["x_att"] = non_ref["x"] * non_ref["attack_dir"]
    non_ref["y_att"] = non_ref["y"] * non_ref["attack_dir"]
    non_ref["is_hi"] = non_ref["speed_zone"] >= 3

    half_counts = (
        non_ref.groupby(["player_id", "half"])["timestamp_ms"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={1: "h1_frames", 2: "h2_frames"})
    )
    for col in ["h1_frames", "h2_frames"]:
        if col not in half_counts.columns:
            half_counts[col] = 0

    half_hi = (
        non_ref.groupby(["player_id", "half"])["is_hi"]
        .mean()
        .mul(100.0)
        .unstack(fill_value=0.0)
        .rename(columns={1: "h1_hi_pct", 2: "h2_hi_pct"})
    )
    for col in ["h1_hi_pct", "h2_hi_pct"]:
        if col not in half_hi.columns:
            half_hi[col] = 0.0

    spatial = non_ref.groupby("player_id").agg(
        total_tracked_frames=("timestamp_ms", "nunique"),
        mean_x_att_m=("x_att", "mean"),
        mean_y_att_m=("y_att", "mean"),
        hi_speed_frames=("is_hi", "sum"),
        hi_speed_rate_pct=("is_hi", lambda s: 100.0 * float(s.mean())),
        max_cum_dist_km=("cumulative_distance_km", "max"),
    )

    df = summary.merge(half_counts, on="player_id", how="left").merge(half_hi, on="player_id", how="left").merge(
        spatial, on="player_id", how="left"
    )
    df["h1_min_tracked"] = (df["h1_frames"] * FRAME_DT_SEC / 60.0).round(1)
    df["h2_min_tracked"] = (df["h2_frames"] * FRAME_DT_SEC / 60.0).round(1)
    df["tracked_minutes_est"] = (df["total_tracked_frames"] * FRAME_DT_SEC / 60.0).round(1)
    df["comparable_both_halves"] = (df["h1_frames"] >= 1500) & (df["h2_frames"] >= 1500)
    return df


def build_key_moments_catalog(events: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """
    Build a searchable, coach-friendly catalog of key moments to review:
    - All 6 Goal events (event_name == 'goal'), linking the associated shot if one exists,
      or noting when no shot row is attached (2 of the 6 goals).
    - All 29 Shot events (event_name == 'shot'), enriched with nearest-frame defensive shape heuristics.
    """
    goals = events[events["event_name"] == "goal"].copy()
    shots = events[events["event_name"] == "shot"].copy()

    # Map sequence_id or nearby timestamp to link the 4 shots with is_goal == True to their goal row
    goal_shot_map: dict[int, pd.Series] = {}
    for _, g_row in goals.iterrows():
        candidates = shots[
            (shots["half"] == g_row["half"])
            & (shots["team"] == g_row["team"])
            & (shots["is_goal"] == True)  # noqa: E712
            & (shots["timestamp_ms"] <= g_row["timestamp_ms"])
            & (shots["timestamp_ms"] >= g_row["timestamp_ms"] - 5000)
        ]
        if not candidates.empty:
            goal_shot_map[int(g_row["event_id"])] = candidates.iloc[-1]

    records: list[dict[str, Any]] = []

    # 1. Add all 6 goals first as primary review moments
    for idx, (_, g_row) in enumerate(goals.sort_values("timestamp_ms").iterrows(), start=1):
        ev_id = int(g_row["event_id"])
        ts_ms = int(g_row["timestamp_ms"])
        half = int(g_row["half"])
        team = str(g_row["team"])
        def_team = "home" if team == "away" else "away"
        linked_shot = goal_shot_map.get(ev_id)

        # Use linked shot timestamp for defensive frame if available, else goal timestamp
        ref_ts_ms = int(linked_shot["timestamp_ms"]) if linked_shot is not None else ts_ms
        frame_df, diff_ms = get_nearest_frame(players, ref_ts_ms, half)
        shape = compute_defensive_shape_heuristic(frame_df, def_team)

        if linked_shot is not None:
            shot_dist = round(float(linked_shot["shot_distance_m"]), 1)
            shot_ang = round(float(linked_shot["shot_angle_deg"]), 1)
            x_loc = float(linked_shot["x_start"])
            y_loc = float(linked_shot["y_start"])
            has_shot_row = True
            coord_note = (
                f"Goal row #{ev_id} has no coordinates in events.csv; location & shot metrics ({shot_dist} m, {shot_ang}°) "
                f"are linked from scoring shot event #{int(linked_shot['event_id'])} ({(ts_ms - ref_ts_ms)/1000:.1f}s earlier)."
            )
        else:
            # Estimate ball position from nearest tracking frame if available
            ball_rows = frame_df.dropna(subset=["ball_x", "ball_y"])
            x_loc = float(ball_rows["ball_x"].iloc[0]) if not ball_rows.empty else np.nan
            y_loc = float(ball_rows["ball_y"].iloc[0]) if not ball_rows.empty else np.nan
            shot_dist = np.nan
            shot_ang = np.nan
            has_shot_row = False
            coord_note = (
                f"Goal row #{ev_id} has no coordinates and NO attached `is_goal` shot row in events.csv "
                "(one of 2 unattached goals). Pitch marker uses nearest-frame tracked ball position."
            )

        vid_clock = format_video_clock(ts_ms)
        inplay_clock = format_inplay_clock(ts_ms, half)
        tags = ["Goal"]
        if shape["is_stretched_heuristic"]:
            tags.append("Stretched Defense (Heuristic)")
        if not has_shot_row:
            tags.append("No Attached Shot Row")

        obs = (
            f"Goal #{idx} ({team.upper()}, scorer `{g_row['player_id']}`) at video time {vid_clock} ({inplay_clock}). "
            f"{coord_note} Defending `{def_team}` outfield block measured {shape['outfield_width_m']} m wide × "
            f"{shape['outfield_depth_m']} m deep (deepest-4 width: {shape['back4_width_m']} m)."
        )

        records.append(
            {
                "moment_key": f"goal_{ev_id}",
                "event_id": ev_id,
                "linked_shot_event_id": int(linked_shot["event_id"]) if linked_shot is not None else None,
                "moment_type": "Goal",
                "timestamp_ms": ts_ms,
                "ref_frame_timestamp_ms": ref_ts_ms,
                "frame_offset_ms": diff_ms,
                "video_clock": vid_clock,
                "video_minute": round(float(g_row["minute"]), 2),
                "inplay_clock": inplay_clock,
                "half": half,
                "team": team,
                "defending_team": def_team,
                "player_id": str(g_row["player_id"]),
                "attack_dir": int(g_row["attack_dir"]),
                "sequence_id": int(g_row["sequence_id"]),
                "x": x_loc,
                "y": y_loc,
                "shot_distance_m": shot_dist,
                "shot_angle_deg": shot_ang,
                "is_goal": True,
                "has_shot_row": has_shot_row,
                "outfield_width_m": shape["outfield_width_m"],
                "outfield_depth_m": shape["outfield_depth_m"],
                "back4_width_m": shape["back4_width_m"],
                "back4_depth_spread_m": shape["back4_depth_spread_m"],
                "is_stretched_heuristic": bool(shape["is_stretched_heuristic"]),
                "stretch_reason": shape["stretch_reason"],
                "tags": ", ".join(tags),
                "label": (
                    f"[GOAL #{idx}] Video {vid_clock} ({inplay_clock}) | {team.upper()} - {g_row['player_id']} "
                    f"| {'Shot ' + str(shot_dist) + 'm' if has_shot_row else 'Unattached Goal'}"
                ),
                "coach_observation": obs,
            }
        )

    # 2. Add all 29 Shot events
    for idx, (_, s_row) in enumerate(shots.sort_values("timestamp_ms").iterrows(), start=1):
        ev_id = int(s_row["event_id"])
        ts_ms = int(s_row["timestamp_ms"])
        half = int(s_row["half"])
        team = str(s_row["team"])
        def_team = "home" if team == "away" else "away"
        is_g = bool(s_row["is_goal"])

        frame_df, diff_ms = get_nearest_frame(players, ts_ms, half)
        shape = compute_defensive_shape_heuristic(frame_df, def_team)

        shot_dist = round(float(s_row["shot_distance_m"]), 1)
        shot_ang = round(float(s_row["shot_angle_deg"]), 1)
        vid_clock = format_video_clock(ts_ms)
        inplay_clock = format_inplay_clock(ts_ms, half)

        tags = ["Shot"]
        if is_g:
            tags.append("Scored (is_goal=True)")
        if shot_dist <= 15.0:
            tags.append("Close Range (<=15m)")
        if shape["is_stretched_heuristic"]:
            tags.append("Stretched Defense (Heuristic)")

        outcome_str = "Resulted in a GOAL (`is_goal=True`)" if is_g else "Did not score (`is_goal=False`)"
        obs = (
            f"Shot #{idx} by `{s_row['player_id']}` ({team.upper()}) at video {vid_clock} ({inplay_clock}) "
            f"from {shot_dist} m ({shot_ang}° goal mouth angle). {outcome_str}. "
            f"At nearest tracking frame (Δt={diff_ms} ms), defending `{def_team}` outfield block was "
            f"{shape['outfield_width_m']} m wide × {shape['outfield_depth_m']} m deep "
            f"({shape['stretch_reason']})."
        )

        records.append(
            {
                "moment_key": f"shot_{ev_id}",
                "event_id": ev_id,
                "linked_shot_event_id": ev_id,
                "moment_type": "Shot",
                "timestamp_ms": ts_ms,
                "ref_frame_timestamp_ms": ts_ms,
                "frame_offset_ms": diff_ms,
                "video_clock": vid_clock,
                "video_minute": round(float(s_row["minute"]), 2),
                "inplay_clock": inplay_clock,
                "half": half,
                "team": team,
                "defending_team": def_team,
                "player_id": str(s_row["player_id"]),
                "attack_dir": int(s_row["attack_dir"]),
                "sequence_id": int(s_row["sequence_id"]),
                "x": float(s_row["x_start"]),
                "y": float(s_row["y_start"]),
                "shot_distance_m": shot_dist,
                "shot_angle_deg": shot_ang,
                "is_goal": is_g,
                "has_shot_row": True,
                "outfield_width_m": shape["outfield_width_m"],
                "outfield_depth_m": shape["outfield_depth_m"],
                "back4_width_m": shape["back4_width_m"],
                "back4_depth_spread_m": shape["back4_depth_spread_m"],
                "is_stretched_heuristic": bool(shape["is_stretched_heuristic"]),
                "stretch_reason": shape["stretch_reason"],
                "tags": ", ".join(tags),
                "label": (
                    f"[SHOT #{idx}{' - GOAL' if is_g else ''}] Video {vid_clock} ({inplay_clock}) | "
                    f"{team.upper()} - {s_row['player_id']} | {shot_dist} m ({shot_ang}°)"
                ),
                "coach_observation": obs,
            }
        )

    return pd.DataFrame(records)


def get_moment_window_data(
    events: pd.DataFrame,
    players: pd.DataFrame,
    moment_row: pd.Series | dict[str, Any],
    lead_seconds: int = 15,
) -> dict[str, Any]:
    """
    Extract lead-up events and same-half tracking frames for [T - lead_seconds, T] around a moment.
    Never crosses half boundaries.
    """
    ts_ms = int(moment_row["timestamp_ms"])
    ref_ts_ms = int(moment_row.get("ref_frame_timestamp_ms", ts_ms))
    half = int(moment_row["half"])
    start_ms = max(HALF_START_MS.get(half, 0), ts_ms - int(lead_seconds) * 1000)

    ev_win = events[
        (events["half"] == half)
        & (events["timestamp_ms"] >= start_ms)
        & (events["timestamp_ms"] <= ts_ms + 500)
    ].copy()
    ev_win["t_rel_s"] = ((ev_win["timestamp_ms"] - ts_ms) / 1000.0).round(1)
    ev_win["video_clock"] = ev_win["timestamp_ms"].apply(format_video_clock)

    track_win = players[
        (players["half"] == half)
        & (players["timestamp_ms"] >= start_ms)
        & (players["timestamp_ms"] <= ts_ms + 500)
    ].copy()
    track_win["t_rel_s"] = ((track_win["timestamp_ms"] - ts_ms) / 1000.0).round(1)

    snapshot_df, diff_ms = get_nearest_frame(players, ref_ts_ms, half)

    # Extract ball trajectory segments (handling ~12% missing ball_x/ball_y and frame gaps)
    ball_frames = track_win.drop_duplicates("timestamp_ms")[
        ["timestamp_ms", "half", "ball_x", "ball_y", "t_rel_s"]
    ].sort_values("timestamp_ms")
    total_ball_frames = len(ball_frames)
    valid_ball_frames = int(ball_frames["ball_x"].notna().sum())
    ball_segments = split_trajectory_segments(ball_frames, x_col="ball_x", y_col="ball_y", max_gap_ms=1000)

    return {
        "events_window": ev_win,
        "tracking_window": track_win,
        "snapshot_frame": snapshot_df,
        "snapshot_offset_ms": diff_ms,
        "ball_segments": ball_segments,
        "total_window_frames": total_ball_frames,
        "valid_ball_frames": valid_ball_frames,
        "missing_ball_frames": total_ball_frames - valid_ball_frames,
    }


def draw_pitch(ax: plt.Axes | None = None, figsize: tuple[float, float] = (10.5, 6.8)) -> plt.Axes:
    """Draw a standard 105.0 m × 67.7 m pitch centered at (0, 0) with metre labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    hl, hw = PITCH_L / 2.0, PITCH_W / 2.0
    ax.set_facecolor("#f7fbf7")
    ax.add_patch(plt.Rectangle((-hl, -hw), PITCH_L, PITCH_W, fill=False, lw=1.6, color="#3a4a3a"))
    ax.plot([0, 0], [-hw, hw], color="#3a4a3a", lw=1.4, ls="-")
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, lw=1.4, color="#3a4a3a"))
    ax.scatter([0], [0], color="#3a4a3a", s=18, zorder=2)

    for s in (-1, 1):
        # Penalty area (16.5 m x 40.3 m)
        ax.add_patch(
            plt.Rectangle((s * hl, -20.15), -s * 16.5, 40.3, fill=False, lw=1.4, color="#3a4a3a")
        )
        # 6-yard box (5.5 m x 18.32 m)
        ax.add_patch(
            plt.Rectangle((s * hl, -9.16), -s * 5.5, 18.32, fill=False, lw=1.4, color="#3a4a3a")
        )
        # Goal mouth (7.32 m wide)
        ax.plot([s * hl, s * hl], [-3.66, 3.66], color="#111111", lw=4.0, zorder=3)

    ax.set_xlim(-hl - 4.0, hl + 4.0)
    ax.set_ylim(-hw - 4.0, hw + 4.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Pitch length x (m, centre spot = 0 m)", fontsize=9, color="#333333")
    ax.set_ylabel("Pitch width y (m, centre spot = 0 m)", fontsize=9, color="#333333")
    ax.tick_params(axis="both", which="major", labelsize=8, colors="#444444")
    return ax


def plot_moment_pitch(
    moment_row: pd.Series | dict[str, Any],
    window_data: dict[str, Any],
    show_trails: bool = True,
    highlight_back4: bool = True,
) -> plt.Figure:
    """
    Plot the pitch snapshot at the selected moment with all 22 players, ball trajectory,
    lead-up pass/shot arrows, attacking direction banner, and optional deepest-4 defender box.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax=ax)

    snap = window_data["snapshot_frame"]
    track_win = window_data["tracking_window"]
    ev_win = window_data["events_window"]
    ball_segments = window_data["ball_segments"]

    att_team = str(moment_row["team"])
    def_team = str(moment_row["defending_team"])
    att_dir = int(moment_row["attack_dir"])
    goal_x = 52.5 * att_dir

    # Attacking direction indicator at top of pitch
    arrow_start_x = -12.0 * att_dir
    arrow_end_x = 12.0 * att_dir
    ax.annotate(
        f"{att_team.upper()} attacking {'+x (Right)' if att_dir == 1 else '-x (Left)'} in Half {int(moment_row['half'])}",
        xy=(arrow_end_x, 35.8),
        xytext=(arrow_start_x, 35.8),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=TEAM_COLOURS[att_team]),
        fontsize=9.5,
        fontweight="bold",
        color=TEAM_COLOURS[att_team],
        ha="center",
        va="center",
    )

    # 1. Player trails in lead-up window (split cleanly on any gaps)
    if show_trails and not track_win.empty:
        for pid, p_group in track_win[track_win["role"] != "referee"].groupby("player_id"):
            team = str(p_group["team"].iloc[0])
            col = TEAM_COLOURS.get(team, "#888888")
            is_actor = str(pid) == str(moment_row["player_id"])
            for seg in split_trajectory_segments(p_group, "x", "y", max_gap_ms=1000):
                ax.plot(
                    seg["x"],
                    seg["y"],
                    color=col,
                    lw=2.0 if is_actor else 0.9,
                    alpha=0.65 if is_actor else 0.22,
                    zorder=2,
                )

    # 2. Ball trajectory segments (dashed black/gold)
    for b_idx, b_seg in enumerate(ball_segments):
        ax.plot(
            b_seg["ball_x"],
            b_seg["ball_y"],
            color="#222222",
            lw=1.5,
            ls="--",
            alpha=0.75,
            label="Tracked ball path" if b_idx == 0 else None,
            zorder=4,
        )

    # 3. Lead-up pass arrows from events_window
    passes = ev_win[
        (ev_win["event_name"] == "pass")
        & ev_win["x_start"].notna()
        & ev_win["y_start"].notna()
        & ev_win["receiver_x"].notna()
        & ev_win["receiver_y"].notna()
    ]
    for _, p_row in passes.iterrows():
        p_col = TEAM_COLOURS.get(str(p_row["team"]), "#555555")
        ax.annotate(
            "",
            xy=(p_row["receiver_x"], p_row["receiver_y"]),
            xytext=(p_row["x_start"], p_row["y_start"]),
            arrowprops=dict(arrowstyle="->", lw=1.4, color=p_col, alpha=0.55),
            zorder=3,
        )

    # 4. Highlight deepest 4 outfield defenders if requested
    if highlight_back4 and not snap.empty:
        def_out = snap[(snap["team"] == def_team) & (snap["role"] == "outfield")].dropna(subset=["x", "y"]).copy()
        if len(def_out) >= 4:
            def_out["x_att"] = def_out["x"] * def_out["attack_dir"]
            back4 = def_out.nsmallest(4, "x_att").sort_values("y")
            ax.plot(
                back4["x"],
                back4["y"],
                color=TEAM_COLOURS[def_team],
                ls=":",
                lw=2.2,
                alpha=0.85,
                label=f"Defending deepest 4 ({moment_row['back4_width_m']} m wide)",
                zorder=4,
            )

    # 5. Plot players at the nearest snapshot frame
    if not snap.empty:
        for team_name in ["home", "away"]:
            t_rows = snap[snap["team"] == team_name].dropna(subset=["x", "y"])
            if t_rows.empty:
                continue
            out_rows = t_rows[t_rows["role"] == "outfield"]
            gk_rows = t_rows[t_rows["role"] == "goalkeeper"]
            col = TEAM_COLOURS[team_name]

            if not out_rows.empty:
                ax.scatter(
                    out_rows["x"],
                    out_rows["y"],
                    s=210,
                    color=col,
                    edgecolors="white",
                    linewidths=1.4,
                    label=f"{team_name.upper()} outfield",
                    zorder=5,
                )
            if not gk_rows.empty:
                ax.scatter(
                    gk_rows["x"],
                    gk_rows["y"],
                    s=230,
                    marker="s",
                    color=col,
                    edgecolors="#111111",
                    linewidths=1.6,
                    label=f"{team_name.upper()} goalkeeper",
                    zorder=5,
                )

            for _, prow in t_rows.iterrows():
                pid_str = str(prow["player_id"])
                shirt = pid_str.split("_")[-1] if "_" in pid_str else pid_str
                ax.text(
                    prow["x"],
                    prow["y"],
                    shirt,
                    color="white",
                    fontsize=7.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=6,
                )

        # Ball position at snapshot frame
        ball_snap = snap.dropna(subset=["ball_x", "ball_y"])
        if not ball_snap.empty:
            bx = float(ball_snap["ball_x"].iloc[0])
            by = float(ball_snap["ball_y"].iloc[0])
            ax.scatter(
                [bx],
                [by],
                s=95,
                color="#ffd700",
                edgecolors="#111111",
                linewidths=1.5,
                label="Ball at nearest frame",
                zorder=7,
            )

    # 6. Highlight the shot/goal event location and line to goal centre
    if pd.notna(moment_row["x"]) and pd.notna(moment_row["y"]):
        mx, my = float(moment_row["x"]), float(moment_row["y"])
        ax.scatter(
            [mx],
            [my],
            s=320,
            marker="*",
            color="#ffaa00",
            edgecolors="#111111",
            linewidths=1.4,
            label=f"{moment_row['moment_type']} location",
            zorder=8,
        )
        ax.plot(
            [mx, goal_x],
            [my, 0.0],
            color="#ffaa00",
            ls="-.",
            lw=1.8,
            alpha=0.85,
            zorder=4,
        )

    title_str = (
        f"{moment_row['label']} | Nearest Frame Δt = {window_data['snapshot_offset_ms']} ms"
    )
    ax.set_title(title_str, fontsize=10.5, fontweight="bold", pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def plot_player_pitch_profile(
    players: pd.DataFrame,
    events: pd.DataFrame,
    player_id: str,
    normalize_attack_dir: bool = True,
    half_filter: str = "All",
    view_mode: str = "Heatmap + On-Ball Events",
) -> plt.Figure:
    """
    Plot a selected player's spatial tracking distribution and on-ball events (passes, shots, goals, tackles/interceptions).
    When normalize_attack_dir=True, transforms coordinates by attack_dir so the player attacks Left -> Right (+x)
    across both halves, preventing half-time end-swap cancellation.
    """
    p_track = players[players["player_id"] == player_id].dropna(subset=["x", "y"]).copy()
    p_events = events[events["player_id"] == player_id].copy()

    if half_filter == "Half 1":
        p_track = p_track[p_track["half"] == 1]
        p_events = p_events[p_events["half"] == 1]
    elif half_filter == "Half 2":
        p_track = p_track[p_track["half"] == 2]
        p_events = p_events[p_events["half"] == 2]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax=ax)

    if p_track.empty:
        ax.set_title(f"No tracking frames for {player_id} in {half_filter}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig

    team = str(p_track["team"].iloc[0])
    team_col = TEAM_COLOURS.get(team, "#1f77b4")

    if normalize_attack_dir:
        p_track["plot_x"] = p_track["x"] * p_track["attack_dir"]
        p_track["plot_y"] = p_track["y"] * p_track["attack_dir"]
        ax.annotate(
            f"Normalized Attacking Direction ({player_id} attacks Left → Right +x)",
            xy=(18.0, 35.8),
            xytext=(-18.0, 35.8),
            arrowprops=dict(arrowstyle="->", lw=2.0, color=team_col),
            fontsize=9.5,
            fontweight="bold",
            color=team_col,
            ha="center",
            va="center",
        )
    else:
        p_track["plot_x"] = p_track["x"]
        p_track["plot_y"] = p_track["y"]
        ax.text(
            0.0,
            35.8,
            "Raw Pitch Coordinates (Teams swap ends at half-time)",
            fontsize=9.5,
            fontweight="bold",
            color="#444444",
            ha="center",
            va="center",
        )

    # 1. Tracking distribution: 2D histogram heatmap or speed-zone scatter
    if "Heatmap" in view_mode:
        cmap = "Reds" if team == "away" else "Blues"
        ax.hist2d(
            p_track["plot_x"],
            p_track["plot_y"],
            bins=[21, 14],
            range=[[-52.5, 52.5], [-33.85, 33.85]],
            cmap=cmap,
            cmin=5,
            alpha=0.55,
            zorder=2,
        )
    else:
        low_sp = p_track[p_track["speed_zone"] < 3]
        hi_sp = p_track[p_track["speed_zone"] >= 3]
        if not low_sp.empty:
            ax.scatter(
                low_sp["plot_x"],
                low_sp["plot_y"],
                s=6,
                color=team_col,
                alpha=0.15,
                label="Tracked frames (<20 km/h)",
                zorder=2,
            )
        if not hi_sp.empty:
            ax.scatter(
                hi_sp["plot_x"],
                hi_sp["plot_y"],
                s=18,
                color="#ff8800",
                alpha=0.75,
                label="High-intensity frames (>=20 km/h, Zone 3-4)",
                zorder=3,
            )

    # Average position marker
    mean_x = float(p_track["plot_x"].mean())
    mean_y = float(p_track["plot_y"].mean())
    ax.scatter(
        [mean_x],
        [mean_y],
        s=260,
        color=team_col,
        edgecolors="white",
        linewidths=2.0,
        label=f"Mean position ({mean_x:+.1f} m, {mean_y:+.1f} m)",
        zorder=6,
    )

    # 2. On-ball events overlay
    if "Events" in view_mode and not p_events.empty:
        ev_valid = p_events.dropna(subset=["x_start", "y_start"]).copy()
        if not ev_valid.empty:
            if normalize_attack_dir:
                ev_valid["ex"] = ev_valid["x_start"] * ev_valid["attack_dir"]
                ev_valid["ey"] = ev_valid["y_start"] * ev_valid["attack_dir"]
                ev_valid["rx"] = ev_valid["receiver_x"] * ev_valid["attack_dir"]
                ev_valid["ry"] = ev_valid["receiver_y"] * ev_valid["attack_dir"]
            else:
                ev_valid["ex"] = ev_valid["x_start"]
                ev_valid["ey"] = ev_valid["y_start"]
                ev_valid["rx"] = ev_valid["receiver_x"]
                ev_valid["ry"] = ev_valid["receiver_y"]

            # Passes
            passes = ev_valid[ev_valid["event_name"] == "pass"]
            comp_p = passes[passes["pass_success"] == True]  # noqa: E712
            inc_p = passes[passes["pass_success"] == False]  # noqa: E712

            for idx_p, (_, r) in enumerate(comp_p.iterrows()):
                if pd.notna(r["rx"]) and pd.notna(r["ry"]):
                    ax.annotate(
                        "",
                        xy=(r["rx"], r["ry"]),
                        xytext=(r["ex"], r["ey"]),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color="#1b8a3d", alpha=0.75),
                        zorder=4,
                    )
                ax.scatter(
                    [r["ex"]],
                    [r["ey"]],
                    s=32,
                    color="#1b8a3d",
                    alpha=0.8,
                    label="Completed pass" if idx_p == 0 else None,
                    zorder=4,
                )

            for idx_p, (_, r) in enumerate(inc_p.iterrows()):
                ax.scatter(
                    [r["ex"]],
                    [r["ey"]],
                    s=36,
                    marker="x",
                    color="#777777",
                    alpha=0.8,
                    label="Incomplete pass" if idx_p == 0 else None,
                    zorder=4,
                )

            # Defensive actions (tackle, interception, duel)
            def_ev = ev_valid[ev_valid["event_name"].isin(["tackle", "interception", "duel"])]
            if not def_ev.empty:
                ax.scatter(
                    def_ev["ex"],
                    def_ev["ey"],
                    s=70,
                    marker="^",
                    color="#8e44ad",
                    edgecolors="white",
                    linewidths=0.8,
                    label=f"Tackles/Interceptions/Duels ({len(def_ev)})",
                    zorder=5,
                )

            # Shots
            shot_ev = ev_valid[ev_valid["event_name"] == "shot"]
            if not shot_ev.empty:
                ax.scatter(
                    shot_ev["ex"],
                    shot_ev["ey"],
                    s=200,
                    marker="*",
                    color="#ffaa00",
                    edgecolors="#111111",
                    linewidths=1.2,
                    label=f"Shots ({len(shot_ev)})",
                    zorder=7,
                )

    tracked_min = len(p_track) * FRAME_DT_SEC / 60.0
    ax.set_title(
        f"Player Profile: {player_id} ({team.upper()}) | {half_filter} | {len(p_track):,} frames (~{tracked_min:.1f} tracked min)",
        fontsize=10.5,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8, frameon=True)
    fig.tight_layout()
    return fig
