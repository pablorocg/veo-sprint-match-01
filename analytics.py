"""Core data loading, defensive shape metrics, and moment ranking for the Veo prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

PITCH_L: float = 105.0
PITCH_W: float = 67.7
EXPECTED_OUTFIELD: int = 10
DEFAULT_TOLERANCE_MS: int = 1000
FRAME_STEP_MS: int = 400

TEAM_COLORS: dict[str, str] = {
    "home": "#1f77b4",
    "away": "#d62728",
}

TEAM_LABELS: dict[str, str] = {
    "home": "Home (Blue)",
    "away": "Away (Red)",
}

METRIC_OPTIONS: dict[str, dict[str, Any]] = {
    "stretch_score": {
        "label": "Composite Stretch Index (z-score)",
        "unit": "z",
        "short_label": "Stretch Index",
        "description": (
            "Weighted within-half z-score combining Team Width (35%), "
            "Team Depth (35%), and Largest Longitudinal Gap (30%)."
        ),
        "default_threshold": 1.25,
        "min_val": -1.0,
        "max_val": 3.0,
        "step": 0.1,
    },
    "width_m": {
        "label": "Outfield Team Width (y-span, m)",
        "unit": "m",
        "short_label": "Team Width",
        "description": "Lateral span from leftmost to rightmost observed outfield player (y_max - y_min).",
        "default_threshold": 48.0,
        "min_val": 20.0,
        "max_val": 67.0,
        "step": 1.0,
    },
    "depth_m": {
        "label": "Outfield Team Depth (x-span, m)",
        "unit": "m",
        "short_label": "Team Depth",
        "description": "Longitudinal span from deepest to highest observed outfield player (x_max - x_min).",
        "default_threshold": 42.0,
        "min_val": 15.0,
        "max_val": 85.0,
        "step": 1.0,
    },
    "max_x_gap_m": {
        "label": "Largest Longitudinal Gap (adjacent x-gap, m)",
        "unit": "m",
        "short_label": "Max Longitudinal Gap",
        "description": "Largest longitudinal distance between two adjacent outfield teammates along the x-axis.",
        "default_threshold": 15.0,
        "min_val": 5.0,
        "max_val": 40.0,
        "step": 0.5,
    },
    "compactness_m": {
        "label": "Mean Centroid Distance (dispersion, m)",
        "unit": "m",
        "short_label": "Mean Centroid Dist.",
        "description": "Average Euclidean distance of observed outfield players from the team's outfield centroid.",
        "default_threshold": 18.0,
        "min_val": 8.0,
        "max_val": 32.0,
        "step": 0.5,
    },
}


def format_video_time(timestamp_ms: float | int, include_seconds: bool = True) -> str:
    """Format a video millisecond timestamp as MM:SS (and optional exact seconds)."""
    if pd.isna(timestamp_ms):
        return "N/A"
    total_seconds = float(timestamp_ms) / 1000.0
    minutes = int(total_seconds // 60)
    seconds = int(round(total_seconds % 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    mm_ss = f"{minutes:02d}:{seconds:02d}"
    if include_seconds:
        return f"{mm_ss} ({total_seconds:.1f}s)"
    return mm_ss


def load_raw_datasets(data_dir: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load player_summary.csv, events.csv, and players.parquet from data_dir."""
    root = Path(data_dir)
    summary_df = pd.read_csv(root / "player_summary.csv")
    events_df = pd.read_csv(root / "events.csv")
    players_df = pd.read_parquet(root / "players.parquet")
    return summary_df, events_df, players_df


def validate_dataset_baselines(
    summary_df: pd.DataFrame,
    events_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> dict[str, Any]:
    """Verify repository dataset baselines documented in AGENTS.md and DATA_DICTIONARY.md."""
    goal_events = events_df[events_df["event_name"] == "goal"]
    shot_goals = events_df[
        (events_df["event_name"] == "shot") & (events_df["is_goal"] == True)  # noqa: E712
    ]
    return {
        "summary_rows": len(summary_df),
        "event_rows": len(events_df),
        "goal_event_rows": len(goal_events),
        "shot_is_goal_rows": len(shot_goals),
        "player_tracking_rows": len(players_df),
        "unique_frames": int(players_df["timestamp_ms"].nunique()),
        "halves": sorted(players_df["half"].dropna().unique().astype(int).tolist()),
    }


def get_nearest_frame(
    players_df: pd.DataFrame,
    timestamp_ms: float | int,
    half: int,
    max_tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> tuple[pd.DataFrame, int | None, int | None]:
    """Return tracking rows for the nearest timestamp strictly within the same half.

    Returns (frame_df, matched_timestamp_ms, time_diff_ms). If no frame exists in the
    requested half within max_tolerance_ms, returns an empty DataFrame and (None, time_diff_ms).
    """
    half_df = players_df[players_df["half"] == int(half)]
    if half_df.empty:
        return half_df.iloc[0:0].copy(), None, None

    unique_ts = half_df["timestamp_ms"].unique()
    diffs = np.abs(unique_ts - float(timestamp_ms))
    min_idx = int(np.argmin(diffs))
    nearest_ts = int(unique_ts[min_idx])
    diff_ms = int(round(float(diffs[min_idx])))

    if diff_ms > max_tolerance_ms:
        return half_df.iloc[0:0].copy(), None, diff_ms

    matched_df = half_df[half_df["timestamp_ms"] == nearest_ts].copy()
    return matched_df, nearest_ts, diff_ms


def compute_team_shape_timeseries(players_df: pd.DataFrame) -> pd.DataFrame:
    """Compute outfield defensive shape metrics for every (half, timestamp_ms, team) frame.

    Filters strictly to role == 'outfield' (excluding goalkeepers and referee).
    Computes width (y span), depth (x span), bounding box area, mean centroid distance
    (compactness_m), and largest adjacent longitudinal gap along x (max_x_gap_m) along with
    the two outfield players defining that gap.
    """
    # Extract frame-level metadata (every frame has a single ball_x, ball_y)
    frame_ball = (
        players_df.groupby(["half", "timestamp_ms"], as_index=False)[["ball_x", "ball_y"]]
        .first()
    )

    # Outfield players only (exclude goalkeeper and referee)
    outfield = players_df[
        (players_df["role"] == "outfield") & (players_df["team"].isin(["home", "away"]))
    ].copy()

    # Sort by half, timestamp_ms, team, x so adjacent longitudinal gaps are vectorized
    outfield = outfield.sort_values(["half", "timestamp_ms", "team", "x", "player_id"])

    group_cols = ["half", "timestamp_ms", "team"]
    grp = outfield.groupby(group_cols, sort=False)

    # Basic group aggregations
    agg_df = grp.agg(
        n_outfield=("player_id", "count"),
        attack_dir=("attack_dir", "first"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        y_min=("y", "min"),
        y_max=("y", "max"),
        centroid_x=("x", "mean"),
        centroid_y=("y", "mean"),
    ).reset_index()

    agg_df["width_m"] = agg_df["y_max"] - agg_df["y_min"]
    agg_df["depth_m"] = agg_df["x_max"] - agg_df["x_min"]
    agg_df["bbox_area_m2"] = agg_df["width_m"] * agg_df["depth_m"]

    # Mean Euclidean distance to team outfield centroid (compactness_m)
    outfield = outfield.merge(
        agg_df[group_cols + ["centroid_x", "centroid_y"]],
        on=group_cols,
        how="left",
    )
    outfield["dist_to_centroid"] = np.hypot(
        outfield["x"] - outfield["centroid_x"],
        outfield["y"] - outfield["centroid_y"],
    )
    compact_series = (
        outfield.groupby(group_cols, sort=False)["dist_to_centroid"]
        .mean()
        .reset_index(name="compactness_m")
    )

    # Adjacent longitudinal gaps along sorted x within each (half, timestamp_ms, team)
    same_group_prev = (
        (outfield["half"] == outfield["half"].shift(1))
        & (outfield["timestamp_ms"] == outfield["timestamp_ms"].shift(1))
        & (outfield["team"] == outfield["team"].shift(1))
    )
    outfield["x_prev"] = np.where(same_group_prev, outfield["x"].shift(1), np.nan)
    outfield["player_prev"] = np.where(same_group_prev, outfield["player_id"].shift(1), None)
    outfield["x_gap"] = outfield["x"] - outfield["x_prev"]

    valid_gaps = outfield.dropna(subset=["x_gap"])
    if not valid_gaps.empty:
        idx_max_gap = valid_gaps.groupby(group_cols, sort=False)["x_gap"].idxmax()
        max_gaps = valid_gaps.loc[
            idx_max_gap,
            group_cols + ["x_gap", "x_prev", "x", "player_prev", "player_id"],
        ].rename(
            columns={
                "x_gap": "max_x_gap_m",
                "x_prev": "gap_x_start",
                "x": "gap_x_end",
                "player_prev": "gap_player_low_x",
                "player_id": "gap_player_high_x",
            }
        )
    else:
        max_gaps = pd.DataFrame(
            columns=group_cols
            + ["max_x_gap_m", "gap_x_start", "gap_x_end", "gap_player_low_x", "gap_player_high_x"]
        )

    shape_df = (
        agg_df.merge(compact_series, on=group_cols, how="left")
        .merge(max_gaps, on=group_cols, how="left")
        .merge(frame_ball, on=["half", "timestamp_ms"], how="left")
    )

    # Normalized attacking coordinates (x_att = x * attack_dir; negative = defending own half)
    shape_df["centroid_x_att"] = shape_df["centroid_x"] * shape_df["attack_dir"]
    shape_df["ball_x_att"] = shape_df["ball_x"] * shape_df["attack_dir"]
    shape_df["deepest_defender_x_att"] = np.where(
        shape_df["attack_dir"] == 1,
        shape_df["x_min"],
        -shape_df["x_max"],
    )
    shape_df["highest_outfield_x_att"] = np.where(
        shape_df["attack_dir"] == 1,
        shape_df["x_max"],
        -shape_df["x_min"],
    )

    # Coverage ratio relative to expected 10 outfield players
    shape_df["coverage_pct"] = np.minimum(100.0, (shape_df["n_outfield"] / EXPECTED_OUTFIELD) * 100.0)
    shape_df["video_seconds"] = shape_df["timestamp_ms"] / 1000.0
    shape_df["video_clock"] = shape_df["timestamp_ms"].map(
        lambda ms: format_video_time(ms, include_seconds=False)
    )

    # Compute transparent within-half, within-team standardized Stretch Score
    # Using frames with >= 8 outfield players as the baseline distribution
    shape_df["stretch_score"] = np.nan
    for (half_val, team_val), sub_idx in shape_df.groupby(["half", "team"]).groups.items():
        sub = shape_df.loc[sub_idx]
        ref_mask = sub["n_outfield"] >= 8
        ref = sub[ref_mask] if ref_mask.sum() >= 10 else sub

        z_parts = []
        weights = [("width_m", 0.35), ("depth_m", 0.35), ("max_x_gap_m", 0.30)]
        for col, weight in weights:
            mean_val = ref[col].mean()
            std_val = ref[col].std()
            if pd.isna(std_val) or std_val <= 1e-6:
                z_col = pd.Series(0.0, index=sub.index)
            else:
                z_col = (sub[col] - mean_val) / std_val
            z_parts.append(weight * z_col)

        shape_df.loc[sub_idx, "stretch_score"] = sum(z_parts)

    shape_df = shape_df.sort_values(["half", "team", "timestamp_ms"]).reset_index(drop=True)
    return shape_df


def annotate_moment_context(
    row: pd.Series,
    events_df: pd.DataFrame,
    window_ms: int = 10000,
) -> str:
    """Summarize nearby events within +-window_ms in the same half for a ranked moment."""
    half = int(row["half"])
    ts = int(row["timestamp_ms"])
    half_events = events_df[
        (events_df["half"] == half)
        & (events_df["timestamp_ms"] >= ts - window_ms)
        & (events_df["timestamp_ms"] <= ts + window_ms)
    ].copy()

    if half_events.empty:
        return "No tagged on-ball events within ±10s"

    half_events["abs_diff"] = (half_events["timestamp_ms"] - ts).abs()
    half_events = half_events.sort_values("abs_diff")

    # Prioritize goals, shots, set pieces, then nearest event
    priority_events = half_events[
        half_events["event_name"].isin(["goal", "shot", "free_kick", "corner", "throw_in", "kick_off"])
    ]
    chosen = priority_events.iloc[0] if not priority_events.empty else half_events.iloc[0]
    rel_s = (int(chosen["timestamp_ms"]) - ts) / 1000.0
    sign_str = f"{rel_s:+.1f}s"
    team_str = str(chosen["team"]).capitalize() if pd.notna(chosen["team"]) else "Match"
    ev_name = str(chosen["event_name"]).replace("_", " ")
    return f"{team_str} {ev_name} ({sign_str})"


def get_ranked_stretched_moments(
    shape_df: pd.DataFrame,
    events_df: pd.DataFrame,
    team: str,
    half: int,
    rank_metric: str = "stretch_score",
    min_outfield: int = 8,
    context_filter: str = "all",
    min_separation_s: float = 15.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank distinct stretched snapshots within the same half and team.

    Parameters
    ----------
    shape_df : pd.DataFrame
        Output of compute_team_shape_timeseries.
    events_df : pd.DataFrame
        Match event log.
    team : str
        'home' or 'away'.
    half : int
        1 or 2.
    rank_metric : str
        Column to sort descending ('stretch_score', 'width_m', 'depth_m', 'max_x_gap_m', 'compactness_m').
    min_outfield : int
        Minimum observed outfield players required for a valid snapshot.
    context_filter : str
        'all' (all adequate-coverage frames),
        'ball_def_half' (ball observed and in defending team's half: ball_x_att < 0),
        'deep_block' (team outfield centroid in own half: centroid_x_att < 0).
    min_separation_s : float
        Minimum time gap (seconds) between ranked peak moments to avoid duplicate adjacent frames.
    top_n : int
        Maximum number of ranked moments to return.
    """
    sub = shape_df[
        (shape_df["team"] == team)
        & (shape_df["half"] == int(half))
        & (shape_df["n_outfield"] >= int(min_outfield))
    ].copy()

    if context_filter == "ball_def_half":
        sub = sub[sub["ball_x_att"].notna() & (sub["ball_x_att"] < 0.0)]
    elif context_filter == "deep_block":
        sub = sub[sub["centroid_x_att"] < 0.0]

    sub = sub.dropna(subset=[rank_metric])
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values([rank_metric, "timestamp_ms"], ascending=[False, True])

    selected_rows: list[pd.Series] = []
    selected_timestamps: list[int] = []
    min_sep_ms = int(round(min_separation_s * 1000.0))

    for _, row in sub.iterrows():
        ts = int(row["timestamp_ms"])
        if all(abs(ts - prev_ts) >= min_sep_ms for prev_ts in selected_timestamps):
            selected_rows.append(row)
            selected_timestamps.append(ts)
            if len(selected_rows) >= top_n:
                break

    ranked_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)
    ranked_df["video_time_label"] = ranked_df["timestamp_ms"].map(
        lambda ms: format_video_time(ms, include_seconds=True)
    )
    ranked_df["nearby_context"] = [
        annotate_moment_context(row, events_df, window_ms=10000)
        for _, row in ranked_df.iterrows()
    ]
    return ranked_df


def compute_convex_hull_vertices(points_xy: np.ndarray) -> np.ndarray:
    """Return closed polygon vertices (N+1, 2) for the 2D convex hull of points_xy."""
    if len(points_xy) < 3:
        return points_xy
    try:
        hull = ConvexHull(points_xy)
        verts = points_xy[hull.vertices]
        return np.vstack([verts, verts[0]])
    except Exception:
        return points_xy


def compute_convex_hull_area(points_xy: np.ndarray) -> float:
    """Return the 2D convex hull area in square metres (m^2), or 0.0 if < 3 points."""
    if len(points_xy) < 3:
        return 0.0
    try:
        hull = ConvexHull(points_xy)
        # In 2D, scipy ConvexHull.volume is area and .area is perimeter
        return float(hull.volume)
    except Exception:
        return 0.0


def get_moment_details(
    players_df: pd.DataFrame,
    shape_df: pd.DataFrame,
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    half: int,
    timestamp_ms: int | float,
    defending_team: str,
    event_window_s: float = 15.0,
    max_tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> dict[str, Any]:
    """Assemble full pitch snapshot, shape overlays, and nearby events for a video moment."""
    frame_df, matched_ts, diff_ms = get_nearest_frame(
        players_df,
        timestamp_ms=timestamp_ms,
        half=half,
        max_tolerance_ms=max_tolerance_ms,
    )
    if frame_df.empty or matched_ts is None:
        return {
            "found": False,
            "requested_timestamp_ms": int(timestamp_ms),
            "matched_timestamp_ms": None,
            "time_diff_ms": diff_ms,
            "half": int(half),
            "defending_team": defending_team,
        }

    # Extract jersey number / short label from player_id (e.g. 'home_7' -> '7')
    frame_df = frame_df.copy()
    frame_df["player_number"] = frame_df["player_id"].astype(str).str.split("_").str[-1]
    if not summary_df.empty and "player_id" in summary_df.columns:
        summary_cols = [
            c for c in ["player_id", "minutes_played", "tackles", "interceptions", "duels"]
            if c in summary_df.columns
        ]
        frame_df = frame_df.merge(
            summary_df[summary_cols].drop_duplicates("player_id"),
            on="player_id",
            how="left",
        )

    def_outfield = frame_df[
        (frame_df["team"] == defending_team) & (frame_df["role"] == "outfield")
    ].sort_values("x")
    def_gk = frame_df[
        (frame_df["team"] == defending_team) & (frame_df["role"] == "goalkeeper")
    ]
    opp_team = "home" if defending_team == "away" else "away"
    opp_players = frame_df[
        (frame_df["team"] == opp_team) & (frame_df["role"].isin(["outfield", "goalkeeper"]))
    ]
    referee_df = frame_df[frame_df["role"] == "referee"]

    shape_row_df = shape_df[
        (shape_df["half"] == int(half))
        & (shape_df["timestamp_ms"] == matched_ts)
        & (shape_df["team"] == defending_team)
    ]
    shape_metrics = shape_row_df.iloc[0].to_dict() if not shape_row_df.empty else {}

    points_xy = def_outfield[["x", "y"]].to_numpy(dtype=float)
    hull_vertices = compute_convex_hull_vertices(points_xy)
    hull_area_m2 = compute_convex_hull_area(points_xy)

    ball_x = float(frame_df["ball_x"].iloc[0]) if pd.notna(frame_df["ball_x"].iloc[0]) else np.nan
    ball_y = float(frame_df["ball_y"].iloc[0]) if pd.notna(frame_df["ball_y"].iloc[0]) else np.nan

    # Nearby events strictly within the same half
    window_ms = int(round(event_window_s * 1000.0))
    nearby_events = events_df[
        (events_df["half"] == int(half))
        & (events_df["timestamp_ms"] >= matched_ts - window_ms)
        & (events_df["timestamp_ms"] <= matched_ts + window_ms)
    ].copy()
    if not nearby_events.empty:
        nearby_events["rel_seconds"] = (nearby_events["timestamp_ms"] - matched_ts) / 1000.0
        nearby_events["video_clock"] = nearby_events["timestamp_ms"].map(
            lambda ms: format_video_time(ms, include_seconds=False)
        )
        nearby_events = nearby_events.sort_values("timestamp_ms")

    return {
        "found": True,
        "requested_timestamp_ms": int(timestamp_ms),
        "matched_timestamp_ms": matched_ts,
        "time_diff_ms": diff_ms,
        "half": int(half),
        "defending_team": defending_team,
        "opposing_team": opp_team,
        "def_outfield": def_outfield,
        "def_gk": def_gk,
        "opp_players": opp_players,
        "referee": referee_df,
        "shape_metrics": shape_metrics,
        "hull_vertices": hull_vertices,
        "hull_area_m2": hull_area_m2,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "nearby_events": nearby_events,
    }


def split_timeseries_on_gaps(
    df: pd.DataFrame,
    max_gap_ms: int = 800,
) -> pd.DataFrame:
    """Insert NaN break rows wherever timestamp_ms jumps by more than max_gap_ms.

    Prevents line charts from drawing false straight lines across missing coverage
    segments or tracking gaps within a half.
    """
    if df.empty:
        return df.copy()

    sorted_df = df.sort_values("timestamp_ms").reset_index(drop=True)
    dt = sorted_df["timestamp_ms"].diff()
    gap_indices = sorted_df.index[dt > max_gap_ms].tolist()
    if not gap_indices:
        return sorted_df

    segments: list[pd.DataFrame] = []
    start_idx = 0
    for idx in gap_indices:
        segments.append(sorted_df.iloc[start_idx:idx])
        nan_row = sorted_df.iloc[[idx - 1]].copy()
        mid_ts = int(sorted_df.loc[idx - 1, "timestamp_ms"]) + FRAME_STEP_MS
        for col in nan_row.columns:
            if col not in ("half", "team", "timestamp_ms", "video_seconds", "video_clock"):
                nan_row[col] = np.nan
        nan_row["timestamp_ms"] = mid_ts
        nan_row["video_seconds"] = mid_ts / 1000.0
        nan_row["video_clock"] = format_video_time(mid_ts, include_seconds=False)
        segments.append(nan_row)
        start_idx = idx

    segments.append(sorted_df.iloc[start_idx:])
    return pd.concat(segments, ignore_index=True)
