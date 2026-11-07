"""Analytical utilities for Second-Half Intensity & Movement Comparison.

Follows DATA_DICTIONARY.md rules:
- Coordinates in metres (-52.5..52.5 x, -33.85..33.85 y)
- Normalize attacking direction across halves via x_att = x * attack_dir, y_att = y * attack_dir
- timestamp_ms is video time in milliseconds; match events to nearest frame WITHIN the same half
- Split tracking gaps and half boundaries; never interpolate or animate across half-time
- Treat high-speed (speed_zone >= 3, >= 20 km/h) and sprint (speed_zone == 4, >= 25 km/h)
  frame shares strictly as sampled frame shares (percentage of observed player frames in that half)
- Compare only players meeting both-half coverage eligibility thresholds
- Distinguish percentage-point (pp) changes from relative percent (%) changes and handle zero denominators
- Explicitly state that observed second-half declines cannot establish physiological fatigue
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 67.7
TEAM_COLOURS = {
    "home": "#1f77b4",
    "away": "#d62728",
    "referee": "#7f7f7f",
}
SPEED_ZONE_LABELS = {
    0: "Zone 0 (0–10 km/h)",
    1: "Zone 1 (10–15 km/h)",
    2: "Zone 2 (15–20 km/h)",
    3: "Zone 3 (20–25 km/h, High-Speed)",
    4: "Zone 4 (25+ km/h, Sprint)",
}


def load_datasets(data_dir: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load player_summary.csv, events.csv, and players.parquet from repository root."""
    root = Path(data_dir)
    summary = pd.read_csv(root / "player_summary.csv")
    events = pd.read_csv(root / "events.csv")
    players = pd.read_parquet(root / "players.parquet")

    # Attack-direction normalized coordinates (DATA_DICTIONARY.md Section 1)
    players = players.copy()
    players["x_att"] = players["x"] * players["attack_dir"]
    players["y_att"] = players["y"] * players["attack_dir"]
    return summary, events, players


def validate_dataset_baselines(
    summary: pd.DataFrame, events: pd.DataFrame, players: pd.DataFrame
) -> dict[str, int | bool]:
    """Check core dataset invariants required by AGENTS.md and DATA_DICTIONARY.md."""
    goal_events = events[events["event_name"] == "goal"]
    shot_goals = events[events["is_goal"] == True]  # noqa: E712
    half_1_max = int(players.loc[players["half"] == 1, "timestamp_ms"].max())
    half_2_min = int(players.loc[players["half"] == 2, "timestamp_ms"].min())

    info = {
        "summary_rows": len(summary),
        "events_rows": len(events),
        "players_rows": len(players),
        "goal_event_rows": len(goal_events),
        "is_goal_shot_rows": len(shot_goals),
        "half_1_frames": int(players.loc[players["half"] == 1, "timestamp_ms"].nunique()),
        "half_2_frames": int(players.loc[players["half"] == 2, "timestamp_ms"].nunique()),
        "half_gap_ms": half_2_min - half_1_max,
        "baselines_ok": (
            len(summary) == 33
            and len(events) == 1289
            and len(goal_events) == 6
            and len(shot_goals) == 4
            and half_2_min > half_1_max
        ),
    }
    return info


def match_event_to_nearest_frame(
    players: pd.DataFrame, timestamp_ms: int, half: int, max_tolerance_ms: int = 1000
) -> pd.DataFrame:
    """Match an event timestamp to the nearest tracking frame WITHIN the same half.

    Returns an empty DataFrame if the half has no frames or the nearest frame exceeds max_tolerance_ms.
    """
    half_df = players[players["half"] == half]
    if half_df.empty:
        return half_df.iloc[0:0].copy()

    unique_ts = half_df["timestamp_ms"].unique()
    diffs = np.abs(unique_ts - int(timestamp_ms))
    min_idx = int(diffs.argmin())
    nearest_ts = int(unique_ts[min_idx])
    min_diff = int(diffs[min_idx])

    if min_diff > max_tolerance_ms:
        return half_df.iloc[0:0].copy()

    out = half_df[half_df["timestamp_ms"] == nearest_ts].copy()
    out["frame_offset_ms"] = nearest_ts - int(timestamp_ms)
    return out


def compute_per_half_player_metrics(
    players: pd.DataFrame, summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute per-half tracking metrics for every non-referee player.

    Metrics per half:
    - n_frames: number of observed tracking frames for the player in that half
    - half_total_frames: total unique tracking frames recorded in that half
    - coverage_pct: 100 * n_frames / half_total_frames
    - hs_frames: frames with speed_zone >= 3 (>= 20 km/h)
    - sprint_frames: frames with speed_zone == 4 (>= 25 km/h)
    - hs_share_pct: 100 * hs_frames / n_frames (sampled frame share)
    - sprint_share_pct: 100 * sprint_frames / n_frames (sampled frame share)
    - z0_share_pct .. z4_share_pct: sampled frame share in each speed_zone
    - dist_km: distance covered in that half from cumulative_distance_km delta
    - mean_x_att, mean_y_att: average attack-normalized pitch position (m)
    - max_gap_s: largest tracking gap (seconds) between consecutive frames for that player in that half
    """
    df = players[players["role"] != "referee"].copy()
    half_totals = players.groupby("half")["timestamp_ms"].nunique().to_dict()

    # Base player list from summary or players
    if summary is not None and not summary.empty:
        base_players = summary[["player_id", "team", "role", "minutes_played", "distance_km", "top_speed_kmh", "sprints"]].copy()
    else:
        base_players = df[["player_id", "team", "role"]].drop_duplicates().copy()
        base_players["minutes_played"] = np.nan
        base_players["distance_km"] = np.nan
        base_players["top_speed_kmh"] = np.nan
        base_players["sprints"] = np.nan

    # Create full grid of (player_id, half) for halves 1 and 2 so absent halves are explicit
    grid_rows = []
    for _, prow in base_players.iterrows():
        for half in (1, 2):
            grid_rows.append(
                {
                    "player_id": prow["player_id"],
                    "team": prow["team"],
                    "role": prow["role"],
                    "half": half,
                    "half_total_frames": int(half_totals.get(half, 0)),
                }
            )
    grid = pd.DataFrame(grid_rows)

    # Aggregate observed frames per (player_id, half)
    df_sorted = df.sort_values(["player_id", "half", "timestamp_ms"])

    def _max_gap_s(ts_series: pd.Series) -> float:
        if len(ts_series) <= 1:
            return 0.0
        diffs = np.diff(ts_series.to_numpy(dtype=np.int64))
        return float(diffs.max() / 1000.0)

    grouped = (
        df_sorted.groupby(["player_id", "half"], as_index=False)
        .agg(
            n_frames=("timestamp_ms", "count"),
            hs_frames=("speed_zone", lambda s: int((s >= 3).sum())),
            sprint_frames=("speed_zone", lambda s: int((s == 4).sum())),
            z0_frames=("speed_zone", lambda s: int((s == 0).sum())),
            z1_frames=("speed_zone", lambda s: int((s == 1).sum())),
            z2_frames=("speed_zone", lambda s: int((s == 2).sum())),
            z3_frames=("speed_zone", lambda s: int((s == 3).sum())),
            z4_frames=("speed_zone", lambda s: int((s == 4).sum())),
            min_cum_dist=("cumulative_distance_km", "min"),
            max_cum_dist=("cumulative_distance_km", "max"),
            mean_x_att=("x_att", "mean"),
            mean_y_att=("y_att", "mean"),
            std_x_att=("x_att", "std"),
            std_y_att=("y_att", "std"),
            max_gap_s=("timestamp_ms", _max_gap_s),
        )
    )

    merged = pd.merge(grid, grouped, on=["player_id", "half"], how="left")
    frame_cols = [
        "n_frames",
        "hs_frames",
        "sprint_frames",
        "z0_frames",
        "z1_frames",
        "z2_frames",
        "z3_frames",
        "z4_frames",
    ]
    for col in frame_cols:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["coverage_pct"] = np.where(
        merged["half_total_frames"] > 0,
        100.0 * merged["n_frames"] / merged["half_total_frames"],
        0.0,
    )

    # Sampled frame shares (handle n_frames == 0 safely with NaN)
    n_obs = merged["n_frames"].replace(0, np.nan)
    merged["hs_share_pct"] = 100.0 * merged["hs_frames"] / n_obs
    merged["sprint_share_pct"] = 100.0 * merged["sprint_frames"] / n_obs
    for z in range(5):
        merged[f"z{z}_share_pct"] = 100.0 * merged[f"z{z}_frames"] / n_obs

    merged["dist_km"] = (merged["max_cum_dist"] - merged["min_cum_dist"]).fillna(0.0)
    merged["max_gap_s"] = merged["max_gap_s"].fillna(0.0)

    return merged


def build_half_comparison_table(
    per_half_df: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    min_coverage_pct: float = 50.0,
    max_coverage_diff_pct: float = 100.0,
    include_goalkeepers: bool = False,
    team_filter: str = "All",
) -> pd.DataFrame:
    """Build a wide per-player comparison table across Half 1 and Half 2.

    Includes:
    - Half 1 and Half 2 frame counts, coverage_pct, hs_share_pct, sprint_share_pct, dist_km, mean_x_att
    - Percentage-point (pp) changes: h2 - h1
    - Relative percent (%) changes: 100 * (h2 - h1) / h1 when h1 > 0, else NaN (explicit zero-denominator handling)
    - Eligibility status & plain-English reason based on min_coverage_pct (in BOTH halves)
      and max_coverage_diff_pct (|cov_h2 - cov_h1| <= max_coverage_diff_pct).
    """
    df = per_half_df.copy()
    if not include_goalkeepers:
        df = df[df["role"] == "outfield"].copy()
    if team_filter in ("home", "away"):
        df = df[df["team"] == team_filter].copy()

    if df.empty:
        return pd.DataFrame()

    h1 = df[df["half"] == 1].set_index("player_id")
    h2 = df[df["half"] == 2].set_index("player_id")

    meta_cols = ["team", "role"]
    comp = h1[meta_cols].copy()

    value_cols = [
        "n_frames",
        "coverage_pct",
        "hs_frames",
        "sprint_frames",
        "hs_share_pct",
        "sprint_share_pct",
        "z0_share_pct",
        "z1_share_pct",
        "z2_share_pct",
        "z3_share_pct",
        "z4_share_pct",
        "dist_km",
        "mean_x_att",
        "mean_y_att",
        "max_gap_s",
    ]
    for col in value_cols:
        comp[f"{col}_h1"] = h1[col]
        comp[f"{col}_h2"] = h2[col]

    comp = comp.reset_index()

    if summary is not None and not summary.empty:
        sum_cols = summary[["player_id", "minutes_played", "distance_km", "top_speed_kmh", "sprints"]].rename(
            columns={"distance_km": "match_distance_km", "sprints": "match_sprints"}
        )
        comp = pd.merge(comp, sum_cols, on="player_id", how="left")

    # Coverage metrics and eligibility
    comp["min_half_coverage_pct"] = np.minimum(comp["coverage_pct_h1"], comp["coverage_pct_h2"])
    comp["coverage_diff_pp"] = comp["coverage_pct_h2"] - comp["coverage_pct_h1"]
    comp["abs_coverage_diff_pp"] = np.abs(comp["coverage_diff_pp"])

    def _classify_status(row: pd.Series) -> tuple[bool, str]:
        c1 = float(row["coverage_pct_h1"])
        c2 = float(row["coverage_pct_h2"])
        diff = abs(c2 - c1)
        if c1 == 0.0 and c2 > 0.0:
            return False, "2nd half only (absent in 1st half)"
        if c2 == 0.0 and c1 > 0.0:
            return False, "1st half only (absent in 2nd half)"
        if c1 == 0.0 and c2 == 0.0:
            return False, "No frames in either half"
        if c1 < min_coverage_pct or c2 < min_coverage_pct:
            return (
                False,
                f"Below min coverage ({min(c1, c2):.1f}% < {min_coverage_pct:.0f}%)",
            )
        if diff > max_coverage_diff_pct:
            return (
                False,
                f"Unequal half observation (|Δcov| {diff:.1f} pp > {max_coverage_diff_pct:.0f} pp)",
            )
        return True, "Eligible (observed in both halves)"

    statuses = comp.apply(_classify_status, axis=1)
    comp["is_eligible"] = [s[0] for s in statuses]
    comp["eligibility_status"] = [s[1] for s in statuses]

    # Compute changes ONLY where both halves have observed frames (and mark comparison availability)
    both_observed = (comp["n_frames_h1"] > 0) & (comp["n_frames_h2"] > 0)
    comp["both_halves_observed"] = both_observed

    # High-speed share changes (speed_zone >= 3)
    comp["hs_diff_pp"] = np.where(
        both_observed,
        comp["hs_share_pct_h2"] - comp["hs_share_pct_h1"],
        np.nan,
    )
    comp["hs_rel_change_pct"] = np.where(
        both_observed & (comp["hs_share_pct_h1"] > 0),
        100.0 * (comp["hs_share_pct_h2"] - comp["hs_share_pct_h1"]) / comp["hs_share_pct_h1"],
        np.nan,
    )
    comp["hs_rel_note"] = np.where(
        ~both_observed,
        "Absent in one half",
        np.where(
            comp["hs_share_pct_h1"] == 0,
            "Undefined (0% in H1)",
            "Valid",
        ),
    )

    # Sprint share changes (speed_zone == 4)
    comp["sprint_diff_pp"] = np.where(
        both_observed,
        comp["sprint_share_pct_h2"] - comp["sprint_share_pct_h1"],
        np.nan,
    )
    comp["sprint_rel_change_pct"] = np.where(
        both_observed & (comp["sprint_share_pct_h1"] > 0),
        100.0 * (comp["sprint_share_pct_h2"] - comp["sprint_share_pct_h1"]) / comp["sprint_share_pct_h1"],
        np.nan,
    )
    comp["sprint_rel_note"] = np.where(
        ~both_observed,
        "Absent in one half",
        np.where(
            comp["sprint_share_pct_h1"] == 0,
            "Undefined (0% in H1)",
            "Valid",
        ),
    )

    # Distance and attack-normalized depth shift
    comp["dist_diff_km"] = np.where(
        both_observed,
        comp["dist_km_h2"] - comp["dist_km_h1"],
        np.nan,
    )
    comp["x_att_shift_m"] = np.where(
        both_observed,
        comp["mean_x_att_h2"] - comp["mean_x_att_h1"],
        np.nan,
    )

    # Sort eligible players first, then by team and shirt number
    def _shirt_num(pid: str) -> int:
        parts = str(pid).split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 999

    comp["shirt_num"] = comp["player_id"].map(_shirt_num)
    comp = comp.sort_values(["is_eligible", "team", "shirt_num"], ascending=[False, True, True]).reset_index(drop=True)
    return comp


def compute_team_half_summary(
    comp_df: pd.DataFrame, only_eligible: bool = True
) -> pd.DataFrame:
    """Aggregate team-level Half 1 vs Half 2 frame shares across the selected player cohort.

    Computes both pooled frame shares (total high-speed frames / total observed frames)
    and unweighted player means across the cohort, handling empty cohorts gracefully.
    """
    if comp_df.empty:
        return pd.DataFrame()

    sub = comp_df[comp_df["is_eligible"]].copy() if only_eligible else comp_df[comp_df["both_halves_observed"]].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for team, g in sub.groupby("team"):
        f1 = int(g["n_frames_h1"].sum())
        f2 = int(g["n_frames_h2"].sum())
        hs1 = int(g["hs_frames_h1"].sum())
        hs2 = int(g["hs_frames_h2"].sum())
        sp1 = int(g["sprint_frames_h1"].sum())
        sp2 = int(g["sprint_frames_h2"].sum())

        hs_share_h1 = 100.0 * hs1 / f1 if f1 > 0 else np.nan
        hs_share_h2 = 100.0 * hs2 / f2 if f2 > 0 else np.nan
        sp_share_h1 = 100.0 * sp1 / f1 if f1 > 0 else np.nan
        sp_share_h2 = 100.0 * sp2 / f2 if f2 > 0 else np.nan

        hs_pp = hs_share_h2 - hs_share_h1 if (f1 > 0 and f2 > 0) else np.nan
        hs_rel = 100.0 * hs_pp / hs_share_h1 if (f1 > 0 and f2 > 0 and hs_share_h1 > 0) else np.nan

        sp_pp = sp_share_h2 - sp_share_h1 if (f1 > 0 and f2 > 0) else np.nan
        sp_rel = 100.0 * sp_pp / sp_share_h1 if (f1 > 0 and f2 > 0 and sp_share_h1 > 0) else np.nan

        rows.append(
            {
                "team": team,
                "n_players": len(g),
                "frames_h1": f1,
                "frames_h2": f2,
                "hs_share_h1_pct": hs_share_h1,
                "hs_share_h2_pct": hs_share_h2,
                "hs_diff_pp": hs_pp,
                "hs_rel_change_pct": hs_rel,
                "sprint_share_h1_pct": sp_share_h1,
                "sprint_share_h2_pct": sp_share_h2,
                "sprint_diff_pp": sp_pp,
                "sprint_rel_change_pct": sp_rel,
                "mean_x_att_h1_m": float(g["mean_x_att_h1"].mean()),
                "mean_x_att_h2_m": float(g["mean_x_att_h2"].mean()),
                "x_att_shift_m": float(g["x_att_shift_m"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_half_period_trends(
    players: pd.DataFrame,
    events: pd.DataFrame | None = None,
    bin_minutes: int = 5,
    eligible_player_ids: list[str] | None = None,
    include_goalkeepers: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute within-half time-binned movement intensity and team compactness trends.

    - Splits Half 1 and Half 2 strictly; never bridges or interpolates across half-time.
    - Uses video timestamp_ms relative to each half's kick-off (min timestamp_ms in that half)
      and also reports exact video minute ranges.
    - If eligible_player_ids is provided, filters high-speed/sprint frame shares to that cohort
      so substitutions don't confound the trend, while computing team shape (length/width/centroid)
      on all active outfield players on the pitch in each frame.
    """
    outfield = players[players["role"] == "outfield"].copy()
    if include_goalkeepers:
        intensity_df = players[players["role"] != "referee"].copy()
    else:
        intensity_df = outfield.copy()

    if eligible_player_ids is not None:
        intensity_df = intensity_df[intensity_df["player_id"].isin(eligible_player_ids)].copy()

    # Half start timestamps
    half_starts = players.groupby("half")["timestamp_ms"].min().to_dict()

    # 1. Intensity per (half, team, bin)
    if not intensity_df.empty:
        intensity_df["half_start_ms"] = intensity_df["half"].map(half_starts)
        intensity_df["half_elapsed_min"] = (intensity_df["timestamp_ms"] - intensity_df["half_start_ms"]) / 60000.0
        intensity_df["bin_start_min"] = (intensity_df["half_elapsed_min"] // bin_minutes).astype(int) * bin_minutes
        intensity_df["period_label"] = (
            "H"
            + intensity_df["half"].astype(str)
            + " "
            + intensity_df["bin_start_min"].astype(str)
            + "–"
            + (intensity_df["bin_start_min"] + bin_minutes).astype(str)
            + "'"
        )

        int_grouped = (
            intensity_df.groupby(["half", "team", "bin_start_min", "period_label"], as_index=False)
            .agg(
                player_frames=("timestamp_ms", "count"),
                hs_frames=("speed_zone", lambda s: int((s >= 3).sum())),
                sprint_frames=("speed_zone", lambda s: int((s == 4).sum())),
                min_video_min=("timestamp_ms", lambda s: float(s.min() / 60000.0)),
                max_video_min=("timestamp_ms", lambda s: float(s.max() / 60000.0)),
            )
        )
        int_grouped["hs_share_pct"] = np.where(
            int_grouped["player_frames"] > 0,
            100.0 * int_grouped["hs_frames"] / int_grouped["player_frames"],
            np.nan,
        )
        int_grouped["sprint_share_pct"] = np.where(
            int_grouped["player_frames"] > 0,
            100.0 * int_grouped["sprint_frames"] / int_grouped["player_frames"],
            np.nan,
        )
    else:
        int_grouped = pd.DataFrame(
            columns=[
                "half",
                "team",
                "bin_start_min",
                "period_label",
                "player_frames",
                "hs_frames",
                "sprint_frames",
                "min_video_min",
                "max_video_min",
                "hs_share_pct",
                "sprint_share_pct",
            ]
        )

    # 2. Outfield team compactness & attacking depth per frame, then averaged per bin
    outfield["half_start_ms"] = outfield["half"].map(half_starts)
    outfield["half_elapsed_min"] = (outfield["timestamp_ms"] - outfield["half_start_ms"]) / 60000.0
    outfield["bin_start_min"] = (outfield["half_elapsed_min"] // bin_minutes).astype(int) * bin_minutes

    frame_shape = (
        outfield.groupby(["half", "team", "bin_start_min", "timestamp_ms"], as_index=False)
        .agg(
            n_outfield=("player_id", "count"),
            x_min=("x_att", "min"),
            x_max=("x_att", "max"),
            y_min=("y_att", "min"),
            y_max=("y_att", "max"),
            mean_x_att=("x_att", "mean"),
            mean_y_att=("y_att", "mean"),
        )
    )
    frame_shape["length_m"] = frame_shape["x_max"] - frame_shape["x_min"]
    frame_shape["width_m"] = frame_shape["y_max"] - frame_shape["y_min"]

    shape_grouped = (
        frame_shape.groupby(["half", "team", "bin_start_min"], as_index=False)
        .agg(
            mean_length_m=("length_m", "mean"),
            mean_width_m=("width_m", "mean"),
            mean_x_att_m=("mean_x_att", "mean"),
            mean_outfield_tracked=("n_outfield", "mean"),
        )
    )

    trends = pd.merge(
        shape_grouped,
        int_grouped,
        on=["half", "team", "bin_start_min"],
        how="left",
    )
    if "period_label" not in trends.columns or trends["period_label"].isna().any():
        trends["period_label"] = (
            "H"
            + trends["half"].astype(str)
            + " "
            + trends["bin_start_min"].astype(str)
            + "–"
            + (trends["bin_start_min"] + bin_minutes).astype(str)
            + "'"
        )

    # Goal markers mapped to nearest tracking frame within the same half
    goal_markers = pd.DataFrame()
    if events is not None and not events.empty:
        goals = events[events["event_name"] == "goal"].copy()
        g_rows = []
        for _, grow in goals.iterrows():
            h = int(grow["half"])
            ts = int(grow["timestamp_ms"])
            matched = match_event_to_nearest_frame(players, ts, h, max_tolerance_ms=2000)
            nearest_ts = int(matched["timestamp_ms"].iloc[0]) if not matched.empty else ts
            h_start = int(half_starts.get(h, ts))
            elapsed_min = (nearest_ts - h_start) / 60000.0
            b_start = int(elapsed_min // bin_minutes) * bin_minutes
            g_rows.append(
                {
                    "event_id": int(grow["event_id"]),
                    "half": h,
                    "team": grow["team"],
                    "player_id": grow["player_id"],
                    "timestamp_ms": ts,
                    "nearest_frame_ms": nearest_ts,
                    "video_minute": float(grow["minute"]),
                    "half_elapsed_min": elapsed_min,
                    "bin_start_min": b_start,
                    "period_label": f"H{h} {b_start}–{b_start + bin_minutes}'",
                }
            )
        goal_markers = pd.DataFrame(g_rows)

    trends = trends.sort_values(["half", "bin_start_min", "team"]).reset_index(drop=True)
    return trends, goal_markers


def get_player_half_spatial_data(
    players: pd.DataFrame, player_id: str
) -> dict[int, pd.DataFrame]:
    """Return Half 1 and Half 2 tracking rows for a specific player, normalized by attack_dir.

    Includes segment_id incremented whenever consecutive frames have a gap > 400 ms so paths
    never connect across tracking gaps or half boundaries.
    """
    sub = players[players["player_id"] == player_id].sort_values(["half", "timestamp_ms"]).copy()
    result: dict[int, pd.DataFrame] = {}
    for half in (1, 2):
        h_df = sub[sub["half"] == half].copy()
        if not h_df.empty:
            dt = h_df["timestamp_ms"].diff().fillna(0)
            h_df["is_gap"] = dt > 450
            h_df["segment_id"] = h_df["is_gap"].cumsum()
        result[half] = h_df
    return result
