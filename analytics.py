"""Analytical functions for the Match Story & Goal Buildup Explorer prototype.

All functions operate on the real Veo Football Analytics Sprint dataset
(`player_summary.csv`, `events.csv`, `players.parquet`, `ball.csv`) while
respecting half boundaries, tracking gaps, attacking directions (`attack_dir`),
role filtering (`outfield` vs `goalkeeper` / `referee`), and missing ball
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 67.7
HALF_LENGTH_M = PITCH_LENGTH_M / 2.0  # 52.5
HALF_WIDTH_M = PITCH_WIDTH_M / 2.0  # 33.85
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_HALF_WIDTH_M = 20.15
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
GOAL_WIDTH_M = 7.32

TEAM_COLORS = {
    "away": "#d62728",
    "home": "#1f77b4",
    "referee": "#888888",
}

DEFAULT_FRAME_TOLERANCE_MS = 1000
TRACKING_GAP_SPLIT_MS = 1200


@dataclass(frozen=True)
class GoalReferenceInfo:
    """Resolved reference coordinate and timing for a goal event."""

    goal_event_id: int
    goal_timestamp_ms: int
    minute: float
    half: int
    team: str
    scorer_id: str
    attack_dir: int
    sequence_id: int
    linked_shot_event_id: int | None
    reference_timestamp_ms: int
    reference_x: float
    reference_y: float
    reference_x_att: float
    reference_y_att: float
    shot_distance_m: float
    shot_angle_deg: float | None
    location_source_label: str
    is_exact_shot_event: bool


def format_video_time(timestamp_ms: int | float) -> str:
    """Format video milliseconds into MM:SS.s video clock format."""
    if pd.isna(timestamp_ms):
        return "N/A"
    total_seconds = float(timestamp_ms) / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def load_datasets(root_dir: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Load the four match dataset files from root_dir and compute normalized coordinates."""
    root = Path(root_dir)
    summary = pd.read_csv(root / "player_summary.csv")
    events = pd.read_csv(root / "events.csv")
    players = pd.read_parquet(root / "players.parquet")
    ball = pd.read_csv(root / "ball.csv")

    # Add attacking-direction normalized coordinates on events and players
    events = events.copy()
    events["x_att"] = events["x_start"] * events["attack_dir"]
    events["y_att"] = events["y_start"] * events["attack_dir"]
    events["video_time_str"] = events["timestamp_ms"].apply(format_video_time)

    players = players.copy()
    players["x_att"] = players["x"] * players["attack_dir"]
    players["y_att"] = players["y"] * players["attack_dir"]

    return {
        "summary": summary,
        "events": events,
        "players": players,
        "ball": ball,
    }


def validate_dataset_baselines(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Check dataset baseline counts required by AGENTS.md and DATA_DICTIONARY.md."""
    summary = data["summary"]
    events = data["events"]
    goals = events[events["event_name"] == "goal"]
    shots_is_goal = events[(events["event_name"] == "shot") & (events["is_goal"] == True)]
    all_shots = events[events["event_name"] == "shot"]

    checks = {
        "summary_rows": len(summary),
        "events_rows": len(events),
        "goal_event_rows": len(goals),
        "shots_with_is_goal": len(shots_is_goal),
        "total_shots": len(all_shots),
        "goal_rows_have_no_coords": bool(
            goals["x_start"].isna().all() and goals["y_start"].isna().all()
        ),
    }
    assert checks["summary_rows"] == 33, f"Expected 33 summary rows, got {checks['summary_rows']}"
    assert checks["events_rows"] == 1289, f"Expected 1289 events rows, got {checks['events_rows']}"
    assert checks["goal_event_rows"] == 6, f"Expected 6 goal rows, got {checks['goal_event_rows']}"
    assert (
        checks["shots_with_is_goal"] == 4
    ), f"Expected 4 shots with is_goal, got {checks['shots_with_is_goal']}"
    assert checks["total_shots"] == 29, f"Expected 29 shots, got {checks['total_shots']}"
    assert checks["goal_rows_have_no_coords"], "Goal rows must have null x_start/y_start"
    return checks


def get_nearest_frame(
    players: pd.DataFrame,
    timestamp_ms: int | float,
    half: int,
    tolerance_ms: int = DEFAULT_FRAME_TOLERANCE_MS,
) -> tuple[pd.DataFrame, int | None, int | None]:
    """Return tracking rows for the nearest player frame WITHIN the same half.

    Returns (frame_df, matched_timestamp_ms, time_diff_ms).
    If no frame exists within tolerance_ms in the given half, returns (empty DataFrame, None, None).
    """
    half_df = players[players["half"] == half]
    if half_df.empty:
        return pd.DataFrame(columns=players.columns), None, None

    unique_ts = half_df["timestamp_ms"].unique()
    diffs = np.abs(unique_ts - int(timestamp_ms))
    min_idx = int(diffs.argmin())
    min_diff = int(diffs[min_idx])
    matched_ts = int(unique_ts[min_idx])

    if min_diff > tolerance_ms:
        return pd.DataFrame(columns=players.columns), None, min_diff

    frame_df = half_df[half_df["timestamp_ms"] == matched_ts].copy()
    return frame_df, matched_ts, min_diff


def split_trajectory_segments(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    max_gap_ms: int = TRACKING_GAP_SPLIT_MS,
) -> list[pd.DataFrame]:
    """Split a time-ordered tracking DataFrame into contiguous segments.

    Splits whenever:
    - `half` changes (never animate or draw lines across half-time),
    - `x_col` or `y_col` is NaN,
    - `timestamp_ms` jumps by more than `max_gap_ms`.
    """
    if df.empty:
        return []

    valid = df.dropna(subset=[x_col, y_col, "timestamp_ms", "half"]).sort_values("timestamp_ms")
    if valid.empty:
        return []

    ts_diff = valid["timestamp_ms"].diff().fillna(0)
    half_diff = valid["half"].diff().fillna(0)
    new_seg = (ts_diff > max_gap_ms) | (half_diff != 0)
    seg_ids = new_seg.cumsum()

    segments: list[pd.DataFrame] = []
    for _, seg in valid.groupby(seg_ids):
        if not seg.empty:
            segments.append(seg.copy())
    return segments


def resolve_all_goals(
    events: pd.DataFrame,
    players: pd.DataFrame,
    ball: pd.DataFrame,
) -> list[GoalReferenceInfo]:
    """Resolve all six `event_name == 'goal'` rows with transparent location/time provenance.

    - Four goals have a linked `shot` row with `is_goal == True` immediately preceding the goal.
    - Two goals (#792 by away_8 and #1157 by home_18) have no shot row in `events.csv`.
      For those, we never invent goal coordinates; we use the nearest tracked ball coordinate
      just before/at the goal timestamp in the same half and explicitly label its source and timestamp.
    """
    goal_rows = events[events["event_name"] == "goal"].sort_values("timestamp_ms")
    shot_rows = events[(events["event_name"] == "shot") & (events["is_goal"] == True)]

    resolved: list[GoalReferenceInfo] = []
    for _, g in goal_rows.iterrows():
        g_id = int(g["event_id"])
        g_ts = int(g["timestamp_ms"])
        g_half = int(g["half"])
        g_team = str(g["team"])
        g_scorer = str(g["player_id"])
        g_att_dir = int(g["attack_dir"])
        g_seq = int(g["sequence_id"])

        # Match linked shot with is_goal == True within 3 seconds prior in the same half & team
        cand_shots = shot_rows[
            (shot_rows["half"] == g_half)
            & (shot_rows["team"] == g_team)
            & (shot_rows["timestamp_ms"] <= g_ts)
            & (shot_rows["timestamp_ms"] >= g_ts - 3000)
        ]

        if not cand_shots.empty:
            s = cand_shots.iloc[-1]
            s_id = int(s["event_id"])
            ref_ts = int(s["timestamp_ms"])
            ref_x = float(s["x_start"])
            ref_y = float(s["y_start"])
            dist_m = float(s["shot_distance_m"])
            angle_deg = float(s["shot_angle_deg"]) if pd.notna(s["shot_angle_deg"]) else None
            src_label = (
                f"Linked shot event #{s_id} at {format_video_time(ref_ts)} "
                f"({(ref_ts - g_ts) / 1000.0:+.1f}s before goal row)"
            )
            is_exact = True
        else:
            s_id = None
            angle_deg = None
            is_exact = False
            # Look for nearest player frame at/before goal_ts with non-null ball_x/ball_y
            half_frames = (
                players[
                    (players["half"] == g_half)
                    & (players["timestamp_ms"] >= g_ts - 3000)
                    & (players["timestamp_ms"] <= g_ts)
                ]
                .drop_duplicates("timestamp_ms")
                .sort_values("timestamp_ms")
            )
            valid_frames = half_frames[half_frames["ball_x"].notna() & half_frames["ball_y"].notna()]
            if not valid_frames.empty:
                last_f = valid_frames.iloc[-1]
                ref_ts = int(last_f["timestamp_ms"])
                ref_x = float(last_f["ball_x"])
                ref_y = float(last_f["ball_y"])
                src_label = (
                    f"No shot event in events.csv; using tracked ball from players.parquet "
                    f"at {format_video_time(ref_ts)} ({(ref_ts - g_ts) / 1000.0:+.1f}s vs goal row)"
                )
            else:
                # Fallback to ball.csv within 3s before goal
                b_win = ball[
                    (ball["half"] == g_half)
                    & (ball["timestamp_ms"] >= g_ts - 3000)
                    & (ball["timestamp_ms"] <= g_ts)
                ].sort_values("timestamp_ms")
                last_b = b_win.iloc[-1]
                ref_ts = int(last_b["timestamp_ms"])
                ref_x = float(last_b["x"])
                ref_y = float(last_b["y"])
                src_label = (
                    f"No shot event in events.csv; using ball.csv trajectory "
                    f"at {format_video_time(ref_ts)} ({(ref_ts - g_ts) / 1000.0:+.1f}s vs goal row)"
                )

            goal_mouth_x = HALF_LENGTH_M * g_att_dir
            dist_m = float(np.hypot(ref_x - goal_mouth_x, ref_y - 0.0))

        resolved.append(
            GoalReferenceInfo(
                goal_event_id=g_id,
                goal_timestamp_ms=g_ts,
                minute=float(g["minute"]),
                half=g_half,
                team=g_team,
                scorer_id=g_scorer,
                attack_dir=g_att_dir,
                sequence_id=g_seq,
                linked_shot_event_id=s_id,
                reference_timestamp_ms=ref_ts,
                reference_x=ref_x,
                reference_y=ref_y,
                reference_x_att=ref_x * g_att_dir,
                reference_y_att=ref_y * g_att_dir,
                shot_distance_m=dist_m,
                shot_angle_deg=angle_deg,
                location_source_label=src_label,
                is_exact_shot_event=is_exact,
            )
        )
    return resolved


def compute_frame_spatial_metrics(
    frame_df: pd.DataFrame,
    attacking_team: str,
    ref_x: float | None = None,
    ref_y: float | None = None,
) -> dict[str, Any]:
    """Compute reproducible descriptive spatial metrics for a single tracking frame.

    Outfield players only (`role == 'outfield'`), excluding goalkeepers and referee.
    Metrics:
    1. `nearest_defender_m`: Euclidean distance (m) from ball (or reference point if ball missing)
       to the closest defending outfield player.
    2. `attacking_width_m`: Lateral spread (`max(y) - min(y)`, m) of attacking outfield players.
    3. `defensive_width_m`: Lateral spread (`max(y) - min(y)`, m) of defending outfield players.
    4. `attackers_near_ball_10m` / `defenders_near_ball_10m`: Outfield players within 10 m of ball.
    5. `attackers_in_box` / `defenders_in_box`: Outfield players inside the attacking penalty box
       (`x_att >= 36.0` m and `|y| <= 20.15` m).
    """
    if frame_df.empty:
        return {
            "ball_tracked": False,
            "ball_x": None,
            "ball_y": None,
            "nearest_defender_m": None,
            "nearest_defender_id": None,
            "attacking_width_m": None,
            "defensive_width_m": None,
            "attackers_near_ball_10m": None,
            "defenders_near_ball_10m": None,
            "attackers_in_box": 0,
            "defenders_in_box": 0,
            "tracked_attackers": 0,
            "tracked_defenders": 0,
        }

    bx = frame_df["ball_x"].iloc[0]
    by = frame_df["ball_y"].iloc[0]
    ball_tracked = bool(pd.notna(bx) and pd.notna(by))

    target_x = float(bx) if ball_tracked else (float(ref_x) if ref_x is not None and pd.notna(ref_x) else None)
    target_y = float(by) if ball_tracked else (float(ref_y) if ref_y is not None and pd.notna(ref_y) else None)

    att_out = frame_df[
        (frame_df["team"] == attacking_team) & (frame_df["role"] == "outfield")
    ].copy()
    def_out = frame_df[
        (frame_df["team"] != attacking_team)
        & (frame_df["team"].isin(["home", "away"]))
        & (frame_df["role"] == "outfield")
    ].copy()

    att_width = float(att_out["y"].max() - att_out["y"].min()) if len(att_out) >= 2 else None
    def_width = float(def_out["y"].max() - def_out["y"].min()) if len(def_out) >= 2 else None

    # Attacking penalty box in attacking direction coordinates
    if not att_out.empty:
        att_dir = int(att_out["attack_dir"].iloc[0])
    elif not def_out.empty:
        att_dir = -int(def_out["attack_dir"].iloc[0])
    else:
        att_dir = 1

    att_in_box = int(
        (((att_out["x"] * att_dir) >= (HALF_LENGTH_M - PENALTY_BOX_DEPTH_M)) & (att_out["y"].abs() <= PENALTY_BOX_HALF_WIDTH_M)).sum()
    )
    def_in_box = int(
        (((def_out["x"] * att_dir) >= (HALF_LENGTH_M - PENALTY_BOX_DEPTH_M)) & (def_out["y"].abs() <= PENALTY_BOX_HALF_WIDTH_M)).sum()
    )

    nearest_def_m: float | None = None
    nearest_def_id: str | None = None
    att_10m: int | None = None
    def_10m: int | None = None

    if target_x is not None and target_y is not None:
        if not def_out.empty:
            dists_def = np.hypot(def_out["x"] - target_x, def_out["y"] - target_y)
            min_idx = dists_def.idxmin()
            nearest_def_m = float(dists_def.loc[min_idx])
            nearest_def_id = str(def_out.loc[min_idx, "player_id"])
            def_10m = int((dists_def <= 10.0).sum())
        if not att_out.empty:
            dists_att = np.hypot(att_out["x"] - target_x, att_out["y"] - target_y)
            att_10m = int((dists_att <= 10.0).sum())

    return {
        "ball_tracked": ball_tracked,
        "ball_x": float(bx) if ball_tracked else None,
        "ball_y": float(by) if ball_tracked else None,
        "nearest_defender_m": nearest_def_m,
        "nearest_defender_id": nearest_def_id,
        "attacking_width_m": att_width,
        "defensive_width_m": def_width,
        "attackers_near_ball_10m": att_10m,
        "defenders_near_ball_10m": def_10m,
        "attackers_in_box": att_in_box,
        "defenders_in_box": def_in_box,
        "tracked_attackers": len(att_out),
        "tracked_defenders": len(def_out),
    }


def extract_buildup_window(
    events: pd.DataFrame,
    players: pd.DataFrame,
    anchor_timestamp_ms: int,
    half: int,
    attacking_team: str,
    lead_in_seconds: float = 18.0,
    post_seconds: float = 1.0,
) -> dict[str, Any]:
    """Extract tracking frames, time-series spatial metrics, and events for a buildup window.

    Strictly filters to `half == half` so half boundaries are never crossed.
    """
    start_ms = int(anchor_timestamp_ms - lead_in_seconds * 1000)
    end_ms = int(anchor_timestamp_ms + post_seconds * 1000)

    win_players = players[
        (players["half"] == half)
        & (players["timestamp_ms"] >= start_ms)
        & (players["timestamp_ms"] <= end_ms)
    ].copy()
    win_players["t_rel_s"] = (win_players["timestamp_ms"] - anchor_timestamp_ms) / 1000.0

    win_events = events[
        (events["half"] == half)
        & (events["timestamp_ms"] >= start_ms)
        & (events["timestamp_ms"] <= end_ms)
    ].copy()
    win_events["t_rel_s"] = (win_events["timestamp_ms"] - anchor_timestamp_ms) / 1000.0

    frame_timestamps = sorted(win_players["timestamp_ms"].unique().tolist())
    metrics_rows: list[dict[str, Any]] = []
    for ts in frame_timestamps:
        f_df = win_players[win_players["timestamp_ms"] == ts]
        m = compute_frame_spatial_metrics(f_df, attacking_team=attacking_team)
        m["timestamp_ms"] = int(ts)
        m["t_rel_s"] = (int(ts) - anchor_timestamp_ms) / 1000.0
        m["video_time_str"] = format_video_time(ts)
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)

    return {
        "anchor_timestamp_ms": anchor_timestamp_ms,
        "half": half,
        "attacking_team": attacking_team,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "players_window": win_players,
        "events_window": win_events,
        "frame_timestamps": frame_timestamps,
        "metrics_timeline": metrics_df,
    }


def find_comparable_unsuccessful_shots(
    events: pd.DataFrame,
    goal_info: GoalReferenceInfo,
    criterion: str = "same_team_half_location",
) -> pd.DataFrame:
    """Rank unsuccessful shots (`event_name == 'shot'` and `is_goal == False`) against a goal.

    Also excludes earlier rebound/blocked shots that occurred inside the exact same
    goal-scoring possession sequence (`sequence_id == goal_info.sequence_id`) so we
    compare against an genuinely unsuccessful attack.

    Supported transparent criteria:
    - `same_team_half_location`: Same team & half, ranked by normalized shot location distance (m)
    - `same_team_any_half_location`: Same team (either half), ranked by normalized shot location distance (m)
    - `closest_location_any_team`: Any team, closest normalized pitch coordinates `(x_att, y_att)`
    - `closest_shot_distance`: Same team, closest distance to goal centre (`shot_distance_m`)
    """
    unsuccessful = events[
        (events["event_name"] == "shot")
        & (~events["is_goal"].fillna(False).astype(bool))
        & (events["sequence_id"] != goal_info.sequence_id)
    ].copy()

    # Compute normalized coordinates distance to the goal's reference point
    unsuccessful["loc_diff_m"] = np.hypot(
        unsuccessful["x_att"] - goal_info.reference_x_att,
        unsuccessful["y_att"] - goal_info.reference_y_att,
    )
    unsuccessful["dist_diff_m"] = np.abs(
        unsuccessful["shot_distance_m"] - goal_info.shot_distance_m
    )
    unsuccessful["same_team"] = unsuccessful["team"] == goal_info.team
    unsuccessful["same_half"] = unsuccessful["half"] == goal_info.half

    if criterion == "same_team_half_location":
        filtered = unsuccessful[unsuccessful["same_team"] & unsuccessful["same_half"]].copy()
        if filtered.empty:
            filtered = unsuccessful[unsuccessful["same_team"]].copy()
        filtered = filtered.sort_values(["loc_diff_m", "dist_diff_m"])
        filtered["selection_reason"] = filtered.apply(
            lambda r: (
                f"Same team ({r['team']}) & Half {int(r['half'])}; "
                f"{r['loc_diff_m']:.1f} m from goal's normalized shot location "
                f"(shot distance {r['shot_distance_m']:.1f} m vs {goal_info.shot_distance_m:.1f} m)"
            ),
            axis=1,
        )
    elif criterion == "same_team_any_half_location":
        filtered = unsuccessful[unsuccessful["same_team"]].copy()
        filtered = filtered.sort_values(["loc_diff_m", "dist_diff_m"])
        filtered["selection_reason"] = filtered.apply(
            lambda r: (
                f"Same team ({r['team']}), Half {int(r['half'])}; "
                f"{r['loc_diff_m']:.1f} m from goal's normalized shot location "
                f"(shot distance {r['shot_distance_m']:.1f} m vs {goal_info.shot_distance_m:.1f} m)"
            ),
            axis=1,
        )
    elif criterion == "closest_location_any_team":
        filtered = unsuccessful.sort_values(["loc_diff_m", "dist_diff_m"]).copy()
        filtered["selection_reason"] = filtered.apply(
            lambda r: (
                f"Closest normalized pitch location ({r['team']}, Half {int(r['half'])}); "
                f"{r['loc_diff_m']:.1f} m away in (x_att, y_att)"
            ),
            axis=1,
        )
    else:  # closest_shot_distance
        filtered = unsuccessful[unsuccessful["same_team"]].copy()
        filtered = filtered.sort_values(["dist_diff_m", "loc_diff_m"])
        filtered["selection_reason"] = filtered.apply(
            lambda r: (
                f"Same team ({r['team']}), Half {int(r['half'])}; "
                f"shot distance {r['shot_distance_m']:.1f} m "
                f"(diff {r['dist_diff_m']:.1f} m vs goal's {goal_info.shot_distance_m:.1f} m)"
            ),
            axis=1,
        )

    return filtered.reset_index(drop=True)


def build_factual_observation(
    goal_info: GoalReferenceInfo,
    goal_metrics: dict[str, Any],
    comp_shot_row: pd.Series,
    comp_metrics: dict[str, Any],
) -> str:
    """Generate a plain-language factual observation comparing the goal and unsuccessful shot.

    Uses only descriptive spatial measurements without claiming causal explanation.
    """
    g_team_label = goal_info.team.upper()
    c_team_label = str(comp_shot_row["team"]).upper()
    g_time = format_video_time(goal_info.reference_timestamp_ms)
    c_time = format_video_time(comp_shot_row["timestamp_ms"])

    parts: list[str] = [
        f"At the key moment for Goal #{goal_info.goal_event_id} ({g_team_label}, {goal_info.scorer_id} at {g_time}), "
        f"the ball/shot was {goal_info.shot_distance_m:.1f} m from goal, compared with "
        f"{float(comp_shot_row['shot_distance_m']):.1f} m on unsuccessful Shot #{int(comp_shot_row['event_id'])} "
        f"({c_team_label}, {comp_shot_row['player_id']} at {c_time})."
    ]

    g_def = goal_metrics.get("nearest_defender_m")
    c_def = comp_metrics.get("nearest_defender_m")
    if g_def is not None and c_def is not None:
        diff_def = g_def - c_def
        relation = "more separation" if diff_def > 0 else "less separation"
        parts.append(
            f"Nearest outfield defender distance was {g_def:.1f} m on the goal frame versus "
            f"{c_def:.1f} m on the unsuccessful shot ({abs(diff_def):.1f} m {relation})."
        )
    elif g_def is not None:
        parts.append(
            f"Nearest outfield defender distance was {g_def:.1f} m on the goal frame "
            "(ball coordinates were missing at the comparison frame)."
        )

    g_w = goal_metrics.get("attacking_width_m")
    c_w = comp_metrics.get("attacking_width_m")
    if g_w is not None and c_w is not None:
        parts.append(
            f"Attacking outfield width spanned {g_w:.1f} m across the pitch during the goal moment "
            f"compared with {c_w:.1f} m on the unsuccessful attempt, with penalty-box occupation (attackers vs defenders) "
            f"at {goal_metrics.get('attackers_in_box', 0)}v{goal_metrics.get('defenders_in_box', 0)} vs "
            f"{comp_metrics.get('attackers_in_box', 0)}v{comp_metrics.get('defenders_in_box', 0)}."
        )

    parts.append(
        "Note: These spatial metrics describe player and ball geometry in the tracked frames; "
        "comparing one goal buildup with one unsuccessful attack illustrates tactical context rather than proving causation."
    )
    return " ".join(parts)


def compute_match_story_summary(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    players: pd.DataFrame,
) -> dict[str, Any]:
    """Compute match-level and half-by-half process vs shot-location summaries."""
    shots = events[events["event_name"] == "shot"].copy()
    goals = events[events["event_name"] == "goal"].copy()
    passes = events[events["event_name"] == "pass"].copy()

    # Enrich each of the 29 shots with nearest-frame defender distance
    shot_records: list[dict[str, Any]] = []
    for _, s in shots.iterrows():
        f_df, matched_ts, diff_ms = get_nearest_frame(
            players, s["timestamp_ms"], int(s["half"]), tolerance_ms=DEFAULT_FRAME_TOLERANCE_MS
        )
        m = compute_frame_spatial_metrics(
            f_df, attacking_team=str(s["team"]), ref_x=float(s["x_start"]), ref_y=float(s["y_start"])
        )
        shot_records.append(
            {
                "event_id": int(s["event_id"]),
                "timestamp_ms": int(s["timestamp_ms"]),
                "video_time_str": format_video_time(s["timestamp_ms"]),
                "minute": float(s["minute"]),
                "half": int(s["half"]),
                "team": str(s["team"]),
                "player_id": str(s["player_id"]),
                "x_start": float(s["x_start"]),
                "y_start": float(s["y_start"]),
                "x_att": float(s["x_att"]),
                "y_att": float(s["y_att"]),
                "shot_distance_m": float(s["shot_distance_m"]),
                "shot_angle_deg": float(s["shot_angle_deg"]),
                "is_goal": bool(s["is_goal"]) if pd.notna(s["is_goal"]) else False,
                "in_penalty_box": bool(
                    (float(s["x_att"]) >= (HALF_LENGTH_M - PENALTY_BOX_DEPTH_M))
                    and (abs(float(s["y_att"])) <= PENALTY_BOX_HALF_WIDTH_M)
                ),
                "nearest_defender_m": m["nearest_defender_m"],
                "attacking_width_m": m["attacking_width_m"],
                "attackers_in_box": m["attackers_in_box"],
                "defenders_in_box": m["defenders_in_box"],
                "matched_frame_ts": matched_ts,
                "frame_diff_ms": diff_ms,
                "sequence_id": int(s["sequence_id"]),
            }
        )
    shots_enriched = pd.DataFrame(shot_records)

    # Team comparison table
    team_stats: dict[str, dict[str, Any]] = {}
    for team in ["away", "home"]:
        t_sum = summary[summary["team"] == team]
        t_out = t_sum[t_sum["role"] == "outfield"]
        t_ev = events[events["team"] == team]
        t_shots = shots_enriched[shots_enriched["team"] == team]
        t_passes = passes[passes["team"] == team]
        t_goals = goals[goals["team"] == team]

        # Final third entries / possession in tracking
        team_stats[team] = {
            "goals": int(len(t_goals)),
            "shots": int(len(t_shots)),
            "shots_h1": int((t_shots["half"] == 1).sum()),
            "shots_h2": int((t_shots["half"] == 2).sum()),
            "shots_in_box": int(t_shots["in_penalty_box"].sum()),
            "shots_within_15m": int((t_shots["shot_distance_m"] <= 15.0).sum()),
            "mean_shot_distance_m": float(t_shots["shot_distance_m"].mean()),
            "median_shot_distance_m": float(t_shots["shot_distance_m"].median()),
            "mean_shot_angle_deg": float(t_shots["shot_angle_deg"].mean()),
            "mean_nearest_defender_at_shot_m": float(t_shots["nearest_defender_m"].dropna().mean()),
            "passes": int(len(t_passes)),
            "pass_completion_pct": float(100.0 * t_passes["pass_success"].mean()),
            "progressive_passes_10m": int(
                (
                    (t_passes["pass_success"] == True)
                    & (
                        (t_passes["receiver_x"] * t_passes["attack_dir"])
                        - (t_passes["x_start"] * t_passes["attack_dir"])
                        >= 10.0
                    )
                ).sum()
            ),
            "final_third_passes": int(
                (
                    (t_passes["pass_success"] == True)
                    & ((t_passes["receiver_x"] * t_passes["attack_dir"]) > 17.5)
                ).sum()
            ),
            "outfield_distance_km": float(t_out["distance_km"].sum()),
            "tackles": int((t_ev["event_name"] == "tackle").sum()),
            "interceptions": int((t_ev["event_name"] == "interception").sum()),
        }

    return {
        "team_stats": team_stats,
        "shots_enriched": shots_enriched,
        "goals_df": goals,
    }
