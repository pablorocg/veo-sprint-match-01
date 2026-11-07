"""Analytical logic for the Veo Football Analytics Sprint: Passing Options Explorer.

Provides data loading, same-half nearest-frame tracking lookup with explicit tolerance,
and explainable geometric heuristic scoring of snapshot-based passing alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 67.7
HALF_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_WIDTH_M = PITCH_WIDTH_M / 2.0

# Reasonable physical coordinate bounds (pitch + generous margin for run-off areas)
MAX_ABS_X_M = HALF_LENGTH_M + 10.0
MAX_ABS_Y_M = HALF_WIDTH_M + 10.0

TEAM_COLORS = {
    "home": "#1f77b4",
    "away": "#d62728",
    "referee": "#999999",
}

DEFAULT_FRAME_TOLERANCE_MS = 500


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for combining heuristic passing option components."""

    lane_safety: float = 0.40
    target_separation: float = 0.30
    forward_progression: float = 0.30

    def normalized(self) -> tuple[float, float, float]:
        total = self.lane_safety + self.target_separation + self.forward_progression
        if total <= 1e-9:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (
            self.lane_safety / total,
            self.target_separation / total,
            self.forward_progression / total,
        )


def load_datasets(root_dir: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Load the core match datasets from the repository root."""
    root = Path(root_dir)
    summary = pd.read_csv(root / "player_summary.csv")
    events = pd.read_csv(root / "events.csv")
    players = pd.read_parquet(root / "players.parquet")
    return {
        "summary": summary,
        "events": events,
        "players": players,
    }


def format_video_clock(timestamp_ms: int | float) -> str:
    """Convert video milliseconds into MM:SS.s video clock string."""
    if pd.isna(timestamp_ms):
        return "N/A"
    total_seconds = float(timestamp_ms) / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def build_half_frame_index(players_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Pre-compute sorted unique frame timestamps per half for fast nearest lookup."""
    index: dict[int, np.ndarray] = {}
    for half_val in sorted(players_df["half"].dropna().unique()):
        h = int(half_val)
        timestamps = players_df.loc[players_df["half"] == h, "timestamp_ms"].unique()
        index[h] = np.sort(timestamps.astype(np.int64))
    return index


def get_pass_events(
    events_df: pd.DataFrame,
    team: str | None = None,
    half: int | None = None,
    outcome: str = "all",
    only_identified_passer: bool = False,
) -> pd.DataFrame:
    """Filter events.csv to pass rows with helpful display labels."""
    passes = events_df[events_df["event_name"] == "pass"].copy()
    if team is not None and team in ("home", "away"):
        passes = passes[passes["team"] == team]
    if half is not None and half in (1, 2):
        passes = passes[passes["half"] == int(half)]
    if outcome == "completed":
        passes = passes[passes["pass_success"] == True]  # noqa: E712
    elif outcome == "incomplete":
        passes = passes[passes["pass_success"] == False]  # noqa: E712
    if only_identified_passer:
        passes = passes[passes["player_id"].notna()]

    passes["video_clock"] = passes["timestamp_ms"].apply(format_video_clock)
    passes["x_att_start"] = passes["x_start"] * passes["attack_dir"]
    passes["x_att_end"] = passes["receiver_x"] * passes["attack_dir"]
    passes["forward_gain_m"] = passes["x_att_end"] - passes["x_att_start"]
    return passes.sort_values("timestamp_ms").reset_index(drop=True)


def get_nearest_same_half_frame(
    players_df: pd.DataFrame,
    timestamp_ms: int,
    half: int,
    tolerance_ms: int = DEFAULT_FRAME_TOLERANCE_MS,
    frame_index: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Retrieve the nearest tracking frame strictly within the same half and tolerance.

    Returns a dictionary with:
    - 'frame_df': DataFrame of tracked rows at the matched frame (empty if no match)
    - 'matched_timestamp_ms': int or None
    - 'offset_ms': signed int (matched - event) or None
    - 'abs_offset_ms': non-negative int or None
    - 'within_tolerance': bool
    - 'status': human-readable status message
    """
    if pd.isna(timestamp_ms) or pd.isna(half):
        return {
            "frame_df": players_df.iloc[0:0].copy(),
            "matched_timestamp_ms": None,
            "offset_ms": None,
            "abs_offset_ms": None,
            "within_tolerance": False,
            "status": "Invalid event timestamp or half.",
        }

    half_int = int(half)
    target_ms = int(timestamp_ms)

    if frame_index is not None and half_int in frame_index:
        candidates = frame_index[half_int]
    else:
        candidates = np.sort(
            players_df.loc[players_df["half"] == half_int, "timestamp_ms"]
            .unique()
            .astype(np.int64)
        )

    if len(candidates) == 0:
        return {
            "frame_df": players_df.iloc[0:0].copy(),
            "matched_timestamp_ms": None,
            "offset_ms": None,
            "abs_offset_ms": None,
            "within_tolerance": False,
            "status": f"No tracking frames available in Half {half_int}.",
        }

    diffs = np.abs(candidates - target_ms)
    best_idx = int(np.argmin(diffs))
    matched_ms = int(candidates[best_idx])
    offset_ms = matched_ms - target_ms
    abs_offset_ms = abs(offset_ms)

    if abs_offset_ms > tolerance_ms:
        return {
            "frame_df": players_df.iloc[0:0].copy(),
            "matched_timestamp_ms": matched_ms,
            "offset_ms": offset_ms,
            "abs_offset_ms": abs_offset_ms,
            "within_tolerance": False,
            "status": (
                f"Nearest Half {half_int} frame at {matched_ms} ms is {abs_offset_ms} ms away "
                f"(exceeds ±{tolerance_ms} ms tolerance)."
            ),
        }

    frame_df = players_df[
        (players_df["half"] == half_int) & (players_df["timestamp_ms"] == matched_ms)
    ].copy()

    return {
        "frame_df": frame_df,
        "matched_timestamp_ms": matched_ms,
        "offset_ms": offset_ms,
        "abs_offset_ms": abs_offset_ms,
        "within_tolerance": True,
        "status": (
            f"Matched Half {half_int} frame at {matched_ms} ms "
            f"({offset_ms:+d} ms from event timestamp)."
        ),
    }


def is_valid_position(x: float, y: float) -> bool:
    """Check whether (x, y) is finite and within plausible pitch + runoff bounds."""
    if pd.isna(x) or pd.isna(y):
        return False
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return abs(float(x)) <= MAX_ABS_X_M and abs(float(y)) <= MAX_ABS_Y_M


def point_to_segment_clamped(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float, float]:
    """Compute distance from point P(px, py) to segment A(ax, ay)->B(bx, by) with clamped projection.

    Returns:
        (distance_m, proj_x, proj_y, t_clamped)
        where t_clamped in [0.0, 1.0] is the normalized position along A->B.
    """
    abx = float(bx) - float(ax)
    aby = float(by) - float(ay)
    apx = float(px) - float(ax)
    apy = float(py) - float(ay)
    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq <= 1e-12:
        dist = float(np.hypot(apx, apy))
        return dist, float(ax), float(ay), 0.0

    t_raw = (apx * abx + apy * aby) / ab_len_sq
    t_clamped = float(np.clip(t_raw, 0.0, 1.0))
    proj_x = float(ax) + t_clamped * abx
    proj_y = float(ay) + t_clamped * aby
    dist = float(np.hypot(float(px) - proj_x, float(py) - proj_y))
    return dist, proj_x, proj_y, t_clamped


def score_lane_safety(clearance_m: float) -> float:
    """Map nearest-defender-to-segment distance (m) to a 0-100 heuristic score.

    Saturates at 100 when clearance >= 4.0 m.
    """
    if pd.isna(clearance_m):
        return 50.0  # neutral fallback if defenders are missing
    return float(np.clip((float(clearance_m) / 4.0) * 100.0, 0.0, 100.0))


def score_target_separation(separation_m: float) -> float:
    """Map target receiver separation from nearest defender (m) to a 0-100 heuristic score.

    Saturates at 100 when separation >= 10.0 m.
    """
    if pd.isna(separation_m):
        return 50.0  # neutral fallback if defenders are missing
    return float(np.clip((float(separation_m) / 10.0) * 100.0, 0.0, 100.0))


def score_forward_progression(forward_progress_m: float) -> float:
    """Map attack_dir-normalized forward progress (m) to a 0-100 heuristic score.

    -15 m (backward) -> 0
      0 m (lateral)  -> 37.5
    +25 m (forward)  -> 100
    """
    if pd.isna(forward_progress_m):
        return 0.0
    return float(np.clip(((float(forward_progress_m) + 15.0) / 40.0) * 100.0, 0.0, 100.0))


def evaluate_pass_options(
    pass_event: pd.Series | dict[str, Any],
    players_df: pd.DataFrame,
    tolerance_ms: int = DEFAULT_FRAME_TOLERANCE_MS,
    include_goalkeeper: bool = False,
    weights: tuple[float, float, float] = (0.40, 0.30, 0.30),
    frame_index: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evaluate actual pass details and snapshot-based teammate alternatives for a pass event.

    Excludes passer, referee, opponents, and rows with invalid coordinates or missing player_id
    from candidate targets. Handles missing passer, no candidates, missing defenders, or missing
    tracking frames gracefully without fabricating values.
    """
    event_id = int(pass_event["event_id"])
    timestamp_ms = int(pass_event["timestamp_ms"])
    half = int(pass_event["half"])
    team = str(pass_event["team"]) if pd.notna(pass_event["team"]) else None
    passer_id = (
        str(pass_event["player_id"])
        if pd.notna(pass_event["player_id"]) and str(pass_event["player_id"]).strip() != ""
        else None
    )
    attack_dir = (
        int(pass_event["attack_dir"])
        if pd.notna(pass_event["attack_dir"]) and int(pass_event["attack_dir"]) in (-1, 1)
        else None
    )

    # Actual pass factual details from events.csv
    x_start = float(pass_event["x_start"]) if is_valid_position(pass_event.get("x_start"), pass_event.get("y_start")) else None
    y_start = float(pass_event["y_start"]) if x_start is not None else None

    has_actual_end = is_valid_position(pass_event.get("receiver_x"), pass_event.get("receiver_y"))
    actual_end_x = float(pass_event["receiver_x"]) if has_actual_end else None
    actual_end_y = float(pass_event["receiver_y"]) if has_actual_end else None
    actual_receiver_id = (
        str(pass_event["receiver_player_id"])
        if pd.notna(pass_event.get("receiver_player_id"))
        and str(pass_event.get("receiver_player_id")).strip() != ""
        else None
    )
    pass_success = (
        bool(pass_event["pass_success"])
        if pd.notna(pass_event.get("pass_success"))
        else None
    )
    pass_distance_m = (
        float(pass_event["pass_distance_m"])
        if pd.notna(pass_event.get("pass_distance_m"))
        else None
    )

    frame_lookup = get_nearest_same_half_frame(
        players_df=players_df,
        timestamp_ms=timestamp_ms,
        half=half,
        tolerance_ms=tolerance_ms,
        frame_index=frame_index,
    )
    frame_df = frame_lookup["frame_df"]

    empty_candidates = pd.DataFrame(
        columns=[
            "rank",
            "target_player_id",
            "role",
            "target_x",
            "target_y",
            "pass_length_m",
            "forward_progress_m",
            "lane_clearance_m",
            "nearest_lane_defender_id",
            "lane_proj_x",
            "lane_proj_y",
            "lane_proj_t",
            "target_separation_m",
            "nearest_target_defender_id",
            "lane_safety_score",
            "target_sep_score",
            "progression_score",
            "composite_score",
            "is_actual_receiver",
        ]
    )

    # Filter frame to valid positions
    if not frame_df.empty:
        valid_mask = frame_df.apply(lambda r: is_valid_position(r["x"], r["y"]), axis=1)
        frame_df = frame_df[valid_mask].copy()

    # Ball position in matched frame (may be null)
    ball_pos: tuple[float, float] | None = None
    if not frame_df.empty:
        b_x = frame_df["ball_x"].iloc[0]
        b_y = frame_df["ball_y"].iloc[0]
        if is_valid_position(b_x, b_y):
            ball_pos = (float(b_x), float(b_y))

    # Locate passer in matched tracking frame by player_id
    passer_row: pd.Series | None = None
    if not frame_df.empty and passer_id is not None:
        matched_passers = frame_df[
            (frame_df["team"] == team) & (frame_df["player_id"] == passer_id)
        ]
        if not matched_passers.empty:
            passer_row = matched_passers.iloc[0]

    # Separate frame entities
    opponents_df = pd.DataFrame()
    teammates_df = pd.DataFrame()
    referee_df = pd.DataFrame()

    if not frame_df.empty and team in ("home", "away"):
        opp_team = "away" if team == "home" else "home"
        opponents_df = frame_df[
            (frame_df["team"] == opp_team) & (frame_df["role"] != "referee")
        ].copy()
        referee_df = frame_df[
            (frame_df["team"] == "referee") | (frame_df["role"] == "referee")
        ].copy()

        # Teammates: same team, exclude referee, exclude passer, require non-null player_id
        tm_mask = (
            (frame_df["team"] == team)
            & (frame_df["role"] != "referee")
            & (frame_df["player_id"].notna())
        )
        if passer_id is not None:
            tm_mask = tm_mask & (frame_df["player_id"] != passer_id)
        if not include_goalkeeper:
            tm_mask = tm_mask & (frame_df["role"] == "outfield")
        teammates_df = frame_df[tm_mask].copy()

    # Check if we can evaluate alternatives
    warnings: list[str] = []
    if not frame_lookup["within_tolerance"]:
        warnings.append(frame_lookup["status"])
    if team not in ("home", "away"):
        warnings.append("Pass event team is unavailable; cannot identify teammates or opponents.")
    if attack_dir not in (-1, 1):
        warnings.append("Attacking direction (attack_dir) is unavailable; cannot normalize forward progress.")
    if passer_id is None:
        warnings.append(
            "Passer player_id is unattributed (null) in events.csv; cannot locate tracked passer by player_id."
        )
    elif frame_lookup["within_tolerance"] and passer_row is None:
        warnings.append(
            f"Passer {passer_id} was not located in the matched tracking frame ({frame_lookup['matched_timestamp_ms']} ms)."
        )

    if opponents_df.empty and frame_lookup["within_tolerance"]:
        warnings.append(
            "No tracked opponent defenders found in this frame; lane clearance and target separation use neutral fallback scores."
        )

    # Compute actual pass geometric metrics if actual end and start exist
    actual_metrics: dict[str, Any] = {
        "has_start": x_start is not None,
        "has_end": has_actual_end,
        "x_start": x_start,
        "y_start": y_start,
        "x_end": actual_end_x,
        "y_end": actual_end_y,
        "receiver_player_id": actual_receiver_id,
        "pass_success": pass_success,
        "pass_distance_m": pass_distance_m,
        "forward_progress_m": (
            (actual_end_x - x_start) * attack_dir
            if (x_start is not None and actual_end_x is not None and attack_dir in (-1, 1))
            else None
        ),
        "passer_tracked_x": float(passer_row["x"]) if passer_row is not None else None,
        "passer_tracked_y": float(passer_row["y"]) if passer_row is not None else None,
        "passer_event_offset_m": (
            float(np.hypot(float(passer_row["x"]) - x_start, float(passer_row["y"]) - y_start))
            if (passer_row is not None and x_start is not None and y_start is not None)
            else None
        ),
    }

    if (
        not frame_lookup["within_tolerance"]
        or team not in ("home", "away")
        or attack_dir not in (-1, 1)
        or passer_row is None
        or teammates_df.empty
    ):
        if frame_lookup["within_tolerance"] and passer_row is not None and teammates_df.empty:
            warnings.append("No eligible tracked teammate candidates found in this frame.")
        return {
            "event_id": event_id,
            "timestamp_ms": timestamp_ms,
            "video_clock": format_video_clock(timestamp_ms),
            "half": half,
            "team": team,
            "passer_id": passer_id,
            "attack_dir": attack_dir,
            "frame_lookup": frame_lookup,
            "frame_df": frame_df,
            "passer_row": passer_row,
            "opponents_df": opponents_df,
            "referee_df": referee_df,
            "ball_pos": ball_pos,
            "actual_pass": actual_metrics,
            "candidates_df": empty_candidates,
            "warnings": warnings,
        }

    # Normalize weights
    w_lane, w_sep, w_prog = ScoringWeights(*weights).normalized()

    px = float(passer_row["x"])
    py = float(passer_row["y"])

    records: list[dict[str, Any]] = []
    for tm in teammates_df.itertuples():
        tx = float(tm.x)
        ty = float(tm.y)
        t_id = str(tm.player_id)
        t_role = str(tm.role)

        pass_len_m = float(np.hypot(tx - px, ty - py))
        fwd_prog_m = (tx - px) * float(attack_dir)

        # Nearest defender to passing segment A(px, py) -> B(tx, ty) with clamped projection
        if not opponents_df.empty:
            best_lane_dist = float("inf")
            best_lane_def_id = None
            best_proj_x = px
            best_proj_y = py
            best_proj_t = 0.0

            best_sep_dist = float("inf")
            best_sep_def_id = None

            for opp in opponents_df.itertuples():
                ox = float(opp.x)
                oy = float(opp.y)
                opp_id = str(opp.player_id) if pd.notna(opp.player_id) else "unattributed_opp"

                d_seg, pr_x, pr_y, pr_t = point_to_segment_clamped(ox, oy, px, py, tx, ty)
                if d_seg < best_lane_dist:
                    best_lane_dist = d_seg
                    best_lane_def_id = opp_id
                    best_proj_x = pr_x
                    best_proj_y = pr_y
                    best_proj_t = pr_t

                d_target = float(np.hypot(ox - tx, oy - ty))
                if d_target < best_sep_dist:
                    best_sep_dist = d_target
                    best_sep_def_id = opp_id
        else:
            best_lane_dist = np.nan
            best_lane_def_id = None
            best_proj_x = np.nan
            best_proj_y = np.nan
            best_proj_t = np.nan
            best_sep_dist = np.nan
            best_sep_def_id = None

        s_lane = score_lane_safety(best_lane_dist)
        s_sep = score_target_separation(best_sep_dist)
        s_prog = score_forward_progression(fwd_prog_m)
        composite = w_lane * s_lane + w_sep * s_sep + w_prog * s_prog

        records.append(
            {
                "target_player_id": t_id,
                "role": t_role,
                "target_x": tx,
                "target_y": ty,
                "pass_length_m": pass_len_m,
                "forward_progress_m": fwd_prog_m,
                "lane_clearance_m": best_lane_dist,
                "nearest_lane_defender_id": best_lane_def_id,
                "lane_proj_x": best_proj_x,
                "lane_proj_y": best_proj_y,
                "lane_proj_t": best_proj_t,
                "target_separation_m": best_sep_dist,
                "nearest_target_defender_id": best_sep_def_id,
                "lane_safety_score": s_lane,
                "target_sep_score": s_sep,
                "progression_score": s_prog,
                "composite_score": composite,
                "is_actual_receiver": (actual_receiver_id is not None and t_id == actual_receiver_id),
            }
        )

    candidates_df = pd.DataFrame(records)
    candidates_df = candidates_df.sort_values(
        by=["composite_score", "forward_progress_m"], ascending=[False, False]
    ).reset_index(drop=True)
    candidates_df["rank"] = np.arange(1, len(candidates_df) + 1)

    # Reorder columns cleanly
    candidates_df = candidates_df[empty_candidates.columns]

    return {
        "event_id": event_id,
        "timestamp_ms": timestamp_ms,
        "video_clock": format_video_clock(timestamp_ms),
        "half": half,
        "team": team,
        "passer_id": passer_id,
        "attack_dir": attack_dir,
        "frame_lookup": frame_lookup,
        "frame_df": frame_df,
        "passer_row": passer_row,
        "opponents_df": opponents_df,
        "referee_df": referee_df,
        "ball_pos": ball_pos,
        "actual_pass": actual_metrics,
        "candidates_df": candidates_df,
        "warnings": warnings,
        "weights_normalized": (w_lane, w_sep, w_prog),
    }
