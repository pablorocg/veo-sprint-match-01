"""Core defensive-transition extraction, attribution, and player evidence ranking.

Implements the shared core API for Veo Football Analytics Sprint Match 01:
- ``extract_transitions``: extract sequence-opening defensive interventions and
  in-window retained events prior to stopping boundaries.
- ``attribute_transitions``: measure attack-normalised forward progression,
  territorial final-third entries, shot/goal separation, and deduplicated roles.
- ``summarise_players``: aggregate direct actions and downstream role associations
  with explicit denominators, net pass metres, failed-pass stops, descriptive
  Wilson intervals, screening Pareto tiers, and pass-only progression ordering.
- ``run_analysis``: execute the complete pipeline for a chosen window and threshold.
- ``find_tied_sequence_starts`` / ``canonicalise_tied_events``: transparent opt-in
  tie-order sensitivity helpers preserving original ``event_id`` and raw columns
  while guarding ``<=100 ms`` co-detected restart pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FINAL_THIRD_X_ATT_M = 17.5
DEFAULT_WINDOW_S = 15.0
DEFAULT_PROGRESSION_THRESHOLD_M = 10.0
DEFAULT_RESTART_GUARD_MS = 100
WILSON_Z_95 = 1.959963984540054
COORD_ROUND_DECIMALS = 9

DEFENSIVE_INTERVENTION_EVENTS = frozenset({"tackle", "interception"})
RESTART_EVENTS = frozenset({"throw_in", "kick_corner", "kick_free_kick"})
FINISHER_EVENTS = frozenset({"shot", "goal"})
RESTART_CO_DETECTED_ACTIONS = frozenset({"pass", "shot"})

REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "timestamp_ms",
    "half",
    "event_name",
    "team",
    "player_id",
    "attack_dir",
    "x_start",
    "pass_success",
    "receiver_player_id",
    "receiver_x",
    "sequence_id",
)

REQUIRED_SUMMARY_COLUMNS = (
    "player_id",
    "team",
    "minutes_played",
)

TRANSITION_COLUMNS = [
    "transition_id",
    "sequence_id",
    "team",
    "half",
    "start_player_id",
    "start_ms",
    "end_ms",
    "start_x_att",
    "start_zone",
    "stop_reason",
    "terminal_event_id",
]

METRICS_ADDED_COLUMNS = [
    "n_completed_passes",
    "has_followup",
    "shot_count",
    "has_shot",
    "goal_count",
    "has_goal",
    "final_third_entries",
    "has_entry",
    "forward_metres",
    "has_progression",
    "outcome_category",
    "measurable_passes",
    "missing_endpoint_passes",
]

METRICS_COLUMNS = TRANSITION_COLUMNS + METRICS_ADDED_COLUMNS

CONTRIBUTION_COLUMNS = [
    "transition_id",
    "player_id",
    "contribution_role",
    "event_id",
]

MINIMUM_RANKING_COLUMNS = [
    "player_id",
    "team",
    "minutes_played",
    "recovery_starts",
    "recovery_followups",
    "recovery_entry_passages",
    "recovery_shot_passages",
    "recovery_outcome_passages",
    "outlet_passages",
    "outlet_entry_passages",
    "outlet_shot_passages",
    "outlet_outcome_passages",
    "completed_transition_passes",
    "measurable_transition_passes",
    "forward_metres",
    "progressive_passes",
    "progression_passages",
    "entries_delivered",
    "entries_received",
    "entry_passages",
    "shot_passages",
    "pareto_tier",
    "progression_rank",
]

SUPPORTING_RANKING_COLUMNS = [
    "net_forward_metres",
    "failed_transition_passes",
    "advancer_passages",
    "involved_passages",
    "missing_endpoint_transition_passes",
    "recovery_defensive_half_starts",
    "recovery_attacking_half_starts",
    "recovery_final_third_starts",
    "recovery_failed_pass_stops",
    "recovery_followup_share",
    "recovery_followup_wilson_low",
    "recovery_followup_wilson_high",
    "recovery_outcome_share",
    "recovery_outcome_wilson_low",
    "recovery_outcome_wilson_high",
    "outlet_outcome_share",
    "outlet_outcome_wilson_low",
    "outlet_outcome_wilson_high",
]

RANKING_COLUMNS = MINIMUM_RANKING_COLUMNS + SUPPORTING_RANKING_COLUMNS

PARETO_OBJECTIVE_COLUMNS = [
    "entries_delivered",
    "forward_metres",
    "recovery_outcome_passages",
    "outlet_outcome_passages",
]


def _is_true_flag(value: object) -> bool:
    """Safely evaluate boolean flags without turning NaN or pd.NA into True."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value == 1)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _clean_player_id(value: object) -> str | None:
    """Return a clean non-empty player identifier, keeping missing IDs None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    return text


def _compute_x_att(x_value: object, attack_dir: object) -> float | None:
    """Normalise pitch x-coordinate (metres) by team attack direction (+1 or -1)."""
    if x_value is None or attack_dir is None or pd.isna(x_value) or pd.isna(attack_dir):
        return None
    direction = float(attack_dir)
    if direction not in (-1.0, 1.0):
        return None
    return float(x_value) * direction


def _classify_start_zone(start_x_att: float | None) -> object:
    """Classify normalised starting x-coordinate into defensive/attacking/final-third zones."""
    if start_x_att is None or pd.isna(start_x_att):
        return pd.NA
    rounded = round(float(start_x_att), COORD_ROUND_DECIMALS)
    if rounded <= 0.0:
        return "defensive_half"
    if rounded <= FINAL_THIRD_X_ATT_M:
        return "attacking_half"
    return "final_third"


def _empty_transitions_df() -> pd.DataFrame:
    """Construct an empty transitions DataFrame with exact contract schema and dtypes."""
    return pd.DataFrame(
        {
            "transition_id": pd.Series(dtype="int64"),
            "sequence_id": pd.Series(dtype="int64"),
            "team": pd.Series(dtype="object"),
            "half": pd.Series(dtype="int64"),
            "start_player_id": pd.Series(dtype="object"),
            "start_ms": pd.Series(dtype="int64"),
            "end_ms": pd.Series(dtype="int64"),
            "start_x_att": pd.Series(dtype="float64"),
            "start_zone": pd.Series(dtype="object"),
            "stop_reason": pd.Series(dtype="object"),
            "terminal_event_id": pd.Series(dtype="Int64"),
        }
    )


def _empty_metrics_df() -> pd.DataFrame:
    """Construct an empty metrics DataFrame with exact contract schema and dtypes."""
    base = _empty_transitions_df()
    extra = pd.DataFrame(
        {
            "n_completed_passes": pd.Series(dtype="int64"),
            "has_followup": pd.Series(dtype="bool"),
            "shot_count": pd.Series(dtype="int64"),
            "has_shot": pd.Series(dtype="bool"),
            "goal_count": pd.Series(dtype="int64"),
            "has_goal": pd.Series(dtype="bool"),
            "final_third_entries": pd.Series(dtype="int64"),
            "has_entry": pd.Series(dtype="bool"),
            "forward_metres": pd.Series(dtype="float64"),
            "has_progression": pd.Series(dtype="bool"),
            "outcome_category": pd.Series(dtype="object"),
            "measurable_passes": pd.Series(dtype="int64"),
            "missing_endpoint_passes": pd.Series(dtype="int64"),
        }
    )
    return pd.concat([base, extra], axis=1)


def _empty_contributions_df() -> pd.DataFrame:
    """Construct an empty contributions DataFrame with exact contract schema and dtypes."""
    return pd.DataFrame(
        {
            "transition_id": pd.Series(dtype="int64"),
            "player_id": pd.Series(dtype="object"),
            "contribution_role": pd.Series(dtype="object"),
            "event_id": pd.Series(dtype="int64"),
        }
    )


def _empty_ranking_df() -> pd.DataFrame:
    """Construct an empty ranking DataFrame with exact minimum and supporting columns."""
    int_cols = {
        "recovery_starts",
        "recovery_followups",
        "recovery_entry_passages",
        "recovery_shot_passages",
        "recovery_outcome_passages",
        "outlet_passages",
        "outlet_entry_passages",
        "outlet_shot_passages",
        "outlet_outcome_passages",
        "completed_transition_passes",
        "measurable_transition_passes",
        "progressive_passes",
        "progression_passages",
        "entries_delivered",
        "entries_received",
        "entry_passages",
        "shot_passages",
        "pareto_tier",
        "progression_rank",
        "failed_transition_passes",
        "advancer_passages",
        "involved_passages",
        "missing_endpoint_transition_passes",
        "recovery_defensive_half_starts",
        "recovery_attacking_half_starts",
        "recovery_final_third_starts",
        "recovery_failed_pass_stops",
    }
    float_cols = {
        "minutes_played",
        "forward_metres",
        "net_forward_metres",
        "recovery_followup_share",
        "recovery_followup_wilson_low",
        "recovery_followup_wilson_high",
        "recovery_outcome_share",
        "recovery_outcome_wilson_low",
        "recovery_outcome_wilson_high",
        "outlet_outcome_share",
        "outlet_outcome_wilson_low",
        "outlet_outcome_wilson_high",
    }
    data: dict[str, pd.Series] = {}
    for col in RANKING_COLUMNS:
        if col in int_cols:
            data[col] = pd.Series(dtype="int64")
        elif col in float_cols:
            data[col] = pd.Series(dtype="float64")
        else:
            data[col] = pd.Series(dtype="object")
    return pd.DataFrame(data)


def wilson_interval(
    successes: int | float | pd.Series | np.ndarray,
    trials: int | float | pd.Series | np.ndarray,
    z: float = WILSON_Z_95,
) -> tuple[float, float] | tuple[pd.Series, pd.Series]:
    """Compute descriptive two-sided Wilson score intervals for conversion shares.

    Note: Intervals summarise binomial sampling uncertainty for small denominators
    descriptively; passages within a single match are not independent observations.
    """
    if z <= 0 or not np.isfinite(z):
        raise ValueError("z must be a positive finite number.")

    is_scalar = np.ndim(successes) == 0 and np.ndim(trials) == 0
    k_arr = np.asarray(successes, dtype=float)
    n_arr = np.asarray(trials, dtype=float)

    if np.any(n_arr < 0) or np.any(k_arr < 0) or np.any(k_arr > n_arr):
        raise ValueError("Require 0 <= successes <= trials.")

    low_arr = np.full_like(n_arr, np.nan, dtype=float)
    high_arr = np.full_like(n_arr, np.nan, dtype=float)

    valid = n_arr > 0
    if np.any(valid):
        k_v = k_arr[valid]
        n_v = n_arr[valid]
        p_hat = k_v / n_v
        z2 = z * z
        denom = 1.0 + z2 / n_v
        centre = (p_hat + z2 / (2.0 * n_v)) / denom
        radicand = np.maximum((p_hat * (1.0 - p_hat)) / n_v + (z2 / (4.0 * n_v * n_v)), 0.0)
        half_width = (z * np.sqrt(radicand)) / denom
        low_arr[valid] = np.clip(centre - half_width, 0.0, 1.0)
        high_arr[valid] = np.clip(centre + half_width, 0.0, 1.0)

    if is_scalar:
        return float(low_arr.item()), float(high_arr.item())

    index = successes.index if isinstance(successes, pd.Series) else (
        trials.index if isinstance(trials, pd.Series) else None
    )
    return pd.Series(low_arr, index=index, dtype="float64"), pd.Series(
        high_arr, index=index, dtype="float64"
    )


def _assign_pareto_tiers(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Assign 1-indexed non-dominated Pareto frontier tiers (smaller tier is better)."""
    if df.empty:
        return pd.Series(dtype="int64", index=df.index)

    mat = df[columns].to_numpy(dtype=float).copy()
    mat = np.round(mat, COORD_ROUND_DECIMALS)

    n = len(df)
    remaining = np.ones(n, dtype=bool)
    tiers = np.zeros(n, dtype=np.int64)
    current_tier = 1

    while np.any(remaining):
        active_idx = np.flatnonzero(remaining)
        active_mat = mat[active_idx]
        m = len(active_idx)
        is_dominated = np.zeros(m, dtype=bool)

        for i in range(m):
            ge_all = np.all(active_mat >= active_mat[i], axis=1)
            gt_any = np.any(active_mat > active_mat[i], axis=1)
            if np.any(ge_all & gt_any):
                is_dominated[i] = True

        front_idx = active_idx[~is_dominated]
        tiers[front_idx] = current_tier
        remaining[front_idx] = False
        current_tier += 1

    return pd.Series(tiers, index=df.index, dtype="int64")


def find_tied_sequence_starts(
    events: pd.DataFrame,
    require_completed_pass: bool = True,
) -> pd.DataFrame:
    """Identify sequence-opening simultaneous same-player pass -> intervention pairs.

    Parameters
    ----------
    events:
        Match event table matching ``events.csv`` schema.
    require_completed_pass:
        When True (default), include only sequence openings where the co-detected
        pass is completed (12 sequences in Match 01). Set False to also include
        failed one-touch passes (adds ``sequence_id == 212``, 13 sequences total).
    """
    cols = [
        "sequence_id",
        "player_id",
        "team",
        "half",
        "timestamp_ms",
        "pass_event_id",
        "pass_success",
        "intervention_event_id",
        "intervention_name",
    ]
    if events.empty:
        return pd.DataFrame(columns=cols)

    work = events.sort_values(["timestamp_ms", "event_id"], kind="mergesort").reset_index(drop=True)
    records: list[dict[str, object]] = []
    for seq_id, grp in work.groupby("sequence_id", sort=False):
        if len(grp) < 2:
            continue
        first = grp.iloc[0]
        second = grp.iloc[1]
        pid0 = _clean_player_id(first["player_id"])
        pid1 = _clean_player_id(second["player_id"])
        if (
            pid0 is not None
            and pid0 == pid1
            and int(first["timestamp_ms"]) == int(second["timestamp_ms"])
            and first["event_name"] == "pass"
            and second["event_name"] in DEFENSIVE_INTERVENTION_EVENTS
        ):
            is_comp = _is_true_flag(first["pass_success"])
            if require_completed_pass and not is_comp:
                continue
            records.append(
                {
                    "sequence_id": int(seq_id),
                    "player_id": pid0,
                    "team": first["team"],
                    "half": int(first["half"]),
                    "timestamp_ms": int(first["timestamp_ms"]),
                    "pass_event_id": int(first["event_id"]),
                    "pass_success": is_comp,
                    "intervention_event_id": int(second["event_id"]),
                    "intervention_name": second["event_name"],
                }
            )
    return pd.DataFrame(records, columns=cols)


def canonicalise_tied_events(
    events: pd.DataFrame,
    *,
    require_completed_pass: bool = True,
    restart_guard_ms: int = DEFAULT_RESTART_GUARD_MS,
) -> pd.DataFrame:
    """Return a canonicalised event copy for opt-in tie-order sensitivity analysis.

    Preserves 100% of original ``event_id`` values, ``timestamp_ms`` values, and raw
    columns without mutating ``events``:
    1. Within each ``sequence_id``, if the first two chronological events share an
       identical ``timestamp_ms`` and non-null ``player_id`` as ``pass -> tackle/interception``
       (completed pass by default), places the defensive intervention first so the
       one-touch recovery+pass is eligible as a candidate start.
    2. Guards co-detected restart pairs within each ``sequence_id``: if a ``pass`` or
       ``shot`` precedes a ``throw_in``, ``kick_corner``, or ``kick_free_kick`` within
       ``0 <= dt_ms <= restart_guard_ms`` (default 100 ms), places the restart first so
       ``extract_transitions`` stops at the restart without leaking the restart delivery
       into open-play transition metrics.
    """
    if restart_guard_ms < 0:
        raise ValueError("restart_guard_ms must be non-negative.")

    if len(events.columns) > 0:
        missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in events.columns]
        if missing:
            raise ValueError(f"events is missing required columns: {missing}")

    if events.empty:
        out_empty = events.copy()
        out_empty.attrs["precanonicalised_order"] = True
        return out_empty

    orig_cols = list(events.columns)
    work = events.sort_values(["timestamp_ms", "event_id"], kind="mergesort").reset_index(drop=True)
    perm = list(range(len(work)))

    for _, grp in work.groupby("sequence_id", sort=False):
        seq_positions = grp.index.tolist()
        if len(seq_positions) < 2:
            continue

        local = [perm[idx] for idx in seq_positions]

        # 1. Swap simultaneous same-player opening pass -> defensive intervention
        r0 = work.iloc[local[0]]
        r1 = work.iloc[local[1]]
        pid0 = _clean_player_id(r0["player_id"])
        pid1 = _clean_player_id(r1["player_id"])
        if (
            pid0 is not None
            and pid0 == pid1
            and int(r0["timestamp_ms"]) == int(r1["timestamp_ms"])
            and r0["event_name"] == "pass"
            and r1["event_name"] in DEFENSIVE_INTERVENTION_EVENTS
            and (not require_completed_pass or _is_true_flag(r0["pass_success"]))
        ):
            local[0], local[1] = local[1], local[0]

        # 2. Guard co-detected pass/shot <= restart_guard_ms before a same-sequence restart
        for k in range(len(local) - 1):
            ra = work.iloc[local[k]]
            rb = work.iloc[local[k + 1]]
            dt_ms = int(rb["timestamp_ms"]) - int(ra["timestamp_ms"])
            if (
                ra["event_name"] in RESTART_CO_DETECTED_ACTIONS
                and rb["event_name"] in RESTART_EVENTS
                and 0 <= dt_ms <= restart_guard_ms
            ):
                local[k], local[k + 1] = local[k + 1], local[k]

        for idx, mapped_pos in zip(seq_positions, local):
            perm[idx] = mapped_pos

    canonical = work.iloc[perm].copy().reset_index(drop=True)[orig_cols]
    canonical.attrs["precanonicalised_order"] = True
    return canonical


def extract_transitions(
    events: pd.DataFrame,
    window_s: float = DEFAULT_WINDOW_S,
    *,
    canonicalise_ties: bool = False,
    restart_guard_ms: int = DEFAULT_RESTART_GUARD_MS,
    require_completed_opening_pass: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract sequence-opening defensive interventions and bounded retained events.

    Parameters
    ----------
    events:
        Match on-ball event table matching ``events.csv`` schema.
    window_s:
        Inclusive post-intervention time window in seconds (default 15.0).
    canonicalise_ties:
        Opt-in sensitivity flag (default False, keeping the 138-start baseline).
        When True, applies ``canonicalise_tied_events`` before extraction.
    restart_guard_ms:
        Co-detection guard threshold (ms) used when ``canonicalise_ties=True``.
    require_completed_opening_pass:
        Pass-completion requirement used when ``canonicalise_ties=True``.
    """
    if window_s < 0 or not np.isfinite(window_s):
        raise ValueError("window_s must be a non-negative finite number.")

    if len(events.columns) > 0:
        missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in events.columns]
        if missing:
            raise ValueError(f"events is missing required columns: {missing}")
        orig_cols = list(events.columns)
    else:
        orig_cols = list(REQUIRED_EVENT_COLUMNS)

    tev_cols = orig_cols + ([] if "transition_id" in orig_cols else ["transition_id"])

    if events.empty:
        if len(events.columns) == 0:
            empty_tev = pd.DataFrame({col: pd.Series(dtype="object") for col in orig_cols})
            empty_tev["event_id"] = pd.Series(dtype="int64")
            empty_tev["timestamp_ms"] = pd.Series(dtype="int64")
            empty_tev["half"] = pd.Series(dtype="int64")
            empty_tev["attack_dir"] = pd.Series(dtype="int64")
            empty_tev["x_start"] = pd.Series(dtype="float64")
            empty_tev["receiver_x"] = pd.Series(dtype="float64")
            empty_tev["sequence_id"] = pd.Series(dtype="int64")
        else:
            empty_tev = events.copy()
        if "transition_id" not in empty_tev.columns:
            empty_tev["transition_id"] = pd.Series(dtype="int64")
        return _empty_transitions_df(), empty_tev[tev_cols]

    is_precanonical = bool(events.attrs.get("precanonicalised_order", False))
    if canonicalise_ties:
        work = canonicalise_tied_events(
            events,
            require_completed_pass=require_completed_opening_pass,
            restart_guard_ms=restart_guard_ms,
        )
        is_precanonical = True
    elif is_precanonical:
        work = events.reset_index(drop=True).copy()
    else:
        work = events.sort_values(["timestamp_ms", "event_id"], kind="mergesort").reset_index(
            drop=True
        )

    window_ms = round(float(window_s) * 1000.0, 6)

    first_in_sequence = ~work["sequence_id"].duplicated(keep="first")
    is_intervention = work["event_name"].isin(DEFENSIVE_INTERVENTION_EVENTS)
    candidate_positions = np.flatnonzero((first_in_sequence & is_intervention).to_numpy())

    if len(candidate_positions) == 0:
        empty_tev = work.iloc[0:0].copy()
        empty_tev["transition_id"] = pd.Series(dtype="int64")
        return _empty_transitions_df(), empty_tev[tev_cols]

    event_ids = work["event_id"].to_numpy()
    timestamps = work["timestamp_ms"].to_numpy()
    halves = work["half"].to_numpy()
    teams = work["team"].to_numpy()
    sequences = work["sequence_id"].to_numpy()
    names = work["event_name"].to_numpy()
    pass_flags = work["pass_success"].to_numpy()
    players = work["player_id"].to_numpy()
    x_starts = work["x_start"].to_numpy()
    attack_dirs = work["attack_dir"].to_numpy()

    n_total = len(work)
    transition_records: list[dict[str, object]] = []
    retained_indices: list[int] = []
    retained_tids: list[int] = []
    failed_pass_player_by_tid: dict[int, str] = {}

    for start_pos in candidate_positions:
        tid = int(event_ids[start_pos])
        start_ms = int(timestamps[start_pos])
        start_half = int(halves[start_pos])
        start_team = teams[start_pos]
        start_seq = int(sequences[start_pos])

        passage_positions = [int(start_pos)]
        stop_reason = "end_of_events"
        terminal_event_id: object = pd.NA

        for pos in range(int(start_pos) + 1, n_total):
            ev_id = int(event_ids[pos])
            ev_ms = int(timestamps[pos])
            ev_name = names[pos]

            if int(halves[pos]) != start_half:
                stop_reason = "half_change"
                terminal_event_id = ev_id
                break
            if (ev_ms - start_ms) > window_ms:
                stop_reason = "window_elapsed"
                terminal_event_id = ev_id
                break
            if teams[pos] != start_team:
                stop_reason = "opponent_event"
                terminal_event_id = ev_id
                break
            if int(sequences[pos]) != start_seq:
                stop_reason = "sequence_change"
                terminal_event_id = ev_id
                break
            if ev_name in RESTART_EVENTS:
                stop_reason = "restart"
                terminal_event_id = ev_id
                break
            if ev_name in DEFENSIVE_INTERVENTION_EVENTS:
                stop_reason = "defensive_intervention"
                terminal_event_id = ev_id
                break
            if ev_name == "pass" and not _is_true_flag(pass_flags[pos]):
                stop_reason = "failed_pass"
                terminal_event_id = ev_id
                failed_pid = _clean_player_id(players[pos])
                if failed_pid is not None:
                    failed_pass_player_by_tid[tid] = failed_pid
                break

            passage_positions.append(pos)
            if ev_name == "goal":
                stop_reason = "goal"
                terminal_event_id = ev_id
                break

        end_ms = int(max(int(timestamps[p]) for p in passage_positions))
        start_x_att = _compute_x_att(x_starts[start_pos], attack_dirs[start_pos])
        start_zone = _classify_start_zone(start_x_att)
        clean_start_pid = _clean_player_id(players[start_pos])

        transition_records.append(
            {
                "transition_id": tid,
                "sequence_id": start_seq,
                "team": start_team,
                "half": start_half,
                "start_player_id": clean_start_pid if clean_start_pid is not None else pd.NA,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_x_att": float(start_x_att) if start_x_att is not None else np.nan,
                "start_zone": start_zone,
                "stop_reason": stop_reason,
                "terminal_event_id": terminal_event_id,
            }
        )
        retained_indices.extend(passage_positions)
        retained_tids.extend([tid] * len(passage_positions))

    transitions = pd.DataFrame(transition_records, columns=TRANSITION_COLUMNS)
    transitions["transition_id"] = transitions["transition_id"].astype("int64")
    transitions["sequence_id"] = transitions["sequence_id"].astype("int64")
    transitions["half"] = transitions["half"].astype("int64")
    transitions["start_ms"] = transitions["start_ms"].astype("int64")
    transitions["end_ms"] = transitions["end_ms"].astype("int64")
    transitions["start_x_att"] = transitions["start_x_att"].astype("float64")
    transitions["terminal_event_id"] = transitions["terminal_event_id"].astype("Int64")
    transitions.attrs["failed_pass_player_by_tid"] = failed_pass_player_by_tid
    transitions.attrs["precanonicalised_order"] = is_precanonical

    transition_events = work.iloc[retained_indices].copy().reset_index(drop=True)
    transition_events["transition_id"] = pd.Series(retained_tids, dtype="int64")
    transition_events = transition_events[tev_cols]
    transition_events.attrs["precanonicalised_order"] = is_precanonical

    return transitions, transition_events


def attribute_transitions(
    transitions: pd.DataFrame,
    transition_events: pd.DataFrame,
    progression_threshold_m: float = DEFAULT_PROGRESSION_THRESHOLD_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute passage-level metrics and deduplicated player role contributions."""
    if progression_threshold_m <= 0 or not np.isfinite(progression_threshold_m):
        raise ValueError("progression_threshold_m must be a positive finite number.")

    missing_tr = [col for col in TRANSITION_COLUMNS if col not in transitions.columns]
    if missing_tr:
        raise ValueError(f"transitions is missing required columns: {missing_tr}")

    if transitions.empty:
        return _empty_metrics_df(), _empty_contributions_df()

    is_precanonical = bool(
        transition_events.attrs.get("precanonicalised_order", False)
        or transitions.attrs.get("precanonicalised_order", False)
    )
    if not transition_events.empty:
        sort_cols = (
            ["transition_id", "timestamp_ms"]
            if is_precanonical
            else ["transition_id", "timestamp_ms", "event_id"]
        )
        sorted_tev = transition_events.sort_values(sort_cols, kind="mergesort")
        grouped = {
            int(tid): grp for tid, grp in sorted_tev.groupby("transition_id", sort=False)
        }
    else:
        grouped = {}

    thresh_rounded = round(float(progression_threshold_m), COORD_ROUND_DECIMALS)
    metric_rows: list[dict[str, object]] = []
    contrib_rows: list[tuple[int, str, str, int]] = []

    for _, tr_row in transitions.iterrows():
        tid = int(tr_row["transition_id"])
        grp = grouped.get(tid)

        opening_pid: str | None = _clean_player_id(tr_row["start_player_id"])
        opening_eid = tid
        if grp is not None and not grp.empty:
            opening_match = grp[grp["event_id"] == tid]
            if not opening_match.empty:
                opening_pid = _clean_player_id(opening_match.iloc[0]["player_id"])
                opening_eid = int(opening_match.iloc[0]["event_id"])
            subsequent = grp[grp["event_id"] != tid]
        else:
            subsequent = None

        if opening_pid is not None:
            contrib_rows.append((tid, opening_pid, "recoverer", opening_eid))

        n_completed_passes = 0
        measurable_passes = 0
        missing_endpoint_passes = 0
        final_third_entries = 0
        forward_metres = 0.0
        has_progression = False
        shot_count = 0
        goal_count = 0

        outlet_assigned = False
        entry_pass_records: list[tuple[int, str | None, str | None]] = []
        positive_pass_records: list[tuple[float, int, int, str | None]] = []
        finisher_records: list[tuple[int, str]] = []

        if subsequent is not None and not subsequent.empty:
            for seq_order, (_, ev) in enumerate(subsequent.iterrows()):
                ev_id = int(ev["event_id"])
                ev_name = ev["event_name"]

                if ev_name == "pass" and _is_true_flag(ev["pass_success"]):
                    n_completed_passes += 1
                    passer_id = _clean_player_id(ev["player_id"])
                    receiver_id = _clean_player_id(ev["receiver_player_id"])

                    if not outlet_assigned:
                        outlet_assigned = True
                        if receiver_id is not None:
                            contrib_rows.append((tid, receiver_id, "outlet", ev_id))

                    start_x_att = _compute_x_att(ev["x_start"], ev["attack_dir"])
                    end_x_att = _compute_x_att(ev["receiver_x"], ev["attack_dir"])

                    if start_x_att is not None and end_x_att is not None:
                        measurable_passes += 1
                        gain = end_x_att - start_x_att
                        gain_rounded = round(gain, COORD_ROUND_DECIMALS)

                        if gain > 0.0:
                            forward_metres += gain
                        if gain_rounded >= thresh_rounded:
                            has_progression = True
                        if (
                            round(start_x_att, COORD_ROUND_DECIMALS) <= FINAL_THIRD_X_ATT_M
                            and round(end_x_att, COORD_ROUND_DECIMALS) > FINAL_THIRD_X_ATT_M
                        ):
                            final_third_entries += 1
                            entry_pass_records.append((ev_id, passer_id, receiver_id))
                        if gain_rounded > 0.0:
                            positive_pass_records.append(
                                (gain_rounded, seq_order, ev_id, passer_id)
                            )
                    else:
                        missing_endpoint_passes += 1

                elif ev_name == "shot":
                    shot_count += 1
                    fin_pid = _clean_player_id(ev["player_id"])
                    if fin_pid is not None:
                        finisher_records.append((ev_id, fin_pid))

                elif ev_name == "goal":
                    goal_count += 1
                    fin_pid = _clean_player_id(ev["player_id"])
                    if fin_pid is not None:
                        finisher_records.append((ev_id, fin_pid))

        has_shot = shot_count > 0
        has_goal = goal_count > 0
        has_entry = final_third_entries > 0
        has_followup = bool(n_completed_passes > 0 or has_shot or has_goal)

        if has_shot or has_goal:
            outcome_category = "shot"
        elif has_entry:
            outcome_category = "final_third_entry"
        elif has_progression:
            outcome_category = "forward_progression"
        elif has_followup:
            outcome_category = "other_completed_play"
        else:
            outcome_category = "no_observed_followup"

        if has_entry:
            for ev_id, passer_id, receiver_id in entry_pass_records:
                if passer_id is not None:
                    contrib_rows.append((tid, passer_id, "advancer", ev_id))
                if receiver_id is not None:
                    contrib_rows.append((tid, receiver_id, "advancer", ev_id))
        elif positive_pass_records:
            best_gain, _, best_eid, best_passer = max(
                positive_pass_records, key=lambda item: (item[0], -item[1], -item[2])
            )
            if best_gain > 0.0 and best_passer is not None:
                contrib_rows.append((tid, best_passer, "advancer", best_eid))

        for ev_id, fin_pid in finisher_records:
            contrib_rows.append((tid, fin_pid, "finisher", ev_id))

        base_dict = {col: tr_row[col] for col in TRANSITION_COLUMNS}
        base_dict.update(
            {
                "n_completed_passes": n_completed_passes,
                "has_followup": has_followup,
                "shot_count": shot_count,
                "has_shot": has_shot,
                "goal_count": goal_count,
                "has_goal": has_goal,
                "final_third_entries": final_third_entries,
                "has_entry": has_entry,
                "forward_metres": float(forward_metres),
                "has_progression": has_progression,
                "outcome_category": outcome_category,
                "measurable_passes": measurable_passes,
                "missing_endpoint_passes": missing_endpoint_passes,
            }
        )
        metric_rows.append(base_dict)

    metrics = pd.DataFrame(metric_rows, columns=METRICS_COLUMNS)
    metrics["transition_id"] = metrics["transition_id"].astype("int64")
    metrics["sequence_id"] = metrics["sequence_id"].astype("int64")
    metrics["half"] = metrics["half"].astype("int64")
    metrics["start_ms"] = metrics["start_ms"].astype("int64")
    metrics["end_ms"] = metrics["end_ms"].astype("int64")
    metrics["start_x_att"] = metrics["start_x_att"].astype("float64")
    metrics["terminal_event_id"] = metrics["terminal_event_id"].astype("Int64")
    for int_col in (
        "n_completed_passes",
        "shot_count",
        "goal_count",
        "final_third_entries",
        "measurable_passes",
        "missing_endpoint_passes",
    ):
        metrics[int_col] = metrics[int_col].astype("int64")
    for bool_col in ("has_followup", "has_shot", "has_goal", "has_entry", "has_progression"):
        metrics[bool_col] = metrics[bool_col].astype("bool")
    metrics["forward_metres"] = metrics["forward_metres"].astype("float64")
    metrics.attrs["failed_pass_player_by_tid"] = dict(
        transitions.attrs.get("failed_pass_player_by_tid", {})
    )

    if not contrib_rows:
        contributions = _empty_contributions_df()
    else:
        contributions = pd.DataFrame(contrib_rows, columns=CONTRIBUTION_COLUMNS)
        sort_con_cols = (
            ["transition_id"] if is_precanonical else ["transition_id", "event_id"]
        )
        contributions = (
            contributions.sort_values(sort_con_cols, kind="mergesort")
            .drop_duplicates(
                subset=["transition_id", "player_id", "contribution_role"], keep="first"
            )
            .reset_index(drop=True)
        )
        contributions["transition_id"] = contributions["transition_id"].astype("int64")
        contributions["event_id"] = contributions["event_id"].astype("int64")

    return metrics, contributions


def summarise_players(
    metrics: pd.DataFrame,
    transition_events: pd.DataFrame,
    contributions: pd.DataFrame,
    summary: pd.DataFrame,
    progression_threshold_m: float = DEFAULT_PROGRESSION_THRESHOLD_M,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise role-specific player transition evidence, diagnostics, and screening tiers.

    Preserves every player in ``summary`` (all 33 squad rows in real data, including
    zero-evidence players, with no minimum-minutes exclusion). Separates direct on-ball
    progression/entry/shot actions from downstream recovery/outlet passage associations,
    deduplicating passage denominators across multiple roles or repeated actions.
    """
    if progression_threshold_m <= 0 or not np.isfinite(progression_threshold_m):
        raise ValueError("progression_threshold_m must be a positive finite number.")

    if len(summary.columns) > 0:
        missing_sum = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in summary.columns]
        if missing_sum:
            raise ValueError(f"summary is missing required columns: {missing_sum}")

    if summary.empty:
        return _empty_ranking_df()

    players_df = summary[["player_id", "team", "minutes_played"]].copy()
    players_df["player_id"] = players_df["player_id"].map(_clean_player_id)
    if players_df["player_id"].isna().any():
        raise ValueError("summary contains missing player_id values.")
    if not players_df["player_id"].is_unique:
        raise ValueError("summary contains duplicate player_id values.")
    players_df["minutes_played"] = players_df["minutes_played"].astype("float64")

    player_ids = list(players_df["player_id"])
    thresh_rounded = round(float(progression_threshold_m), COORD_ROUND_DECIMALS)

    if not metrics.empty:
        m_sub = metrics[
            [
                "transition_id",
                "start_zone",
                "stop_reason",
                "terminal_event_id",
                "has_followup",
                "has_entry",
                "has_shot",
                "has_goal",
            ]
        ].copy()
        m_sub["is_shot_or_goal"] = m_sub["has_shot"].astype(bool) | m_sub["has_goal"].astype(bool)
        m_sub["is_attacking_outcome"] = m_sub["has_entry"].astype(bool) | m_sub["is_shot_or_goal"]
        m_by_tid = m_sub.set_index("transition_id")
    else:
        m_by_tid = pd.DataFrame()

    # Attribute terminal failed passes to the passer who played the excluded stopping pass
    failed_pass_counts: dict[str, int] = {pid: 0 for pid in player_ids}
    if not metrics.empty:
        failed_stops = metrics[metrics["stop_reason"] == "failed_pass"]
        if not failed_stops.empty:
            if events is not None and not events.empty and "event_id" in events.columns:
                pid_by_eid = {
                    int(eid): _clean_player_id(pid)
                    for eid, pid in zip(events["event_id"], events["player_id"])
                }
                for term_eid in failed_stops["terminal_event_id"].dropna():
                    f_pid = pid_by_eid.get(int(term_eid))
                    if f_pid is not None and f_pid in failed_pass_counts:
                        failed_pass_counts[f_pid] += 1
            else:
                attr_map: dict[int, str] = metrics.attrs.get("failed_pass_player_by_tid", {})
                for tid in failed_stops["transition_id"]:
                    f_pid = attr_map.get(int(tid))
                    if f_pid is not None and f_pid in failed_pass_counts:
                        failed_pass_counts[f_pid] += 1

    rec_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}
    out_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}
    adv_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}
    fin_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}
    all_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}

    if not contributions.empty:
        dedup_con = contributions.drop_duplicates(
            subset=["transition_id", "player_id", "contribution_role"]
        )
        for _, crow in dedup_con.iterrows():
            pid = _clean_player_id(crow["player_id"])
            if pid is None or pid not in all_tids:
                continue
            tid = int(crow["transition_id"])
            role = crow["contribution_role"]
            all_tids[pid].add(tid)
            if role == "recoverer":
                rec_tids[pid].add(tid)
            elif role == "outlet":
                out_tids[pid].add(tid)
            elif role == "advancer":
                adv_tids[pid].add(tid)
            elif role == "finisher":
                fin_tids[pid].add(tid)

    comp_pass_count: dict[str, int] = {pid: 0 for pid in player_ids}
    meas_pass_count: dict[str, int] = {pid: 0 for pid in player_ids}
    fwd_metres_sum: dict[str, float] = {pid: 0.0 for pid in player_ids}
    net_fwd_metres_sum: dict[str, float] = {pid: 0.0 for pid in player_ids}
    prog_pass_count: dict[str, int] = {pid: 0 for pid in player_ids}
    prog_passage_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}
    entries_deliv_count: dict[str, int] = {pid: 0 for pid in player_ids}
    entries_recv_count: dict[str, int] = {pid: 0 for pid in player_ids}
    entry_passage_tids: dict[str, set[int]] = {pid: set() for pid in player_ids}

    if not transition_events.empty:
        subsequent_events = transition_events[
            transition_events["event_id"] != transition_events["transition_id"]
        ]
        for _, ev in subsequent_events.iterrows():
            if ev["event_name"] != "pass" or not _is_true_flag(ev["pass_success"]):
                continue
            tid = int(ev["transition_id"])
            passer = _clean_player_id(ev["player_id"])
            receiver = _clean_player_id(ev["receiver_player_id"])

            if passer is not None and passer in comp_pass_count:
                comp_pass_count[passer] += 1

            sx = _compute_x_att(ev["x_start"], ev["attack_dir"])
            ex = _compute_x_att(ev["receiver_x"], ev["attack_dir"])
            if sx is None or ex is None:
                continue

            gain = ex - sx
            gain_rounded = round(gain, COORD_ROUND_DECIMALS)
            is_entry = (
                round(sx, COORD_ROUND_DECIMALS) <= FINAL_THIRD_X_ATT_M
                and round(ex, COORD_ROUND_DECIMALS) > FINAL_THIRD_X_ATT_M
            )

            if passer is not None and passer in meas_pass_count:
                meas_pass_count[passer] += 1
                net_fwd_metres_sum[passer] += gain
                if gain > 0.0:
                    fwd_metres_sum[passer] += gain
                if gain_rounded >= thresh_rounded:
                    prog_pass_count[passer] += 1
                    prog_passage_tids[passer].add(tid)
                if is_entry:
                    entries_deliv_count[passer] += 1
                    entry_passage_tids[passer].add(tid)

            if is_entry and receiver is not None and receiver in entries_recv_count:
                entries_recv_count[receiver] += 1
                entry_passage_tids[receiver].add(tid)

    rows: list[dict[str, object]] = []
    for _, prow in players_df.iterrows():
        pid = prow["player_id"]
        r_set = rec_tids[pid]
        o_set = out_tids[pid]

        if r_set and not m_by_tid.empty:
            r_df = m_by_tid.loc[sorted(r_set)]
            rec_starts = len(r_df)
            rec_followups = int(r_df["has_followup"].sum())
            rec_entry_p = int(r_df["has_entry"].sum())
            rec_shot_p = int(r_df["is_shot_or_goal"].sum())
            rec_outcome_p = int(r_df["is_attacking_outcome"].sum())
            rec_def_half = int((r_df["start_zone"] == "defensive_half").sum())
            rec_att_half = int((r_df["start_zone"] == "attacking_half").sum())
            rec_fin_third = int((r_df["start_zone"] == "final_third").sum())
            rec_failed_stops = int((r_df["stop_reason"] == "failed_pass").sum())
        else:
            rec_starts = len(r_set)
            rec_followups = 0
            rec_entry_p = 0
            rec_shot_p = 0
            rec_outcome_p = 0
            rec_def_half = 0
            rec_att_half = 0
            rec_fin_third = 0
            rec_failed_stops = 0

        if o_set and not m_by_tid.empty:
            o_df = m_by_tid.loc[sorted(o_set)]
            out_passages = len(o_df)
            out_entry_p = int(o_df["has_entry"].sum())
            out_shot_p = int(o_df["is_shot_or_goal"].sum())
            out_outcome_p = int(o_df["is_attacking_outcome"].sum())
        else:
            out_passages = len(o_set)
            out_entry_p = 0
            out_shot_p = 0
            out_outcome_p = 0

        comp_p = comp_pass_count[pid]
        meas_p = meas_pass_count[pid]

        rows.append(
            {
                "player_id": pid,
                "team": prow["team"],
                "minutes_played": float(prow["minutes_played"]),
                "recovery_starts": rec_starts,
                "recovery_followups": rec_followups,
                "recovery_entry_passages": rec_entry_p,
                "recovery_shot_passages": rec_shot_p,
                "recovery_outcome_passages": rec_outcome_p,
                "outlet_passages": out_passages,
                "outlet_entry_passages": out_entry_p,
                "outlet_shot_passages": out_shot_p,
                "outlet_outcome_passages": out_outcome_p,
                "completed_transition_passes": comp_p,
                "measurable_transition_passes": meas_p,
                "forward_metres": float(fwd_metres_sum[pid]),
                "progressive_passes": prog_pass_count[pid],
                "progression_passages": len(prog_passage_tids[pid]),
                "entries_delivered": entries_deliv_count[pid],
                "entries_received": entries_recv_count[pid],
                "entry_passages": len(entry_passage_tids[pid]),
                "shot_passages": len(fin_tids[pid]),
                "pareto_tier": 0,
                "progression_rank": 0,
                "net_forward_metres": float(net_fwd_metres_sum[pid]),
                "failed_transition_passes": failed_pass_counts[pid],
                "advancer_passages": len(adv_tids[pid]),
                "involved_passages": len(all_tids[pid]),
                "missing_endpoint_transition_passes": comp_p - meas_p,
                "recovery_defensive_half_starts": rec_def_half,
                "recovery_attacking_half_starts": rec_att_half,
                "recovery_final_third_starts": rec_fin_third,
                "recovery_failed_pass_stops": rec_failed_stops,
            }
        )

    ranking = pd.DataFrame(rows)

    ranking["recovery_followup_share"] = np.where(
        ranking["recovery_starts"] > 0,
        ranking["recovery_followups"] / ranking["recovery_starts"],
        np.nan,
    )
    rf_low, rf_high = wilson_interval(ranking["recovery_followups"], ranking["recovery_starts"])
    ranking["recovery_followup_wilson_low"] = rf_low
    ranking["recovery_followup_wilson_high"] = rf_high

    ranking["recovery_outcome_share"] = np.where(
        ranking["recovery_starts"] > 0,
        ranking["recovery_outcome_passages"] / ranking["recovery_starts"],
        np.nan,
    )
    ro_low, ro_high = wilson_interval(
        ranking["recovery_outcome_passages"], ranking["recovery_starts"]
    )
    ranking["recovery_outcome_wilson_low"] = ro_low
    ranking["recovery_outcome_wilson_high"] = ro_high

    ranking["outlet_outcome_share"] = np.where(
        ranking["outlet_passages"] > 0,
        ranking["outlet_outcome_passages"] / ranking["outlet_passages"],
        np.nan,
    )
    oo_low, oo_high = wilson_interval(
        ranking["outlet_outcome_passages"], ranking["outlet_passages"]
    )
    ranking["outlet_outcome_wilson_low"] = oo_low
    ranking["outlet_outcome_wilson_high"] = oo_high

    ranking["pareto_tier"] = _assign_pareto_tiers(ranking, PARETO_OBJECTIVE_COLUMNS)

    ranking["_fm_round"] = ranking["forward_metres"].round(COORD_ROUND_DECIMALS)
    prog_order_idx = ranking.sort_values(
        by=["_fm_round", "entries_delivered", "player_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).index
    prog_ranks = pd.Series(
        np.arange(1, len(ranking) + 1, dtype=np.int64),
        index=prog_order_idx,
        dtype="int64",
    )
    ranking["progression_rank"] = prog_ranks
    ranking = ranking.drop(columns=["_fm_round"])

    ranking = (
        ranking[RANKING_COLUMNS]
        .sort_values(
            by=["pareto_tier", "progression_rank"], ascending=[True, True], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    return ranking


def run_analysis(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    window_s: float = DEFAULT_WINDOW_S,
    progression_threshold_m: float = DEFAULT_PROGRESSION_THRESHOLD_M,
    *,
    canonicalise_ties: bool = False,
    restart_guard_ms: int = DEFAULT_RESTART_GUARD_MS,
    require_completed_opening_pass: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run full defensive-transition extraction, attribution, and player ranking."""
    transitions, transition_events = extract_transitions(
        events,
        window_s=window_s,
        canonicalise_ties=canonicalise_ties,
        restart_guard_ms=restart_guard_ms,
        require_completed_opening_pass=require_completed_opening_pass,
    )
    metrics, contributions = attribute_transitions(
        transitions,
        transition_events,
        progression_threshold_m=progression_threshold_m,
    )
    ranking = summarise_players(
        metrics,
        transition_events,
        contributions,
        summary,
        progression_threshold_m=progression_threshold_m,
        events=events,
    )
    return {
        "transitions": transitions,
        "transition_events": transition_events,
        "metrics": metrics,
        "contributions": contributions,
        "ranking": ranking,
    }
