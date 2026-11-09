"""Validation suite for transition_analysis.py (synthetic checks + real-data baselines)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_analysis import (
    attribute_transitions,
    canonicalise_tied_events,
    extract_transitions,
    find_tied_sequence_starts,
    run_analysis,
    summarise_players,
    wilson_interval,
)

EXPECTED_TRANSITION_COLS = [
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

EXPECTED_METRICS_COLS = EXPECTED_TRANSITION_COLS + [
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

EXPECTED_CONTRIBUTION_COLS = [
    "transition_id",
    "player_id",
    "contribution_role",
    "event_id",
]

MINIMUM_RANKING_COLS = [
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

PARETO_COLS = [
    "entries_delivered",
    "forward_metres",
    "recovery_outcome_passages",
    "outlet_outcome_passages",
]


def _make_event(
    event_id: int,
    timestamp_ms: int,
    sequence_id: int,
    event_name: str,
    team: str = "home",
    half: int = 1,
    player_id: str | float | None = "home_6",
    attack_dir: int = 1,
    x_start: float | None = -10.0,
    y_start: float | None = 0.0,
    pass_success: bool | float | None = np.nan,
    receiver_player_id: str | float | None = np.nan,
    receiver_x: float | None = np.nan,
    receiver_y: float | None = np.nan,
    is_goal: bool = False,
) -> dict:
    """Build one synthetic event row matching events.csv schema."""
    return {
        "event_id": event_id,
        "timestamp_ms": timestamp_ms,
        "minute": timestamp_ms / 60000.0,
        "half": half,
        "event_name": event_name,
        "team": team,
        "role": "outfield",
        "player_id": player_id,
        "attack_dir": attack_dir,
        "x_start": x_start,
        "y_start": y_start,
        "x_end": np.nan,
        "y_end": np.nan,
        "pass_success": pass_success,
        "receiver_player_id": receiver_player_id,
        "receiver_x": receiver_x,
        "receiver_y": receiver_y,
        "pass_distance_m": np.nan,
        "pass_angle_deg": np.nan,
        "shot_distance_m": np.nan,
        "shot_angle_deg": np.nan,
        "is_goal": is_goal,
        "sequence_id": sequence_id,
    }


def _assert_pareto_invariants(ranking: pd.DataFrame) -> None:
    """Verify strict Pareto dominance and non-domination within every Pareto tier."""
    if ranking.empty:
        return
    vals = ranking[PARETO_COLS].copy()
    vals["forward_metres"] = vals["forward_metres"].round(9)
    arr = vals.to_numpy(dtype=float)
    tiers = ranking["pareto_tier"].to_numpy(dtype=int)
    assert tiers.min() == 1, f"Minimum pareto_tier must be 1, got {tiers.min()}"

    n = len(ranking)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = bool(np.all(arr[i] >= arr[j]) and np.any(arr[i] > arr[j]))
            if dominates:
                assert tiers[i] < tiers[j], (
                    f"Row {i} ({ranking.iloc[i]['player_id']}) dominates row {j} "
                    f"({ranking.iloc[j]['player_id']}) on {PARETO_COLS}, "
                    f"but tiers are {tiers[i]} vs {tiers[j]}"
                )
            if tiers[i] == tiers[j]:
                assert not dominates, "Players in the same Pareto tier cannot dominate each other"

    # Also check every player in tier t > 1 is dominated by at least one player in tier t - 1
    for t in range(2, int(tiers.max()) + 1):
        prev_idx = np.where(tiers == t - 1)[0]
        curr_idx = np.where(tiers == t)[0]
        assert len(curr_idx) > 0, f"Contiguous Pareto tier {t} cannot be empty"
        for j in curr_idx:
            dominated_by_prev = any(
                bool(np.all(arr[i] >= arr[j]) and np.any(arr[i] > arr[j]))
                for i in prev_idx
            )
            assert dominated_by_prev, (
                f"Player {ranking.iloc[j]['player_id']} in tier {t} is not dominated "
                f"by any player in tier {t - 1}"
            )


def check_empty_inputs() -> None:
    """Validate empty event/summary handling and invalid window rejection."""
    empty_events = pd.DataFrame([_make_event(1, 1000, 1, "tackle")]).iloc[0:0]
    summary = pd.DataFrame(
        [
            {"player_id": "home_2", "team": "home", "minutes_played": 96.0},
            {"player_id": "away_9", "team": "away", "minutes_played": 96.0},
        ]
    )

    try:
        extract_transitions(empty_events, window_s=-1.0)
        raise AssertionError("Expected ValueError for negative window_s")
    except ValueError:
        pass

    res = run_analysis(empty_events, summary, window_s=15.0)
    assert set(res.keys()) == {"transitions", "transition_events", "metrics", "contributions", "ranking"}
    assert list(res["transitions"].columns) == EXPECTED_TRANSITION_COLS
    assert list(res["transition_events"].columns) == list(empty_events.columns) + ["transition_id"]
    assert list(res["metrics"].columns) == EXPECTED_METRICS_COLS
    assert list(res["contributions"].columns) == EXPECTED_CONTRIBUTION_COLS
    assert list(res["ranking"].columns[: len(MINIMUM_RANKING_COLS)]) == MINIMUM_RANKING_COLS
    assert len(res["transitions"]) == 0
    assert len(res["transition_events"]) == 0
    assert len(res["metrics"]) == 0
    assert len(res["contributions"]) == 0
    assert len(res["ranking"]) == 2
    assert set(res["ranking"]["pareto_tier"]) == {1}
    assert list(res["ranking"].sort_values("progression_rank")["player_id"]) == ["away_9", "home_2"]

    empty_summary = summary.iloc[0:0]
    empty_rank = summarise_players(
        res["metrics"], res["transition_events"], res["contributions"], empty_summary
    )
    assert len(empty_rank) == 0
    assert list(empty_rank.columns[: len(MINIMUM_RANKING_COLS)]) == MINIMUM_RANKING_COLS

    # Also support completely column-free empty DataFrame
    tr0, tev0 = extract_transitions(pd.DataFrame(), window_s=15.0)
    assert len(tr0) == 0 and len(tev0) == 0

    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def check_extraction_boundaries() -> None:
    """Validate inclusive window, restarts, failed passes, second interventions, goals, and zones."""
    rows = [
        # Seq 1: Non-defensive sequence start (pass) -> NOT a candidate start even with tackle later
        _make_event(1, 1000, 1, "pass", pass_success=True, receiver_player_id="home_8", receiver_x=5.0, receiver_y=0.0),
        _make_event(2, 2000, 1, "tackle", player_id="home_8"),
        # Seq 2: Exact 15.0s inclusive boundary (10000 + 15000 = 25000 included; 25001 excluded); x_att == 0.0 -> defensive_half
        _make_event(10, 10000, 2, "interception", player_id="home_4", x_start=0.0, attack_dir=1),
        _make_event(11, 25000, 2, "pass", player_id="home_4", pass_success=True, receiver_player_id="home_8", x_start=2.0, receiver_x=15.0, receiver_y=0.0),
        _make_event(12, 25001, 2, "pass", player_id="home_8", pass_success=True, receiver_player_id="home_10", x_start=15.0, receiver_x=30.0, receiver_y=0.0),
        # Seq 3: Same-sequence failed pass stops passage and does NOT reconnect; x_att == 17.5 -> attacking_half
        _make_event(20, 40000, 3, "tackle", player_id="home_6", x_start=17.5, attack_dir=1),
        _make_event(21, 42000, 3, "pass", player_id="home_6", pass_success=False, x_start=17.5),
        _make_event(22, 44000, 3, "pass", player_id="home_8", pass_success=True, receiver_player_id="home_9", x_start=20.0, receiver_x=35.0, receiver_y=0.0),
        # Seq 4: Same-sequence restart (throw_in) stops passage and does NOT reconnect; x_att > 17.5 -> final_third
        _make_event(30, 60000, 4, "interception", player_id="away_5", team="away", attack_dir=-1, x_start=-18.0),
        _make_event(31, 63000, 4, "throw_in", player_id="away_2", team="away", attack_dir=-1, x_start=-20.0),
        _make_event(32, 64000, 4, "pass", player_id="away_2", team="away", attack_dir=-1, pass_success=True, receiver_player_id="away_9", x_start=-20.0, receiver_x=-35.0, receiver_y=0.0),
        # Seq 5: Second defensive intervention inside same sequence stops passage
        _make_event(40, 80000, 5, "tackle", player_id="home_3", x_start=-25.0),
        _make_event(41, 82000, 5, "interception", player_id="home_6", x_start=-20.0),
        _make_event(42, 84000, 5, "pass", player_id="home_6", pass_success=True, receiver_player_id="home_10", x_start=-20.0, receiver_x=10.0, receiver_y=0.0),
        # Seq 6: Goal event IS retained, then ends passage immediately
        _make_event(50, 100000, 6, "interception", player_id="home_10", x_start=20.0),
        _make_event(51, 104000, 6, "shot", player_id="home_10", x_start=38.0, is_goal=True),
        _make_event(52, 105000, 6, "goal", player_id="home_10", x_start=np.nan, y_start=np.nan),
        _make_event(53, 106000, 6, "pass", player_id="home_10", pass_success=True, receiver_player_id="home_8", x_start=0.0, receiver_x=-5.0, receiver_y=0.0),
        # Seq 7: Unresolved start stopped by half boundary
        _make_event(60, 120000, 7, "tackle", player_id="home_2", half=1, x_start=-30.0),
        _make_event(61, 125000, 8, "pass", player_id="away_9", team="away", half=2, attack_dir=1, pass_success=True, receiver_player_id="away_8", x_start=0.0, receiver_x=-10.0, receiver_y=0.0),
        # Seq 8: Stopped by kick_corner restart in Half 2
        _make_event(70, 140000, 9, "interception", player_id="home_3", half=2, x_start=-20.0),
        _make_event(71, 142000, 9, "kick_corner", player_id="home_7", half=2, x_start=52.0),
        # Seq 9: Stopped by kick_free_kick restart in Half 2
        _make_event(80, 160000, 10, "tackle", player_id="home_4", half=2, x_start=-10.0),
        _make_event(81, 162000, 10, "kick_free_kick", player_id="home_4", half=2, x_start=-10.0),
        # Seq 10: Pass with missing pass_success (NaN) must NOT be parsed as True -> stops as failed_pass
        _make_event(90, 180000, 11, "interception", player_id="home_6", half=2, x_start=0.0),
        _make_event(91, 181000, 11, "pass", player_id="home_6", half=2, pass_success=np.nan, x_start=0.0),
        # Seq 11: Same-team sequence_change and opponent_event stops
        _make_event(95, 200000, 12, "tackle", player_id="home_8", half=2, x_start=5.0),
        _make_event(96, 202000, 13, "tackle", player_id="home_9", half=2, x_start=8.0),
        _make_event(97, 204000, 14, "pass", player_id="away_9", team="away", half=2, pass_success=True, receiver_player_id="away_8", x_start=0.0, receiver_x=5.0, receiver_y=0.0),
    ]
    # Deliberately reverse row order to verify internal chronological sorting and input immutability
    raw_df = pd.DataFrame(rows[::-1])
    raw_copy = raw_df.copy(deep=True)

    tr, tev = extract_transitions(raw_df, window_s=15.0)
    pd.testing.assert_frame_equal(raw_df, raw_copy)
    assert list(tr.columns) == EXPECTED_TRANSITION_COLS
    assert list(tev.columns) == list(raw_df.columns) + ["transition_id"]

    assert list(tr["transition_id"]) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 96]
    by_id = tr.set_index("transition_id")

    # Check exact inclusive boundary on tid 10
    assert by_id.loc[10, "end_ms"] == 25000
    assert by_id.loc[10, "stop_reason"] == "window_elapsed"
    assert by_id.loc[10, "terminal_event_id"] == 12
    assert by_id.loc[10, "start_zone"] == "defensive_half"
    assert list(tev.loc[tev["transition_id"] == 10, "event_id"]) == [10, 11]

    # Check failed pass stop on tid 20
    assert by_id.loc[20, "end_ms"] == 40000
    assert by_id.loc[20, "stop_reason"] == "failed_pass"
    assert by_id.loc[20, "terminal_event_id"] == 21
    assert by_id.loc[20, "start_zone"] == "attacking_half"
    assert list(tev.loc[tev["transition_id"] == 20, "event_id"]) == [20]

    # Check restart stop on tid 30 & attack_dir=-1 start_zone
    assert by_id.loc[30, "start_x_att"] == 18.0
    assert by_id.loc[30, "start_zone"] == "final_third"
    assert by_id.loc[30, "stop_reason"] == "restart"
    assert by_id.loc[30, "terminal_event_id"] == 31
    assert list(tev.loc[tev["transition_id"] == 30, "event_id"]) == [30]

    # Check second intervention stop on tid 40
    assert by_id.loc[40, "stop_reason"] == "defensive_intervention"
    assert by_id.loc[40, "terminal_event_id"] == 41
    assert list(tev.loc[tev["transition_id"] == 40, "event_id"]) == [40]

    # Check goal retained then terminated on tid 50
    assert by_id.loc[50, "stop_reason"] == "goal"
    assert by_id.loc[50, "terminal_event_id"] == 52
    assert by_id.loc[50, "end_ms"] == 105000
    assert list(tev.loc[tev["transition_id"] == 50, "event_id"]) == [50, 51, 52]

    # Check half boundary on tid 60
    assert by_id.loc[60, "stop_reason"] == "half_change"
    assert by_id.loc[60, "terminal_event_id"] == 61
    assert list(tev.loc[tev["transition_id"] == 60, "event_id"]) == [60]

    # Check corner/free-kick restarts, NaN pass_success, sequence_change, and opponent_event
    assert by_id.loc[70, "stop_reason"] == "restart"
    assert by_id.loc[80, "stop_reason"] == "restart"
    assert by_id.loc[90, "stop_reason"] == "failed_pass"
    assert by_id.loc[95, "stop_reason"] == "sequence_change"
    assert by_id.loc[96, "stop_reason"] == "opponent_event"


def check_attribution_and_outcomes() -> None:
    """Validate direction symmetry, missing vs zero endpoints, shot/goal separation, and priorities."""
    rows = [
        # T100: attack_dir=+1 final-third entry on exact 17.5 boundary (17.5 -> 27.5, +10.0m)
        _make_event(100, 1000, 1, "tackle", team="home", player_id="home_2", attack_dir=1, x_start=5.0),
        _make_event(101, 2000, 1, "pass", team="home", player_id="home_2", attack_dir=1, pass_success=True, receiver_player_id="home_8", x_start=17.5, receiver_x=27.5, receiver_y=5.0),
        # T200: mirror-image attack_dir=-1 final-third entry (-17.5 -> -27.5, +10.0m)
        _make_event(200, 5000, 2, "tackle", team="away", player_id="away_2", attack_dir=-1, x_start=-5.0),
        _make_event(201, 6000, 2, "pass", team="away", player_id="away_2", attack_dir=-1, pass_success=True, receiver_player_id="away_8", x_start=-17.5, receiver_x=-27.5, receiver_y=-5.0),
        # T300: missing endpoint pass (x_start=-25 -> receiver_x=NaN) and missing receiver_player_id
        _make_event(300, 10000, 3, "interception", team="home", player_id="home_4", attack_dir=1, x_start=-25.0),
        _make_event(301, 11000, 3, "pass", team="home", player_id="home_4", attack_dir=1, pass_success=True, receiver_player_id=np.nan, x_start=-25.0, receiver_x=np.nan, receiver_y=np.nan),
        # T400: true zero-gain pass (x_start=-25 -> receiver_x=-25) and pass landing on 17.5 (not >17.5, +5.1->15.1 float test)
        _make_event(400, 20000, 4, "interception", team="home", player_id="home_3", attack_dir=1, x_start=-25.0),
        _make_event(401, 21000, 4, "pass", team="home", player_id="home_3", attack_dir=1, pass_success=True, receiver_player_id="home_6", x_start=-25.0, receiver_x=-25.0, receiver_y=10.0),
        # T500: forward progression (5.1 -> 15.1 = exact 10.0m IEEE-754 test, tie-break advancer on +10.0m)
        _make_event(500, 30000, 5, "tackle", team="home", player_id="home_6", attack_dir=1, x_start=0.0),
        _make_event(501, 31000, 5, "pass", team="home", player_id="home_6", attack_dir=1, pass_success=True, receiver_player_id="home_7", x_start=5.1, receiver_x=15.1, receiver_y=0.0),
        _make_event(502, 32000, 5, "pass", team="home", player_id="home_7", attack_dir=1, pass_success=True, receiver_player_id="home_9", x_start=7.5, receiver_x=17.5, receiver_y=0.0),
        # T600: scoring shot + goal pair (must NOT double-count as two chances or two finisher rows)
        _make_event(600, 40000, 6, "interception", team="away", player_id="away_6", attack_dir=-1, x_start=-10.0),
        _make_event(601, 41000, 6, "shot", team="away", player_id="away_9", attack_dir=-1, x_start=-35.0, is_goal=True),
        _make_event(602, 41500, 6, "goal", team="away", player_id="away_9", attack_dir=-1, x_start=np.nan, y_start=np.nan),
        # T700: standalone goal without a shot row -> still outcome_category == "shot"
        _make_event(700, 50000, 7, "tackle", team="away", player_id="away_10", attack_dir=-1, x_start=-40.0),
        _make_event(701, 51000, 7, "goal", team="away", player_id="away_10", attack_dir=-1, x_start=np.nan, y_start=np.nan),
        # T800: no observed followup
        _make_event(800, 60000, 8, "tackle", team="home", player_id="home_12", attack_dir=1, x_start=-5.0),
    ]
    df = pd.DataFrame(rows)
    tr, tev = extract_transitions(df, window_s=15.0)
    metrics, contrib = attribute_transitions(tr, tev)

    assert list(metrics.columns) == EXPECTED_METRICS_COLS
    assert list(contrib.columns) == EXPECTED_CONTRIBUTION_COLS
    assert contrib["player_id"].isna().sum() == 0
    assert not contrib["player_id"].astype(str).isin(["nan", "<NA>", "None", ""]).any()
    assert not contrib.duplicated(["transition_id", "player_id", "contribution_role"]).any()

    m = metrics.set_index("transition_id")

    # T100 vs T200 direction symmetry
    for tid in (100, 200):
        assert m.loc[tid, "final_third_entries"] == 1
        assert bool(m.loc[tid, "has_entry"]) is True
        assert np.isclose(m.loc[tid, "forward_metres"], 10.0)
        assert bool(m.loc[tid, "has_progression"]) is True
        assert m.loc[tid, "outcome_category"] == "final_third_entry"

    # T300 missing endpoint vs T400 true zero gain
    assert m.loc[300, "n_completed_passes"] == 1
    assert m.loc[300, "measurable_passes"] == 0
    assert m.loc[300, "missing_endpoint_passes"] == 1
    assert m.loc[300, "forward_metres"] == 0.0
    assert bool(m.loc[300, "has_followup"]) is True
    assert m.loc[300, "outcome_category"] == "other_completed_play"
    assert len(contrib[(contrib["transition_id"] == 300) & (contrib["contribution_role"] == "advancer")]) == 0
    assert len(contrib[(contrib["transition_id"] == 300) & (contrib["contribution_role"] == "outlet")]) == 0

    assert m.loc[400, "n_completed_passes"] == 1
    assert m.loc[400, "measurable_passes"] == 1
    assert m.loc[400, "missing_endpoint_passes"] == 0
    assert m.loc[400, "forward_metres"] == 0.0
    assert m.loc[400, "outcome_category"] == "other_completed_play"
    assert len(contrib[(contrib["transition_id"] == 400) & (contrib["contribution_role"] == "advancer")]) == 0

    # T500 IEEE-754 10.0m progression, landing on 17.5 is NOT entry, tie-break earliest event_id (501 -> home_6)
    assert m.loc[500, "final_third_entries"] == 0
    assert bool(m.loc[500, "has_entry"]) is False
    assert bool(m.loc[500, "has_progression"]) is True
    assert np.isclose(m.loc[500, "forward_metres"], 20.0)
    assert m.loc[500, "outcome_category"] == "forward_progression"
    adv_500 = contrib[(contrib["transition_id"] == 500) & (contrib["contribution_role"] == "advancer")]
    assert len(adv_500) == 1
    assert adv_500.iloc[0]["player_id"] == "home_6"
    assert adv_500.iloc[0]["event_id"] == 501

    # T600 shot + goal separation & single finisher
    assert m.loc[600, "shot_count"] == 1
    assert bool(m.loc[600, "has_shot"]) is True
    assert m.loc[600, "goal_count"] == 1
    assert bool(m.loc[600, "has_goal"]) is True
    assert m.loc[600, "outcome_category"] == "shot"
    fin_600 = contrib[(contrib["transition_id"] == 600) & (contrib["contribution_role"] == "finisher")]
    assert len(fin_600) == 1
    assert fin_600.iloc[0]["player_id"] == "away_9"
    assert fin_600.iloc[0]["event_id"] == 601

    # T700 standalone goal -> outcome_category == "shot"
    assert m.loc[700, "shot_count"] == 0
    assert bool(m.loc[700, "has_shot"]) is False
    assert m.loc[700, "goal_count"] == 1
    assert bool(m.loc[700, "has_goal"]) is True
    assert bool(m.loc[700, "has_followup"]) is True
    assert m.loc[700, "outcome_category"] == "shot"

    # T800 no followup
    assert bool(m.loc[800, "has_followup"]) is False
    assert m.loc[800, "outcome_category"] == "no_observed_followup"

    # Verify summarise_players preserves distinction between missing endpoint (home_4) and true zero gain (home_3)
    syn_summary = pd.DataFrame(
        [
            {"player_id": pid, "team": "home" if pid.startswith("home") else "away", "minutes_played": 90.0}
            for pid in ["home_2", "home_8", "away_2", "away_8", "home_4", "home_3", "home_6", "home_7", "home_9", "away_6", "away_9", "away_10", "home_12"]
        ]
    )
    syn_rk = summarise_players(metrics, tev, contrib, syn_summary).set_index("player_id")
    assert syn_rk.loc["home_4", "completed_transition_passes"] == 1
    assert syn_rk.loc["home_4", "measurable_transition_passes"] == 0
    assert syn_rk.loc["home_4", "missing_endpoint_transition_passes"] == 1
    assert syn_rk.loc["home_3", "completed_transition_passes"] == 1
    assert syn_rk.loc["home_3", "measurable_transition_passes"] == 1
    assert syn_rk.loc["home_3", "missing_endpoint_transition_passes"] == 0
    assert "net_forward_metres" in syn_rk.columns
    assert "failed_transition_passes" in syn_rk.columns


def check_duplicated_roles_and_pareto() -> None:
    """Validate unique passage denominators under multi-role involvement and exact Pareto tiers."""
    rows = [
        # T1: p_multi recovers, plays progressive entry pass to p_out, gets entry back from p_out, shoots & scores!
        _make_event(1, 1000, 1, "tackle", team="home", player_id="p_multi", attack_dir=1, x_start=0.0),
        _make_event(2, 2000, 1, "pass", team="home", player_id="p_multi", attack_dir=1, pass_success=True, receiver_player_id="p_out", x_start=0.0, receiver_x=20.0, receiver_y=0.0),
        _make_event(3, 3000, 1, "pass", team="home", player_id="p_out", attack_dir=1, pass_success=True, receiver_player_id="p_multi", x_start=20.0, receiver_x=15.0, receiver_y=0.0),
        _make_event(4, 4000, 1, "pass", team="home", player_id="p_multi", attack_dir=1, pass_success=True, receiver_player_id="p_out", x_start=15.0, receiver_x=26.0, receiver_y=0.0),
        _make_event(5, 5000, 1, "pass", team="home", player_id="p_out", attack_dir=1, pass_success=True, receiver_player_id="p_multi", x_start=16.0, receiver_x=30.0, receiver_y=0.0),
        _make_event(6, 6000, 1, "shot", team="home", player_id="p_multi", attack_dir=1, x_start=35.0, is_goal=True),
        _make_event(7, 6500, 1, "goal", team="home", player_id="p_multi", attack_dir=1, x_start=np.nan, y_start=np.nan),
        # T2: p_prog makes a 45m pass without final-third entry (-40 -> +5)
        _make_event(10, 20000, 2, "interception", team="away", player_id="p_rec", attack_dir=1, x_start=-42.0),
        _make_event(11, 21000, 2, "pass", team="away", player_id="p_rec", attack_dir=1, pass_success=True, receiver_player_id="p_prog", x_start=-42.0, receiver_x=-40.0, receiver_y=0.0),
        _make_event(12, 22000, 2, "pass", team="away", player_id="p_prog", attack_dir=1, pass_success=True, receiver_player_id="p_sub", x_start=-40.0, receiver_x=5.0, receiver_y=0.0),
        # T3: p_sub makes a 20m pass without entry (-30 -> -10), strictly dominated by p_prog (45m vs 20m, 0 others)
        _make_event(20, 40000, 3, "tackle", team="away", player_id="p_sub", attack_dir=1, x_start=-30.0),
        _make_event(21, 41000, 3, "pass", team="away", player_id="p_sub", attack_dir=1, pass_success=True, receiver_player_id="p_zero1", x_start=-30.0, receiver_x=-10.0, receiver_y=0.0),
    ]
    summary = pd.DataFrame(
        [
            {"player_id": "p_multi", "team": "home", "minutes_played": 90.0},
            {"player_id": "p_out", "team": "home", "minutes_played": 90.0},
            {"player_id": "p_rec", "team": "away", "minutes_played": 90.0},
            {"player_id": "p_prog", "team": "away", "minutes_played": 90.0},
            {"player_id": "p_sub", "team": "away", "minutes_played": 45.0},
            {"player_id": "p_zero1", "team": "away", "minutes_played": 15.0},
            {"player_id": "p_zero2", "team": "home", "minutes_played": 0.0},
        ]
    )

    res = run_analysis(pd.DataFrame(rows), summary, window_s=15.0)
    rk = res["ranking"].set_index("player_id")

    # Deduplicated passage checks on p_multi (2 entries delivered, 1 entry received, 2 progressive passes, shot+goal, all in T1!)
    assert rk.loc["p_multi", "recovery_starts"] == 1
    assert rk.loc["p_multi", "recovery_entry_passages"] == 1
    assert rk.loc["p_multi", "recovery_shot_passages"] == 1
    assert rk.loc["p_multi", "recovery_outcome_passages"] == 1  # Union of entry and shot in T1 = 1, NOT 2!
    assert rk.loc["p_multi", "progressive_passes"] == 2
    assert rk.loc["p_multi", "progression_passages"] == 1  # Unique passage = 1, NOT 2!
    assert rk.loc["p_multi", "entries_delivered"] == 2
    assert rk.loc["p_multi", "entries_received"] == 1
    assert rk.loc["p_multi", "entry_passages"] == 1  # Unique passage = 1, NOT 3!
    assert rk.loc["p_multi", "shot_passages"] == 1  # Shot + goal = 1 passage!
    assert rk.loc["p_multi", "involved_passages"] == 1

    # Pareto checks:
    # p_multi: (entries=2, fm=31.0, rec_out=1, out_out=0) -> Tier 1
    # p_out:   (entries=1, fm=14.0, rec_out=0, out_out=1) -> Tier 1 (sole outlet_outcome=1!)
    # p_prog:  (entries=0, fm=45.0, rec_out=0, out_out=0) -> Tier 1 (sole max fm=45.0!)
    # p_sub:   (entries=0, fm=20.0, rec_out=0, out_out=0) -> Tier 2 (dominated by p_prog & p_multi)
    # p_rec:   (entries=0, fm=2.0,  rec_out=0, out_out=0) -> Tier 3 (dominated by p_sub)
    # p_zero1, p_zero2: (0, 0.0, 0, 0)                    -> Tier 4 (dominated by p_rec; equal bottom tier)
    _assert_pareto_invariants(res["ranking"])
    assert rk.loc["p_multi", "pareto_tier"] == 1
    assert rk.loc["p_out", "pareto_tier"] == 1
    assert rk.loc["p_prog", "pareto_tier"] == 1
    assert rk.loc["p_sub", "pareto_tier"] == 2
    assert rk.loc["p_rec", "pareto_tier"] == 3
    assert rk.loc["p_zero1", "pareto_tier"] == 4
    assert rk.loc["p_zero2", "pareto_tier"] == 4

    # Progression rank: p_prog (45m)=1, p_multi (31m)=2, p_sub (20m)=3, p_out (14m)=4, p_rec (2m)=5, p_zero1 (0m)=6, p_zero2 (0m)=7
    assert list(res["ranking"].sort_values("progression_rank")["player_id"]) == [
        "p_prog",
        "p_multi",
        "p_sub",
        "p_out",
        "p_rec",
        "p_zero1",
        "p_zero2",
    ]


def check_real_data_baselines_and_windows() -> dict[float, dict[str, pd.DataFrame]]:
    """Validate real dataset baselines and run 10s, 15s, and 20s window analyses."""
    root = Path(__file__).resolve().parent
    events = pd.read_csv(root / "events.csv")
    summary = pd.read_csv(root / "player_summary.csv")

    # Raw dataset baselines from DATA_DICTIONARY.md and contract
    assert len(events) == 1289, f"Expected 1289 events, got {len(events)}"
    assert len(summary) == 33, f"Expected 33 players, got {len(summary)}"
    assert int((events["event_name"] == "goal").sum()) == 6
    assert int((events["is_goal"] == True).sum()) == 4  # noqa: E712
    assert int(events["event_name"].isin(["tackle", "interception"]).sum()) == 177

    comp_passes = events[(events["event_name"] == "pass") & (events["pass_success"] == True)]  # noqa: E712
    assert len(comp_passes) == 536
    assert int((comp_passes["receiver_x"].notna() & comp_passes["receiver_player_id"].notna()).sum()) == 523

    results: dict[float, dict[str, pd.DataFrame]] = {}
    for w in (10.0, 15.0, 20.0):
        res = run_analysis(events, summary, window_s=w, progression_threshold_m=10.0)
        tr = res["transitions"]
        tev = res["transition_events"]
        met = res["metrics"]
        con = res["contributions"]
        rk = res["ranking"]

        assert len(tr) == 138, f"Expected 138 candidate starts at {w}s, got {len(tr)}"
        assert len(met) == 138
        assert int(met["outcome_category"].notna().sum()) == 138
        assert set(met["outcome_category"]).issubset(
            {
                "shot",
                "final_third_entry",
                "forward_progression",
                "other_completed_play",
                "no_observed_followup",
            }
        )
        assert tev["event_id"].is_unique, "No event may belong to multiple candidate passages"

        # Verify inclusive window bounds on every retained event
        merged = tev.merge(tr[["transition_id", "start_ms", "end_ms"]], on="transition_id", how="left")
        dt_ms = merged["timestamp_ms"] - merged["start_ms"]
        assert bool((dt_ms >= 0).all())
        assert bool((dt_ms <= round(w * 1000.0, 6)).all())
        assert bool((merged["timestamp_ms"] <= merged["end_ms"]).all())

        # Verify all 33 players preserved, unattributed opening tackle (event_id=754) kept unknown,
        # and Pareto + progression_rank invariants
        assert len(rk) == 33
        assert set(rk["player_id"]) == set(summary["player_id"])
        assert set(rk["progression_rank"]) == set(range(1, 34))
        assert int(tr["start_player_id"].isna().sum()) == 1
        assert int(tr.loc[tr["start_player_id"].isna(), "transition_id"].iloc[0]) == 754
        assert int(rk["recovery_starts"].sum()) == int(tr["start_player_id"].notna().sum()) == 137
        _assert_pareto_invariants(rk)

        # Verify denominators >= numerators everywhere
        assert bool((rk["recovery_starts"] >= rk["recovery_followups"]).all())
        assert bool((rk["recovery_starts"] >= rk["recovery_outcome_passages"]).all())
        assert bool((rk["outlet_passages"] >= rk["outlet_outcome_passages"]).all())
        assert bool((rk["completed_transition_passes"] >= rk["measurable_transition_passes"]).all())
        assert bool((rk["measurable_transition_passes"] >= rk["progressive_passes"]).all())
        assert bool((rk["progressive_passes"] >= rk["progression_passages"]).all())

        results[w] = res

    # Identical stable transition_ids across 10s, 15s, and 20s
    assert list(results[10.0]["transitions"]["transition_id"]) == list(results[15.0]["transitions"]["transition_id"])
    assert list(results[15.0]["transitions"]["transition_id"]) == list(results[20.0]["transitions"]["transition_id"])

    # Contract 15s pilot baseline: 3 shots and 10 final-third entry passes
    m15 = results[15.0]["metrics"]
    assert int(m15["shot_count"].sum()) == 3, f"Expected 3 shots at 15s, got {m15['shot_count'].sum()}"
    assert int(m15["final_third_entries"].sum()) == 10, (
        f"Expected 10 final-third entry passes at 15s, got {m15['final_third_entries'].sum()}"
    )

    return results


def check_tie_canonicalisation_and_diagnostics() -> None:
    """Validate opt-in tied-start canonicalisation, <=100ms restart guard, and net/failure cols."""
    rows = [
        # Seq 1: Same-timestamp same-player completed pass (eid=1) before interception (eid=2),
        # followed by legitimate open-play pass (+500ms, eid=3), then co-detected throw-in pass (+34ms before throw_in, eid=4,5)
        _make_event(1, 10000, 1, "pass", team="away", player_id="away_12", attack_dir=-1, pass_success=True, receiver_player_id="away_8", x_start=10.0, receiver_x=-10.0, receiver_y=0.0),
        _make_event(2, 10000, 1, "interception", team="away", player_id="away_12", attack_dir=-1, x_start=10.0),
        _make_event(3, 12000, 1, "pass", team="away", player_id="away_8", attack_dir=-1, pass_success=True, receiver_player_id="away_12", x_start=-10.0, receiver_x=5.0, receiver_y=0.0),
        _make_event(4, 15000, 1, "pass", team="away", player_id="away_2", attack_dir=-1, pass_success=True, receiver_player_id="away_9", x_start=-15.0, receiver_x=-35.0, receiver_y=0.0),
        _make_event(5, 15034, 1, "throw_in", team="away", player_id="away_2", attack_dir=-1, x_start=-15.0),
        # Seq 2: Same-timestamp DIFFERENT-player pass before interception -> must NOT swap
        _make_event(10, 30000, 2, "pass", team="home", player_id="home_4", attack_dir=1, pass_success=True, receiver_player_id="home_6", x_start=-20.0, receiver_x=-10.0, receiver_y=0.0),
        _make_event(11, 30000, 2, "interception", team="home", player_id="home_6", attack_dir=1, x_start=-10.0),
        # Seq 3: Normal tackle followed by failed pass by home_6 (tests failed_transition_passes)
        _make_event(20, 50000, 3, "tackle", team="home", player_id="home_6", attack_dir=1, x_start=0.0),
        _make_event(21, 52000, 3, "pass", team="home", player_id="home_6", attack_dir=1, pass_success=False, x_start=0.0),
    ]
    syn_events = pd.DataFrame(rows)
    syn_copy = syn_events.copy(deep=True)
    syn_summary = pd.DataFrame(
        [
            {"player_id": "away_12", "team": "away", "minutes_played": 90.0},
            {"player_id": "away_8", "team": "away", "minutes_played": 90.0},
            {"player_id": "away_2", "team": "away", "minutes_played": 90.0},
            {"player_id": "away_9", "team": "away", "minutes_played": 90.0},
            {"player_id": "home_4", "team": "home", "minutes_played": 90.0},
            {"player_id": "home_6", "team": "home", "minutes_played": 90.0},
        ]
    )

    # Default baseline ignores Seq 1 tied start (only Seq 3 starts with tackle)
    base_res = run_analysis(syn_events, syn_summary, window_s=15.0)
    assert list(base_res["transitions"]["transition_id"]) == [20]
    base_rk = base_res["ranking"].set_index("player_id")
    assert base_rk.loc["home_6", "failed_transition_passes"] == 1

    # Opt-in canonicalisation preserves raw columns, event_ids, and timestamps, recovers Seq 1 (tid=2),
    # ignores Seq 2 (different players), and stops BEFORE throw_in (eid=5) without leaking eid=4!
    canon = canonicalise_tied_events(syn_events)
    pd.testing.assert_frame_equal(syn_events, syn_copy)
    assert list(canon.columns) == list(syn_events.columns)
    assert set(canon["event_id"]) == set(syn_events["event_id"])
    assert (
        canon.sort_values("event_id")["timestamp_ms"].tolist()
        == syn_events.sort_values("event_id")["timestamp_ms"].tolist()
    )

    alt_res = run_analysis(syn_events, syn_summary, window_s=15.0, canonicalise_ties=True)
    alt_via_df = run_analysis(canon, syn_summary, window_s=15.0)
    pd.testing.assert_frame_equal(alt_res["transitions"], alt_via_df["transitions"])
    pd.testing.assert_frame_equal(alt_res["ranking"], alt_via_df["ranking"])

    assert list(alt_res["transitions"]["transition_id"]) == [2, 20]
    t2 = alt_res["transitions"].set_index("transition_id").loc[2]
    assert t2["stop_reason"] == "restart"
    assert t2["terminal_event_id"] == 5
    # Retained events for tid=2 must be [2, 1, 3] — eid=4 (34ms before throw_in eid=5) is excluded!
    assert list(alt_res["transition_events"].loc[alt_res["transition_events"]["transition_id"] == 2, "event_id"]) == [2, 1, 3]

    # Check net_forward_metres: away_8 passed backward (-10 -> +5 with attack_dir=-1 means +10 -> -5 = -15.0m net, 0.0m positive)
    alt_rk = alt_res["ranking"].set_index("player_id")
    assert np.isclose(alt_rk.loc["away_8", "forward_metres"], 0.0)
    assert np.isclose(alt_rk.loc["away_8", "net_forward_metres"], -15.0)
    assert np.isclose(alt_rk.loc["away_12", "forward_metres"], 20.0)
    assert np.isclose(alt_rk.loc["away_12", "net_forward_metres"], 20.0)

    # Real-data checks for net_forward_metres, failed_transition_passes, and tied-start sensitivity
    root = Path(__file__).resolve().parent
    real_events = pd.read_csv(root / "events.csv")
    real_summary = pd.read_csv(root / "player_summary.csv")

    tied_comp = find_tied_sequence_starts(real_events, require_completed_pass=True)
    tied_all = find_tied_sequence_starts(real_events, require_completed_pass=False)
    assert len(tied_comp) == 12
    assert len(tied_all) == 13
    assert int((tied_comp["player_id"] == "away_12").sum()) == 4

    real_15 = run_analysis(real_events, real_summary, window_s=15.0)
    rk_15 = real_15["ranking"].set_index("player_id")
    # home_7 has positive forward_metres ~55.04m but negative net_forward_metres ~-8.64m; home_2 has 4 failed passes
    assert rk_15.loc["home_7", "forward_metres"] > 55.0
    assert rk_15.loc["home_7", "net_forward_metres"] < -8.0
    assert rk_15.loc["home_2", "failed_transition_passes"] == 4

    real_alt_comp = run_analysis(real_events, real_summary, window_s=15.0, canonicalise_ties=True)
    real_alt_all = run_analysis(
        real_events,
        real_summary,
        window_s=15.0,
        canonicalise_ties=True,
        require_completed_opening_pass=False,
    )
    assert len(real_alt_comp["transitions"]) == 150
    assert len(real_alt_all["transitions"]) == 151
    rk_alt = real_alt_comp["ranking"].set_index("player_id")
    assert np.isclose(rk_alt.loc["away_12", "forward_metres"], 152.4899, atol=1e-3)
    assert rk_alt.loc["away_12", "entries_delivered"] == 1

    # Verify canonicalise_tied_events orders restart before co-detected pass in seq 143 (425 before 424)
    # and stops seq 317 (TID=908) at kick_free_kick eid=910 without leaking 66ms co-detected pass eid=909
    real_canon = canonicalise_tied_events(real_events)
    seq143_eids = real_canon.loc[real_canon["sequence_id"] == 143, "event_id"].tolist()
    assert seq143_eids[:2] == [418, 417]
    assert seq143_eids.index(425) < seq143_eids.index(424)

    real_alt_30 = run_analysis(real_events, real_summary, window_s=30.0, canonicalise_ties=True)
    tr_908 = real_alt_30["transitions"].set_index("transition_id").loc[908]
    assert tr_908["stop_reason"] == "restart"
    assert tr_908["terminal_event_id"] == 910
    assert 909 not in set(real_alt_30["transition_events"]["event_id"])


def main() -> None:
    check_empty_inputs()
    print("[PASS] Synthetic empty-input and validation checks")
    check_extraction_boundaries()
    print("[PASS] Synthetic extraction boundary checks")
    check_attribution_and_outcomes()
    print("[PASS] Synthetic attribution, missing-endpoint, direction, and outcome checks")
    check_duplicated_roles_and_pareto()
    print("[PASS] Synthetic multi-role deduplication, Pareto tier, and progression rank checks")
    results = check_real_data_baselines_and_windows()
    print("[PASS] Real-data baselines and 10s / 15s / 20s window invariants")
    check_tie_canonicalisation_and_diagnostics()
    print("[PASS] Opt-in tied-start canonicalisation, <=100ms restart guard, net metres, and failed passes")

    for w in (10.0, 15.0, 20.0):
        met = results[w]["metrics"]
        rk = results[w]["ranking"]
        cats = met["outcome_category"].value_counts().to_dict()
        tier1 = rk[rk["pareto_tier"] == 1][
            [
                "player_id",
                "team",
                "minutes_played",
                "entries_delivered",
                "forward_metres",
                "recovery_outcome_passages",
                "outlet_outcome_passages",
                "progression_rank",
            ]
        ]
        print(f"\n--- Window {w:.0f}s Outcome Categories (n={len(met)}) ---")
        print(cats)
        print(f"--- Window {w:.0f}s Pareto Tier 1 Players (n={len(tier1)}) ---")
        print(tier1.to_string(index=False))


if __name__ == "__main__":
    main()
