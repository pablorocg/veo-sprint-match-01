"""Automated validation suite for Second-Half Intensity & Movement prototype.

Verifies:
1. Python syntax compilation of analytics.py and app.py
2. Source dataset baselines (33 summary rows, 1,289 events, 6 goal rows, 4 is_goal shots)
3. Nearest-frame event matching within the same half and half-boundary separation
4. Per-half high-speed (speed_zone >= 3) and sprint (speed_zone == 4) sampled frame shares
5. Eligibility thresholds, zero-denominator handling, absent-half handling, and empty-set safety
6. Headless Streamlit AppTest execution across default and modified control states
"""

from __future__ import annotations

import math
import py_compile
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analytics as an


def test_syntax_compilation() -> None:
    py_compile.compile("analytics.py", doraise=True)
    py_compile.compile("app.py", doraise=True)
    print("[PASS] Python syntax compilation (analytics.py, app.py)")


def test_analytical_invariants() -> None:
    summary, events, players = an.load_datasets(".")
    info = an.validate_dataset_baselines(summary, events, players)
    assert info["baselines_ok"], f"Dataset baselines failed: {info}"
    assert info["summary_rows"] == 33
    assert info["events_rows"] == 1289
    assert info["goal_event_rows"] == 6
    assert info["is_goal_shot_rows"] == 4
    assert info["half_1_frames"] == 7148
    assert info["half_2_frames"] == 7253
    assert info["half_gap_ms"] > 300_000, "Half 1 and Half 2 must have a distinct half-time gap"

    # Attack-direction normalization check (keepers defend -x_att in both halves)
    for gk_id in ("away_1", "home_1"):
        for half in (1, 2):
            gk_x_att = players.loc[
                (players["player_id"] == gk_id) & (players["half"] == half), "x_att"
            ].mean()
            assert gk_x_att < -35.0, f"{gk_id} in half {half} should have negative x_att, got {gk_x_att}"

    # Nearest-frame matching within the same half
    goal_83 = events[events["event_name"] == "goal"].iloc[4]  # timestamp_ms = 4982299, half 2
    matched = an.match_event_to_nearest_frame(players, int(goal_83["timestamp_ms"]), half=2, max_tolerance_ms=500)
    assert not matched.empty
    assert (matched["half"] == 2).all()
    assert abs(int(matched["timestamp_ms"].iloc[0]) - int(goal_83["timestamp_ms"])) <= 400

    # Cross-half mismatch must return empty when tolerance is respected
    wrong_half = an.match_event_to_nearest_frame(players, int(goal_83["timestamp_ms"]), half=1, max_tolerance_ms=1000)
    assert wrong_half.empty, "Matching a Half 2 timestamp against Half 1 within 1s tolerance must be empty"

    # Per-half player metrics
    per_half = an.compute_per_half_player_metrics(players, summary)
    assert len(per_half) == 66  # 33 players * 2 halves

    # Frame shares must sum to 100% for every observed player-half
    observed = per_half[per_half["n_frames"] > 0]
    zone_sum = (
        observed["z0_share_pct"]
        + observed["z1_share_pct"]
        + observed["z2_share_pct"]
        + observed["z3_share_pct"]
        + observed["z4_share_pct"]
    )
    assert np.allclose(zone_sum, 100.0, atol=1e-6)
    assert np.allclose(
        observed["hs_share_pct"], observed["z3_share_pct"] + observed["z4_share_pct"], atol=1e-6
    )
    assert np.allclose(observed["sprint_share_pct"], observed["z4_share_pct"], atol=1e-6)

    # Half comparison table & edge cases
    comp = an.build_half_comparison_table(
        per_half, summary, min_coverage_pct=50.0, max_coverage_diff_pct=30.0, include_goalkeepers=False
    )
    assert len(comp) == 31  # 31 outfield players
    assert int(comp["is_eligible"].sum()) == 11  # 5 away + 6 home outfield players >=50% cov in both halves

    # Zero denominator check: away_12 had 0 sprints in H1 and 13 sprint frames in H2
    away_12 = comp[comp["player_id"] == "away_12"].iloc[0]
    assert away_12["sprint_share_pct_h1"] == 0.0
    assert away_12["sprint_share_pct_h2"] > 0.0
    assert away_12["sprint_diff_pp"] > 0.0
    assert math.isnan(away_12["sprint_rel_change_pct"]), "Zero H1 denominator must yield NaN relative %"
    assert away_12["sprint_rel_note"] == "Undefined (0% in H1)"

    # Single-half player check: away_16 played H1 only
    away_16 = comp[comp["player_id"] == "away_16"].iloc[0]
    assert not away_16["is_eligible"]
    assert not away_16["both_halves_observed"]
    assert math.isnan(away_16["hs_diff_pp"])
    assert away_16["hs_rel_note"] == "Absent in one half"

    # Low coverage in H1 check: away_9 had only 12 frames (0.17%) in H1 and 50.86% in H2
    away_9 = comp[comp["player_id"] == "away_9"].iloc[0]
    assert not away_9["is_eligible"]
    assert "Below min coverage" in away_9["eligibility_status"]

    # Empty filtered set handling
    comp_impossible = an.build_half_comparison_table(
        per_half, summary, min_coverage_pct=100.0, max_coverage_diff_pct=0.0, include_goalkeepers=False
    )
    # away_8 has 100% in both halves; test empty summary when filtering to empty DataFrame
    empty_team_sum = an.compute_team_half_summary(comp_impossible.iloc[0:0], only_eligible=True)
    assert empty_team_sum.empty

    # Trends & spatial trajectory gap splitting
    trends, goals = an.compute_half_period_trends(players, events, bin_minutes=5)
    assert not trends.empty
    assert set(trends["half"].unique()) == {1, 2}
    assert len(goals) == 6

    spatial_a6 = an.get_player_half_spatial_data(players, "away_6")
    assert spatial_a6[2]["segment_id"].max() >= 1, "away_6 has tracking gaps in H2 that must increment segment_id"

    print("[PASS] Analytical invariants, coverage eligibility, zero-denominator & gap splitting checks")


def test_streamlit_apptest() -> None:
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Default AppTest raised exception: {at.exception}"
    assert len(at.metric) >= 4, "Expected top KPI metrics to render on default load"

    # Change metric radio from High-Speed ('hs') to Sprint ('sprint')
    at.sidebar.radio[0].set_value("sprint").run()
    assert not at.exception, f"Switching to sprint metric raised exception: {at.exception}"

    # Change team filter to 'away'
    at.sidebar.selectbox[0].set_value("away").run()
    assert not at.exception, f"Filtering to away team raised exception: {at.exception}"

    # Test extreme threshold where 0 players are selected (empty state handling)
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception, f"Empty player selection raised exception: {at.exception}"
    assert len(at.warning) >= 1, "Expected user-friendly warning when player selection is empty"

    print("[PASS] Headless Streamlit AppTest (default load, metric toggle, team filter, empty selection state)")


if __name__ == "__main__":
    test_syntax_compilation()
    test_analytical_invariants()
    test_streamlit_apptest()
    print("ALL VALIDATION CHECKS PASSED.")
