"""Automated analytical & headless Streamlit AppTest validation for Off-Ball Space Creators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analytics as an


def test_dataset_baselines() -> None:
    """Verify required dataset row counts and goal/shot baselines from AGENTS.md & DATA_DICTIONARY.md."""
    datasets = an.load_datasets(".")
    summary = datasets["summary"]
    events = datasets["events"]
    players = datasets["players"]

    assert len(summary) == 33, f"Expected 33 summary rows, got {len(summary)}"
    assert len(events) == 1289, f"Expected 1,289 event rows, got {len(events)}"
    assert len(players) == 322451, f"Expected 322,451 player tracking rows, got {len(players)}"

    goals = events[events["event_name"] == "goal"]
    shots = events[events["event_name"] == "shot"]
    shots_is_goal = events[events["is_goal"] == True]  # noqa: E712

    assert len(goals) == 6, "Expected 6 goal event rows"
    assert len(shots) == 29, "Expected 29 shot event rows"
    assert len(shots_is_goal) == 4, "Expected 4 shots with is_goal == True"
    assert goals["x_start"].isna().all(), "Goal rows should lack coordinates"


def test_nearest_frame_same_half_and_boundaries() -> None:
    """Verify nearest-frame matching stays strictly within the requested half and respects tolerance."""
    datasets = an.load_datasets(".")
    players = datasets["players"]

    # Valid match in Half 2 (Goal at 4982299 ms -> 83:02)
    res_valid = an.find_nearest_frame_in_half(players, 4982299, half=2, tolerance_ms=500)
    assert res_valid["matched"] is True
    assert res_valid["frame_timestamp_ms"] == 4982399
    assert abs(res_valid["offset_ms"]) == 100
    assert (res_valid["frame_df"]["half"] == 2).all()

    # Cross-half timestamp should fail tolerance check in wrong half
    res_wrong_half = an.find_nearest_frame_in_half(players, 4982299, half=1, tolerance_ms=1000)
    assert res_wrong_half["matched"] is False


def test_continuous_window_and_gap_splitting() -> None:
    """Verify continuous window extraction and per-player gap splitting never bridge gaps or halves."""
    datasets = an.load_datasets(".")
    players = datasets["players"]
    events = datasets["events"]

    # Check Event 883 (away shot where home_15 has a 6400ms gap inside the 10s window)
    e883 = events[events["event_id"] == 883].iloc[0]
    w_info = an.extract_continuous_same_half_window(players, int(e883["timestamp_ms"]), int(e883["half"]), 10.0)
    assert w_info["valid"] is True
    assert (w_info["window_df"]["half"] == int(e883["half"])).all()

    h15_rows = w_info["window_df"][w_info["window_df"]["player_id"] == "home_15"]
    segs = an.split_player_continuous_segments(h15_rows)
    assert len(segs) >= 2, "home_15 should be split across its 6.4s tracking gap"
    for seg in segs:
        if len(seg) > 1:
            diffs = np.diff(seg["timestamp_ms"].to_numpy())
            assert np.max(diffs) <= an.MAX_FRAME_GAP_MS


def test_off_ball_runs_and_missing_ball_handling() -> None:
    """Verify outfield filtering, attack_dir normalization, and honest Unavailable handling on missing ball coords."""
    datasets = an.load_datasets(".")
    players = datasets["players"]
    events = datasets["events"]

    # 1. Goal #1012 (83:02 away goal, well-tracked ball)
    e1012 = events[events["event_id"] == 1012].iloc[0]
    w1012 = an.extract_continuous_same_half_window(players, int(e1012["timestamp_ms"]), int(e1012["half"]), 10.0)
    runs_1012 = an.compute_off_ball_runs(w1012, events, str(e1012["team"]), str(e1012["player_id"]))

    assert not runs_1012.empty
    # Goalkeeper (away_1) and referee must never appear in outfield space creator rankings
    assert "away_1" not in runs_1012["player_id"].values
    assert "referee" not in runs_1012["player_id"].values

    # Scorer away_9 is marked on-ball shooter/scorer and not given an off-ball rank
    scorer_row = runs_1012[runs_1012["player_id"] == "away_9"].iloc[0]
    assert bool(scorer_row["is_eligible_off_ball"]) is False
    assert pd.isna(scorer_row["off_ball_rank"])

    # Top ranked runner has rank 1 and valid metrics
    top_row = runs_1012.iloc[0]
    assert int(top_row["off_ball_rank"]) == 1
    assert bool(top_row["metrics_available"]) is True
    assert 0.0 <= float(top_row["heuristic_score"]) <= 100.0

    # 2. Goal #1157 (94:55 home goal, 18/25 frames missing ball coordinates -> 30.8% coverage)
    e1157 = events[events["event_id"] == 1157].iloc[0]
    w1157 = an.extract_continuous_same_half_window(players, int(e1157["timestamp_ms"]), int(e1157["half"]), 10.0)
    runs_1157 = an.compute_off_ball_runs(w1157, events, str(e1157["team"]), str(e1157["player_id"]))

    assert w1157["ball_coverage_pct"] < 50.0
    # Because ball tracking is missing in most of the window, all players must have metrics_available == False
    # and no fabricated numeric off-ball rank or zero heuristic score
    assert (~runs_1157["metrics_available"]).all()
    assert runs_1157["off_ball_rank"].isna().all()
    assert runs_1157["heuristic_score"].isna().all()


def test_streamlit_app_headless() -> None:
    """Run headless Streamlit AppTest on app.py and verify default state and control interactions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit app raised an exception on startup: {at.exception}"

    # Verify metrics rendered on startup (defaulting to away team and Goal #1012 at 83:02)
    assert len(at.metric) >= 5
    assert "83:02.3" in at.metric[0].value

    # Change team selectbox from 'away' to 'home' and re-run
    team_select = at.sidebar.selectbox[0]
    team_select.set_value("home").run()
    assert not at.exception, f"Exception after switching team to home: {at.exception}"

    # Filter to Goals Only (All 6 Goal Rows) and select Goal #1157 (missing ball coords) to verify warning
    filter_radio = at.sidebar.radio[0]
    filter_radio.set_value("Goals Only (All 6 Goal Rows)").run()
    assert not at.exception

    moment_select = at.sidebar.selectbox[1]
    moment_select.set_value(1157).run()
    assert not at.exception
    assert len(at.warning) >= 1, "Expected data availability warning when inspecting Goal #1157"

    # Change lead-up window slider to 15.0 seconds and switch pitch display mode
    window_slider = at.sidebar.slider[0]
    window_slider.set_value(15.0).run()
    assert not at.exception

    mode_radio = at.sidebar.radio[1]
    mode_radio.set_value("Before / After Vectors").run()
    assert not at.exception
