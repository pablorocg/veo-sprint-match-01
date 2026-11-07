"""Automated analytical and Streamlit AppTest validation suite for the coach dashboard."""

from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

import analytics as an


def test_dataset_baselines_and_goal_shot_schema() -> None:
    """Verify exact dataset baselines from AGENTS.md and DATA_DICTIONARY.md."""
    summary, events, players = an.load_datasets(".")
    assert len(summary) == 33, "Expected 33 rows in player_summary.csv"
    assert len(events) == 1289, "Expected 1,289 rows in events.csv"
    assert len(players) == 322451, "Expected 322,451 rows in players.parquet"

    goals = events[events["event_name"] == "goal"]
    shots = events[events["event_name"] == "shot"]
    shots_is_goal = events[events["is_goal"] == True]  # noqa: E712

    assert len(goals) == 6, "Expected 6 goal event rows"
    assert len(shots) == 29, "Expected 29 shot event rows"
    assert len(shots_is_goal) == 4, "Only 4 shots have is_goal == True"
    assert goals["x_start"].isna().all() and goals["y_start"].isna().all(), "Goal rows carry no coordinates"


def test_team_summary_metrics_invariants() -> None:
    """Verify team-level metrics match README.md and DATA_DICTIONARY.md."""
    summary, events, players = an.load_datasets(".")
    tm = an.compute_team_summary_metrics(summary, events, players)

    assert tm["away"]["goals"] == 4
    assert tm["home"]["goals"] == 2
    assert tm["away"]["shots"] == 14
    assert tm["home"]["shots"] == 15
    assert tm["away"]["passes"] == 393
    assert tm["home"]["passes"] == 439
    assert tm["away"]["pass_completion_pct"] == 64.4
    assert tm["home"]["pass_completion_pct"] == 64.5
    assert tm["away"]["outfield_distance_km"] == 103.6
    assert tm["home"]["outfield_distance_km"] == 101.0
    assert tm["away"]["tackles"] == 34
    assert tm["home"]["tackles"] == 45
    assert tm["away"]["interceptions"] == 54
    assert tm["home"]["interceptions"] == 44


def test_nearest_frame_same_half_and_tolerance() -> None:
    """Verify nearest-frame matching stays within the same half and respects tolerance."""
    _, _, players = an.load_datasets(".")

    # Goal at timestamp_ms = 4982299 in Half 2 -> 83:02 video clock
    assert an.format_video_clock(4982299) == "83:02"
    frame_df, diff_ms = an.get_nearest_frame(players, 4982299, half=2, max_tolerance_ms=1000)
    assert not frame_df.empty
    assert diff_ms is not None and diff_ms <= 250
    assert (frame_df["half"] == 2).all()

    # Querying a Half 2 timestamp against Half 1 with 1000ms tolerance must return empty
    empty_df, bad_diff = an.get_nearest_frame(players, 4982299, half=1, max_tolerance_ms=1000)
    assert empty_df.empty
    assert bad_diff is not None and bad_diff > 1000


def test_split_trajectory_segments_half_and_gaps() -> None:
    """Verify trajectory segmentation splits across half boundaries, missing ball coords, and time gaps."""
    fake_track = pd.DataFrame(
        [
            {"timestamp_ms": 1000, "half": 1, "x": 0.0, "y": 0.0},
            {"timestamp_ms": 1400, "half": 1, "x": 1.0, "y": 1.0},
            # Time gap > 1000 ms in same half
            {"timestamp_ms": 3000, "half": 1, "x": 5.0, "y": 5.0},
            # Half jump even with small timestamp diff
            {"timestamp_ms": 3400, "half": 2, "x": -5.0, "y": -5.0},
            {"timestamp_ms": 3800, "half": 2, "x": -4.0, "y": -4.0},
        ]
    )
    segs = an.split_trajectory_segments(fake_track, "x", "y", max_gap_ms=1000)
    assert len(segs) == 3
    assert len(segs[0]) == 2
    assert len(segs[1]) == 1
    assert len(segs[2]) == 2


def test_moments_and_player_catalogs() -> None:
    """Verify key moments catalog includes all 6 goals + 29 shots and player catalog tracks H1/H2 frames."""
    summary, events, players = an.load_datasets(".")
    moments = an.build_key_moments_catalog(events, players)
    assert len(moments) == 35
    goals_m = moments[moments["moment_type"] == "Goal"]
    shots_m = moments[moments["moment_type"] == "Shot"]
    assert len(goals_m) == 6
    assert len(shots_m) == 29
    assert int(goals_m["has_shot_row"].sum()) == 4

    p_cat = an.build_player_catalog(summary, players)
    assert len(p_cat) == 33
    assert int(p_cat["comparable_both_halves"].sum()) >= 13


def test_streamlit_app_headless_and_interactions() -> None:
    """Run headless Streamlit AppTest on app.py and verify control state changes."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Default app run raised exception: {at.exception}"

    # Verify top-level metrics rendered
    assert len(at.metric) >= 12
    assert "Away 4 – 2 Home" in at.metric[0].value

    # Change team filter in sidebar radio to 'home'
    at.sidebar.radio[0].set_value("home").run()
    assert not at.exception, f"Team filter change raised exception: {at.exception}"

    # Change moment category to 'Stretched Defense Moments (Heuristic)'
    at.sidebar.selectbox[0].set_value("Stretched Defense Moments (Heuristic)").run()
    assert not at.exception, f"Moment category change raised exception: {at.exception}"

    # Change selected player in sidebar player inspector
    at.sidebar.selectbox[1].set_value("home_10").run()
    assert not at.exception, f"Player selection change raised exception: {at.exception}"
