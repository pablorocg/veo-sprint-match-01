"""Automated validation and Streamlit AppTest suite for Match Story & Goal Buildup Explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analytics


def test_dataset_baselines_and_invariants() -> None:
    """Verify exact dataset row counts, goal rows vs is_goal shots, and coordinate extents."""
    data = analytics.load_datasets(".")
    checks = analytics.validate_dataset_baselines(data)
    assert checks["summary_rows"] == 33
    assert checks["events_rows"] == 1289
    assert checks["goal_event_rows"] == 6
    assert checks["shots_with_is_goal"] == 4
    assert checks["total_shots"] == 29
    assert checks["goal_rows_have_no_coords"] is True


def test_all_six_goals_resolved_with_provenance() -> None:
    """Verify all 6 goals resolve with explicit provenance and no fabricated goal coordinates."""
    data = analytics.load_datasets(".")
    goals = analytics.resolve_all_goals(data["events"], data["players"], data["ball"])
    assert len(goals) == 6

    exact_shot_goals = [g for g in goals if g.is_exact_shot_event]
    tracked_ball_goals = [g for g in goals if not g.is_exact_shot_event]
    assert len(exact_shot_goals) == 4
    assert len(tracked_ball_goals) == 2

    # Goals #792 and #1157 have no shot row in events.csv and must clearly state their tracked-ball source
    no_shot_ids = {g.goal_event_id for g in tracked_ball_goals}
    assert no_shot_ids == {792, 1157}
    for g in tracked_ball_goals:
        assert g.linked_shot_event_id is None
        assert "No shot event in events.csv" in g.location_source_label
        assert np.isfinite(g.reference_x) and np.isfinite(g.reference_y)
        assert g.shot_distance_m > 0


def test_nearest_frame_same_half_and_tolerance() -> None:
    """Verify nearest-frame matching never crosses half boundaries and respects tolerance_ms."""
    data = analytics.load_datasets(".")
    players = data["players"]

    # Half 1 end is around 3210.3 s (3210300 ms); Half 2 starts around 3585.0 s (3585000 ms)
    f_h1, ts_h1, diff_h1 = analytics.get_nearest_frame(players, 3210000, half=1, tolerance_ms=1000)
    assert not f_h1.empty
    assert ts_h1 is not None
    assert (f_h1["half"] == 1).all()

    # Querying a Half 2 timestamp while specifying half=1 with 1000ms tolerance must return empty
    f_cross, ts_cross, diff_cross = analytics.get_nearest_frame(
        players, 3700099, half=1, tolerance_ms=1000
    )
    assert f_cross.empty
    assert ts_cross is None
    assert diff_cross is not None and diff_cross > 1000


def test_split_trajectory_segments_handles_gaps_and_halves() -> None:
    """Verify trajectory segmentation splits on missing ball coordinates, time gaps, and half changes."""
    df = pd.DataFrame(
        [
            {"timestamp_ms": 1000, "half": 1, "x": 0.0, "y": 0.0},
            {"timestamp_ms": 1400, "half": 1, "x": 1.0, "y": 1.0},
            {"timestamp_ms": 1800, "half": 1, "x": np.nan, "y": np.nan},  # missing coords
            {"timestamp_ms": 3500, "half": 1, "x": 5.0, "y": 2.0},  # gap > 1200ms
            {"timestamp_ms": 3900, "half": 2, "x": -5.0, "y": -2.0},  # half boundary jump
            {"timestamp_ms": 4300, "half": 2, "x": -6.0, "y": -3.0},
        ]
    )
    segs = analytics.split_trajectory_segments(df, "x", "y", max_gap_ms=1200)
    assert len(segs) == 3
    assert len(segs[0]) == 2 and (segs[0]["half"] == 1).all()
    assert len(segs[1]) == 1 and (segs[1]["half"] == 1).all()
    assert len(segs[2]) == 2 and (segs[2]["half"] == 2).all()


def test_buildup_window_and_spatial_metrics() -> None:
    """Verify buildup window extraction and outfield-only spatial metrics across all 6 goals."""
    data = analytics.load_datasets(".")
    goals = analytics.resolve_all_goals(data["events"], data["players"], data["ball"])

    for g in goals:
        win = analytics.extract_buildup_window(
            events=data["events"],
            players=data["players"],
            anchor_timestamp_ms=g.goal_timestamp_ms,
            half=g.half,
            attacking_team=g.team,
            lead_in_seconds=18.0,
        )
        assert len(win["frame_timestamps"]) >= 35
        assert (win["players_window"]["half"] == g.half).all()
        assert not win["metrics_timeline"].empty

        # Check comparable unsuccessful shot ranking
        for crit in [
            "same_team_half_location",
            "same_team_any_half_location",
            "closest_location_any_team",
            "closest_shot_distance",
        ]:
            comp_df = analytics.find_comparable_unsuccessful_shots(
                data["events"], g, criterion=crit
            )
            assert not comp_df.empty
            assert (~comp_df["is_goal"].fillna(False)).all()
            assert (comp_df["sequence_id"] != g.sequence_id).all()


def test_streamlit_app_headless_and_interactions() -> None:
    """Run headless Streamlit AppTest on app.py and verify control state transitions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Default app.py run raised exception: {at.exception}"

    # Verify metrics, dataframes, and info/success banners rendered
    assert len(at.metric) >= 8
    assert len(at.dataframe) >= 4
    assert len(at.info) == 1
    assert len(at.success) == 1

    # Change selected goal in selectbox 0 to Goal #792 (index 1, goal without shot row)
    at.sidebar.selectbox[0].set_value(1).run()
    assert not at.exception, f"Selecting Goal index 1 raised exception: {at.exception}"
    assert "event_id=792" in at.info[0].value
    assert "No shot event in events.csv" in at.info[0].value

    # Change selected goal to Goal #1157 (index 5, home_18 corner goal without shot row)
    at.sidebar.selectbox[0].set_value(5).run()
    assert not at.exception, f"Selecting Goal index 5 raised exception: {at.exception}"
    assert "event_id=1157" in at.info[0].value

    # Change lead-in window slider and comparable criterion
    at.sidebar.slider[0].set_value(20).run()
    assert not at.exception
    at.sidebar.selectbox[1].set_value("closest_shot_distance").run()
    assert not at.exception

    # Change half filter radio on the Match Story shot map
    at.radio[0].set_value("Second Half (Away 4–2 Home)").run()
    assert not at.exception
