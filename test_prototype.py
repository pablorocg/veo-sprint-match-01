"""Automated validation suite for the Defensive Shape Explorer prototype."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import analytics


@pytest.fixture(scope="module")
def raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load real match datasets once for the test module."""
    return analytics.load_raw_datasets(".")


@pytest.fixture(scope="module")
def shape_df(raw_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Compute outfield shape timeseries once for the test module."""
    _, _, players_df = raw_data
    return analytics.compute_team_shape_timeseries(players_df)


def test_dataset_baselines(raw_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Verify exact repository baselines required by AGENTS.md and DATA_DICTIONARY.md."""
    summary_df, events_df, players_df = raw_data
    baselines = analytics.validate_dataset_baselines(summary_df, events_df, players_df)

    assert baselines["summary_rows"] == 33
    assert baselines["event_rows"] == 1289
    assert baselines["goal_event_rows"] == 6
    assert baselines["shot_is_goal_rows"] == 4
    assert baselines["halves"] == [1, 2]

    # Verify pitch coordinate bounds (105 x 67.7 m pitch plus small touchline/goal-line run-off)
    assert players_df["x"].min() >= -analytics.PITCH_L / 2.0 - 3.0
    assert players_df["x"].max() <= analytics.PITCH_L / 2.0 + 3.0
    assert players_df["y"].min() >= -analytics.PITCH_W / 2.0 - 3.0
    assert players_df["y"].max() <= analytics.PITCH_W / 2.0 + 3.0


def test_nearest_frame_same_half_and_tolerance(
    raw_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Ensure nearest-frame lookup stays strictly within the requested half and respects tolerance."""
    _, _, players_df = raw_data

    # Known Half 1 start is 351199 ms; offset by +120 ms should match 351199 ms in Half 1
    frame_df, matched_ts, diff_ms = analytics.get_nearest_frame(
        players_df, timestamp_ms=351319, half=1, max_tolerance_ms=500
    )
    assert not frame_df.empty
    assert matched_ts == 351199
    assert diff_ms == 120
    assert (frame_df["half"] == 1).all()

    # Querying a Half 1 timestamp while requesting Half 2 must NOT cross the half boundary
    frame_cross, matched_cross, diff_cross = analytics.get_nearest_frame(
        players_df, timestamp_ms=3209999, half=2, max_tolerance_ms=1000
    )
    assert frame_cross.empty
    assert matched_cross is None
    assert diff_cross is not None and diff_cross > 1000


def test_outfield_filtering_and_shape_invariants(
    raw_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    shape_df: pd.DataFrame,
) -> None:
    """Verify goalkeepers and referee are excluded from outfield shape metrics and math is exact."""
    _, _, players_df = raw_data

    # Check a specific frame manually against shape_df
    sample_ts = 3585199
    sample_half = 2
    sample_team = "away"

    raw_frame = players_df[
        (players_df["half"] == sample_half) & (players_df["timestamp_ms"] == sample_ts)
    ]
    # Confirm goalkeeper and referee exist in raw frame
    assert "goalkeeper" in raw_frame[raw_frame["team"] == sample_team]["role"].values
    assert "referee" in raw_frame["role"].values

    outfield_only = raw_frame[
        (raw_frame["team"] == sample_team) & (raw_frame["role"] == "outfield")
    ]
    expected_width = float(outfield_only["y"].max() - outfield_only["y"].min())
    expected_depth = float(outfield_only["x"].max() - outfield_only["x"].min())
    sorted_x = np.sort(outfield_only["x"].to_numpy())
    expected_max_gap = float(np.diff(sorted_x).max())

    row = shape_df[
        (shape_df["half"] == sample_half)
        & (shape_df["timestamp_ms"] == sample_ts)
        & (shape_df["team"] == sample_team)
    ].iloc[0]

    assert int(row["n_outfield"]) == len(outfield_only)
    assert np.isclose(row["width_m"], expected_width)
    assert np.isclose(row["depth_m"], expected_depth)
    assert np.isclose(row["max_x_gap_m"], expected_max_gap)
    assert (shape_df["width_m"] >= 0).all()
    assert (shape_df["depth_m"] >= 0).all()
    assert (shape_df["max_x_gap_m"] <= shape_df["depth_m"] + 1e-6).all()


def test_attack_dir_switches_across_halves(shape_df: pd.DataFrame) -> None:
    """Verify attack_dir flips sign between Half 1 and Half 2 for each team."""
    for team in ["home", "away"]:
        h1_dir = shape_df[(shape_df["team"] == team) & (shape_df["half"] == 1)]["attack_dir"].unique()
        h2_dir = shape_df[(shape_df["team"] == team) & (shape_df["half"] == 2)]["attack_dir"].unique()
        assert len(h1_dir) == 1
        assert len(h2_dir) == 1
        assert h1_dir[0] == -h2_dir[0]


def test_ranked_moments_decluster_and_missing_ball_handling(
    raw_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    shape_df: pd.DataFrame,
) -> None:
    """Ensure ranked moments respect minimum separation, half boundaries, and missing ball coords."""
    summary_df, events_df, players_df = raw_data

    ranked = analytics.get_ranked_stretched_moments(
        shape_df=shape_df,
        events_df=events_df,
        team="away",
        half=2,
        rank_metric="stretch_score",
        min_outfield=8,
        context_filter="all",
        min_separation_s=15.0,
        top_n=8,
    )
    assert not ranked.empty
    assert (ranked["half"] == 2).all()
    assert (ranked["n_outfield"] >= 8).all()
    assert ranked["stretch_score"].is_monotonic_decreasing

    # Verify de-clustering: every pair of ranked timestamps is at least 15,000 ms apart
    ts_vals = ranked["timestamp_ms"].to_numpy()
    for i in range(len(ts_vals)):
        for j in range(i + 1, len(ts_vals)):
            assert abs(int(ts_vals[i]) - int(ts_vals[j])) >= 15000

    # Test get_moment_details on a frame where ball_x is missing (NaN)
    missing_ball_row = shape_df[shape_df["ball_x"].isna()].iloc[0]
    details = analytics.get_moment_details(
        players_df=players_df,
        shape_df=shape_df,
        events_df=events_df,
        summary_df=summary_df,
        half=int(missing_ball_row["half"]),
        timestamp_ms=int(missing_ball_row["timestamp_ms"]),
        defending_team="home",
    )
    assert details["found"] is True
    assert np.isnan(details["ball_x"])
    assert np.isnan(details["ball_y"])
    assert details["hull_area_m2"] > 0.0


def test_streamlit_apptest_default_and_interactions() -> None:
    """Run headless Streamlit AppTest on app.py and verify control state updates."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Default AppTest raised exception: {at.exception}"
    assert len(at.metric) == 6

    # Change Defending Team to 'home' and Shape Metric to 'width_m'
    at.sidebar.selectbox[0].set_value("home")
    at.sidebar.selectbox[1].set_value("width_m")
    at.sidebar.radio[0].set_value(1)  # Switch to Half 1
    at.run()
    assert not at.exception, f"Updated control state raised exception: {at.exception}"
    assert "Home (Blue), Half 1" in at.subheader[0].value

    # Switch Moment Selection Mode (radio[1] in main body) to manual video clock scrubber
    for radio_widget in at.radio:
        if radio_widget.label == "Moment Selection Mode":
            radio_widget.set_value("Scrub Video Clock (All Half Frames)")
            break
    at.run()
    assert not at.exception, f"Scrubber mode raised exception: {at.exception}"
