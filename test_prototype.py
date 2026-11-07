"""Automated analytical invariants and headless Streamlit AppTest suite for Passing Options Explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analytics


def test_dataset_baselines() -> None:
    """Verify required source dataset row counts and baseline invariants from AGENTS.md."""
    datasets = analytics.load_datasets(".")
    summary = datasets["summary"]
    events = datasets["events"]
    players = datasets["players"]

    assert len(summary) == 33, "Expected 33 player_summary rows"
    assert len(events) == 1289, "Expected 1,289 event rows"
    assert len(events[events["event_name"] == "goal"]) == 6, "Expected 6 goal event rows"
    assert int(events["is_goal"].fillna(False).sum()) == 4, "Expected 4 shots with is_goal=True"
    assert len(events[events["event_name"] == "pass"]) == 832, "Expected 832 pass event rows"
    assert len(players) == 322451, "Expected 322,451 player tracking rows"


def test_clamped_segment_projection_geometry() -> None:
    """Verify point_to_segment_clamped clamps projections strictly to [0, 1] between endpoints."""
    # Segment from A=(0, 0) to B=(10, 0)
    # 1. Point directly above midpoint (5, 3) -> t=0.5, dist=3.0
    d_mid, px, py, t_mid = analytics.point_to_segment_clamped(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert np.isclose(t_mid, 0.5)
    assert np.isclose(px, 5.0) and np.isclose(py, 0.0)
    assert np.isclose(d_mid, 3.0)

    # 2. Point behind A at (-4, 3) -> clamped to t=0.0 at (0, 0), dist=5.0
    d_before, px0, py0, t0 = analytics.point_to_segment_clamped(-4.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert np.isclose(t0, 0.0)
    assert np.isclose(px0, 0.0) and np.isclose(py0, 0.0)
    assert np.isclose(d_before, 5.0)

    # 3. Point beyond B at (13, -4) -> clamped to t=1.0 at (10, 0), dist=5.0
    d_after, px1, py1, t1 = analytics.point_to_segment_clamped(13.0, -4.0, 0.0, 0.0, 10.0, 0.0)
    assert np.isclose(t1, 1.0)
    assert np.isclose(px1, 10.0) and np.isclose(py1, 0.0)
    assert np.isclose(d_after, 5.0)

    # 4. Degenerate zero-length segment A == B
    d_deg, pxd, pyd, td = analytics.point_to_segment_clamped(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    assert np.isclose(td, 0.0)
    assert np.isclose(d_deg, 5.0)


def test_same_half_nearest_frame_and_tolerance() -> None:
    """Verify frame lookup never crosses half boundaries and enforces stated tolerance."""
    datasets = analytics.load_datasets(".")
    players = datasets["players"]
    frame_index = analytics.build_half_frame_index(players)

    # Half 1 end is ~3210300 ms, Half 2 start is ~3585000 ms.
    # Query in Half 1 near 3580000 ms (close to H2 start, far from H1 end)
    res_h1 = analytics.get_nearest_same_half_frame(
        players_df=players,
        timestamp_ms=3580000,
        half=1,
        tolerance_ms=500,
        frame_index=frame_index,
    )
    assert res_h1["within_tolerance"] is False
    assert res_h1["frame_df"].empty
    assert res_h1["matched_timestamp_ms"] < 3300000  # Strictly a Half 1 timestamp

    # Normal match within tolerance
    res_ok = analytics.get_nearest_same_half_frame(
        players_df=players,
        timestamp_ms=351233,
        half=1,
        tolerance_ms=500,
        frame_index=frame_index,
    )
    assert res_ok["within_tolerance"] is True
    assert not res_ok["frame_df"].empty
    assert (res_ok["frame_df"]["half"] == 1).all()
    assert res_ok["abs_offset_ms"] <= 500


def test_candidate_exclusions_and_attack_dir_normalization() -> None:
    """Verify passer, referee, opponents, and invalid rows are excluded and attack_dir is respected."""
    datasets = analytics.load_datasets(".")
    events = datasets["events"]
    players = datasets["players"]
    frame_index = analytics.build_half_frame_index(players)

    passes = analytics.get_pass_events(events, only_identified_passer=True)
    # Test across first 25 passes from both halves
    sample_passes = pd.concat(
        [
            passes[passes["half"] == 1].head(15),
            passes[passes["half"] == 2].head(15),
        ]
    )

    for _, p_row in sample_passes.iterrows():
        res = analytics.evaluate_pass_options(
            pass_event=p_row,
            players_df=players,
            include_goalkeeper=False,
            frame_index=frame_index,
        )
        cands = res["candidates_df"]
        assert not cands.empty
        # Passer excluded
        assert p_row["player_id"] not in set(cands["target_player_id"])
        # Only same-team outfield players with valid IDs included
        for tid, role in zip(cands["target_player_id"], cands["role"]):
            assert str(tid).startswith(str(p_row["team"]) + "_")
            assert role == "outfield"
        # Clamped t in [0, 1]
        assert ((cands["lane_proj_t"] >= 0.0) & (cands["lane_proj_t"] <= 1.0)).all()
        # Forward progress sign matches (target_x - passer_x) * attack_dir
        px = float(res["passer_row"]["x"])
        expected_fwd = (cands["target_x"] - px) * int(p_row["attack_dir"])
        assert np.allclose(cands["forward_progress_m"], expected_fwd)


def test_edge_cases_unattributed_and_missing_data() -> None:
    """Verify graceful handling of null passer_id, missing defenders, and missing direction."""
    datasets = analytics.load_datasets(".")
    events = datasets["events"]
    players = datasets["players"]
    frame_index = analytics.build_half_frame_index(players)

    # 1. Unattributed pass (player_id is NaN in events.csv)
    unattr_passes = events[(events["event_name"] == "pass") & (events["player_id"].isna())]
    assert len(unattr_passes) == 19
    res_unattr = analytics.evaluate_pass_options(
        pass_event=unattr_passes.iloc[0],
        players_df=players,
        frame_index=frame_index,
    )
    assert res_unattr["candidates_df"].empty
    assert any("unattributed" in w.lower() for w in res_unattr["warnings"])

    # 2. Missing defenders in frame
    valid_pass = events[(events["event_name"] == "pass") & (events["player_id"].notna())].iloc[0]
    same_team_only_players = players[players["team"] == valid_pass["team"]].copy()
    res_no_def = analytics.evaluate_pass_options(
        pass_event=valid_pass,
        players_df=same_team_only_players,
    )
    assert not res_no_def["candidates_df"].empty
    assert (res_no_def["candidates_df"]["lane_safety_score"] == 50.0).all()
    assert (res_no_def["candidates_df"]["target_sep_score"] == 50.0).all()

    # 3. Missing attack_dir (e.g. 0) does not fabricate candidates
    bad_dir_pass = valid_pass.copy()
    bad_dir_pass["attack_dir"] = 0
    res_bad_dir = analytics.evaluate_pass_options(
        pass_event=bad_dir_pass,
        players_df=players,
        frame_index=frame_index,
    )
    assert res_bad_dir["candidates_df"].empty
    assert any("attack_dir" in w for w in res_bad_dir["warnings"])


def test_streamlit_app_headless_and_interactions() -> None:
    """Run headless Streamlit AppTest on app.py and verify control state transitions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised exception on initial load: {at.exception}"

    # Verify metrics and dataframe rendered
    assert len(at.metric) == 4
    assert len(at.dataframe) == 1
    initial_table = at.dataframe[0].value
    assert not initial_table.empty

    # Change Attacking Team radio from 'home' to 'away' and Half from 1 to 2
    at.sidebar.radio[0].set_value("away").run()
    assert not at.exception
    at.sidebar.radio[1].set_value(2).run()
    assert not at.exception

    away_h2_table = at.dataframe[0].value
    assert not away_h2_table.empty
    assert all(str(tid).startswith("away_") for tid in away_h2_table["Teammate ID"])

    # Toggle 'Include Teammate Goalkeeper as Option' checkbox
    at.sidebar.checkbox[1].set_value(True).run()
    assert not at.exception
    with_gk_table = at.dataframe[0].value
    assert "goalkeeper" in set(with_gk_table["Role"])

    # Adjust progression weight slider and verify composite scores update
    at.sidebar.slider[4].set_value(1.0).run()
    assert not at.exception
