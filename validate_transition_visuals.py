"""Validation suite for transition_visuals.py.

Runs deterministic synthetic and temporary-media checks without mutating
any repository datasets or generating match video assets.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd

import transition_visuals as tv


OUTCOME_CATEGORIES = (
    "shot",
    "final_third_entry",
    "forward_progression",
    "other_completed_play",
    "no_observed_followup",
)


def _render_png_bytes(fig: matplotlib.figure.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    data = buf.getvalue()
    assert len(data) > 2000, "Rendered PNG buffer is unexpectedly small"
    return data


def _make_synthetic_metrics(counts_by_team_outcome: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tid = 10
    seq_id = 1
    zones = ("defensive_half", "attacking_half", "final_third")
    stops = ("window_end", "failed_pass", "opponent_event", "restart", "goal")

    for (team, cat), count in counts_by_team_outcome.items():
        for i in range(count):
            has_shot = cat == "shot"
            has_goal = cat == "shot" and (i % 3 == 0)
            has_entry = cat in ("final_third_entry",) or (has_shot and i % 2 == 0)
            has_prog = cat in ("forward_progression",) or has_entry
            has_followup = cat != "no_observed_followup"
            n_passes = 0 if not has_followup else (1 + (i % 4))
            fwd_m = 0.0
            if cat == "forward_progression":
                fwd_m = 14.5 + i
            elif cat == "final_third_entry":
                fwd_m = 22.0 + i
            elif cat == "shot":
                fwd_m = 28.0 + i
            elif cat == "other_completed_play":
                fwd_m = 4.0 + (i % 5)

            rows.append(
                {
                    "transition_id": tid,
                    "sequence_id": seq_id,
                    "team": team,
                    "half": 1 if (tid % 2 == 0) else 2,
                    "start_player_id": f"{team}_{(i % 10) + 2}",
                    "start_ms": 400_000 + tid * 5_000,
                    "end_ms": 400_000 + tid * 5_000 + (0 if not has_followup else 8_500),
                    "start_x_att": -22.0 + (i % 3) * 18.0,
                    "start_zone": zones[i % 3],
                    "stop_reason": "goal" if has_goal else stops[i % len(stops)],
                    "terminal_event_id": tid + 3,
                    "n_completed_passes": n_passes,
                    "has_followup": has_followup,
                    "shot_count": 1 if (has_shot and not has_goal) else (1 if has_shot else 0),
                    "has_shot": has_shot,
                    "goal_count": 1 if has_goal else 0,
                    "has_goal": has_goal,
                    "final_third_entries": 1 if has_entry else 0,
                    "has_entry": has_entry,
                    "forward_metres": fwd_m,
                    "has_progression": has_prog,
                    "outcome_category": cat,
                    "measurable_passes": max(n_passes - (1 if i % 7 == 0 and n_passes > 0 else 0), 0),
                    "missing_endpoint_passes": 1 if (i % 7 == 0 and n_passes > 0) else 0,
                }
            )
            tid += 5
            seq_id += 1

    return pd.DataFrame(rows)


def _extract_patch_heights(ax: matplotlib.axes.Axes) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    node_heights: dict[str, float] = {}
    flow_spans: dict[str, tuple[float, float]] = {}

    for patch_obj in ax.patches:
        gid = patch_obj.get_gid()
        if not gid:
            continue
        if gid.startswith("node:") and isinstance(patch_obj, Rectangle):
            node_heights[gid] = float(patch_obj.get_height())
        elif gid.startswith("flow:") and isinstance(patch_obj, PathPatch):
            verts = patch_obj.get_path().vertices
            xs = verts[:, 0]
            x_min, x_max = float(xs.min()), float(xs.max())
            left_ys = verts[np.isclose(xs, x_min, atol=1e-6), 1]
            right_ys = verts[np.isclose(xs, x_max, atol=1e-6), 1]
            left_span = float(left_ys.max() - left_ys.min())
            right_span = float(right_ys.max() - right_ys.min())
            flow_spans[gid] = (left_span, right_span)

    return node_heights, flow_spans


def _assert_no_text_overlaps(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    bboxes = []
    for txt in ax.texts:
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        bb = txt.get_window_extent(renderer=renderer)
        if bb.width > 0 and bb.height > 0:
            assert bb.x0 >= fig_bbox.x0 - 2.0 and bb.x1 <= fig_bbox.x1 + 2.0, (
                f"Text {txt.get_text()!r} clipped horizontally ({bb.x0:.1f}..{bb.x1:.1f} vs 0..{fig_bbox.x1:.1f})"
            )
            assert bb.y0 >= fig_bbox.y0 - 2.0 and bb.y1 <= fig_bbox.y1 + 2.0, (
                f"Text {txt.get_text()!r} clipped vertically"
            )
            bboxes.append((txt.get_text(), bb))

    for i in range(len(bboxes)):
        t1, b1 = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            t2, b2 = bboxes[j]
            # Shrink by 1.5 px tolerance for line-height padding
            ix0 = max(b1.x0, b2.x0) + 1.5
            iy0 = max(b1.y0, b2.y0) + 1.5
            ix1 = min(b1.x1, b2.x1) - 1.5
            iy1 = min(b1.y1, b2.y1) - 1.5
            overlaps = (ix1 > ix0) and (iy1 > iy0)
            assert not overlaps, f"Text overlap detected between {t1!r} and {t2!r}"


def check_alluvial_flow() -> None:
    # 1. Baseline 138-transition conservation check across both teams and all 5 categories
    # Mirrors real 15s structure: 3 home shots (1 also had an entry -> 9 exclusive entries, 10 total),
    # plus 1 unattributed start_player_id (like TID=754).
    counts_138 = {
        ("home", "shot"): 3,
        ("away", "shot"): 0,
        ("home", "final_third_entry"): 4,
        ("away", "final_third_entry"): 5,
        ("home", "forward_progression"): 12,
        ("away", "forward_progression"): 11,
        ("home", "other_completed_play"): 15,
        ("away", "other_completed_play"): 13,
        ("home", "no_observed_followup"): 37,
        ("away", "no_observed_followup"): 38,
    }
    metrics = _make_synthetic_metrics(counts_138)
    # Ensure exact 10 total entry passages (9 exclusive + 1 shot passage that also entered)
    metrics["has_entry"] = metrics["outcome_category"] == "final_third_entry"
    shot_idx = metrics.index[metrics["outcome_category"] == "shot"][0]
    metrics.loc[shot_idx, "has_entry"] = True
    metrics.loc[shot_idx, "final_third_entries"] = 1
    # Include one unknown start_player_id (mirroring TID=754)
    away_idx = metrics.index[metrics["team"] == "away"][0]
    metrics.loc[away_idx, "start_player_id"] = pd.NA
    assert len(metrics) == 138

    fig = tv.plot_transition_flow(metrics)
    assert isinstance(fig, matplotlib.figure.Figure)
    _render_png_bytes(fig)
    ax = fig.axes[0]
    _assert_no_text_overlaps(fig, ax)

    node_heights, flow_spans = _extract_patch_heights(ax)
    assert np.isclose(node_heights["node:root"], 138.0), f"Root height {node_heights}"
    assert np.isclose(node_heights["node:team:home"] + node_heights["node:team:away"], 138.0)
    assert np.isclose(
        sum(node_heights[f"node:outcome:{c}"] for c in OUTCOME_CATEGORIES),
        138.0,
    )

    # Check Stage 1 and Stage 2 ribbon conservation and exact counts
    stage1_total = 0.0
    for team, expected_team_n in (("home", 71), ("away", 67)):
        l_span, r_span = flow_spans[f"flow:root->{team}"]
        assert np.isclose(l_span, expected_team_n)
        assert np.isclose(r_span, expected_team_n)
        stage1_total += l_span
    assert np.isclose(stage1_total, 138.0)

    stage2_total = 0.0
    for (team, cat), expected_n in counts_138.items():
        if expected_n == 0:
            assert f"flow:{team}->{cat}" not in flow_spans
            continue
        l_span, r_span = flow_spans[f"flow:{team}->{cat}"]
        assert np.isclose(l_span, expected_n), f"Mismatch for {team}->{cat}: {l_span} != {expected_n}"
        assert np.isclose(r_span, expected_n), f"Right span mismatch for {team}->{cat}"
        stage2_total += l_span
    assert np.isclose(stage2_total, 138.0)

    all_text = "\n".join(t.get_text() for t in ax.texts) + "\n" + ax.get_title()
    assert "138" in all_text
    assert "sequence-opening" in all_text.lower()
    assert "no observed follow-up" in all_text.lower()
    assert "failed recover" not in all_text.lower()
    assert "chance" not in all_text.lower()
    assert "10 total" in all_text.lower() or "+1" in all_text.lower()
    for keyword in ("shot", "territor", "10 m"):
        assert keyword.lower() in all_text.lower(), f"Expected clarifying label {keyword!r} in diagram"
    plt.close(fig)

    # 2. Empty outcome categories and zero team->outcome flows (including adjacent zeros)
    sparse_counts = {
        ("home", "shot"): 0,
        ("away", "shot"): 0,
        ("home", "final_third_entry"): 0,
        ("away", "final_third_entry"): 0,
        ("home", "forward_progression"): 5,
        ("away", "forward_progression"): 0,
        ("home", "other_completed_play"): 4,
        ("away", "other_completed_play"): 3,
        ("home", "no_observed_followup"): 6,
        ("away", "no_observed_followup"): 2,
    }
    sparse_metrics = _make_synthetic_metrics(sparse_counts)
    assert len(sparse_metrics) == 20
    fig_sparse = tv.plot_transition_flow(sparse_metrics)
    _render_png_bytes(fig_sparse)
    ax_sparse = fig_sparse.axes[0]
    _assert_no_text_overlaps(fig_sparse, ax_sparse)
    sparse_nodes, sparse_flows = _extract_patch_heights(ax_sparse)

    assert np.isclose(sparse_nodes.get("node:outcome:shot", 0.0), 0.0)
    assert np.isclose(sparse_nodes.get("node:outcome:final_third_entry", 0.0), 0.0)
    assert "flow:home->shot" not in sparse_flows
    assert "flow:away->forward_progression" not in sparse_flows
    assert np.isclose(sum(v[0] for k, v in sparse_flows.items() if k.startswith("flow:root->")), 20.0)
    assert np.isclose(
        sum(v[0] for k, v in sparse_flows.items() if not k.startswith("flow:root->")),
        20.0,
    )
    plt.close(fig_sparse)

    # 3. Single-team active (other canonical team has zero passages)
    one_team = _make_synthetic_metrics({("home", "forward_progression"): 7, ("home", "no_observed_followup"): 5})
    fig_one = tv.plot_transition_flow(one_team)
    _render_png_bytes(fig_one)
    ax_one = fig_one.axes[0]
    _assert_no_text_overlaps(fig_one, ax_one)
    one_nodes, one_flows = _extract_patch_heights(ax_one)
    assert np.isclose(one_nodes.get("node:team:away", 0.0), 0.0)
    assert "flow:root->away" not in one_flows
    assert np.isclose(one_flows["flow:root->home"][0], 12.0)
    plt.close(fig_one)

    # 4. Guardrails against double-counting / invalid categories / empty input
    dup_metrics = pd.concat([sparse_metrics, sparse_metrics.iloc[[0]]], ignore_index=True)
    try:
        tv.plot_transition_flow(dup_metrics)
        raise AssertionError("Expected ValueError on duplicate transition_id")
    except ValueError:
        pass

    bad_cat = sparse_metrics.copy()
    bad_cat.loc[0, "outcome_category"] = "invented_bucket"
    try:
        tv.plot_transition_flow(bad_cat)
        raise AssertionError("Expected ValueError on unknown outcome_category")
    except ValueError:
        pass

    try:
        tv.plot_transition_flow(sparse_metrics.iloc[0:0])
        raise AssertionError("Expected ValueError on empty metrics DataFrame")
    except ValueError:
        pass


def _make_synthetic_ranking(n_players: int = 33, include_optional: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(n_players):
        team = "away" if idx < 17 else "home"
        shirt = idx + 1 if team == "away" else (idx - 16)
        pid = f"{team}_{shirt}"
        is_gk = shirt == 1
        zero_player = idx >= 27
        rec_starts = 3 if is_gk else (0 if zero_player else int((33 - idx) % 9))
        rec_follow = 0 if is_gk else min(rec_starts, max(0, rec_starts - 2))
        rec_entry = min(rec_follow, 1 if idx in (1, 3, 18) else 0)
        rec_shot = 1 if idx in (1, 19) and rec_starts > 0 else 0
        rec_out = min(rec_follow, rec_entry + rec_shot)

        out_pass = 2 if is_gk else (0 if zero_player else int((idx * 3 + 2) % 7))
        out_entry = 1 if idx in (2, 3, 20) and out_pass > 0 else 0
        out_shot = 1 if idx == 2 and out_pass > 1 else 0
        out_out = min(out_pass, out_entry + out_shot)

        comp_passes = 3 if is_gk else (0 if zero_player else max(out_pass, int((30 - idx) // 3)))
        meas_passes = max(0, comp_passes - (1 if idx % 5 == 0 and comp_passes > 0 and not is_gk else 0))
        # Give away_1 (GK) high positive metres (58.4m), away_12 high outfield metres (106.0m),
        # home_7 negative net metres (-8.6m), & home_10 low pass metres (4.4m) + 4 entries_received
        if pid == "away_1":
            fwd_m = 58.4
            net_m = 58.4
            ent_del, ent_rec = 0, 0
        elif pid == "away_12":
            comp_passes, meas_passes = 12, 12
            fwd_m = 106.0
            net_m = 80.2
            ent_del, ent_rec = 0, 1
        elif pid == "home_7":
            fwd_m = 55.0
            net_m = -8.6  # net-backward recycler
            ent_del, ent_rec = 1, 1
        elif pid == "home_10":
            fwd_m = 4.4
            net_m = -1.8
            ent_del, ent_rec = 0, 4  # primary final-third target
        else:
            fwd_m = 0.0 if meas_passes == 0 else round(float((33 - idx) * 2.4), 1)
            net_m = round(fwd_m - (6.0 if idx % 4 == 0 and meas_passes > 0 else 0.0), 1)
            ent_del = 0 if (zero_player or is_gk) else (2 if idx in (3, 18) else (1 if idx < 7 else 0))
            ent_rec = 0 if (zero_player or is_gk) else (2 if idx in (2, 19) else (1 if idx < 5 else 0))

        prog_passes = 0 if meas_passes == 0 else min(meas_passes, max(ent_del, 1 if fwd_m >= 15 else 0))
        row_dict: dict[str, object] = {
            "player_id": pid,
            "team": team,
            "minutes_played": 96.0 if is_gk else (6.5 if zero_player else round(96.0 - (idx % 7) * 8.2, 1)),
            "recovery_starts": rec_starts,
            "recovery_followups": rec_follow,
            "recovery_entry_passages": rec_entry,
            "recovery_shot_passages": rec_shot,
            "recovery_outcome_passages": rec_out,
            "outlet_passages": out_pass,
            "outlet_entry_passages": out_entry,
            "outlet_shot_passages": out_shot,
            "outlet_outcome_passages": out_out,
            "completed_transition_passes": comp_passes,
            "measurable_transition_passes": meas_passes,
            "forward_metres": fwd_m,
            "progressive_passes": prog_passes,
            "progression_passages": prog_passes,
            "entries_delivered": ent_del,
            "entries_received": ent_rec,
            "entry_passages": min(ent_del + ent_rec, comp_passes + ent_rec),
            "shot_passages": 2 if pid == "home_10" else (1 if idx in (1, 2, 19) and not zero_player else 0),
            # Deliberately put GK away_1 and backward recycler home_7 in Pareto Tier 1 to ensure plot ignores Pareto
            "pareto_tier": 1 if pid in ("away_1", "home_7", "away_2") else (2 if idx < 12 else 3),
            "progression_rank": idx + 1,
        }
        if include_optional:
            row_dict["role"] = "goalkeeper" if is_gk else "outfield"
            row_dict["net_pass_metres"] = net_m
            row_dict["failed_passes_at_stop"] = 3 if is_gk else (4 if pid == "home_2" else (idx % 3))
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["forward_metres", "entries_delivered", "player_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    df["progression_rank"] = np.arange(1, len(df) + 1)
    return df


def _collect_figure_text(fig: matplotlib.figure.Figure) -> str:
    parts: list[str] = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        parts.extend([ax.get_title(), ax.get_xlabel(), ax.get_ylabel()])
        parts.extend(t.get_text() for t in ax.texts)
        parts.extend(lbl.get_text() for lbl in ax.get_xticklabels())
        parts.extend(lbl.get_text() for lbl in ax.get_yticklabels())
        leg = ax.get_legend()
        if leg is not None:
            parts.extend(t.get_text() for t in leg.get_texts())
    return "\n".join(parts)


def _assert_axes_texts_inside_spines(fig: matplotlib.figure.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    for ftxt in fig.texts:
        if not ftxt.get_visible() or not ftxt.get_text().strip():
            continue
        fbb = ftxt.get_window_extent(renderer=renderer)
        assert fbb.x0 >= fig_bbox.x0 + 18.0 and fbb.x1 <= fig_bbox.x1 - 18.0, (
            f"Figure text {ftxt.get_text()!r} runs too close to canvas edge ({fbb.x0:.1f}..{fbb.x1:.1f} vs {fig_bbox.x1:.1f})"
        )
    for ax in fig.axes:
        ax_bbox = ax.get_window_extent(renderer=renderer)
        for txt in ax.texts:
            if not txt.get_visible() or not txt.get_text().strip():
                continue
            bb = txt.get_window_extent(renderer=renderer)
            if bb.width > 0 and bb.height > 0:
                assert bb.x1 <= ax_bbox.x1 - 2.0, (
                    f"Bar label {txt.get_text()!r} overflows subplot right spine ({bb.x1:.1f} > {ax_bbox.x1:.1f})"
                )


def check_player_evidence() -> None:
    ranking = _make_synthetic_ranking(33, include_optional=True)

    # 1. Default view (player_ids=None): must compare outfielders only (no GKs in bars),
    # contextualise goalkeepers separately, include high-entries_received target (home_10),
    # and contain ZERO Pareto references/rings/tier labels.
    fig1 = tv.plot_player_evidence(ranking)
    assert isinstance(fig1, matplotlib.figure.Figure)
    assert len(fig1.axes) >= 3, "Expected separate multi-panel evidence display"
    _render_png_bytes(fig1)
    _assert_axes_texts_inside_spines(fig1)

    text_default = _collect_figure_text(fig1)
    assert "pareto" not in text_default.lower(), "Must not display Pareto claims or labels"
    assert "mvp" not in text_default.lower(), "Must not display MVP score/claims"
    for tier_tag in (", t1)", ", t2)", ", t3)", "tier 1", "tier 2"):
        assert tier_tag not in text_default.lower(), f"Forbidden tier tag {tier_tag!r} found"

    # Check y-tick comparison bars exclude goalkeepers when role is provided, while GK context note exists
    for ax in fig1.axes:
        ytick_str = " ".join(lbl.get_text() for lbl in ax.get_yticklabels())
        assert "away_1 (" not in ytick_str and "home_1 (" not in ytick_str, (
            f"Goalkeeper leaked into default outfield comparison bars: {ytick_str}"
        )
    assert "goalkeeper" in text_default.lower() and "away_1" in text_default, (
        "Goalkeepers must be contextualised separately when role column is provided"
    )
    first_panel_yticks = " ".join(lbl.get_text() for lbl in fig1.axes[0].get_yticklabels())
    assert "home_10" in first_panel_yticks, "Target leader home_10 (4 entries_received) should appear in default view"
    assert "received" in text_default.lower() and "target" in text_default.lower(), (
        "entries_received must be visibly labelled to represent targets"
    )
    plt.close(fig1)

    # 2. Explicit player_ids selection: must use EXACTLY those IDs in EXACT requested order,
    # with NO automatic extra cohort rows and NO goalkeeper padding; show net metres & failed passes.
    chosen = ["away_12", "home_10", "home_7", "away_4", "home_15"]
    fig2 = tv.plot_player_evidence(ranking, player_ids=chosen)
    _render_png_bytes(fig2)
    _assert_axes_texts_inside_spines(fig2)

    for ax in fig2.axes:
        tick_texts = [lbl.get_text() for lbl in ax.get_yticklabels() if lbl.get_text().strip()]
        assert len(tick_texts) == len(chosen), (
            f"Expected exact cohort of {len(chosen)} rows, got {len(tick_texts)}: {tick_texts}"
        )
        extracted_pids = [t.split()[0] for t in tick_texts]
        assert extracted_pids == chosen, (
            f"Cohort order {extracted_pids} does not preserve requested order {chosen} (no Pareto re-sort allowed)"
        )

    text_chosen = _collect_figure_text(fig2)
    assert "pareto" not in text_chosen.lower()
    assert "-8.6" in text_chosen, "Expected negative net_pass_metres (-8.6 m) for home_7 to expose backward recycling"
    assert "failed" in text_chosen.lower(), "Expected failed-pass context when column is present"
    assert "min" in text_chosen.lower(), "Expected minutes played context in evidence plot"
    assert "/" in text_chosen, "Expected explicit numerator/denominator sample sizes"
    plt.close(fig2)

    # 3. Also support alias optional column names (net_forward_metres & failed_transition_passes)
    alias_ranking = ranking.rename(
        columns={
            "net_pass_metres": "net_forward_metres",
            "failed_passes_at_stop": "failed_transition_passes",
        }
    )
    fig_alias = tv.plot_player_evidence(alias_ranking, player_ids=["home_7", "away_2"])
    _render_png_bytes(fig_alias)
    _assert_axes_texts_inside_spines(fig_alias)
    alias_text = _collect_figure_text(fig_alias)
    assert "-8.6" in alias_text and "failed" in alias_text.lower()
    plt.close(fig_alias)

    # 4. Minimum contract columns only (no role / net / failed optional columns) & all-zero table
    min_ranking = _make_synthetic_ranking(33, include_optional=False)
    fig_min = tv.plot_player_evidence(min_ranking, player_ids=["away_12", "home_10", "away_2"])
    _render_png_bytes(fig_min)
    _assert_axes_texts_inside_spines(fig_min)
    assert "net metres" not in fig_min.axes[0].get_title().lower(), (
        "Panel A title should not advertise Net Metres when net column is absent"
    )
    plt.close(fig_min)

    zero_ranking = min_ranking.copy()
    numeric_cols = [
        c
        for c in zero_ranking.columns
        if c not in ("player_id", "team", "minutes_played", "pareto_tier", "progression_rank")
    ]
    for col in numeric_cols:
        zero_ranking[col] = 0
    fig_zero = tv.plot_player_evidence(zero_ranking)
    _render_png_bytes(fig_zero)
    plt.close(fig_zero)

    # Unknown player_id or empty player_ids raises ValueError
    for bad_ids in (["unknown_99"], []):
        try:
            tv.plot_player_evidence(ranking, player_ids=bad_ids)
            raise AssertionError(f"Expected ValueError for player_ids={bad_ids}")
        except ValueError:
            pass


def check_nearest_frame_and_pitch() -> None:
    players = pd.DataFrame(
        [
            # Half 1 frame at 1000 ms
            {"timestamp_ms": 1000, "half": 1, "team": "away", "role": "outfield", "player_id": "away_4", "attack_dir": 1, "x": -10.0, "y": 5.0, "ball_x": -9.5, "ball_y": 4.8},
            {"timestamp_ms": 1000, "half": 1, "team": "home", "role": "outfield", "player_id": "home_9", "attack_dir": -1, "x": -6.0, "y": 3.0, "ball_x": -9.5, "ball_y": 4.8},
            # Half 1 frame at 1040 ms
            {"timestamp_ms": 1040, "half": 1, "team": "away", "role": "outfield", "player_id": "away_4", "attack_dir": 1, "x": -8.0, "y": 5.2, "ball_x": -7.5, "ball_y": 5.0},
            {"timestamp_ms": 1040, "half": 1, "team": "home", "role": "outfield", "player_id": "home_9", "attack_dir": -1, "x": -5.5, "y": 3.2, "ball_x": -7.5, "ball_y": 5.0},
            # Half 2 frame at 3_600_100 ms with missing ball_x/ball_y and referee
            {"timestamp_ms": 3_600_100, "half": 2, "team": "away", "role": "outfield", "player_id": "away_13", "attack_dir": -1, "x": 18.0, "y": -12.0, "ball_x": np.nan, "ball_y": np.nan},
            {"timestamp_ms": 3_600_100, "half": 2, "team": "away", "role": "outfield", "player_id": "away_4", "attack_dir": -1, "x": 5.0, "y": -4.0, "ball_x": np.nan, "ball_y": np.nan},
            {"timestamp_ms": 3_600_100, "half": 2, "team": "home", "role": "goalkeeper", "player_id": "home_1", "attack_dir": 1, "x": -46.0, "y": 0.5, "ball_x": np.nan, "ball_y": np.nan},
            {"timestamp_ms": 3_600_100, "half": 2, "team": "referee", "role": "referee", "player_id": pd.NA, "attack_dir": 0, "x": 0.0, "y": 8.0, "ball_x": np.nan, "ball_y": np.nan},
        ]
    )

    # Exact frame
    f_exact, gap_exact = tv.nearest_frame(players, 1000, half=1)
    assert gap_exact == 0 and isinstance(gap_exact, int)
    assert set(f_exact["timestamp_ms"].unique()) == {1000}
    assert len(f_exact) == 2

    # Between frames -> closer to 1040
    f_near, gap_near = tv.nearest_frame(players, 1030, half=1)
    assert gap_near == 10
    assert set(f_near["timestamp_ms"].unique()) == {1040}

    # Exact tie (1020 is 20 ms from both 1000 and 1040) -> earlier frame (1000)
    f_tie, gap_tie = tv.nearest_frame(players, 1020, half=1)
    assert gap_tie == 20
    assert set(f_tie["timestamp_ms"].unique()) == {1000}, "Exact tie must choose earlier frame"

    # Missing / wrong half -> ValueError, never substitute half 1 when half 2 is absent
    only_h1 = players[players["half"] == 1].copy()
    try:
        tv.nearest_frame(only_h1, 1000, half=2)
        raise AssertionError("Expected ValueError when requested half is missing")
    except ValueError:
        pass

    try:
        tv.nearest_frame(players, 1000, half=3)
        raise AssertionError("Expected ValueError for invalid half=3")
    except ValueError:
        pass

    try:
        tv.nearest_frame(players.iloc[0:0], 1000, half=1)
        raise AssertionError("Expected ValueError for empty players DataFrame")
    except ValueError:
        pass

    # Pitch plot in Half 2 (attack_dir == -1, away attacking -x in raw coords -> +x normalised)
    # Includes a pass with missing receiver_x/receiver_y and a coordinate-less goal row.
    transition_events = pd.DataFrame(
        [
            {
                "transition_id": 801,
                "event_id": 801,
                "timestamp_ms": 3_600_020,
                "minute": 60.00,
                "half": 2,
                "event_name": "interception",
                "team": "away",
                "role": "outfield",
                "player_id": "away_13",
                "attack_dir": -1,
                "x_start": 20.0,
                "y_start": -14.0,
                "x_end": np.nan,
                "y_end": np.nan,
                "pass_success": pd.NA,
                "receiver_player_id": pd.NA,
                "receiver_x": np.nan,
                "receiver_y": np.nan,
                "shot_distance_m": np.nan,
                "is_goal": pd.NA,
                "sequence_id": 250,
            },
            {
                "transition_id": 801,
                "event_id": 802,
                "timestamp_ms": 3_602_500,
                "minute": 60.04,
                "half": 2,
                "event_name": "pass",
                "team": "away",
                "role": "outfield",
                "player_id": "away_13",
                "attack_dir": -1,
                "x_start": 18.0,
                "y_start": -12.0,
                "x_end": np.nan,
                "y_end": np.nan,
                "pass_success": True,
                "receiver_player_id": "away_4",
                "receiver_x": -22.0,
                "receiver_y": -5.0,
                "shot_distance_m": np.nan,
                "is_goal": pd.NA,
                "sequence_id": 250,
            },
            {
                "transition_id": 801,
                "event_id": 803,
                "timestamp_ms": 3_605_000,
                "minute": 60.08,
                "half": 2,
                "event_name": "pass",
                "team": "away",
                "role": "outfield",
                "player_id": "away_4",
                "attack_dir": -1,
                "x_start": -22.0,
                "y_start": -5.0,
                "x_end": np.nan,
                "y_end": np.nan,
                "pass_success": True,
                "receiver_player_id": pd.NA,
                "receiver_x": np.nan,
                "receiver_y": np.nan,
                "shot_distance_m": np.nan,
                "is_goal": pd.NA,
                "sequence_id": 250,
            },
            {
                "transition_id": 801,
                "event_id": 804,
                "timestamp_ms": 3_608_100,
                "minute": 60.135,
                "half": 2,
                "event_name": "shot",
                "team": "away",
                "role": "outfield",
                "player_id": "away_9",
                "attack_dir": -1,
                "x_start": -39.0,
                "y_start": 3.0,
                "x_end": np.nan,
                "y_end": np.nan,
                "pass_success": pd.NA,
                "receiver_player_id": pd.NA,
                "receiver_x": np.nan,
                "receiver_y": np.nan,
                "shot_distance_m": 13.8,
                "is_goal": True,
                "sequence_id": 250,
            },
            {
                "transition_id": 801,
                "event_id": 805,
                "timestamp_ms": 3_608_800,
                "minute": 60.147,
                "half": 2,
                "event_name": "goal",
                "team": "away",
                "role": "outfield",
                "player_id": "away_9",
                "attack_dir": -1,
                "x_start": np.nan,
                "y_start": np.nan,
                "x_end": np.nan,
                "y_end": np.nan,
                "pass_success": pd.NA,
                "receiver_player_id": pd.NA,
                "receiver_x": np.nan,
                "receiver_y": np.nan,
                "shot_distance_m": np.nan,
                "is_goal": pd.NA,
                "sequence_id": 250,
            },
        ]
    )

    fig_pitch = tv.plot_transition_pitch(transition_events, players=players, focus_player="away_4")
    _render_png_bytes(fig_pitch)
    ax_p = fig_pitch.axes[0]
    pitch_text = "\n".join(
        [ax_p.get_title(), ax_p.get_xlabel(), ax_p.get_ylabel()]
        + [t.get_text() for t in ax_p.texts]
        + [t.get_text() for t in fig_pitch.texts]
    )
    assert "(m" in pitch_text.lower() or "metres" in pitch_text.lower(), "Pitch units (m) must be labelled"
    assert "video" in pitch_text.lower(), "Video clock must be explicitly labelled"
    assert "ball" in pitch_text.lower() and "missing" in pitch_text.lower(), "Missing ball coords must be reported"
    assert "missing" in pitch_text.lower() or "unmeasured" in pitch_text.lower()
    plt.close(fig_pitch)

    # Event-only fallback when players is None or missing requested half
    fig_ev_only = tv.plot_transition_pitch(transition_events, players=None)
    _render_png_bytes(fig_ev_only)
    plt.close(fig_ev_only)

    fig_wrong_half = tv.plot_transition_pitch(transition_events, players=only_h1)
    _render_png_bytes(fig_wrong_half)
    plt.close(fig_wrong_half)

    # Reject mixed transition_ids
    mixed = transition_events.copy()
    mixed.loc[mixed.index[-1], "transition_id"] = 999
    try:
        tv.plot_transition_pitch(mixed, players=None)
        raise AssertionError("Expected ValueError when multiple transition_ids are passed")
    except ValueError:
        pass


def _create_lavfi_mp4(path: Path, duration_s: float = 2.6, width: int = 320, height: int = 180, fps: int = 15) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=22050",
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def check_video_probe_and_export() -> None:
    repo_root = Path(__file__).resolve().parent
    real_mp4 = repo_root / "match_01_full.mp4"
    real_stat_before = real_mp4.stat() if real_mp4.exists() else None

    if real_mp4.exists():
        real_info = tv.probe_video(real_mp4)
        assert real_info["width"] == 1280
        assert real_info["height"] == 720
        assert abs(real_info["duration_s"] - 6622.616) < 1.0
        assert 29.0 < real_info["fps"] < 30.5

        # Source protection on real match video (must refuse before touching file)
        try:
            tv.export_transition_clip(real_mp4, real_mp4, 10.0, 12.0)
            raise AssertionError("Expected ValueError when output_path matches source video")
        except ValueError:
            pass

        real_stat_after = real_mp4.stat()
        assert real_stat_after.st_size == real_stat_before.st_size
        assert real_stat_after.st_mtime_ns == real_stat_before.st_mtime_ns

    with tempfile.TemporaryDirectory(prefix="veo_vis_check_") as tmp:
        tmpdir = Path(tmp)
        src_mp4 = tmpdir / "synthetic_source.mp4"
        _create_lavfi_mp4(src_mp4, duration_s=2.6, width=320, height=180, fps=15)
        src_stat_initial = src_mp4.stat()

        info = tv.probe_video(src_mp4)
        assert info["width"] == 320 and info["height"] == 180
        assert info["has_audio"] is True
        assert abs(info["duration_s"] - 2.6) < 0.15

        # Missing & corrupt file checks
        try:
            tv.probe_video(tmpdir / "missing_video.mp4")
            raise AssertionError("Expected FileNotFoundError for missing video")
        except FileNotFoundError:
            pass

        corrupt = tmpdir / "corrupt.mp4"
        corrupt.write_bytes(b"not-a-valid-video-stream")
        try:
            tv.probe_video(corrupt)
            raise AssertionError("Expected RuntimeError for corrupt video")
        except RuntimeError:
            pass

        try:
            tv.export_transition_clip(corrupt, tmpdir / "never_created.mp4", 0.0, 1.0)
            raise AssertionError("Expected RuntimeError when exporting from corrupt source")
        except RuntimeError:
            assert not (tmpdir / "never_created.mp4").exists()

        # Invalid / empty intervals
        for bad_start, bad_end in ((1.2, 1.2), (1.8, 0.5), (-3.0, -0.2), (5.0, 8.0), (float("nan"), 1.0)):
            try:
                tv.export_transition_clip(src_mp4, tmpdir / "bad.mp4", bad_start, bad_end)
                raise AssertionError(f"Expected ValueError for invalid interval ({bad_start}, {bad_end})")
            except ValueError:
                assert not (tmpdir / "bad.mp4").exists()

        # Source protection: direct overwrite AND unrelated pre-existing file without owned sidecar
        try:
            tv.export_transition_clip(src_mp4, src_mp4, 0.2, 1.4)
            raise AssertionError("Expected ValueError when overwriting source video")
        except ValueError:
            pass

        unrelated_existing = tmpdir / "unrelated_existing.mp4"
        unrelated_bytes = b"pre-existing-unrelated-asset"
        unrelated_existing.write_bytes(unrelated_bytes)
        try:
            tv.export_transition_clip(src_mp4, unrelated_existing, 0.2, 1.4)
            raise AssertionError("Expected ValueError when overwriting unowned existing file")
        except ValueError:
            assert unrelated_existing.read_bytes() == unrelated_bytes

        # Simulated ffmpeg subprocess failure after successful ffprobe
        real_run = subprocess.run

        def _failing_ffmpeg(cmd: list[str], *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            assert kwargs.get("shell", False) is False, "subprocess must never use shell=True"
            if Path(cmd[0]).name == "ffmpeg":
                return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="simulated encode error")
            return real_run(cmd, *args, **kwargs)  # type: ignore[arg-type]

        with patch("transition_visuals.subprocess.run", side_effect=_failing_ffmpeg):
            try:
                tv.export_transition_clip(src_mp4, tmpdir / "failed_out.mp4", 0.2, 1.2, make_gif=True)
                raise AssertionError("Expected RuntimeError when ffmpeg fails")
            except RuntimeError as exc:
                assert "simulated encode error" in str(exc)
                assert not (tmpdir / "failed_out.mp4").exists()

        # Relative filename, boundary clamping, muted audio, GIF realtime/no-upscale, and sidecar cache
        prev_cwd = Path.cwd()
        try:
            os.chdir(tmpdir)
            res1 = tv.export_transition_clip(
                "synthetic_source.mp4",
                "rel_clip.mp4",
                start_s=-0.9,
                end_s=1.5,
                make_gif=True,
            )
            assert res1["cached"] is False
            assert res1["clamped"] is True
            assert np.isclose(res1["start_s"], 0.0)
            assert np.isclose(res1["end_s"], 1.5)
            assert np.isclose(res1["duration_s"], 1.5)
            assert Path(res1["mp4_path"]).exists()
            assert res1["gif_path"] is not None and Path(res1["gif_path"]).exists()

            mp4_probe = tv.probe_video(res1["mp4_path"])
            assert mp4_probe["has_audio"] is False, "Exported MP4 must have audio muted"
            assert abs(mp4_probe["duration_s"] - 1.5) < 0.20

            gif_probe = tv.probe_video(res1["gif_path"])
            assert gif_probe["codec_name"] == "gif"
            assert abs(gif_probe["fps"] - 10.0) < 0.5
            assert gif_probe["width"] == 320 and gif_probe["height"] == 180, "Tiny test source must not be upscaled"
            assert abs(gif_probe["duration_s"] - 1.5) < 0.25, "GIF must preserve real-time duration"

            # Ensure no dumped frame sequences exist
            dumped_frames = list(tmpdir.rglob("*.png")) + list(tmpdir.rglob("*.jpg"))
            assert not dumped_frames, f"Unexpected dumped frames found: {dumped_frames}"

            # Cache hit with identical source, clamped times, and settings
            mp4_mtime_1 = Path(res1["mp4_path"]).stat().st_mtime_ns
            res_cached = tv.export_transition_clip(
                "synthetic_source.mp4",
                "rel_clip.mp4",
                start_s=-0.9,
                end_s=1.5,
                make_gif=True,
            )
            assert res_cached["cached"] is True
            assert Path(res1["mp4_path"]).stat().st_mtime_ns == mp4_mtime_1

            # Cache miss & regeneration when end-clamped interval changes on owned output
            res_regen = tv.export_transition_clip(
                "synthetic_source.mp4",
                "rel_clip.mp4",
                start_s=1.2,
                end_s=9.0,
                make_gif=True,
            )
            assert res_regen["cached"] is False
            assert res_regen["clamped"] is True
            assert np.isclose(res_regen["start_s"], 1.2)
            assert abs(res_regen["duration_s"] - (info["duration_s"] - 1.2)) < 0.05
            regen_probe = tv.probe_video(res_regen["mp4_path"])
            assert abs(regen_probe["duration_s"] - res_regen["duration_s"]) < 0.20

            # Cache invalidation when sidecar metadata does not match settings
            sidecar = Path(res_regen["sidecar_path"])
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            meta["settings"]["gif_fps"] = 5
            sidecar.write_text(json.dumps(meta), encoding="utf-8")
            res_invalidated = tv.export_transition_clip(
                "synthetic_source.mp4",
                "rel_clip.mp4",
                start_s=1.2,
                end_s=9.0,
                make_gif=True,
            )
            assert res_invalidated["cached"] is False
        finally:
            os.chdir(prev_cwd)

        # Verify HD source downscales GIF to 854 width preserving 16:9 aspect ratio (854x480)
        hd_src = tmpdir / "hd_source.mp4"
        _create_lavfi_mp4(hd_src, duration_s=1.2, width=1280, height=720, fps=30)
        hd_res = tv.export_transition_clip(hd_src, tmpdir / "clips" / "hd_clip.mp4", 0.1, 1.1, make_gif=True)
        hd_gif_probe = tv.probe_video(hd_res["gif_path"])
        assert hd_gif_probe["width"] == 854
        assert hd_gif_probe["height"] in (480, 481)
        assert abs(hd_gif_probe["duration_s"] - 1.0) < 0.20

        # Confirm synthetic source was never modified
        src_stat_final = src_mp4.stat()
        assert src_stat_final.st_size == src_stat_initial.st_size
        assert src_stat_final.st_mtime_ns == src_stat_initial.st_mtime_ns


def main() -> int:
    checks = [
        ("alluvial_flow", check_alluvial_flow),
        ("player_evidence", check_player_evidence),
        ("nearest_frame_and_pitch", check_nearest_frame_and_pitch),
        ("video_probe_and_export", check_video_probe_and_export),
    ]
    for name, fn in checks:
        fn()
        print(f"[PASS] {name}")
    print("All transition_visuals checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
