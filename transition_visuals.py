"""Visualisation and media helpers for the defence-to-attack transition study.

Implements the shared visual/media API defined in
``docs/superpowers/implementation/defence_to_attack_contract.md``:
- ``plot_transition_flow``
- ``plot_player_evidence``
- ``nearest_frame``
- ``plot_transition_pitch``
- ``probe_video``
- ``export_transition_clip``
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 67.7
FINAL_THIRD_X_ATT_M = 17.5

# Okabe-Ito colourblind-friendly palette
TEAM_COLOURS: dict[str, str] = {
    "home": "#0072B2",     # Blue
    "away": "#D55E00",     # Vermilion
    "referee": "#8C8C8C",  # Neutral grey
}
FALLBACK_TEAM_COLOURS = ("#CC79A7", "#009E73", "#E69F00", "#56B4E9")
ROOT_NODE_COLOUR = "#2C3E50"

OUTCOME_CATEGORIES: tuple[str, ...] = (
    "shot",
    "final_third_entry",
    "forward_progression",
    "other_completed_play",
    "no_observed_followup",
)

OUTCOME_LABELS: dict[str, str] = {
    "shot": "Shot outcome (shot / standalone goal)",
    "final_third_entry": "Final-third entry, no shot (territorial progress >17.5 m)",
    "forward_progression": "Forward progression, no entry/shot (≥10 m pass)",
    "other_completed_play": "Other completed play (<10 m / unmeasured pass)",
    "no_observed_followup": "No observed follow-up (no completed pass/shot before stop)",
}

OUTCOME_COLOURS: dict[str, str] = {
    "shot": "#009E73",                 # Bluish green
    "final_third_entry": "#56B4E9",    # Sky blue
    "forward_progression": "#E69F00",  # Amber / orange
    "other_completed_play": "#CC79A7", # Reddish purple
    "no_observed_followup": "#7F8C8D", # Slate grey
}

SIDECAR_GENERATOR_ID = "transition_visuals.export_transition_clip"
GIF_DEFAULT_FPS = 10
GIF_MAX_WIDTH_PX = 854

PROTECTED_REPO_FILENAMES = frozenset(
    {
        "match_01_full.mp4",
        "events.csv",
        "player_summary.csv",
        "players.parquet",
        "ball.csv",
    }
)


def _team_colour(team: str, idx: int = 0) -> str:
    if team in TEAM_COLOURS:
        return TEAM_COLOURS[team]
    return FALLBACK_TEAM_COLOURS[idx % len(FALLBACK_TEAM_COLOURS)]


def _format_video_time(timestamp_ms: int | float) -> str:
    total_s = float(timestamp_ms) / 1000.0
    minutes = int(total_s // 60)
    rem_s = total_s - 60 * minutes
    return f"{minutes:02d}:{rem_s:04.1f} ({total_s:.1f} s)"


def _make_sankey_ribbon(
    x_left: float,
    y_left_bottom: float,
    y_left_top: float,
    x_right: float,
    y_right_bottom: float,
    y_right_top: float,
    facecolor: str,
    alpha: float = 0.40,
    gid: str | None = None,
) -> PathPatch:
    x_mid = 0.5 * (x_left + x_right)
    vertices = [
        (x_left, y_left_top),
        (x_mid, y_left_top),
        (x_mid, y_right_top),
        (x_right, y_right_top),
        (x_right, y_right_bottom),
        (x_mid, y_right_bottom),
        (x_mid, y_left_bottom),
        (x_left, y_left_bottom),
        (x_left, y_left_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=facecolor,
        linewidth=0.6,
        alpha=alpha,
        zorder=2,
    )
    if gid is not None:
        patch.set_gid(gid)
    return patch


def plot_transition_flow(metrics: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot a 3-stage Sankey/alluvial diagram conserving all candidate passages.

    Stages: all candidate transitions -> recovering team -> exclusive outcome_category.
    Stream thickness strictly represents passage counts; empty categories and zero
    flows are preserved without visual inflation or double-counting.
    """
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("metrics must be a non-empty DataFrame of candidate transitions.")

    required_cols = {"team", "outcome_category"}
    missing_cols = required_cols - set(metrics.columns)
    if missing_cols:
        raise ValueError(f"metrics is missing required columns: {sorted(missing_cols)}")

    if metrics["team"].isna().any() or metrics["outcome_category"].isna().any():
        raise ValueError("metrics contains null values in 'team' or 'outcome_category'.")

    if "transition_id" in metrics.columns:
        if metrics["transition_id"].isna().any():
            raise ValueError("metrics contains null 'transition_id' values.")
        if metrics["transition_id"].duplicated().any():
            raise ValueError("Duplicate transition_id detected; refusing to double-count pathways.")

    observed_cats = set(metrics["outcome_category"].astype(str).unique())
    unknown_cats = observed_cats - set(OUTCOME_CATEGORIES)
    if unknown_cats:
        raise ValueError(
            f"Unknown outcome_category values {sorted(unknown_cats)}; "
            f"expected subset of {OUTCOME_CATEGORIES}."
        )

    total_starts = int(len(metrics))
    observed_teams = [str(t) for t in metrics["team"].unique()]
    if set(observed_teams).issubset({"home", "away"}):
        teams_order = ["home", "away"]
    else:
        teams_order = sorted(dict.fromkeys(observed_teams))

    team_counts: dict[str, int] = {
        t: int((metrics["team"].astype(str) == t).sum()) for t in teams_order
    }
    outcome_counts: dict[str, int] = {
        cat: int((metrics["outcome_category"].astype(str) == cat).sum())
        for cat in OUTCOME_CATEGORIES
    }
    flow_counts: dict[tuple[str, str], int] = {}
    for t in teams_order:
        t_mask = metrics["team"].astype(str) == t
        for cat in OUTCOME_CATEGORIES:
            flow_counts[(t, cat)] = int((t_mask & (metrics["outcome_category"].astype(str) == cat)).sum())

    # Strict flow conservation assertions
    if sum(team_counts.values()) != total_starts:
        raise ValueError("Team counts do not conserve total candidate transitions.")
    if sum(outcome_counts.values()) != total_starts:
        raise ValueError("Outcome counts do not conserve total candidate transitions.")
    if sum(flow_counts.values()) != total_starts:
        raise ValueError("Team-to-outcome flow counts do not conserve total candidate transitions.")

    # Vertical coordinate geometry in exact transition-count units (1 unit == 1 passage)
    outcome_gap = 0.18 * total_starts
    n_out_gaps = max(len(OUTCOME_CATEGORIES) - 1, 0)
    col2_total_span = float(total_starts) + n_out_gaps * outcome_gap
    y_center = 0.5 * col2_total_span

    team_gap = (0.26 * total_starts) if len(teams_order) > 1 else 0.0
    col1_total_span = float(total_starts) + max(len(teams_order) - 1, 0) * team_gap

    # Column x coordinates and bar width (generous right margin for two-line labels)
    x0, x1, x2 = 0.00, 0.35, 0.72
    bar_w = 0.040

    # Top-down placement so priority 1 ("shot") is at top and priority 5 at bottom
    root_top = y_center + 0.5 * total_starts
    root_bot = y_center - 0.5 * total_starts

    team_intervals: dict[str, tuple[float, float]] = {}
    cursor_team = y_center + 0.5 * col1_total_span
    for t in teams_order:
        h = float(team_counts[t])
        team_intervals[t] = (cursor_team - h, cursor_team)
        cursor_team = (cursor_team - h) - team_gap

    outcome_intervals: dict[str, tuple[float, float]] = {}
    cursor_out = col2_total_span
    for cat in OUTCOME_CATEGORIES:
        h = float(outcome_counts[cat])
        outcome_intervals[cat] = (cursor_out - h, cursor_out)
        cursor_out = (cursor_out - h) - outcome_gap

    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Stage 0: Root node ("Sequence-opening interventions")
    root_rect = Rectangle(
        (x0, root_bot),
        bar_w,
        float(total_starts),
        facecolor=ROOT_NODE_COLOUR,
        edgecolor="#1B2631",
        linewidth=1.2,
        zorder=4,
    )
    root_rect.set_gid("node:root")
    ax.add_patch(root_rect)
    unattributed_starts = (
        int(metrics["start_player_id"].isna().sum())
        if "start_player_id" in metrics.columns
        else 0
    )
    unattr_line = (
        f"\n(incl. {unattributed_starts} unattributed ID)"
        if unattributed_starts > 0
        else "\n(tackles & interceptions)"
    )
    ax.text(
        x0 - 0.022,
        0.5 * (root_bot + root_top),
        (
            f"Sequence-opening\ninterventions\n"
            f"n = {total_starts} (100.0%){unattr_line}"
        ),
        ha="right",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color="#1B2631",
    )

    # Stage 1: Root -> Team ribbons and Team nodes
    root_out_cursor = root_top
    for idx, t in enumerate(teams_order):
        t_count = team_counts[t]
        t_bot, t_top = team_intervals[t]
        t_col = _team_colour(t, idx)

        t_rect = Rectangle(
            (x1, t_bot),
            bar_w,
            float(t_count),
            facecolor=t_col,
            edgecolor="#1B2631" if t_count > 0 else t_col,
            linewidth=1.1 if t_count > 0 else 0.0,
            zorder=4,
        )
        t_rect.set_gid(f"node:team:{t}")
        ax.add_patch(t_rect)

        if t_count == 0:
            ax.plot(
                [x1 - 0.008, x1 + bar_w + 0.008],
                [t_bot, t_bot],
                color=t_col,
                linestyle="--",
                linewidth=1.4,
                zorder=4,
            )

        if t_count > 0:
            ribbon_s1 = _make_sankey_ribbon(
                x0 + bar_w,
                root_out_cursor - t_count,
                root_out_cursor,
                x1,
                t_bot,
                t_top,
                facecolor=t_col,
                alpha=0.38,
                gid=f"flow:root->{t}",
            )
            ax.add_patch(ribbon_s1)
            root_out_cursor -= t_count

        pct_team = 100.0 * t_count / total_starts
        ax.text(
            x1 + 0.5 * bar_w,
            0.5 * (t_bot + t_top),
            f"{t.upper()}\nn = {t_count} ({pct_team:.1f}%)",
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#1B2631",
            bbox={
                "boxstyle": "round,pad=0.26",
                "facecolor": "white",
                "edgecolor": t_col,
                "linewidth": 1.3,
                "alpha": 0.94,
            },
            zorder=6,
        )

    # Stage 2: Team -> Exclusive Outcome ribbons and Outcome nodes
    team_out_cursors: dict[str, float] = {t: team_intervals[t][1] for t in teams_order}
    outcome_in_cursors: dict[str, float] = {cat: outcome_intervals[cat][1] for cat in OUTCOME_CATEGORIES}

    for cat in OUTCOME_CATEGORIES:
        c_count = outcome_counts[cat]
        c_bot, c_top = outcome_intervals[cat]
        c_col = OUTCOME_COLOURS[cat]

        c_rect = Rectangle(
            (x2, c_bot),
            bar_w,
            float(c_count),
            facecolor=c_col,
            edgecolor="#1B2631" if c_count > 0 else c_col,
            linewidth=1.1 if c_count > 0 else 0.0,
            zorder=4,
        )
        c_rect.set_gid(f"node:outcome:{cat}")
        ax.add_patch(c_rect)

        if c_count == 0:
            ax.plot(
                [x2 - 0.008, x2 + bar_w + 0.008],
                [c_bot, c_bot],
                color=c_col,
                linestyle="--",
                linewidth=1.5,
                zorder=4,
            )

        for idx, t in enumerate(teams_order):
            f_count = flow_counts[(t, cat)]
            if f_count <= 0:
                continue
            t_col = _team_colour(t, idx)
            y_l_top = team_out_cursors[t]
            y_l_bot = y_l_top - f_count
            team_out_cursors[t] = y_l_bot

            y_r_top = outcome_in_cursors[cat]
            y_r_bot = y_r_top - f_count
            outcome_in_cursors[cat] = y_r_bot

            ribbon_s2 = _make_sankey_ribbon(
                x1 + bar_w,
                y_l_bot,
                y_l_top,
                x2,
                y_r_bot,
                y_r_top,
                facecolor=t_col,
                alpha=0.38,
                gid=f"flow:{t}->{cat}",
            )
            ax.add_patch(ribbon_s2)

        pct_cat = 100.0 * c_count / total_starts
        extra_entry_note = ""
        if cat == "final_third_entry" and "has_entry" in metrics.columns:
            total_entry_passages = int(metrics["has_entry"].fillna(False).astype(bool).sum())
            if total_entry_passages > c_count:
                diff_entry = total_entry_passages - c_count
                extra_entry_note = f"  (+{diff_entry} also shot → {total_entry_passages} total)"

        if c_count > 0:
            split_str = " · ".join(f"{t} {flow_counts[(t, cat)]}" for t in teams_order)
            label_text = (
                f"{OUTCOME_LABELS[cat]} — n = {c_count} ({pct_cat:.1f}%)\n"
                f"By team: {split_str}{extra_entry_note}"
            )
        else:
            label_text = (
                f"{OUTCOME_LABELS[cat]} — n = 0 (0.0%)\n"
                f"Empty category (0 passages){extra_entry_note}"
            )

        ax.text(
            x2 + bar_w + 0.018,
            0.5 * (c_bot + c_top),
            label_text,
            ha="left",
            va="center",
            fontsize=8.7,
            linespacing=1.18,
            fontweight="bold" if c_count > 0 else "normal",
            color="#1B2631" if c_count > 0 else "#5D6D7E",
            zorder=6,
        )

    # Column stage headers
    header_y = col2_total_span + 0.13 * total_starts
    stage_headers = [
        (x0 + 0.5 * bar_w, "1. Sequence-Opening\nInterventions"),
        (x1 + 0.5 * bar_w, "2. Recovering Team\n(passages)"),
        (x2 + 0.5 * bar_w, "3. Exclusive Outcome\n(priority hierarchy)"),
    ]
    for hx, htext in stage_headers:
        ax.text(
            hx,
            header_y,
            htext,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#34495E",
        )

    ax.set_xlim(-0.33, 1.56)
    ax.set_ylim(-0.10 * total_starts, header_y + 0.13 * total_starts)
    ax.axis("off")

    ax.set_title(
        "Who Turned Defensive Interventions into Attack? Candidate Transition Flow & Exclusive Outcomes\n"
        f"Unit: sequence-opening interventions (all {total_starts} starts conserved; exclusive "
        "priority separates shot, territorial progress & pass gain)",
        fontsize=11.3,
        fontweight="bold",
        color="#1B2631",
        pad=10,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.04)
    return fig


def _select_default_outfield_cohort(pool_df: pd.DataFrame, net_col: str | None) -> pd.DataFrame:
    """Build a balanced outfield comparison shortlist across connectors, targets, recoverers, and outlets."""
    work = pool_df.copy()
    metric_tie = net_col if net_col is not None else "forward_metres"
    selected: list[str] = []

    def _add_top(mask_col: str, sort_cols: list[str], ascending: list[bool], limit: int) -> None:
        sub = work[work[mask_col] > 0].sort_values(sort_cols, ascending=ascending)
        for pid in sub["player_id"]:
            if pid not in selected:
                selected.append(str(pid))
                if len(selected) >= 9:
                    return
            if sum(1 for s in selected if s in set(sub["player_id"].head(limit))) >= limit:
                break

    # 1. Connectors delivering passes into the final third
    _add_top("entries_delivered", ["entries_delivered", metric_tie, "player_id"], [False, False, True], 3)
    # 2. Targets receiving final-third entries (ensures target forwards like home_10 are included)
    _add_top("entries_received", ["entries_received", "shot_passages", "player_id"], [False, False, True], 3)
    # 3. High-volume sequence-opening recoverers
    _add_top("recovery_starts", ["recovery_starts", "recovery_followups", "player_id"], [False, False, True], 2)
    # 4. Primary first-outlet receivers
    _add_top("outlet_passages", ["outlet_passages", metric_tie, "player_id"], [False, False, True], 2)

    if len(selected) < min(8, len(work)):
        work["_total_entries"] = work["entries_delivered"] + work["entries_received"]
        work["_total_roles"] = work["recovery_starts"] + work["outlet_passages"]
        fallback = work.sort_values(
            ["_total_entries", "_total_roles", metric_tie, "player_id"],
            ascending=[False, False, False, True],
        )
        for pid in fallback["player_id"]:
            if len(selected) >= min(8, len(work)):
                break
            if pid not in selected:
                selected.append(str(pid))

    return pool_df.set_index("player_id").loc[selected].reset_index()


def plot_player_evidence(
    ranking: pd.DataFrame,
    player_ids: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Plot role-separated transition evidence across connectors, targets, recoverers, and outlets.

    - When ``player_ids`` is supplied, plots *only* those players in exact requested order.
    - When ``player_ids`` is ``None``, compares outfielders (if ``role`` is present) and
      contextualises goalkeepers separately.
    - Never orders, labels, or claims a Pareto or composite ranking; displays optional
      net pass metres and failed-pass stops when present to expose backward recycling.
    """
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        raise ValueError("ranking must be a non-empty DataFrame.")

    required_cols = {
        "player_id",
        "team",
        "minutes_played",
        "recovery_starts",
        "recovery_followups",
        "recovery_shot_passages",
        "recovery_outcome_passages",
        "outlet_passages",
        "outlet_shot_passages",
        "outlet_outcome_passages",
        "completed_transition_passes",
        "measurable_transition_passes",
        "forward_metres",
        "entries_delivered",
        "entries_received",
    }
    missing = required_cols - set(ranking.columns)
    if missing:
        raise ValueError(f"ranking is missing required columns: {sorted(missing)}")

    df = ranking.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["team"] = df["team"].astype(str)
    known_ids = set(df["player_id"])

    net_col = (
        "net_pass_metres"
        if "net_pass_metres" in df.columns
        else ("net_forward_metres" if "net_forward_metres" in df.columns else None)
    )
    fail_col = (
        "failed_passes_at_stop"
        if "failed_passes_at_stop" in df.columns
        else ("failed_transition_passes" if "failed_transition_passes" in df.columns else None)
    )
    has_role = "role" in df.columns

    if has_role:
        gk_mask = df["role"].notna() & (df["role"].astype(str).str.lower() == "goalkeeper")
        gk_df = df[gk_mask].copy()
        outfield_pool = df[~gk_mask].copy()
        if outfield_pool.empty:
            outfield_pool = df.copy()
    else:
        gk_df = df.iloc[0:0].copy()
        outfield_pool = df.copy()

    if player_ids is not None:
        requested_ids = list(dict.fromkeys(str(pid) for pid in player_ids))
        if not requested_ids:
            raise ValueError("player_ids must contain at least one player_id when provided.")
        unknown_ids = [pid for pid in requested_ids if pid not in known_ids]
        if unknown_ids:
            raise ValueError(f"Requested player_ids not found in ranking: {unknown_ids}")
        cohort = df.set_index("player_id").loc[requested_ids].reset_index()
        cohort_mode_label = f"Selected cohort (n={len(cohort)}, requested order preserved)"
    else:
        cohort = _select_default_outfield_cohort(outfield_pool, net_col=net_col)
        cohort_mode_label = (
            f"Outfield role shortlist (n={len(cohort)} of {len(outfield_pool)} outfielders)"
            if has_role
            else f"Role shortlist (n={len(cohort)} of {len(df)} squad players)"
        )

    fig, axes = plt.subplots(2, 2, figsize=(16.4, 10.0))
    fig.patch.set_facecolor("white")
    ax_prog, ax_terr, ax_rec, ax_out = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    y_pos = np.arange(len(cohort))
    y_labels: list[str] = []
    for r in cohort.itertuples():
        role_tag = (
            ", GK"
            if has_role and hasattr(r, "role") and pd.notna(r.role) and str(r.role).lower() == "goalkeeper"
            else ""
        )
        y_labels.append(f"{r.player_id} ({r.team}, {float(r.minutes_played):.0f} min{role_tag})")
    bar_colours = [_team_colour(str(t)) for t in cohort["team"]]

    # Panel A: Connector Pass Progression — Positive vs Net Metres & Failed-Pass Stops
    fwd_vals = cohort["forward_metres"].astype(float).to_numpy()
    if net_col is not None:
        net_vals = cohort[net_col].astype(float).to_numpy()
        ax_prog.barh(
            y_pos,
            fwd_vals,
            height=0.64,
            color=bar_colours,
            alpha=0.28,
            edgecolor=bar_colours,
            linewidth=1.1,
            label="Positive forward m (clipped ≥0)",
        )
        ax_prog.barh(
            y_pos,
            net_vals,
            height=0.36,
            color=bar_colours,
            alpha=0.92,
            edgecolor="#1B2631",
            linewidth=0.8,
            label="Net pass m (incl. back-passes)",
        )
        ax_prog.axvline(0.0, color="#2C3E50", linewidth=1.1, zorder=3)
        min_x = min(0.0, float(net_vals.min()) if len(net_vals) else 0.0)
        max_x = max(float(max(fwd_vals.max(), net_vals.max())) if len(fwd_vals) else 0.0, 5.0)
        span_x = max(max_x - min_x, 5.0)
        for y, r in zip(y_pos, cohort.itertuples()):
            fwd_v = float(r.forward_metres)
            net_v = float(getattr(r, net_col))
            fail_str = (
                f"; {int(getattr(r, fail_col))} failed stop"
                if fail_col is not None and pd.notna(getattr(r, fail_col))
                else ""
            )
            anchor_right = max(0.0, fwd_v, net_v)
            ax_prog.text(
                anchor_right + 0.025 * span_x,
                y,
                (
                    f"+{fwd_v:.1f} m pos. | net {net_v:+.1f} m "
                    f"[{int(r.measurable_transition_passes)}/{int(r.completed_transition_passes)} meas.{fail_str}]"
                ),
                va="center",
                ha="left",
                fontsize=7.7,
                color="#1B2631",
            )
        ax_prog.set_xlim(min_x - 0.06 * span_x if min_x < 0 else 0.0, max_x + 1.58 * span_x)
        ax_prog.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=2,
            fontsize=7.9,
            frameon=False,
        )
    else:
        ax_prog.barh(
            y_pos,
            fwd_vals,
            height=0.58,
            color=bar_colours,
            alpha=0.86,
            edgecolor="#1B2631",
            linewidth=0.8,
        )
        max_x = max(float(fwd_vals.max()) if len(fwd_vals) else 0.0, 5.0)
        for y, r in zip(y_pos, cohort.itertuples()):
            prog_p = int(r.progressive_passes) if hasattr(r, "progressive_passes") else 0
            fail_str = (
                f"; {int(getattr(r, fail_col))} failed stop"
                if fail_col is not None and pd.notna(getattr(r, fail_col))
                else f"; {prog_p} ≥10m"
            )
            ax_prog.text(
                float(r.forward_metres) + 0.025 * max_x,
                y,
                (
                    f"+{float(r.forward_metres):.1f} m pos. "
                    f"[{int(r.measurable_transition_passes)}/{int(r.completed_transition_passes)} meas.{fail_str}]"
                ),
                va="center",
                ha="left",
                fontsize=7.8,
                color="#1B2631",
            )
        ax_prog.set_xlim(0.0, max_x * 2.32)

    if net_col is not None and fail_col is not None:
        panel_a_title = "A. Connector Pass Progression: Positive vs Net Metres & Turnovers"
    elif net_col is not None:
        panel_a_title = "A. Connector Pass Progression: Positive vs Net Pass Metres"
    elif fail_col is not None:
        panel_a_title = "A. Connector Pass Progression: Positive Metres & Turnovers"
    else:
        panel_a_title = "A. Connector Pass Progression: Positive Completed-Pass Metres"

    ax_prog.set_yticks(y_pos, labels=y_labels, fontsize=8.5)
    ax_prog.invert_yaxis()
    ax_prog.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax_prog.set_xlabel("Completed transition pass distance (m)", fontsize=9.4)
    ax_prog.set_title(
        panel_a_title,
        fontsize=10.3,
        fontweight="bold",
        pad=22 if net_col is not None else 10,
    )

    # Panel B: Final-Third Territorial Progress — Connectors (entries_delivered) vs Targets (entries_received)
    ent_del_vals = cohort["entries_delivered"].astype(float).to_numpy()
    ent_rec_vals = cohort["entries_received"].astype(float).to_numpy()
    ax_terr.barh(
        y_pos - 0.17,
        ent_del_vals,
        height=0.32,
        color=bar_colours,
        alpha=0.92,
        edgecolor="#1B2631",
        linewidth=0.7,
        label="Entries delivered (connector >17.5 m)",
    )
    ax_terr.barh(
        y_pos + 0.17,
        ent_rec_vals,
        height=0.32,
        color="#E69F00",
        alpha=0.88,
        edgecolor="#1B2631",
        linewidth=0.7,
        hatch="//",
        label="Entries received (final-third target)",
    )
    max_ent = max(
        float(max(ent_del_vals.max(), ent_rec_vals.max())) if len(ent_del_vals) else 0.0,
        2.0,
    )
    for y, r in zip(y_pos, cohort.itertuples()):
        ed = int(r.entries_delivered)
        er = int(r.entries_received)
        if hasattr(r, "shots_taken") and pd.notna(r.shots_taken):
            shot_note = f"{int(r.shots_taken)} direct shot(s)"
        elif hasattr(r, "shot_passages") and pd.notna(r.shot_passages):
            shot_note = f"{int(r.shot_passages)} shot passage(s)"
        else:
            shot_note = "0 shots"
        ax_terr.text(
            max(ed, er) + 0.04 * max_ent,
            y,
            f"{ed} delivered · {er} received (target)  [{shot_note}]",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#1B2631",
        )
    ax_terr.set_yticks(y_pos, labels=y_labels, fontsize=8.5)
    ax_terr.invert_yaxis()
    ax_terr.set_xlim(0.0, max_ent * 2.25)
    ax_terr.xaxis.get_major_locator().set_params(integer=True)
    ax_terr.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax_terr.set_xlabel("Completed passes crossing x_att = +17.5 m (territorial progress)", fontsize=9.4)
    ax_terr.set_title(
        "B. Final-Third Territory: Connectors (Delivered) vs Targets (Received)",
        fontsize=10.3,
        fontweight="bold",
        pad=22,
    )
    ax_terr.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=7.9,
        frameon=False,
    )

    # Panel C: Recoverer role association (outcomes nested inside sequence-opening interventions)
    rec_starts = cohort["recovery_starts"].astype(float).to_numpy()
    rec_outcomes = cohort["recovery_outcome_passages"].astype(float).to_numpy()
    ax_rec.barh(
        y_pos,
        rec_starts,
        height=0.66,
        color=bar_colours,
        alpha=0.25,
        edgecolor=bar_colours,
        linewidth=1.1,
        label="Sequence-opening interventions (denom.)",
    )
    ax_rec.barh(
        y_pos,
        rec_outcomes,
        height=0.38,
        color=bar_colours,
        alpha=0.92,
        edgecolor="#1B2631",
        linewidth=0.7,
        label="Downstream entry/shot passages",
    )
    max_rec = max(float(rec_starts.max()) if len(rec_starts) else 0.0, 2.0)
    for y, r in zip(y_pos, cohort.itertuples()):
        ax_rec.text(
            float(r.recovery_starts) + 0.03 * max_rec,
            y,
            (
                f"{int(r.recovery_outcome_passages)}/{int(r.recovery_starts)} entry/shot "
                f"({int(r.recovery_followups)}/{int(r.recovery_starts)} follow-up; "
                f"{int(r.recovery_shot_passages)} shot)"
            ),
            va="center",
            ha="left",
            fontsize=8.0,
            color="#1B2631",
        )
    ax_rec.set_yticks(y_pos, labels=y_labels, fontsize=8.5)
    ax_rec.invert_yaxis()
    ax_rec.set_xlim(0.0, max_rec * 2.18)
    ax_rec.xaxis.get_major_locator().set_params(integer=True)
    ax_rec.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax_rec.set_xlabel("Sequence-opening tackles & interceptions (passages)", fontsize=9.4)
    ax_rec.set_title(
        "C. Recoverer Association: Downstream Outcomes / Opening Interventions",
        fontsize=10.3,
        fontweight="bold",
        pad=22,
    )
    ax_rec.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=7.9,
        frameon=False,
    )

    # Panel D: First-outlet role association (outcomes nested inside outlet_passages denominator)
    out_passages = cohort["outlet_passages"].astype(float).to_numpy()
    out_outcomes = cohort["outlet_outcome_passages"].astype(float).to_numpy()
    ax_out.barh(
        y_pos,
        out_passages,
        height=0.66,
        color=bar_colours,
        alpha=0.25,
        edgecolor=bar_colours,
        linewidth=1.1,
        label="First-outlet receptions (denom.)",
    )
    ax_out.barh(
        y_pos,
        out_outcomes,
        height=0.38,
        color=bar_colours,
        alpha=0.92,
        edgecolor="#1B2631",
        linewidth=0.7,
        label="Downstream entry/shot passages",
    )
    max_out = max(float(out_passages.max()) if len(out_passages) else 0.0, 2.0)
    for y, r in zip(y_pos, cohort.itertuples()):
        ax_out.text(
            float(r.outlet_passages) + 0.03 * max_out,
            y,
            (
                f"{int(r.outlet_outcome_passages)}/{int(r.outlet_passages)} entry/shot "
                f"({int(r.outlet_shot_passages)} shot)"
            ),
            va="center",
            ha="left",
            fontsize=8.0,
            color="#1B2631",
        )
    ax_out.set_yticks(y_pos, labels=y_labels, fontsize=8.5)
    ax_out.invert_yaxis()
    ax_out.set_xlim(0.0, max_out * 1.95)
    ax_out.xaxis.get_major_locator().set_params(integer=True)
    ax_out.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax_out.set_xlabel("Distinct passages as first completed-pass recipient (passages)", fontsize=9.4)
    ax_out.set_title(
        "D. First-Outlet Association: Downstream Outcomes / Outlet Receptions",
        fontsize=10.3,
        fontweight="bold",
        pad=22,
    )
    ax_out.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=7.9,
        frameon=False,
    )

    if not gk_df.empty:
        gk_summaries: list[str] = []
        for g in gk_df.sort_values("player_id").itertuples():
            net_part = f", net {float(getattr(g, net_col)):+.1f} m" if net_col is not None else ""
            fail_part = f", {int(getattr(g, fail_col))} failed stops" if fail_col is not None else ""
            gk_summaries.append(
                f"{g.player_id} ({float(g.minutes_played):.0f} min: "
                f"+{float(g.forward_metres):.1f} m pos.{net_part} on "
                f"{int(g.completed_transition_passes)} comp. passes{fail_part}, "
                f"{int(g.entries_delivered)} entries)"
            )
        footnote = (
            "Goalkeeper long-kick context (contextualised separately due to deep defensive runway):\n"
            + " · ".join(gk_summaries)
        )
    else:
        footnote = (
            "Direct on-ball actions (Panels A–B) and downstream passage associations (Panels C–D) "
            "use distinct role denominators; no composite score applied."
        )

    fig.text(
        0.5,
        0.012,
        footnote,
        ha="center",
        va="bottom",
        fontsize=8.3,
        linespacing=1.22,
        color="#34495E",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8F9F9", "edgecolor": "#D5DBDB"},
    )

    fig.suptitle(
        f"Who Turned Defensive Interventions into Attack? Role-Separated Player Evidence ({cohort_mode_label})\n"
        "Direct Actions (Connectors & Final-Third Targets) Separated from Downstream Recoverer & Outlet Associations",
        fontsize=11.5,
        fontweight="bold",
        color="#1B2631",
        y=0.975,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.87, bottom=0.105, wspace=0.38, hspace=0.38)
    return fig


def nearest_frame(
    players: pd.DataFrame,
    timestamp_ms: int,
    half: int,
) -> tuple[pd.DataFrame, int]:
    """Return tracking rows for the nearest frame within ``half`` and absolute ms gap.

    Chooses the earlier frame on an exact timestamp tie and raises ``ValueError``
    if ``players`` is empty or has no frames for ``half``.
    """
    if not isinstance(players, pd.DataFrame) or players.empty:
        raise ValueError("players must be a non-empty tracking DataFrame.")

    required_cols = {"timestamp_ms", "half"}
    missing = required_cols - set(players.columns)
    if missing:
        raise ValueError(f"players is missing required columns: {sorted(missing)}")

    if isinstance(timestamp_ms, bool) or pd.isna(timestamp_ms):
        raise ValueError("timestamp_ms must be a valid integer millisecond timestamp.")
    if isinstance(half, bool) or pd.isna(half):
        raise ValueError("half must be 1 or 2.")

    target_ms = int(timestamp_ms)
    target_half = int(half)
    if target_half not in (1, 2):
        raise ValueError(f"Invalid half={half!r}; expected 1 or 2.")

    in_half = players[players["half"] == target_half]
    if in_half.empty:
        raise ValueError(f"No tracking frames available for half={target_half}.")

    valid_ts = in_half["timestamp_ms"].dropna().to_numpy(dtype=np.int64)
    if valid_ts.size == 0:
        raise ValueError(f"No non-null frame timestamps found for half={target_half}.")

    unique_ts = np.sort(np.unique(valid_ts))
    diffs = np.abs(unique_ts - target_ms)
    # unique_ts is sorted strictly ascending, so argmin picks the earlier frame on a tie
    best_idx = int(np.argmin(diffs))
    chosen_ts = int(unique_ts[best_idx])
    gap_ms = int(abs(chosen_ts - target_ms))

    frame_df = in_half.loc[in_half["timestamp_ms"] == chosen_ts].copy()
    return frame_df, gap_ms


def _draw_normalised_pitch(ax: matplotlib.axes.Axes) -> None:
    hl = PITCH_LENGTH_M / 2.0
    hw = PITCH_WIDTH_M / 2.0
    line_col = "#4A5568"

    ax.add_patch(
        Rectangle(
            (-hl, -hw),
            PITCH_LENGTH_M,
            PITCH_WIDTH_M,
            facecolor="#F7FBF7",
            edgecolor=line_col,
            linewidth=1.6,
            zorder=1,
        )
    )
    ax.plot([0.0, 0.0], [-hw, hw], color=line_col, linewidth=1.3, zorder=2)
    ax.add_patch(Circle((0.0, 0.0), 9.15, fill=False, edgecolor=line_col, linewidth=1.3, zorder=2))
    ax.scatter([0.0], [0.0], s=18, color=line_col, zorder=2)

    for side in (-1.0, 1.0):
        # Penalty area (16.5 m x 40.3 m) and 6-yard area (5.5 m x 18.32 m)
        x_pa = (hl - 16.5) if side > 0 else -hl
        x_six = (hl - 5.5) if side > 0 else -hl
        ax.add_patch(
            Rectangle((x_pa, -20.15), 16.5, 40.30, fill=False, edgecolor=line_col, linewidth=1.3, zorder=2)
        )
        ax.add_patch(
            Rectangle((x_six, -9.16), 5.5, 18.32, fill=False, edgecolor=line_col, linewidth=1.2, zorder=2)
        )
        # Goal mouth (7.32 m wide)
        x_goal = hl if side > 0 else (-hl - 1.6)
        ax.add_patch(
            Rectangle((x_goal, -3.66), 1.6, 7.32, facecolor="#E2E8F0", edgecolor=line_col, linewidth=1.3, zorder=2)
        )

    # Final-third boundary at x_att = +17.5 m
    ax.plot(
        [FINAL_THIRD_X_ATT_M, FINAL_THIRD_X_ATT_M],
        [-hw, hw],
        color="#009E73",
        linestyle="--",
        linewidth=1.4,
        alpha=0.80,
        zorder=2,
    )
    ax.text(
        FINAL_THIRD_X_ATT_M + 0.8,
        hw - 1.5,
        "Final third (+17.5 m)",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#007A59",
        zorder=3,
    )

    ax.set_xlim(-hl - 4.5, hl + 4.5)
    ax.set_ylim(-hw - 4.5, hw + 5.5)
    ax.set_xticks([-52.5, -35.0, -17.5, 0.0, 17.5, 35.0, 52.5])
    ax.set_yticks([-33.8, -16.9, 0.0, 16.9, 33.8])
    ax.set_aspect("equal")
    ax.set_xlabel("Normalised pitch length x_att (m; transitioning side attacks left → right)", fontsize=9.8)
    ax.set_ylabel("Normalised pitch width y_att (m; + left flank / − right flank)", fontsize=9.8)


def plot_transition_pitch(
    transition_events: pd.DataFrame,
    players: pd.DataFrame | None = None,
    focus_player: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot a single transition passage on a normalised pitch with optional tracking snapshot.

    Normalises coordinates by the transitioning team's ``attack_dir`` so play moves
    left-to-right (+x_att), labels metres and video clock explicitly, and reports
    missing ball/event coordinates rather than inventing positions.
    """
    if not isinstance(transition_events, pd.DataFrame) or transition_events.empty:
        raise ValueError("transition_events must be a non-empty DataFrame for one transition.")

    required_cols = {"timestamp_ms", "half", "event_name", "team", "attack_dir"}
    missing_cols = required_cols - set(transition_events.columns)
    if missing_cols:
        raise ValueError(f"transition_events is missing required columns: {sorted(missing_cols)}")

    if "transition_id" in transition_events.columns:
        non_null_tids = transition_events["transition_id"].dropna().unique()
        if len(non_null_tids) > 1:
            raise ValueError(
                f"transition_events must contain a single transition; got IDs {list(non_null_tids)}."
            )
        trans_id_label = f"Transition #{int(non_null_tids[0])}" if len(non_null_tids) == 1 else "Transition"
    else:
        trans_id_label = "Transition"

    sort_cols = ["timestamp_ms"] + (["event_id"] if "event_id" in transition_events.columns else [])
    ev_df = transition_events.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    first_ev = ev_df.iloc[0]
    last_ev = ev_df.iloc[-1]
    trans_team = str(first_ev["team"])
    half = int(first_ev["half"])
    trans_dir = int(first_ev["attack_dir"])
    if trans_dir not in (-1, 1):
        raise ValueError(f"Transitioning team attack_dir must be +1 or -1; got {trans_dir}.")

    start_ms = int(first_ev["timestamp_ms"])
    end_ms = int(last_ev["timestamp_ms"])

    fig, ax = plt.subplots(figsize=(13.0, 8.6))
    fig.patch.set_facecolor("white")
    _draw_normalised_pitch(ax)

    # Attacking direction banner inside top margin of pitch axes (offset right of upper-left legend)
    ax.annotate(
        "",
        xy=(30.0, PITCH_WIDTH_M / 2.0 + 2.4),
        xytext=(-10.0, PITCH_WIDTH_M / 2.0 + 2.4),
        arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": _team_colour(trans_team)},
        zorder=4,
    )
    ax.text(
        10.0,
        PITCH_WIDTH_M / 2.0 + 3.0,
        f"{trans_team.upper()} attacking direction (Half {half}, raw attack_dir={trans_dir:+d} → normalised +x_att)",
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=_team_colour(trans_team),
    )

    caveat_notes: list[str] = []
    legend_handles: list[Any] = []

    # Determine snapshot target time (focus_player's first action in passage if present, else start_ms)
    snap_target_ms = start_ms
    if focus_player is not None:
        fp_str = str(focus_player)
        for row in ev_df.itertuples():
            pid = str(row.player_id) if hasattr(row, "player_id") and pd.notna(row.player_id) else ""
            rid = (
                str(row.receiver_player_id)
                if hasattr(row, "receiver_player_id") and pd.notna(row.receiver_player_id)
                else ""
            )
            if fp_str in (pid, rid):
                snap_target_ms = int(row.timestamp_ms)
                break

    # Optional single tracking snapshot
    snapshot_summary = "Event-only diagram (no tracking snapshot supplied)"
    if players is not None:
        try:
            snap_df, gap_ms = nearest_frame(players, snap_target_ms, half=half)
            snap_actual_ms = int(snap_df["timestamp_ms"].iloc[0])
            snapshot_summary = (
                f"Tracking snapshot at video {_format_video_time(snap_actual_ms)} "
                f"[nearest-frame gap: {gap_ms} ms; later event arrows occur after snapshot]"
            )
            if gap_ms > 1000:
                caveat_notes.append(f"large tracking frame gap ({gap_ms} ms)")

            for idx, (t_name, grp) in enumerate(snap_df.groupby("team", sort=True)):
                t_str = str(t_name)
                col = _team_colour(t_str, idx)
                valid_xy = grp[grp["x"].notna() & grp["y"].notna()]
                if valid_xy.empty:
                    continue
                px = valid_xy["x"].astype(float).to_numpy() * trans_dir
                py = valid_xy["y"].astype(float).to_numpy() * trans_dir
                marker_sz = 120 if t_str == "referee" else 255
                ax.scatter(
                    px,
                    py,
                    s=marker_sz,
                    color=col,
                    alpha=0.45 if t_str == "referee" else 0.80,
                    edgecolor="white",
                    linewidth=1.3,
                    zorder=4,
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=col,
                        markersize=8.5,
                        label=f"{t_str} snapshot pos.",
                    )
                )

                for pr in valid_xy.itertuples():
                    if hasattr(pr, "player_id") and pd.notna(pr.player_id):
                        pid_s = str(pr.player_id)
                        shirt_txt = pid_s.split("_")[-1]
                        ax.text(
                            float(pr.x) * trans_dir,
                            float(pr.y) * trans_dir,
                            shirt_txt,
                            ha="center",
                            va="center",
                            fontsize=7.6,
                            fontweight="bold",
                            color="white",
                            zorder=5,
                        )

            if focus_player is not None and "player_id" in snap_df.columns:
                fp_rows = snap_df[
                    snap_df["player_id"].notna()
                    & (snap_df["player_id"].astype(str) == str(focus_player))
                    & snap_df["x"].notna()
                    & snap_df["y"].notna()
                ]
                if not fp_rows.empty:
                    fx = float(fp_rows["x"].iloc[0]) * trans_dir
                    fy = float(fp_rows["y"].iloc[0]) * trans_dir
                    ax.scatter(
                        [fx],
                        [fy],
                        s=520,
                        facecolor="none",
                        edgecolor="#F1C40F",
                        linewidth=2.8,
                        zorder=6,
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor="none",
                            markeredgecolor="#F1C40F",
                            markeredgewidth=2.2,
                            markersize=10,
                            label=f"Focus player ({focus_player})",
                        )
                    )
                else:
                    caveat_notes.append(f"focus player {focus_player} absent from snapshot frame")

            # Ball position in tracking snapshot (never invent if missing)
            if {"ball_x", "ball_y"}.issubset(snap_df.columns):
                bx_val = snap_df["ball_x"].iloc[0]
                by_val = snap_df["ball_y"].iloc[0]
                if pd.notna(bx_val) and pd.notna(by_val):
                    bx = float(bx_val) * trans_dir
                    by = float(by_val) * trans_dir
                    ax.scatter(
                        [bx],
                        [by],
                        s=105,
                        color="white",
                        edgecolor="#1B2631",
                        linewidth=2.0,
                        zorder=7,
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor="white",
                            markeredgecolor="#1B2631",
                            markeredgewidth=1.8,
                            markersize=7.5,
                            label="Ball at snapshot",
                        )
                    )
                else:
                    caveat_notes.append("snapshot ball coordinates missing in tracking")
            else:
                caveat_notes.append("ball_x/ball_y columns absent from tracking")
        except ValueError as exc:
            snapshot_summary = f"Event-only fallback ({exc})"
            caveat_notes.append(f"tracking unavailable: {exc}")

    # Plot sequential transition events
    team_col = _team_colour(trans_team)
    has_pass_arrow = False
    has_entry_arrow = False
    placed_labels: list[tuple[float, float]] = []

    def _choose_label_xy(anchor_x: float, anchor_y: float, prefer_below: bool = False) -> tuple[float, float]:
        candidates = (
            [(0.0, -3.5), (0.0, 3.5), (-8.5, -4.2), (8.5, 4.2), (-9.0, 4.2), (9.0, -4.2), (0.0, -6.8), (0.0, 6.8)]
            if prefer_below
            else [(0.0, 3.2), (0.0, -3.5), (8.5, 3.8), (-8.5, -3.8), (-9.0, 4.2), (9.0, -4.2), (0.0, 6.8), (0.0, -6.8)]
        )
        best_pt = (anchor_x + candidates[0][0], anchor_y + candidates[0][1])
        best_score = -1.0
        for dx, dy in candidates:
            cx = min(max(anchor_x + dx, -45.0), 45.0)
            cy = min(max(anchor_y + dy, -31.0), 31.0)
            if not placed_labels:
                best_pt = (cx, cy)
                break
            non_overlapping = all(
                (abs(cx - px) > 19.5) or (abs(cy - py) > 3.9) for px, py in placed_labels
            )
            if non_overlapping:
                best_pt = (cx, cy)
                break
            score = min(max(abs(cx - px) / 19.5, abs(cy - py) / 3.9) for px, py in placed_labels)
            if score > best_score:
                best_score = score
                best_pt = (cx, cy)
        placed_labels.append(best_pt)
        return best_pt

    for step_idx, row in enumerate(ev_df.itertuples(), start=1):
        ev_name = str(row.event_name)
        pid_str = str(row.player_id) if hasattr(row, "player_id") and pd.notna(row.player_id) else "unknown"
        t_rel_s = (int(row.timestamp_ms) - start_ms) / 1000.0
        has_start = (
            hasattr(row, "x_start")
            and hasattr(row, "y_start")
            and pd.notna(row.x_start)
            and pd.notna(row.y_start)
        )

        if ev_name == "goal":
            caveat_notes.append(
                f"step {step_idx} goal ({pid_str} at video {_format_video_time(row.timestamp_ms)}; "
                f"goal rows carry no x/y in schema)"
            )
            continue

        if not has_start:
            caveat_notes.append(f"step {step_idx} {ev_name} ({pid_str}) missing start coordinates")
            continue

        xs = float(row.x_start) * trans_dir
        ys = float(row.y_start) * trans_dir
        involves_focus = focus_player is not None and (
            pid_str == str(focus_player)
            or (
                hasattr(row, "receiver_player_id")
                and pd.notna(row.receiver_player_id)
                and str(row.receiver_player_id) == str(focus_player)
            )
        )

        if step_idx == 1 and ev_name in ("tackle", "interception"):
            ax.scatter(
                [xs],
                [ys],
                s=230,
                marker="D",
                color=team_col,
                edgecolor="#1B2631",
                linewidth=1.8,
                zorder=8,
            )
            lx, ly = _choose_label_xy(xs, ys, prefer_below=True)
            ax.text(
                lx,
                ly,
                f"1. {ev_name} ({pid_str}, +0.0s)",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color="#1B2631",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.90, "edgecolor": team_col},
                zorder=9,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="w",
                    markerfacecolor=team_col,
                    markeredgecolor="#1B2631",
                    markersize=8,
                    label="Opening intervention",
                )
            )
        elif ev_name == "pass":
            has_rec = (
                hasattr(row, "receiver_x")
                and hasattr(row, "receiver_y")
                and pd.notna(row.receiver_x)
                and pd.notna(row.receiver_y)
            )
            rec_id = (
                str(row.receiver_player_id)
                if hasattr(row, "receiver_player_id") and pd.notna(row.receiver_player_id)
                else "?"
            )
            if has_rec:
                xr = float(row.receiver_x) * trans_dir
                yr = float(row.receiver_y) * trans_dir
                is_entry = (xs <= FINAL_THIRD_X_ATT_M) and (xr > FINAL_THIRD_X_ATT_M)
                arrow_col = "#009E73" if is_entry else team_col
                has_entry_arrow = has_entry_arrow or is_entry
                has_pass_arrow = has_pass_arrow or (not is_entry)

                arrow = FancyArrowPatch(
                    (xs, ys),
                    (xr, yr),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=2.6 if involves_focus else 2.1,
                    color=arrow_col,
                    alpha=0.92,
                    shrinkA=4,
                    shrinkB=4,
                    zorder=7,
                )
                ax.add_patch(arrow)
                gain_m = xr - xs
                mx, my = 0.5 * (xs + xr), 0.5 * (ys + yr)
                lx, ly = _choose_label_xy(mx, my, prefer_below=(step_idx % 2 == 1))
                ax.text(
                    lx,
                    ly,
                    f"{step_idx}. {pid_str}→{rec_id} ({gain_m:+.1f} m)",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                    color=arrow_col,
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
                    zorder=9,
                )
            else:
                ax.scatter(
                    [xs],
                    [ys],
                    s=140,
                    facecolor="white",
                    edgecolor=team_col,
                    linewidth=2.0,
                    linestyle="--",
                    zorder=8,
                )
                lx, ly = _choose_label_xy(xs, ys, prefer_below=False)
                ax.text(
                    lx,
                    ly,
                    f"{step_idx}. pass ({pid_str}, endpoint unmeasured)",
                    ha="center",
                    va="center",
                    fontsize=7.9,
                    color="#7F8C8D",
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.90, "edgecolor": "#BDC3C7"},
                    zorder=9,
                )
                caveat_notes.append(
                    f"step {step_idx} pass ({pid_str}, +{t_rel_s:.1f}s) missing receiver_x/receiver_y (unmeasured)"
                )
        elif ev_name == "shot":
            ax.scatter(
                [xs],
                [ys],
                s=340,
                marker="*",
                color="#009E73",
                edgecolor="#1B2631",
                linewidth=1.3,
                zorder=9,
            )
            shot_arrow = FancyArrowPatch(
                (xs, ys),
                (PITCH_LENGTH_M / 2.0, 0.0),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.9,
                linestyle="--",
                color="#009E73",
                alpha=0.85,
                zorder=7,
            )
            ax.add_patch(shot_arrow)
            dist_txt = (
                f", {float(row.shot_distance_m):.1f} m"
                if hasattr(row, "shot_distance_m") and pd.notna(row.shot_distance_m)
                else ""
            )
            lx, ly = _choose_label_xy(xs, ys, prefer_below=False)
            ax.text(
                lx,
                ly,
                f"{step_idx}. SHOT ({pid_str}{dist_txt}, +{t_rel_s:.1f}s)",
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color="#007A59",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.92, "edgecolor": "#009E73"},
                zorder=9,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="#009E73",
                    markeredgecolor="#1B2631",
                    markersize=11,
                    label="Shot (dashed line → goal centre)",
                )
            )
        else:
            ax.scatter(
                [xs],
                [ys],
                s=130,
                marker="^",
                color=team_col,
                edgecolor="white",
                linewidth=1.2,
                zorder=8,
            )
            ax.text(
                xs,
                ys + 2.0,
                f"{step_idx}. {ev_name} ({pid_str}, +{t_rel_s:.1f}s)",
                ha="center",
                va="bottom",
                fontsize=7.9,
                color="#1B2631",
                zorder=9,
            )

    if has_pass_arrow:
        legend_handles.append(
            Line2D([0], [0], color=team_col, lw=2.2, label="Completed pass (event coords)")
        )
    if has_entry_arrow:
        legend_handles.append(
            Line2D([0], [0], color="#009E73", lw=2.4, label="Final-third entry pass (> +17.5 m)")
        )

    # Deduplicate legend labels
    dedup_handles: dict[str, Any] = {}
    for h in legend_handles:
        dedup_handles.setdefault(h.get_label(), h)
    if dedup_handles:
        ax.legend(
            list(dedup_handles.values()),
            list(dedup_handles.keys()),
            loc="upper left",
            fontsize=8.2,
            framealpha=0.92,
        )

    ax.set_title(
        f"{trans_id_label} · {trans_team.upper()} · Half {half} · "
        f"Video time {_format_video_time(start_ms)} → {_format_video_time(end_ms)}\n"
        f"{snapshot_summary}",
        fontsize=10.6,
        fontweight="bold",
        color="#1B2631",
        pad=10,
    )

    note_text = (
        "Data notes: " + "; ".join(caveat_notes)
        if caveat_notes
        else "Data notes: all plotted events and snapshot ball coordinates present."
    )
    fig.text(
        0.5,
        0.015,
        note_text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#34495E",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8F9F9", "edgecolor": "#D5DBDB"},
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.10)
    return fig


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str in ("0/0", "N/A"):
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", maxsplit=1)
        num, den = float(num_s), float(den_s)
        return (num / den) if den != 0.0 else 0.0
    return float(rate_str)


def probe_video(path: str | Path) -> dict[str, Any]:
    """Probe a media file with ``ffprobe`` using an argument list (no shell).

    Raises ``FileNotFoundError`` if ``path`` does not exist and ``RuntimeError``
    if ``ffprobe`` fails or finds no usable video stream.
    """
    src_path = Path(path)
    if not src_path.exists():
        raise FileNotFoundError(f"Media file does not exist: {src_path}")
    if not src_path.is_file():
        raise ValueError(f"Media path is not a regular file: {src_path}")

    resolved = src_path.resolve()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(resolved),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to execute ffprobe: {exc}") from exc

    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "unknown ffprobe error").strip()
        raise RuntimeError(f"ffprobe failed for '{src_path}' (exit {proc.returncode}): {err_msg}")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for '{src_path}': {exc}") from exc

    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"No video stream found in '{src_path}'.")

    v_stream = video_streams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fps = _parse_fps(v_stream.get("avg_frame_rate")) or _parse_fps(v_stream.get("r_frame_rate"))

    duration_raw = fmt.get("duration") or v_stream.get("duration")
    duration_s = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0
    if duration_s <= 0.0 and fps > 0.0 and v_stream.get("nb_frames") not in (None, "N/A"):
        duration_s = float(v_stream["nb_frames"]) / fps
    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise RuntimeError(f"Could not determine positive duration for '{src_path}'.")

    width = int(v_stream.get("width") or 0)
    height = int(v_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions ({width}x{height}) for '{src_path}'.")

    st = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "available": True,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "fps": fps,
        "codec_name": str(v_stream.get("codec_name") or "unknown"),
        "has_audio": has_audio,
        "nb_streams": len(streams),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _sidecar_path_for(mp4_path: Path) -> Path:
    return mp4_path.with_name(f"{mp4_path.name}.meta.json")


def _read_owned_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    if not sidecar_path.is_file():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("generator") == SIDECAR_GENERATOR_ID:
        return data
    return None


def _run_ffmpeg_checked(cmd: list[str], action_desc: str, cleanup_paths: list[Path]) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        for p in cleanup_paths:
            p.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to launch ffmpeg during {action_desc}: {exc}") from exc

    if proc.returncode != 0:
        for p in cleanup_paths:
            p.unlink(missing_ok=True)
        err_msg = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg failed during {action_desc} (exit {proc.returncode}): {err_msg}")


def export_transition_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_s: float,
    end_s: float,
    make_gif: bool = False,
) -> dict[str, Any]:
    """Export a clamped, muted MP4 clip and optional 10 fps aspect-preserving GIF.

    Reuses an existing output only when its sidecar metadata matches the source
    file identity, clamped bounds, and encoding settings; regenerates owned
    outputs on mismatch; and refuses to overwrite source or unowned files.
    """
    if isinstance(start_s, bool) or isinstance(end_s, bool):
        raise ValueError("start_s and end_s must be finite numbers.")
    start_f = float(start_s)
    end_f = float(end_s)
    if not (math.isfinite(start_f) and math.isfinite(end_f)):
        raise ValueError(f"start_s ({start_s}) and end_s ({end_s}) must be finite.")
    if end_f <= start_f:
        raise ValueError(
            f"Invalid or empty interval: start_s={start_f}, end_s={end_f} (require start_s < end_s)."
        )

    src_path = Path(video_path)
    out_raw = Path(output_path)
    if str(out_raw).strip() in ("", ".", ".."):
        raise ValueError(f"Invalid output_path: {output_path!r}")

    suffix = out_raw.suffix.lower()
    if suffix == "":
        mp4_path = out_raw.with_suffix(".mp4")
    elif suffix in (".mp4", ".gif"):
        mp4_path = out_raw.with_suffix(".mp4")
    else:
        raise ValueError(f"Unsupported output extension {suffix!r}; expected '.mp4' (or no suffix).")

    gif_path = mp4_path.with_suffix(".gif")
    sidecar_path = _sidecar_path_for(mp4_path)

    # Guard against overwriting input video or protected repository files BEFORE probing/writing
    if src_path.exists():
        src_resolved = src_path.resolve()
        for candidate in (mp4_path, gif_path, sidecar_path):
            if candidate.resolve() == src_resolved or (
                candidate.exists() and os.path.samefile(candidate, src_resolved)
            ):
                raise ValueError(f"Refusing to overwrite source video file: {src_path}")

    for candidate in (mp4_path, gif_path):
        if candidate.name in PROTECTED_REPO_FILENAMES:
            raise ValueError(f"Refusing to overwrite protected dataset/source filename: {candidate.name}")

    # Probe source video (raises FileNotFoundError / RuntimeError on unavailable or invalid media)
    source_info = probe_video(src_path)
    source_duration_s = float(source_info["duration_s"])

    clamped_start_s = round(max(0.0, start_f), 6)
    clamped_end_s = round(min(source_duration_s, end_f), 6)
    clamped = (not math.isclose(clamped_start_s, start_f, abs_tol=1e-6)) or (
        not math.isclose(clamped_end_s, end_f, abs_tol=1e-6)
    )
    clip_duration_s = round(clamped_end_s - clamped_start_s, 6)
    if clip_duration_s <= 1e-3:
        raise ValueError(
            f"Requested interval [{start_f}, {end_f}] becomes empty [{clamped_start_s}, {clamped_end_s}] "
            f"after clamping to [0.0, {source_duration_s:.3f}]."
        )

    # Ownership check: do not overwrite pre-existing unrelated files lacking our sidecar
    existing_sidecar = _read_owned_sidecar(sidecar_path)
    if existing_sidecar is None:
        if mp4_path.exists():
            raise ValueError(
                f"Refusing to overwrite existing unowned file '{mp4_path}' (no sidecar metadata)."
            )
        if make_gif and gif_path.exists():
            raise ValueError(
                f"Refusing to overwrite existing unowned GIF '{gif_path}' (no sidecar metadata)."
            )

    expected_signature = {
        "source_path": source_info["path"],
        "source_size_bytes": source_info["size_bytes"],
        "source_mtime_ns": source_info["mtime_ns"],
        "start_s": clamped_start_s,
        "end_s": clamped_end_s,
        "settings": {
            "video_codec": "libx264",
            "audio_muted": True,
            "make_gif": bool(make_gif),
            "gif_fps": GIF_DEFAULT_FPS if make_gif else None,
            "gif_max_width": GIF_MAX_WIDTH_PX if make_gif else None,
        },
    }

    # Check cache validity
    if existing_sidecar is not None:
        sig_matches = (
            existing_sidecar.get("source_path") == expected_signature["source_path"]
            and existing_sidecar.get("source_size_bytes") == expected_signature["source_size_bytes"]
            and existing_sidecar.get("source_mtime_ns") == expected_signature["source_mtime_ns"]
            and existing_sidecar.get("start_s") == expected_signature["start_s"]
            and existing_sidecar.get("end_s") == expected_signature["end_s"]
            and existing_sidecar.get("settings") == expected_signature["settings"]
        )
        files_present = mp4_path.is_file() and ((not make_gif) or gif_path.is_file())
        if sig_matches and files_present:
            try:
                mp4_info = probe_video(mp4_path)
                gif_info = probe_video(gif_path) if make_gif else None
                if not mp4_info["has_audio"]:
                    return {
                        "start_s": clamped_start_s,
                        "end_s": clamped_end_s,
                        "duration_s": clip_duration_s,
                        "mp4_path": str(mp4_path),
                        "gif_path": str(gif_path) if make_gif else None,
                        "cached": True,
                        "clamped": clamped,
                        "requested_start_s": start_f,
                        "requested_end_s": end_f,
                        "source_video": source_info,
                        "mp4_info": mp4_info,
                        "gif_info": gif_info,
                        "sidecar_path": str(sidecar_path),
                    }
            except (OSError, RuntimeError, ValueError):
                pass  # Regenerate corrupted owned asset

    # Ensure parent directory exists (safe for bare relative filenames where parent is Path('.'))
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    tmp_mp4 = mp4_path.with_name(f".{mp4_path.stem}.tmp_{pid}.mp4")
    tmp_gif = gif_path.with_name(f".{gif_path.stem}.tmp_{pid}.gif")

    mp4_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{clamped_start_s:.6f}",
        "-i",
        source_info["path"],
        "-t",
        f"{clip_duration_s:.6f}",
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_mp4),
    ]
    _run_ffmpeg_checked(mp4_cmd, "MP4 clip extraction", [tmp_mp4, tmp_gif])

    if make_gif:
        gif_filter = (
            f"[0:v]fps={GIF_DEFAULT_FPS},"
            f"scale='min({GIF_MAX_WIDTH_PX},iw)':-1:flags=lanczos,"
            "split[s0][s1];"
            "[s0]palettegen=stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        )
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp_mp4),
            "-filter_complex",
            gif_filter,
            "-loop",
            "0",
            str(tmp_gif),
        ]
        _run_ffmpeg_checked(gif_cmd, "palette GIF encoding", [tmp_mp4, tmp_gif])

    os.replace(tmp_mp4, mp4_path)
    if make_gif:
        os.replace(tmp_gif, gif_path)
    elif existing_sidecar is not None and gif_path.exists():
        # Remove stale owned GIF if re-exporting without make_gif
        gif_path.unlink(missing_ok=True)

    mp4_info = probe_video(mp4_path)
    gif_info = probe_video(gif_path) if make_gif else None

    sidecar_payload = {
        "generator": SIDECAR_GENERATOR_ID,
        **expected_signature,
        "duration_s": clip_duration_s,
        "mp4_name": mp4_path.name,
        "gif_name": gif_path.name if make_gif else None,
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

    return {
        "start_s": clamped_start_s,
        "end_s": clamped_end_s,
        "duration_s": clip_duration_s,
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path) if make_gif else None,
        "cached": False,
        "clamped": clamped,
        "requested_start_s": start_f,
        "requested_end_s": end_f,
        "source_video": source_info,
        "mp4_info": mp4_info,
        "gif_info": gif_info,
        "sidecar_path": str(sidecar_path),
    }
