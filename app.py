"""Streamlit Dashboard: Second-Half Intensity & Movement Comparison.

Launch command:
    uv run streamlit run app.py --server.port 8504
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import analytics as an

st.set_page_config(
    page_title="Veo Match 01 · Second-Half Intensity & Movement",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner="Loading Veo match tracking & event datasets...")
def get_cached_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    summary, events, players = an.load_datasets(".")
    baselines = an.validate_dataset_baselines(summary, events, players)
    per_half = an.compute_per_half_player_metrics(players, summary)
    return summary, events, players, per_half, baselines


def draw_plotly_pitch(fig: go.Figure, row: int | None = None, col: int | None = None) -> None:
    """Add standard 105.0 x 67.7 m pitch markings (origin at centre spot) to a Plotly figure."""
    hl = an.PITCH_LENGTH_M / 2.0
    hw = an.PITCH_WIDTH_M / 2.0
    line_style = dict(color="#666666", width=1.5)

    shapes = [
        # Outer boundary
        dict(type="rect", x0=-hl, y0=-hw, x1=hl, y1=hw, line=line_style, fillcolor="rgba(235, 245, 235, 0.35)", layer="below"),
        # Halfway line
        dict(type="line", x0=0, y0=-hw, x1=0, y1=hw, line=line_style, layer="below"),
        # Centre circle (r = 9.15 m)
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line_style, layer="below"),
        # Penalty areas (16.5 m deep, 40.3 m wide)
        dict(type="rect", x0=-hl, y0=-20.15, x1=-hl + 16.5, y1=20.15, line=line_style, layer="below"),
        dict(type="rect", x0=hl - 16.5, y0=-20.15, x1=hl, y1=20.15, line=line_style, layer="below"),
        # Six-yard boxes (5.5 m deep, 18.32 m wide)
        dict(type="rect", x0=-hl, y0=-9.16, x1=-hl + 5.5, y1=9.16, line=line_style, layer="below"),
        dict(type="rect", x0=hl - 5.5, y0=-9.16, x1=hl, y1=9.16, line=line_style, layer="below"),
        # Final third reference line (+17.5 m attacking)
        dict(type="line", x0=17.5, y0=-hw, x1=17.5, y1=hw, line=dict(color="#999999", width=1, dash="dot"), layer="below"),
    ]

    for shp in shapes:
        if row is not None and col is not None:
            fig.add_shape(shp, row=row, col=col)
        else:
            fig.add_shape(shp)


def format_Signed(val: float, decimals: int = 2, suffix: str = "") -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:+.{decimals}f}{suffix}"


def main() -> None:
    summary, events, players, per_half, baselines = get_cached_data()

    # --- Header & Coaching Caveat ---
    st.title("⚽ Second-Half Intensity & Movement Comparison")
    st.caption(
        "**Veo Football Analytics Sprint · Match 01** (`away` 4 – 2 `home`, all 6 goals scored in the 2nd half) · "
        "Pitch: 105.0 m × 67.7 m · Tracking clock: video `timestamp_ms` (400 ms sampling in-play)"
    )

    st.info(
        "**Coaching & Analytical Interpretation Note:** "
        "**Observed second-half decline in high-speed or sprint frame share cannot establish physiological fatigue.** "
        "In-game running output is heavily shaped by scoreline state (Away scored 3 goals between video min 61.7 and 69.0 to lead 3–0), "
        "tactical block height, game-state pacing, ball-in-play interruptions, and substitutions. "
        "All intensity rates below are **sampled frame shares** (% of a player's observed tracking frames in that half), "
        "not continuous elapsed duration.",
        icon="ℹ️",
    )

    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Filter & Eligibility Controls")

    team_choice = st.sidebar.selectbox(
        "Team Filter",
        options=["All", "away", "home"],
        index=0,
        format_func=lambda x: "Both Teams (away & home)" if x == "All" else f"{x.upper()} Team ({x})",
        help="Filter players by team (`away` won 4–2; `home` lost 2–4).",
    )

    metric_choice = st.sidebar.radio(
        "Primary Intensity Metric",
        options=["hs", "sprint"],
        index=0,
        format_func=lambda m: (
            "High-Speed Frame Share (Zone ≥ 3, ≥ 20 km/h)"
            if m == "hs"
            else "Sprint Frame Share (Zone == 4, ≥ 25 km/h)"
        ),
        help="Choose between High-Speed running (speed_zone 3 & 4, ≥20 km/h) or Sprinting (speed_zone 4, ≥25 km/h).",
    )

    min_cov_pct = st.sidebar.slider(
        "Min Per-Half Tracking Coverage (%)",
        min_value=0,
        max_value=95,
        value=50,
        step=5,
        help="A player must be observed in at least this percentage of total half tracking frames in BOTH Half 1 and Half 2 to be eligible for paired comparison.",
    )

    max_cov_diff_pct = st.sidebar.slider(
        "Max Coverage Imbalance |H2 − H1| (pp)",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Equal-observation filter: maximum allowed percentage-point difference between Half 1 and Half 2 frame coverage.",
    )

    include_gk = st.sidebar.checkbox(
        "Include Goalkeepers (`away_1`, `home_1`)",
        value=False,
        help="Outfield movement metrics exclude goalkeepers by default per DATA_DICTIONARY.md.",
    )

    show_ineligible_in_table = st.sidebar.checkbox(
        "Show Ineligible / Single-Half Players in Table",
        value=True,
        help="Keep substituted or single-half players visible in the table with explicit ineligibility reasons and N/A comparisons.",
    )

    # Build comparison table with current controls
    comp_df = an.build_half_comparison_table(
        per_half_df=per_half,
        summary=summary,
        min_coverage_pct=float(min_cov_pct),
        max_coverage_diff_pct=float(max_cov_diff_pct),
        include_goalkeepers=include_gk,
        team_filter=team_choice,
    )

    eligible_df = comp_df[comp_df["is_eligible"]].copy() if not comp_df.empty else pd.DataFrame()
    eligible_ids = eligible_df["player_id"].tolist() if not eligible_df.empty else []

    selected_players = st.sidebar.multiselect(
        "Select Players to Compare",
        options=eligible_ids,
        default=eligible_ids,
        help="Narrow the paired charts and KPIs to a custom subset of eligible players.",
    )

    active_eligible_df = (
        eligible_df[eligible_df["player_id"].isin(selected_players)].copy()
        if not eligible_df.empty
        else pd.DataFrame()
    )

    # Metric column mapping
    if metric_choice == "hs":
        metric_title = "High-Speed Frame Share (≥ 20 km/h)"
        col_h1 = "hs_share_pct_h1"
        col_h2 = "hs_share_pct_h2"
        col_frames_h1 = "hs_frames_h1"
        col_frames_h2 = "hs_frames_h2"
        col_diff_pp = "hs_diff_pp"
        col_rel_pct = "hs_rel_change_pct"
        col_rel_note = "hs_rel_note"
        trend_col = "hs_share_pct"
    else:
        metric_title = "Sprint Frame Share (≥ 25 km/h)"
        col_h1 = "sprint_share_pct_h1"
        col_h2 = "sprint_share_pct_h2"
        col_frames_h1 = "sprint_frames_h1"
        col_frames_h2 = "sprint_frames_h2"
        col_diff_pp = "sprint_diff_pp"
        col_rel_pct = "sprint_rel_change_pct"
        col_rel_note = "sprint_rel_note"
        trend_col = "sprint_share_pct"

    # --- Top KPI Summary Row ---
    st.subheader(f"📊 Cohort Summary: {metric_title}")
    total_candidates = len(comp_df)
    n_elig = len(active_eligible_df)

    if active_eligible_df.empty:
        st.warning(
            "No players match the current eligibility thresholds and player selection. "
            "Try lowering **Min Per-Half Tracking Coverage (%)**, increasing **Max Coverage Imbalance (pp)**, "
            "or selecting players in the sidebar."
        )
    else:
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.metric(
                label="Eligible Selected Players",
                value=f"{n_elig} / {total_candidates}",
                delta=f"Min Cov ≥ {min_cov_pct}%, |ΔCov| ≤ {max_cov_diff_pct} pp",
                delta_color="off",
            )

        team_summary = an.compute_team_half_summary(active_eligible_df, only_eligible=True)

        # Away KPI
        away_row = team_summary[team_summary["team"] == "away"]
        with kpi_cols[1]:
            if not away_row.empty:
                r = away_row.iloc[0]
                v1 = r["hs_share_h1_pct"] if metric_choice == "hs" else r["sprint_share_h1_pct"]
                v2 = r["hs_share_h2_pct"] if metric_choice == "hs" else r["sprint_share_h2_pct"]
                d_pp = r["hs_diff_pp"] if metric_choice == "hs" else r["sprint_diff_pp"]
                d_rel = r["hs_rel_change_pct"] if metric_choice == "hs" else r["sprint_rel_change_pct"]
                rel_str = f"{d_rel:+.1f}% rel" if pd.notna(d_rel) else "undef rel %"
                st.metric(
                    label=f"AWAY Pooled Share (n={int(r['n_players'])})",
                    value=f"H1: {v1:.2f}% → H2: {v2:.2f}%",
                    delta=f"{d_pp:+.2f} pp ({rel_str})",
                )
            else:
                st.metric(label="AWAY Pooled Share", value="No away players", delta="Filtered out", delta_color="off")

        # Home KPI
        home_row = team_summary[team_summary["team"] == "home"]
        with kpi_cols[2]:
            if not home_row.empty:
                r = home_row.iloc[0]
                v1 = r["hs_share_h1_pct"] if metric_choice == "hs" else r["sprint_share_h1_pct"]
                v2 = r["hs_share_h2_pct"] if metric_choice == "hs" else r["sprint_share_h2_pct"]
                d_pp = r["hs_diff_pp"] if metric_choice == "hs" else r["sprint_diff_pp"]
                d_rel = r["hs_rel_change_pct"] if metric_choice == "hs" else r["sprint_rel_change_pct"]
                rel_str = f"{d_rel:+.1f}% rel" if pd.notna(d_rel) else "undef rel %"
                st.metric(
                    label=f"HOME Pooled Share (n={int(r['n_players'])})",
                    value=f"H1: {v1:.2f}% → H2: {v2:.2f}%",
                    delta=f"{d_pp:+.2f} pp ({rel_str})",
                )
            else:
                st.metric(label="HOME Pooled Share", value="No home players", delta="Filtered out", delta_color="off")

        # Attacking Depth Shift KPI
        with kpi_cols[3]:
            mean_shift = active_eligible_df["x_att_shift_m"].mean()
            decliners = int((active_eligible_df[col_diff_pp] < 0).sum())
            increasers = int((active_eligible_df[col_diff_pp] > 0).sum())
            st.metric(
                label="Player Direction Split (H2 vs H1)",
                value=f"{decliners} Lower · {increasers} Higher",
                delta=f"Mean depth shift: {mean_shift:+.1f} m (x_att)",
                delta_color="off",
            )

    # --- Main Visual Tabs ---
    tab_paired, tab_trends, tab_pitch, tab_table = st.tabs(
        [
            "📈 Paired Half Comparison (Bars & Scatter)",
            "⏱️ Within-Half Movement & Compactness Trends",
            "🏟️ Player Half-by-Half Pitch Footprint",
            "📋 Per-Player Change & Eligibility Table",
        ]
    )

    # =========================================================================
    # TAB 1: PAIRED BARS & SCATTER
    # =========================================================================
    with tab_paired:
        if active_eligible_df.empty:
            st.info("No eligible players selected to plot. Adjust sidebar filters.")
        else:
            col_left, col_right = st.columns([1.15, 1.0])

            with col_left:
                st.markdown(f"#### Paired Half 1 vs Half 2 {metric_title}")
                st.caption(
                    "Sampled frame share (% of player's observed frames in each half). "
                    "Sorted by team and shirt number."
                )

                bar_fig = go.Figure()
                x_labels = [
                    f"{row['player_id']} ({row['coverage_pct_h1']:.0f}%/{row['coverage_pct_h2']:.0f}%)"
                    for _, row in active_eligible_df.iterrows()
                ]

                bar_fig.add_trace(
                    go.Bar(
                        name="1st Half Share (%)",
                        x=x_labels,
                        y=active_eligible_df[col_h1],
                        marker_color="#4c78a8",
                        text=[f"{v:.2f}%" for v in active_eligible_df[col_h1]],
                        textposition="auto",
                        customdata=np.stack(
                            [
                                active_eligible_df["n_frames_h1"],
                                active_eligible_df[col_frames_h1],
                                active_eligible_df["coverage_pct_h1"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>%{x} · 1st Half</b><br>"
                            "Frame Share: %{y:.2f}%<br>"
                            "Target Frames: %{customdata[1]} / %{customdata[0]} observed<br>"
                            "Half Coverage: %{customdata[2]:.1f}%<extra></extra>"
                        ),
                    )
                )

                bar_fig.add_trace(
                    go.Bar(
                        name="2nd Half Share (%)",
                        x=x_labels,
                        y=active_eligible_df[col_h2],
                        marker_color="#f58518",
                        text=[f"{v:.2f}%" for v in active_eligible_df[col_h2]],
                        textposition="auto",
                        customdata=np.stack(
                            [
                                active_eligible_df["n_frames_h2"],
                                active_eligible_df[col_frames_h2],
                                active_eligible_df["coverage_pct_h2"],
                                active_eligible_df[col_diff_pp],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>%{x} · 2nd Half</b><br>"
                            "Frame Share: %{y:.2f}%<br>"
                            "Target Frames: %{customdata[1]} / %{customdata[0]} observed<br>"
                            "Half Coverage: %{customdata[2]:.1f}%<br>"
                            "Change (H2 − H1): %{customdata[3]:+.2f} pp<extra></extra>"
                        ),
                    )
                )

                bar_fig.update_layout(
                    barmode="group",
                    xaxis_title="Player ID (H1 / H2 Coverage %)",
                    yaxis_title=f"{metric_title} (% of observed frames)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=20, t=40, b=60),
                    height=430,
                )
                st.plotly_chart(bar_fig, width="stretch")

            with col_right:
                st.markdown(f"#### 1st Half vs 2nd Half Parity Scatter")
                st.caption(
                    "Points **below** the dashed `y = x` line had a lower sampled frame share in H2; "
                    "points **above** increased their share in H2."
                )

                scat_fig = go.Figure()
                max_val = float(
                    max(
                        active_eligible_df[col_h1].max(),
                        active_eligible_df[col_h2].max(),
                        0.5,
                    )
                ) * 1.18

                # Parity diagonal y = x
                scat_fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode="lines",
                        line=dict(color="#888888", dash="dash", width=1.5),
                        name="Equal Share (H2 = H1)",
                        hoverinfo="skip",
                    )
                )

                for tm in ("away", "home"):
                    tm_df = active_eligible_df[active_eligible_df["team"] == tm]
                    if tm_df.empty:
                        continue
                    rel_strings = [
                        f"{r[col_rel_pct]:+.1f}%" if pd.notna(r[col_rel_pct]) else r[col_rel_note]
                        for _, r in tm_df.iterrows()
                    ]
                    scat_fig.add_trace(
                        go.Scatter(
                            x=tm_df[col_h1],
                            y=tm_df[col_h2],
                            mode="markers+text",
                            name=f"{tm.upper()} ({tm})",
                            text=tm_df["player_id"],
                            textposition="top center",
                            marker=dict(
                                size=13,
                                color=an.TEAM_COLOURS[tm],
                                line=dict(width=1.5, color="white"),
                            ),
                            customdata=np.stack(
                                [
                                    tm_df[col_diff_pp],
                                    rel_strings,
                                    tm_df["coverage_pct_h1"],
                                    tm_df["coverage_pct_h2"],
                                ],
                                axis=-1,
                            ),
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "1st Half: %{x:.2f}% (cov %{customdata[2]:.1f}%)<br>"
                                "2nd Half: %{y:.2f}% (cov %{customdata[3]:.1f}%)<br>"
                                "Diff (pp): %{customdata[0]} pp<br>"
                                "Rel Change (%): %{customdata[1]}<extra></extra>"
                            ),
                        )
                    )

                scat_fig.update_layout(
                    xaxis=dict(title=f"1st Half {metric_title} (%)", range=[-0.05, max_val]),
                    yaxis=dict(title=f"2nd Half {metric_title} (%)", range=[-0.05, max_val]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=20, t=40, b=60),
                    height=430,
                )
                st.plotly_chart(scat_fig, width="stretch")

            # Percentage-Point (pp) vs Relative Percent (%) Comparison Bar
            st.markdown("#### Percentage-Point Change (`pp`) vs Relative Percent Change (`%`)")
            st.caption(
                "**Why distinguish `pp` from `%`?** Small first-half baselines produce huge relative percentages "
                "(e.g. `away_12` rose from `0.67%` to `0.92%` high-speed share: `+0.24 pp`, which is `+35.8%` relative; "
                "and for sprints `away_12` went from `0.00%` to `0.18%` (`+0.18 pp`), where relative `%` change has a **zero denominator** and is undefined)."
            )

            diff_sorted = active_eligible_df.sort_values(col_diff_pp, ascending=True)
            pp_fig = go.Figure()
            bar_colors = [an.TEAM_COLOURS.get(t, "#555555") for t in diff_sorted["team"]]
            bar_labels = [
                f"{row[col_diff_pp]:+.2f} pp ({row[col_rel_pct]:+.1f}%)"
                if pd.notna(row[col_rel_pct])
                else f"{row[col_diff_pp]:+.2f} pp ({row[col_rel_note]})"
                for _, row in diff_sorted.iterrows()
            ]
            pp_fig.add_trace(
                go.Bar(
                    x=diff_sorted[col_diff_pp],
                    y=diff_sorted["player_id"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=bar_labels,
                    textposition="outside",
                    hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
                )
            )
            max_abs_pp = float(diff_sorted[col_diff_pp].abs().max() or 0.5) * 1.35
            pp_fig.update_layout(
                xaxis=dict(title=f"Change in {metric_title} (H2 − H1, percentage points pp)", range=[-max_abs_pp, max_abs_pp]),
                yaxis=dict(title="Player ID"),
                margin=dict(l=40, r=60, t=20, b=40),
                height=max(280, 32 * len(diff_sorted)),
            )
            pp_fig.add_vline(x=0, line_width=1.5, line_dash="solid", line_color="#333333")
            st.plotly_chart(pp_fig, width="stretch")

    # =========================================================================
    # TAB 2: WITHIN-HALF MOVEMENT & COMPACTNESS TRENDS
    # =========================================================================
    with tab_trends:
        st.markdown("#### Within-Half Intensity & Tactical Shape Progression (Split by Half)")
        st.caption(
            "Tracking is segmented strictly within **Half 1** (video `5:51–53:30`) and **Half 2** (video `59:45–108:06`) "
            "— never interpolated across the half-time interval. "
            "Notice how `away` drops deeper (`x_att` shifts from `-2.7 m` in H1 to `-5.0 m` in H2) after scoring 3 early second-half goals, "
            "while `home` pushes higher (`-2.4 m` to `-1.3 m`) and narrows width (`47.3 m` to `38.1 m`) chasing the game."
        )

        t_ctrl1, t_ctrl2 = st.columns([1, 2])
        with t_ctrl1:
            bin_mins = st.selectbox(
                "Time Bin Width (Minutes)",
                options=[5, 10, 15],
                index=0,
                format_func=lambda b: f"{b}-Minute Bins Within Each Half",
            )
        with t_ctrl2:
            cohort_mode = st.radio(
                "Intensity Trend Cohort",
                options=["eligible", "all"],
                index=0,
                horizontal=True,
                format_func=lambda c: (
                    f"Selected Eligible Players Only (n={len(active_eligible_df)}) — controls for fresh substitutes"
                    if c == "eligible"
                    else "All Active Outfield Players on Pitch (includes 2nd-half substitutes)"
                ),
            )

        trend_pids = active_eligible_df["player_id"].tolist() if cohort_mode == "eligible" else None
        if cohort_mode == "eligible" and not trend_pids:
            st.warning("No eligible players selected. Showing team shape only.")

        trends_df, goal_markers = an.compute_half_period_trends(
            players=players,
            events=events,
            bin_minutes=int(bin_mins),
            eligible_player_ids=trend_pids,
            include_goalkeepers=include_gk,
        )

        trend_fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=False,
            subplot_titles=(
                f"1st Half · {metric_title}",
                f"2nd Half · {metric_title} (6 Goals Marked)",
                "1st Half · Mean Attacking Depth x_att (m)",
                "2nd Half · Mean Attacking Depth x_att (m)",
            ),
            vertical_spacing=0.18,
            horizontal_spacing=0.08,
        )

        for half_idx in (1, 2):
            h_trends = trends_df[trends_df["half"] == half_idx]
            for tm in ("away", "home"):
                if team_choice in ("away", "home") and tm != team_choice:
                    continue
                tm_sub = h_trends[h_trends["team"] == tm].sort_values("bin_start_min")
                if tm_sub.empty:
                    continue

                show_leg = half_idx == 1
                # Row 1: Intensity share
                trend_fig.add_trace(
                    go.Scatter(
                        x=tm_sub["period_label"],
                        y=tm_sub[trend_col],
                        mode="lines+markers",
                        name=f"{tm.upper()} ({tm})",
                        legendgroup=tm,
                        showlegend=show_leg,
                        line=dict(color=an.TEAM_COLOURS[tm], width=2.5),
                        marker=dict(size=8),
                        customdata=np.stack(
                            [
                                tm_sub["min_video_min"].fillna(0),
                                tm_sub["max_video_min"].fillna(0),
                                tm_sub["player_frames"].fillna(0),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            f"<b>{tm.upper()} · %{{x}}</b><br>"
                            "Share: %{y:.2f}%<br>"
                            "Video clock: %{customdata[0]:.1f}–%{customdata[1]:.1f} min<br>"
                            "Observed player-frames: %{customdata[2]:.0f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=half_idx,
                )

                # Row 2: Attacking depth (x_att)
                trend_fig.add_trace(
                    go.Scatter(
                        x=tm_sub["period_label"],
                        y=tm_sub["mean_x_att_m"],
                        mode="lines+markers",
                        name=f"{tm.upper()} Depth",
                        legendgroup=tm,
                        showlegend=False,
                        line=dict(color=an.TEAM_COLOURS[tm], width=2.5, dash="dot"),
                        marker=dict(size=8, symbol="diamond"),
                        customdata=np.stack(
                            [tm_sub["mean_length_m"], tm_sub["mean_width_m"]],
                            axis=-1,
                        ),
                        hovertemplate=(
                            f"<b>{tm.upper()} · %{{x}}</b><br>"
                            "Mean Attacking Depth (x_att): %{y:+.1f} m<br>"
                            "Team Length: %{customdata[0]:.1f} m · Width: %{customdata[1]:.1f} m<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=half_idx,
                )

        # Annotate the 6 second-half goals from event_name == 'goal'
        if not goal_markers.empty:
            for _, grow in goal_markers.iterrows():
                g_team = grow["team"]
                g_label = grow["period_label"]
                trend_fig.add_vline(
                    x=g_label,
                    line_width=1,
                    line_dash="dash",
                    line_color=an.TEAM_COLOURS.get(g_team, "#333"),
                    row=1,
                    col=2,
                )

        trend_fig.update_yaxes(title_text="Frame Share (%)", row=1, col=1)
        trend_fig.update_yaxes(title_text="Frame Share (%)", row=1, col=2)
        trend_fig.update_yaxes(title_text="Mean x_att (m, + = forward)", row=2, col=1)
        trend_fig.update_yaxes(title_text="Mean x_att (m, + = forward)", row=2, col=2)
        trend_fig.update_layout(
            height=580,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        )
        st.plotly_chart(trend_fig, width="stretch")

        # Compactness & Goal Event Log
        c_shape, c_goals = st.columns([1.1, 1.0])
        with c_shape:
            st.markdown("##### Outfield Team Shape Averages by Half (`attack_dir` Normalized)")
            out_only = players[players["role"] == "outfield"]
            f_shape = (
                out_only.groupby(["half", "team", "timestamp_ms"], as_index=False)
                .agg(
                    length_m=("x_att", lambda s: float(s.max() - s.min())),
                    width_m=("y_att", lambda s: float(s.max() - s.min())),
                    mean_x_att_m=("x_att", "mean"),
                )
                .groupby(["team", "half"], as_index=False)
                .agg(
                    mean_length_m=("length_m", "mean"),
                    mean_width_m=("width_m", "mean"),
                    mean_x_att_m=("mean_x_att_m", "mean"),
                )
            )
            f_shape_disp = f_shape.rename(
                columns={
                    "team": "Team",
                    "half": "Half",
                    "mean_length_m": "Mean Length (m)",
                    "mean_width_m": "Mean Width (m)",
                    "mean_x_att_m": "Mean Attacking Depth x_att (m)",
                }
            )
            st.dataframe(
                f_shape_disp.style.format(
                    {
                        "Mean Length (m)": "{:.1f}",
                        "Mean Width (m)": "{:.1f}",
                        "Mean Attacking Depth x_att (m)": "{:+.1f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        with c_goals:
            st.markdown("##### Match Goal Timeline (`event_name == 'goal'`, all 6 in Half 2)")
            if not goal_markers.empty:
                g_disp = goal_markers[
                    ["video_minute", "half_elapsed_min", "team", "player_id", "period_label", "nearest_frame_ms"]
                ].rename(
                    columns={
                        "video_minute": "Video Min",
                        "half_elapsed_min": "H2 Elapsed Min",
                        "team": "Team",
                        "player_id": "Scorer",
                        "period_label": "Half Bin",
                        "nearest_frame_ms": "Nearest Frame (ms)",
                    }
                )
                st.dataframe(
                    g_disp.style.format({"Video Min": "{:.2f}", "H2 Elapsed Min": "{:.1f}"}),
                    width="stretch",
                    hide_index=True,
                )

    # =========================================================================
    # TAB 3: PLAYER PITCH FOOTPRINT (ATTACK-DIRECTION NORMALIZED)
    # =========================================================================
    with tab_pitch:
        st.markdown("#### Player Half-by-Half Pitch Footprint & High-Speed Locations (`x_att`, `y_att`)")
        st.caption(
            "Coordinates are normalized by `attack_dir` (`x_att = x * attack_dir`, `y_att = y * attack_dir`) "
            "so **both halves attack Left → Right (`+x`)**. "
            "Shows where a player operated in each half and highlights high-speed (`speed_zone == 3`) and sprint (`speed_zone == 4`) frames."
        )

        all_player_options = comp_df["player_id"].tolist() if not comp_df.empty else []
        if not all_player_options:
            st.info("No players available under current filter.")
        else:
            default_idx = 0
            # Default to away_8 or home_10 if present
            for preferred in ("away_8", "home_10", "away_2"):
                if preferred in all_player_options:
                    default_idx = all_player_options.index(preferred)
                    break

            focus_pid = st.selectbox(
                "Select Player to Inspect Spatial Movement Across Halves",
                options=all_player_options,
                index=default_idx,
                format_func=lambda pid: (
                    f"{pid} · {comp_df.loc[comp_df['player_id'] == pid, 'eligibility_status'].iloc[0]} "
                    f"(H1 cov {comp_df.loc[comp_df['player_id'] == pid, 'coverage_pct_h1'].iloc[0]:.1f}%, "
                    f"H2 cov {comp_df.loc[comp_df['player_id'] == pid, 'coverage_pct_h2'].iloc[0]:.1f}%)"
                ),
            )

            prow = comp_df[comp_df["player_id"] == focus_pid].iloc[0]
            spatial = an.get_player_half_spatial_data(players, focus_pid)

            pitch_fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    f"1st Half · {focus_pid} (Cov: {prow['coverage_pct_h1']:.1f}%, n={int(prow['n_frames_h1'])})",
                    f"2nd Half · {focus_pid} (Cov: {prow['coverage_pct_h2']:.1f}%, n={int(prow['n_frames_h2'])})",
                ),
                horizontal_spacing=0.06,
            )

            for h_idx in (1, 2):
                draw_plotly_pitch(pitch_fig, row=1, col=h_idx)
                h_df = spatial[h_idx]
                if h_df.empty:
                    pitch_fig.add_annotation(
                        x=0,
                        y=0,
                        text=f"No observed frames in Half {h_idx}",
                        showarrow=False,
                        font=dict(size=14, color="#666666"),
                        row=1,
                        col=h_idx,
                    )
                    continue

                # Low/moderate speed frames (decimated every 3rd frame for clean rendering)
                low_df = h_df[h_df["speed_zone"] < 3].iloc[::3]
                z3_df = h_df[h_df["speed_zone"] == 3]
                z4_df = h_df[h_df["speed_zone"] == 4]

                show_leg = h_idx == 1
                pitch_fig.add_trace(
                    go.Scatter(
                        x=low_df["x_att"],
                        y=low_df["y_att"],
                        mode="markers",
                        name="Zones 0–2 (< 20 km/h)",
                        legendgroup="z02",
                        showlegend=show_leg,
                        marker=dict(size=4, color="rgba(120, 130, 140, 0.28)"),
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=h_idx,
                )

                pitch_fig.add_trace(
                    go.Scatter(
                        x=z3_df["x_att"],
                        y=z3_df["y_att"],
                        mode="markers",
                        name="Zone 3 (20–25 km/h, High-Speed)",
                        legendgroup="z3",
                        showlegend=show_leg,
                        marker=dict(size=7, color="#f58518", opacity=0.85),
                        customdata=z3_df["timestamp_ms"] / 60000.0,
                        hovertemplate="Zone 3 (20–25 km/h)<br>x_att: %{x:.1f} m, y_att: %{y:.1f} m<br>Video min: %{customdata:.1f}'<extra></extra>",
                    ),
                    row=1,
                    col=h_idx,
                )

                pitch_fig.add_trace(
                    go.Scatter(
                        x=z4_df["x_att"],
                        y=z4_df["y_att"],
                        mode="markers",
                        name="Zone 4 (25+ km/h, Sprint)",
                        legendgroup="z4",
                        showlegend=show_leg,
                        marker=dict(size=9, color="#e41a1c", symbol="star", line=dict(width=1, color="black")),
                        customdata=z4_df["timestamp_ms"] / 60000.0,
                        hovertemplate="Zone 4 (≥25 km/h Sprint)<br>x_att: %{x:.1f} m, y_att: %{y:.1f} m<br>Video min: %{customdata:.1f}'<extra></extra>",
                    ),
                    row=1,
                    col=h_idx,
                )

                # Mean centroid marker
                mx = float(h_df["x_att"].mean())
                my = float(h_df["y_att"].mean())
                pitch_fig.add_trace(
                    go.Scatter(
                        x=[mx],
                        y=[my],
                        mode="markers+text",
                        name="Half Centroid (Mean x_att, y_att)",
                        legendgroup="centroid",
                        showlegend=show_leg,
                        text=[f"Mean ({mx:+.1f}m, {my:+.1f}m)"],
                        textposition="top center",
                        marker=dict(
                            size=14,
                            color=an.TEAM_COLOURS.get(prow["team"], "#1f77b4"),
                            symbol="cross",
                            line=dict(width=2, color="white"),
                        ),
                        hovertemplate=f"<b>Half {h_idx} Centroid</b><br>x_att: {mx:+.1f} m<br>y_att: {my:+.1f} m<extra></extra>",
                    ),
                    row=1,
                    col=h_idx,
                )

            pitch_fig.update_xaxes(
                title_text="Attacking Direction x_att (m, Attacking → +52.5 m)",
                range=[-56, 56],
                constrain="domain",
            )
            pitch_fig.update_yaxes(
                title_text="Pitch Width y_att (m, Left Flank = +y)",
                range=[-37, 37],
                scaleanchor="x",
                scaleratio=1,
            )
            pitch_fig.update_layout(
                height=480,
                margin=dict(l=30, r=30, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
            )
            st.plotly_chart(pitch_fig, width="stretch")

            # Speed zone distribution table for focus player
            z_rows = []
            for z in range(5):
                z1_val = prow[f"z{z}_share_pct_h1"]
                z2_val = prow[f"z{z}_share_pct_h2"]
                diff_z = z2_val - z1_val if (pd.notna(z1_val) and pd.notna(z2_val)) else np.nan
                z_rows.append(
                    {
                        "Speed Zone": an.SPEED_ZONE_LABELS[z],
                        "1st Half Share (%)": f"{z1_val:.2f}%" if pd.notna(z1_val) else "Absent",
                        "2nd Half Share (%)": f"{z2_val:.2f}%" if pd.notna(z2_val) else "Absent",
                        "Change (pp)": format_Signed(diff_z, 2, " pp"),
                    }
                )
            st.markdown(f"##### Speed Zone Breakdown for `{focus_pid}` (% of Observed Frames)")
            st.dataframe(pd.DataFrame(z_rows), width="stretch", hide_index=True)

    # =========================================================================
    # TAB 4: PER-PLAYER CHANGE & ELIGIBILITY TABLE
    # =========================================================================
    with tab_table:
        st.markdown("#### Per-Player Half Comparison & Coverage Eligibility Table")
        st.caption(
            "Explicitly separates **percentage-point (`pp`)** changes from **relative percent (`%`)** changes, "
            "flags **zero denominators** (`0% in H1`), and marks **absent comparisons** when a player only appeared in one half."
        )

        table_source = comp_df if show_ineligible_in_table else active_eligible_df
        if table_source.empty:
            st.info("No rows to display for the current filter selection.")
        else:
            display_rows = []
            for _, r in table_source.iterrows():
                both = bool(r["both_halves_observed"])
                hs_rel_str = (
                    f"{r['hs_rel_change_pct']:+.1f}%"
                    if pd.notna(r["hs_rel_change_pct"])
                    else str(r["hs_rel_note"])
                )
                sp_rel_str = (
                    f"{r['sprint_rel_change_pct']:+.1f}%"
                    if pd.notna(r["sprint_rel_change_pct"])
                    else str(r["sprint_rel_note"])
                )
                display_rows.append(
                    {
                        "Player": r["player_id"],
                        "Team": r["team"],
                        "Role": r["role"],
                        "Eligible": "✅ Yes" if r["is_eligible"] else "⚠️ No",
                        "Eligibility / Coverage Note": r["eligibility_status"],
                        "H1 Frames (Cov %)": f"{int(r['n_frames_h1'])} ({r['coverage_pct_h1']:.1f}%)",
                        "H2 Frames (Cov %)": f"{int(r['n_frames_h2'])} ({r['coverage_pct_h2']:.1f}%)",
                        "H1 High-Speed (%)": f"{r['hs_share_pct_h1']:.2f}%" if pd.notna(r["hs_share_pct_h1"]) else "Absent",
                        "H2 High-Speed (%)": f"{r['hs_share_pct_h2']:.2f}%" if pd.notna(r["hs_share_pct_h2"]) else "Absent",
                        "Δ High-Speed (pp)": format_Signed(r["hs_diff_pp"], 2, " pp") if both else "N/A (1 half)",
                        "Δ High-Speed (Rel %)": hs_rel_str,
                        "H1 Sprint (%)": f"{r['sprint_share_pct_h1']:.2f}%" if pd.notna(r["sprint_share_pct_h1"]) else "Absent",
                        "H2 Sprint (%)": f"{r['sprint_share_pct_h2']:.2f}%" if pd.notna(r["sprint_share_pct_h2"]) else "Absent",
                        "Δ Sprint (pp)": format_Signed(r["sprint_diff_pp"], 2, " pp") if both else "N/A (1 half)",
                        "Δ Sprint (Rel %)": sp_rel_str,
                        "H1 Dist (km)": f"{r['dist_km_h1']:.2f}" if r["n_frames_h1"] > 0 else "0.00",
                        "H2 Dist (km)": f"{r['dist_km_h2']:.2f}" if r["n_frames_h2"] > 0 else "0.00",
                        "Δ Mean x_att (m)": format_Signed(r["x_att_shift_m"], 1, " m") if both else "N/A",
                    }
                )
            st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True, height=520)


if __name__ == "__main__":
    main()
