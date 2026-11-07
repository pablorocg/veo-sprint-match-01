"""Streamlit dashboard prototype: Defensive Shape Explorer (Port 8503)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics

st.set_page_config(
    page_title="Veo Defensive Shape Explorer",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(show_spinner="Loading tracking and event datasets...")
def get_cached_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw datasets and precompute frame-by-frame outfield shape metrics once."""
    summary_df, events_df, players_df = analytics.load_raw_datasets(".")
    shape_df = analytics.compute_team_shape_timeseries(players_df)
    return summary_df, events_df, players_df, shape_df


def build_pitch_figure(
    details: dict,
    show_hull: bool = True,
    show_bbox: bool = True,
    show_max_gap: bool = True,
    show_opponents: bool = True,
) -> go.Figure:
    """Create an interactive 105 x 67.7 m pitch plot with defensive shape overlays."""
    fig = go.Figure()

    half_l = analytics.PITCH_L / 2.0  # 52.5
    half_w = analytics.PITCH_W / 2.0  # 33.85

    # Pitch markings (white lines on green turf)
    line_style = dict(color="rgba(255, 255, 255, 0.85)", width=2)
    pitch_shapes = [
        # Outer boundary
        dict(type="rect", x0=-half_l, y0=-half_w, x1=half_l, y1=half_w, line=line_style, layer="below"),
        # Halfway line
        dict(type="line", x0=0, y0=-half_w, x1=0, y1=half_w, line=line_style, layer="below"),
        # Center circle (r = 9.15m)
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line_style, layer="below"),
        # Left penalty area (16.5m x 40.32m)
        dict(type="rect", x0=-half_l, y0=-20.16, x1=-half_l + 16.5, y1=20.16, line=line_style, layer="below"),
        # Right penalty area
        dict(type="rect", x0=half_l - 16.5, y0=-20.16, x1=half_l, y1=20.16, line=line_style, layer="below"),
        # Left 6-yard box (5.5m x 18.32m)
        dict(type="rect", x0=-half_l, y0=-9.16, x1=-half_l + 5.5, y1=9.16, line=line_style, layer="below"),
        # Right 6-yard box
        dict(type="rect", x0=half_l - 5.5, y0=-9.16, x1=half_l, y1=9.16, line=line_style, layer="below"),
    ]

    metrics = details.get("shape_metrics", {})
    def_team = details["defending_team"]
    opp_team = details["opposing_team"]
    def_color = analytics.TEAM_COLORS.get(def_team, "#1f77b4")
    opp_color = analytics.TEAM_COLORS.get(opp_team, "#d62728")

    # 1. Largest longitudinal gap band
    if (
        show_max_gap
        and metrics
        and pd.notna(metrics.get("max_x_gap_m"))
        and pd.notna(metrics.get("gap_x_start"))
        and pd.notna(metrics.get("gap_x_end"))
    ):
        gx0 = float(metrics["gap_x_start"])
        gx1 = float(metrics["gap_x_end"])
        ym0 = float(metrics.get("y_min", -25.0))
        ym1 = float(metrics.get("y_max", 25.0))
        fig.add_trace(
            go.Scatter(
                x=[gx0, gx1, gx1, gx0, gx0],
                y=[ym0, ym0, ym1, ym1, ym0],
                fill="toself",
                fillcolor="rgba(255, 193, 7, 0.25)",
                line=dict(color="rgba(255, 193, 7, 0.9)", width=1.5, dash="dot"),
                name=f"Max Longitudinal Gap ({metrics['max_x_gap_m']:.1f} m)",
                hoverinfo="name",
            )
        )
        fig.add_annotation(
            x=(gx0 + gx1) / 2.0,
            y=ym1 + 2.2,
            text=f"Max x-gap: {metrics['max_x_gap_m']:.1f} m ({metrics.get('gap_player_low_x', '')} ↔ {metrics.get('gap_player_high_x', '')})",
            showarrow=False,
            font=dict(color="#ffe082", size=11, family="sans-serif"),
            bgcolor="rgba(20, 30, 20, 0.75)",
            borderpad=3,
        )

    # 2. Convex hull overlay
    hull_verts = details.get("hull_vertices", np.empty((0, 2)))
    if show_hull and len(hull_verts) >= 3:
        fig.add_trace(
            go.Scatter(
                x=hull_verts[:, 0],
                y=hull_verts[:, 1],
                fill="toself",
                fillcolor=(
                    "rgba(31, 119, 180, 0.22)"
                    if def_team == "home"
                    else "rgba(214, 39, 40, 0.22)"
                ),
                line=dict(color=def_color, width=2.5),
                name=f"Outfield Convex Hull ({details.get('hull_area_m2', 0.0):.0f} m²)",
                hoverinfo="name",
            )
        )

    # 3. Bounding box overlay (Width x Depth)
    if show_bbox and metrics and pd.notna(metrics.get("x_min")):
        x0, x1 = float(metrics["x_min"]), float(metrics["x_max"])
        y0, y1 = float(metrics["y_min"]), float(metrics["y_max"])
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                line=dict(color="#ffffff", width=1.5, dash="dash"),
                name=f"Bounding Box ({metrics['depth_m']:.1f} m × {metrics['width_m']:.1f} m)",
                hoverinfo="name",
            )
        )
        # Dimension annotations for Depth and Width
        fig.add_annotation(
            x=(x0 + x1) / 2.0,
            y=y0 - 2.6,
            text=f"Depth (x-span): {metrics['depth_m']:.1f} m",
            showarrow=False,
            font=dict(color="#ffffff", size=11),
            bgcolor="rgba(0, 0, 0, 0.65)",
            borderpad=2,
        )
        fig.add_annotation(
            x=x1 + 2.0,
            y=(y0 + y1) / 2.0,
            text=f"Width: {metrics['width_m']:.1f} m",
            textangle=90,
            showarrow=False,
            font=dict(color="#ffffff", size=11),
            bgcolor="rgba(0, 0, 0, 0.65)",
            borderpad=2,
        )

    # 4. Opposing players
    opp_df = details.get("opp_players", pd.DataFrame())
    if show_opponents and not opp_df.empty:
        fig.add_trace(
            go.Scatter(
                x=opp_df["x"],
                y=opp_df["y"],
                mode="markers+text",
                text=opp_df["player_number"],
                textposition="middle center",
                textfont=dict(color="white", size=9),
                marker=dict(
                    size=18,
                    color=opp_color,
                    opacity=0.75,
                    line=dict(color="white", width=1.2),
                ),
                customdata=np.stack([opp_df["player_id"], opp_df["role"]], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "x: %{x:.1f} m, y: %{y:.1f} m<extra>Opponent</extra>"
                ),
                name=f"Opponent ({opp_team.capitalize()})",
            )
        )

    # 5. Defending goalkeeper
    gk_df = details.get("def_gk", pd.DataFrame())
    if not gk_df.empty:
        fig.add_trace(
            go.Scatter(
                x=gk_df["x"],
                y=gk_df["y"],
                mode="markers+text",
                text=["GK"],
                textposition="middle center",
                textfont=dict(color="black", size=9),
                marker=dict(
                    size=20,
                    symbol="diamond",
                    color="#aee2ff" if def_team == "home" else "#ffb3b3",
                    line=dict(color="white", width=2),
                ),
                customdata=gk_df[["player_id"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (Goalkeeper - excluded from outfield shape)<br>"
                    "x: %{x:.1f} m, y: %{y:.1f} m<extra>Defending GK</extra>"
                ),
                name=f"{def_team.capitalize()} Goalkeeper (excl. from shape)",
            )
        )

    # 6. Defending outfield players + Team Centroid
    def_df = details.get("def_outfield", pd.DataFrame())
    if not def_df.empty:
        fig.add_trace(
            go.Scatter(
                x=def_df["x"],
                y=def_df["y"],
                mode="markers+text",
                text=def_df["player_number"],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                marker=dict(
                    size=23,
                    color=def_color,
                    line=dict(color="white", width=2.2),
                ),
                customdata=def_df[["player_id", "speed_zone"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (Outfield)<br>"
                    "x: %{x:.1f} m, y: %{y:.1f} m<br>"
                    "Speed zone: %{customdata[1]}<extra>Defending Outfield</extra>"
                ),
                name=f"{def_team.capitalize()} Outfield ({len(def_df)} observed)",
            )
        )

    if metrics and pd.notna(metrics.get("centroid_x")):
        fig.add_trace(
            go.Scatter(
                x=[metrics["centroid_x"]],
                y=[metrics["centroid_y"]],
                mode="markers",
                marker=dict(
                    size=13,
                    symbol="x",
                    color="#fff176",
                    line=dict(color="black", width=1.5),
                ),
                hovertemplate=(
                    "<b>Outfield Centroid</b><br>"
                    "x: %{x:.1f} m, y: %{y:.1f} m<extra></extra>"
                ),
                name="Outfield Centroid",
            )
        )

    # 7. Ball position (if observed in frame)
    ball_x = details.get("ball_x", np.nan)
    ball_y = details.get("ball_y", np.nan)
    if pd.notna(ball_x) and pd.notna(ball_y):
        fig.add_trace(
            go.Scatter(
                x=[ball_x],
                y=[ball_y],
                mode="markers",
                marker=dict(
                    size=14,
                    symbol="circle",
                    color="#ffd700",
                    line=dict(color="#000000", width=2),
                ),
                hovertemplate="<b>Ball</b><br>x: %{x:.1f} m, y: %{y:.1f} m<extra></extra>",
                name="Ball",
            )
        )

    # Attacking & Defending direction indicator at top of pitch
    attack_dir = int(metrics.get("attack_dir", 1)) if metrics else 1
    own_goal_x = -half_l if attack_dir == 1 else half_l
    opp_goal_x = half_l if attack_dir == 1 else -half_l
    arrow_text = (
        f"◀ Defending Own Goal (x = {own_goal_x:+.1f} m)   |   "
        f"{def_team.capitalize()} Attacking Direction ({'→ +x' if attack_dir == 1 else '← -x'}) ▶"
        if attack_dir == 1
        else f"◀ {def_team.capitalize()} Attacking Direction (← -x)   |   "
        f"Defending Own Goal (x = {own_goal_x:+.1f} m) ▶"
    )
    fig.add_annotation(
        x=0,
        y=half_w + 3.2,
        text=arrow_text,
        showarrow=False,
        font=dict(color="#e0e0e0", size=12),
        bgcolor="rgba(15, 25, 15, 0.85)",
        borderpad=4,
    )

    fig.update_layout(
        shapes=pitch_shapes,
        plot_bgcolor="#234d20",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Pitch Longitudinal Coordinate x (m)",
            range=[-half_l - 5.5, half_l + 5.5],
            showgrid=False,
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="Pitch Lateral Coordinate y (m)",
            range=[-half_w - 5.5, half_w + 5.5],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1.0,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=35, b=65),
        height=520,
    )
    return fig


def build_half_timeline_figure(
    half_shape_df: pd.DataFrame,
    events_df: pd.DataFrame,
    half: int,
    defending_team: str,
    selected_metric: str,
    threshold_val: float,
    min_outfield: int,
    selected_ts: int,
) -> go.Figure:
    """Build a half-specific timeline chart of the selected metric plus Width & Depth trends."""
    metric_info = analytics.METRIC_OPTIONS[selected_metric]
    unit = metric_info["unit"]
    def_color = analytics.TEAM_COLORS.get(defending_team, "#1f77b4")

    valid_df = half_shape_df[half_shape_df["n_outfield"] >= min_outfield].copy()
    gapped_df = analytics.split_timeseries_on_gaps(valid_df, max_gap_ms=800)

    fig = go.Figure()

    # Plot main selected metric
    fig.add_trace(
        go.Scatter(
            x=gapped_df["video_seconds"],
            y=gapped_df[selected_metric],
            mode="lines",
            line=dict(color=def_color, width=2),
            customdata=np.stack(
                [
                    gapped_df["video_clock"].fillna(""),
                    gapped_df["n_outfield"].fillna(0),
                    gapped_df["width_m"].fillna(0),
                    gapped_df["depth_m"].fillna(0),
                    gapped_df["max_x_gap_m"].fillna(0),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Video Time: %{customdata[0]} (%{x:.1f}s)</b><br>"
                f"{metric_info['short_label']}: %{{y:.2f}} {unit}<br>"
                "Width: %{customdata[2]:.1f} m | Depth: %{customdata[3]:.1f} m<br>"
                "Max x-gap: %{customdata[4]:.1f} m | Outfielders: %{customdata[1]:.0f}<extra></extra>"
            ),
            name=f"{metric_info['short_label']} ({unit})",
        )
    )

    # Also show companion Width / Depth traces if a single metre metric is chosen
    if selected_metric in ("width_m", "depth_m", "max_x_gap_m", "compactness_m"):
        companion_col = "depth_m" if selected_metric == "width_m" else "width_m"
        companion_label = "Team Depth (m)" if companion_col == "depth_m" else "Team Width (m)"
        fig.add_trace(
            go.Scatter(
                x=gapped_df["video_seconds"],
                y=gapped_df[companion_col],
                mode="lines",
                line=dict(color="rgba(160, 160, 160, 0.55)", width=1.3),
                name=f"{companion_label} (reference)",
                hoverinfo="skip",
            )
        )

    # Highlight points exceeding the adjustable threshold
    exceed_df = valid_df[valid_df[selected_metric] >= threshold_val]
    if not exceed_df.empty:
        fig.add_trace(
            go.Scatter(
                x=exceed_df["video_seconds"],
                y=exceed_df[selected_metric],
                mode="markers",
                marker=dict(color="#ff9800", size=5, opacity=0.8),
                name=f"Above Threshold (≥ {threshold_val:.2f} {unit})",
                hoverinfo="skip",
            )
        )

    # Adjustable threshold horizontal line
    fig.add_hline(
        y=threshold_val,
        line_dash="dash",
        line_color="#ff9800",
        annotation_text=f"Alert Threshold: {threshold_val:.2f} {unit}",
        annotation_position="top left",
    )

    # Selected moment vertical cursor
    sel_sec = selected_ts / 1000.0
    fig.add_vline(
        x=sel_sec,
        line_width=2.5,
        line_color="#00e676",
        annotation_text=f"Selected: {analytics.format_video_time(selected_ts, include_seconds=False)}",
        annotation_position="top right",
    )

    # Mark match goals in this half (using event_name == 'goal', the complete goal record)
    half_goals = events_df[
        (events_df["half"] == int(half)) & (events_df["event_name"] == "goal")
    ]
    for _, goal_row in half_goals.iterrows():
        g_sec = float(goal_row["timestamp_ms"]) / 1000.0
        g_team = str(goal_row["team"]).capitalize()
        fig.add_vline(
            x=g_sec,
            line_dash="dot",
            line_color="#ffd700",
            opacity=0.8,
            annotation_text=f"⚽ {g_team} Goal ({analytics.format_video_time(goal_row['timestamp_ms'], False)})",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"Half {half} Timeline — {analytics.TEAM_LABELS[defending_team]} ({metric_info['label']})",
        xaxis_title="Video Timestamp (seconds, s)",
        yaxis_title=f"{metric_info['short_label']} ({unit})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=55, b=35),
        height=310,
    )
    return fig


def main() -> None:
    summary_df, events_df, players_df, shape_df = get_cached_data()

    st.title("🛡️ Defensive Shape Explorer")
    st.caption(
        "Inspect outfield team width, depth, compactness, and longitudinal line gaps from 2.5 Hz tracking data. "
        "All spatial metrics are in metres (m) on a 105 × 67.7 m pitch; timestamps are video clock time."
    )

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("1. Team & Period")
        defending_team = st.selectbox(
            "Defending Team",
            options=["away", "home"],
            index=0,
            format_func=lambda t: analytics.TEAM_LABELS[t],
            help="Select which team's outfield defensive shape to analyze.",
        )
        half = st.radio(
            "Match Half",
            options=[1, 2],
            index=1,
            format_func=lambda h: f"Half {h}",
            horizontal=True,
            help="Halves are analyzed independently; tracking is never interpolated across half-time.",
        )

        st.header("2. Metric & Coverage Rules")
        selected_metric = st.selectbox(
            "Shape Metric (for Ranking & Alerting)",
            options=list(analytics.METRIC_OPTIONS.keys()),
            index=0,
            format_func=lambda k: analytics.METRIC_OPTIONS[k]["label"],
            help="Choose the composite stretch heuristic or a single physical dimension (in metres).",
        )
        m_cfg = analytics.METRIC_OPTIONS[selected_metric]
        threshold_val = st.slider(
            f"Stretch Alert Threshold ({m_cfg['unit']})",
            min_value=float(m_cfg["min_val"]),
            max_value=float(m_cfg["max_val"]),
            value=float(m_cfg["default_threshold"]),
            step=float(m_cfg["step"]),
            help=m_cfg["description"],
        )

        min_outfield = st.slider(
            "Minimum Observed Outfield Players",
            min_value=6,
            max_value=10,
            value=8,
            step=1,
            help=(
                "Require at least this many outfield players detected in frame (out of 10 expected) "
                "before computing or ranking team shape."
            ),
        )

        context_filter = st.selectbox(
            "Tactical Context Filter",
            options=["all", "ball_def_half", "deep_block"],
            index=0,
            format_func=lambda c: {
                "all": "All adequate-coverage frames",
                "ball_def_half": "Ball in defending half (ball x_att < 0 m)",
                "deep_block": "Team centroid in own half (centroid x_att < 0 m)",
            }[c],
            help=(
                "Teams naturally spread out when attacking in possession. Filter to frames where "
                "the ball or team centroid is in the defending half to focus on defensive phases."
            ),
        )

        st.header("3. Pitch Overlays")
        show_hull = st.checkbox("Show Outfield Convex Hull", value=True)
        show_bbox = st.checkbox("Show Width × Depth Bounding Box", value=True)
        show_max_gap = st.checkbox("Highlight Largest Longitudinal Gap", value=True)
        show_opponents = st.checkbox("Show Opponent Players", value=True)

    # Filter shape timeseries to selected team & half
    half_team_df = shape_df[
        (shape_df["team"] == defending_team) & (shape_df["half"] == int(half))
    ].copy()
    total_half_frames = len(half_team_df)
    adequate_df = half_team_df[half_team_df["n_outfield"] >= min_outfield]
    adequate_frames = len(adequate_df)
    half_coverage_rate = (
        (adequate_frames / total_half_frames) * 100.0 if total_half_frames > 0 else 0.0
    )

    # Compute ranked stretched moments within the selected half
    ranked_df = analytics.get_ranked_stretched_moments(
        shape_df=shape_df,
        events_df=events_df,
        team=defending_team,
        half=half,
        rank_metric=selected_metric,
        min_outfield=min_outfield,
        context_filter=context_filter,
        min_separation_s=15.0,
        top_n=10,
    )

    # --- Moment Selection Bar ---
    st.subheader(f"Inspect Moment — {analytics.TEAM_LABELS[defending_team]}, Half {half}")
    col_mode, col_picker = st.columns([1, 2])

    with col_mode:
        selection_mode = st.radio(
            "Moment Selection Mode",
            options=["Top Ranked Stretched Moments", "Scrub Video Clock (All Half Frames)"],
            index=0,
            help="Jump directly to de-clustered peak stretched snapshots or scrub any second in the half.",
        )

    min_ts = int(half_team_df["timestamp_ms"].min())
    max_ts = int(half_team_df["timestamp_ms"].max())

    with col_picker:
        if selection_mode == "Top Ranked Stretched Moments" and not ranked_df.empty:
            rank_options = ranked_df["rank"].tolist()

            def _format_rank_option(r_idx: int) -> str:
                row = ranked_df.loc[ranked_df["rank"] == r_idx].iloc[0]
                val_str = f"{row[selected_metric]:.2f} {m_cfg['unit']}"
                return (
                    f"#{r_idx} | Video {row['video_time_label']} | "
                    f"{m_cfg['short_label']}: {val_str} | "
                    f"W: {row['width_m']:.1f}m, D: {row['depth_m']:.1f}m, Gap: {row['max_x_gap_m']:.1f}m | "
                    f"{row['nearby_context']}"
                )

            chosen_rank = st.selectbox(
                "Select Ranked Stretched Snapshot (separated by ≥15s)",
                options=rank_options,
                index=0,
                format_func=_format_rank_option,
            )
            selected_ts = int(
                ranked_df.loc[ranked_df["rank"] == chosen_rank, "timestamp_ms"].iloc[0]
            )
        else:
            if selection_mode == "Top Ranked Stretched Moments" and ranked_df.empty:
                st.warning(
                    "No frames matched the current coverage and context filters. "
                    "Falling back to manual video timestamp scrubber."
                )
            default_sec = float(
                ranked_df["timestamp_ms"].iloc[0] / 1000.0
                if not ranked_df.empty
                else min_ts / 1000.0
            )
            chosen_sec = st.slider(
                "Video Timestamp (seconds, s)",
                min_value=float(min_ts / 1000.0),
                max_value=float(max_ts / 1000.0),
                value=default_sec,
                step=0.4,
            )
            selected_ts = int(round(chosen_sec * 1000.0))

    # Fetch nearest-frame details strictly within the same half
    details = analytics.get_moment_details(
        players_df=players_df,
        shape_df=shape_df,
        events_df=events_df,
        summary_df=summary_df,
        half=half,
        timestamp_ms=selected_ts,
        defending_team=defending_team,
        event_window_s=15.0,
    )

    if not details["found"]:
        st.error(
            f"No tracking frame found within tolerance ({analytics.DEFAULT_TOLERANCE_MS} ms) "
            f"for {selected_ts} ms in Half {half}."
        )
        return

    matched_ts = int(details["matched_timestamp_ms"])
    cur_metrics = details["shape_metrics"]
    n_obs = int(cur_metrics.get("n_outfield", 0))

    # --- Top KPI Strip ---
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric(
        "Video Timestamp",
        analytics.format_video_time(matched_ts, include_seconds=False),
        f"{matched_ts / 1000.0:.1f} s ({matched_ts} ms)",
    )
    kpi2.metric(
        "Outfield Coverage",
        f"{n_obs} / {analytics.EXPECTED_OUTFIELD} players",
        f"Half ≥{min_outfield}p: {half_coverage_rate:.1f}% ({adequate_frames}/{total_half_frames})",
    )

    mean_w = adequate_df["width_m"].mean() if not adequate_df.empty else np.nan
    mean_d = adequate_df["depth_m"].mean() if not adequate_df.empty else np.nan
    mean_g = adequate_df["max_x_gap_m"].mean() if not adequate_df.empty else np.nan
    mean_c = adequate_df["compactness_m"].mean() if not adequate_df.empty else np.nan

    cur_w = float(cur_metrics.get("width_m", np.nan))
    cur_d = float(cur_metrics.get("depth_m", np.nan))
    cur_g = float(cur_metrics.get("max_x_gap_m", np.nan))
    cur_c = float(cur_metrics.get("compactness_m", np.nan))

    kpi3.metric(
        "Outfield Width (y-span)",
        f"{cur_w:.1f} m",
        f"{cur_w - mean_w:+.1f} m vs half avg ({mean_w:.1f} m)",
        delta_color="inverse",
    )
    kpi4.metric(
        "Outfield Depth (x-span)",
        f"{cur_d:.1f} m",
        f"{cur_d - mean_d:+.1f} m vs half avg ({mean_d:.1f} m)",
        delta_color="inverse",
    )
    kpi5.metric(
        "Largest Longitudinal Gap",
        f"{cur_g:.1f} m",
        f"{cur_g - mean_g:+.1f} m vs half avg ({mean_g:.1f} m)",
        delta_color="inverse",
    )
    kpi6.metric(
        "Compactness (Centroid Dist)",
        f"{cur_c:.1f} m",
        f"Hull area: {details['hull_area_m2']:.0f} m² (avg {mean_c:.1f} m)",
        delta_color="inverse",
    )

    if n_obs < min_outfield:
        st.warning(
            f"⚠️ **Low Outfield Coverage at Selected Frame**: Only **{n_obs}** outfield players are observed "
            f"(minimum threshold set to **{min_outfield}**). Width, depth, and longitudinal gap values at "
            "this timestamp may be understated or distorted by out-of-frame players."
        )

    # --- Main Pitch & Context Columns ---
    col_pitch, col_context = st.columns([3, 2])

    with col_pitch:
        pitch_fig = build_pitch_figure(
            details=details,
            show_hull=show_hull,
            show_bbox=show_bbox,
            show_max_gap=show_max_gap,
            show_opponents=show_opponents,
        )
        st.plotly_chart(pitch_fig, width="stretch")

    with col_context:
        st.markdown("#### Snapshot Diagnostic & Coaching Notes")
        ball_status = (
            f"Observed at ({details['ball_x']:.1f} m, {details['ball_y']:.1f} m)"
            if pd.notna(details["ball_x"]) and pd.notna(details["ball_y"])
            else "Missing in this frame (ball_x / ball_y is NaN)"
        )
        gap_pair_str = (
            f"`{cur_metrics.get('gap_player_low_x')}` (x={cur_metrics.get('gap_x_start', 0):.1f} m) ↔ "
            f"`{cur_metrics.get('gap_player_high_x')}` (x={cur_metrics.get('gap_x_end', 0):.1f} m)"
            if pd.notna(cur_metrics.get("max_x_gap_m"))
            else "N/A"
        )
        st.markdown(
            f"""
- **Review Timestamp**: `{analytics.format_video_time(matched_ts, include_seconds=True)}` (`{matched_ts} ms`, Δ `{details['time_diff_ms']} ms` to nearest frame in Half {half})
- **Composite Stretch Index**: `{cur_metrics.get('stretch_score', np.nan):.2f} z` (Bounding Box: `{cur_metrics.get('bbox_area_m2', 0):.0f} m²`, Convex Hull: `{details['hull_area_m2']:.0f} m²`)
- **Largest Longitudinal Gap**: `{cur_g:.1f} m` between {gap_pair_str}
- **Defending Line Depth (`x_att`)**: Deepest outfielder at `{cur_metrics.get('deepest_defender_x_att', np.nan):.1f} m` | Centroid at `{cur_metrics.get('centroid_x_att', np.nan):.1f} m` (negative = defending own half)
- **Ball Tracking**: {ball_status}
"""
        )

        st.info(
            "💡 **Coach Interpretation Caveat**: A high stretch score or wide/deep shape is a **geometric "
            "heuristic**, not an automatic defensive error. Teams purposefully expand width and depth during "
            "attacking build-up, goal kicks, and counter-attacking transitions. Use the **Tactical Context Filter** "
            "and the **Nearby Events** table below to verify whether the team was out of possession or in transition."
        )

        st.markdown("#### Nearby Match Events (±15s in Same Half)")
        nearby_ev = details["nearby_events"]
        if nearby_ev.empty:
            st.caption(
                "No tagged on-ball events within ±15s of this frame. (Note: missing event rows do not "
                "prove absence of possession or touches—only logged events appear in `events.csv`.)"
            )
        else:
            display_ev = nearby_ev[
                ["video_clock", "rel_seconds", "team", "event_name", "player_id", "is_goal"]
            ].copy()
            display_ev = display_ev.rename(
                columns={
                    "video_clock": "Video Clock",
                    "rel_seconds": "Rel. Time (s)",
                    "team": "Team",
                    "event_name": "Event",
                    "player_id": "Player ID",
                    "is_goal": "Shot Goal Flag",
                }
            )
            display_ev["Rel. Time (s)"] = display_ev["Rel. Time (s)"].map(lambda s: f"{s:+.1f}s")
            st.dataframe(display_ev, width="stretch", hide_index=True, height=210)

    # --- Half Timeline Chart ---
    timeline_fig = build_half_timeline_figure(
        half_shape_df=half_team_df,
        events_df=events_df,
        half=half,
        defending_team=defending_team,
        selected_metric=selected_metric,
        threshold_val=threshold_val,
        min_outfield=min_outfield,
        selected_ts=matched_ts,
    )
    st.plotly_chart(timeline_fig, width="stretch")

    # --- Ranked Stretched Moments Table ---
    st.subheader(
        f"Top Ranked Stretched Moments in Half {half} — {analytics.TEAM_LABELS[defending_team]} "
        f"(Ranked by {m_cfg['short_label']}, ≥{min_outfield} Outfielders)"
    )
    if ranked_df.empty:
        st.info("No moments satisfy the current filter combination.")
    else:
        table_df = ranked_df[
            [
                "rank",
                "video_time_label",
                "timestamp_ms",
                "n_outfield",
                "stretch_score",
                "width_m",
                "depth_m",
                "max_x_gap_m",
                "compactness_m",
                "centroid_x_att",
                "nearby_context",
            ]
        ].copy()
        table_df.columns = [
            "Rank",
            "Video Time",
            "Timestamp (ms)",
            "Observed Outfield",
            "Stretch Index (z)",
            "Width (m)",
            "Depth (m)",
            "Max x-Gap (m)",
            "Centroid Dist (m)",
            "Centroid x_att (m)",
            "Nearby Event Context (±10s)",
        ]
        for col in ["Stretch Index (z)", "Width (m)", "Depth (m)", "Max x-Gap (m)", "Centroid Dist (m)", "Centroid x_att (m)"]:
            table_df[col] = table_df[col].round(2)
        st.dataframe(table_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
