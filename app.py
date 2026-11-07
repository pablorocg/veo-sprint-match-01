"""Streamlit Dashboard: Match Story & Goal Buildup Explorer (Veo Match 01).

Launch with:
    uv run streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics

st.set_page_config(
    page_title="Veo Match Story & Goal Buildup Explorer",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner="Loading Veo match tracking and event data...")
def get_cached_data() -> dict[str, pd.DataFrame]:
    """Load and cache the match datasets."""
    return analytics.load_datasets(".")


@st.cache_data(show_spinner=False)
def get_cached_goals_and_story() -> tuple[list[analytics.GoalReferenceInfo], dict]:
    """Compute and cache resolved goal references and overall match story summary."""
    data = get_cached_data()
    goals = analytics.resolve_all_goals(data["events"], data["players"], data["ball"])
    story = analytics.compute_match_story_summary(data["summary"], data["events"], data["players"])
    return goals, story


@st.cache_data(show_spinner=False)
def get_cached_buildup_window(
    anchor_timestamp_ms: int,
    half: int,
    attacking_team: str,
    lead_in_seconds: float,
) -> dict:
    """Extract and cache a buildup tracking/event window."""
    data = get_cached_data()
    return analytics.extract_buildup_window(
        events=data["events"],
        players=data["players"],
        anchor_timestamp_ms=anchor_timestamp_ms,
        half=half,
        attacking_team=attacking_team,
        lead_in_seconds=lead_in_seconds,
        post_seconds=0.8,
    )


def add_pitch_shapes(fig: go.Figure, attack_dir: int | None = None) -> None:
    """Add standard 105.0 m x 67.7 m pitch markings (origin at centre spot) to a Plotly figure."""
    hl = analytics.HALF_LENGTH_M
    hw = analytics.HALF_WIDTH_M
    line_style = dict(color="#cbd5e1", width=1.8)

    shapes = [
        # Outer boundary
        dict(type="rect", x0=-hl, y0=-hw, x1=hl, y1=hw, line=line_style, layer="below"),
        # Halfway line
        dict(type="line", x0=0, y0=-hw, x1=0, y1=hw, line=line_style, layer="below"),
        # Centre circle (r = 9.15 m)
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line_style, layer="below"),
    ]

    for s in (-1, 1):
        # Penalty area (16.5 m x 40.3 m)
        x_edge = s * hl
        x_box = s * (hl - analytics.PENALTY_BOX_DEPTH_M)
        shapes.append(
            dict(
                type="rect",
                x0=min(x_edge, x_box),
                y0=-analytics.PENALTY_BOX_HALF_WIDTH_M,
                x1=max(x_edge, x_box),
                y1=analytics.PENALTY_BOX_HALF_WIDTH_M,
                line=line_style,
                layer="below",
            )
        )
        # Goal area (5.5 m x 18.32 m)
        x_ga = s * (hl - analytics.GOAL_AREA_DEPTH_M)
        shapes.append(
            dict(
                type="rect",
                x0=min(x_edge, x_ga),
                y0=-analytics.GOAL_AREA_HALF_WIDTH_M,
                x1=max(x_edge, x_ga),
                y1=analytics.GOAL_AREA_HALF_WIDTH_M,
                line=line_style,
                layer="below",
            )
        )
        # Goal mouth (7.32 m wide)
        x_goal_back = s * (hl + 1.8)
        shapes.append(
            dict(
                type="rect",
                x0=min(x_edge, x_goal_back),
                y0=-analytics.GOAL_WIDTH_M / 2.0,
                x1=max(x_edge, x_goal_back),
                y1=analytics.GOAL_WIDTH_M / 2.0,
                line=dict(color="#f8fafc", width=2.5),
                layer="below",
            )
        )

    fig.update_layout(shapes=shapes)

    if attack_dir in (-1, 1):
        arrow_x = 18.0 * attack_dir
        fig.add_annotation(
            x=arrow_x,
            y=hw + 2.3,
            ax=-arrow_x,
            ay=hw + 2.3,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#94a3b8",
            text=f"Attacking direction ({'+x' if attack_dir == 1 else '−x'} goal)",
            font=dict(size=11, color="#cbd5e1"),
        )


def build_single_frame_pitch_figure(
    window_data: dict,
    selected_ts: int,
    attacking_team: str,
    attack_dir: int,
    title: str,
    show_trails: bool = True,
    highlight_player_id: str | None = None,
) -> go.Figure:
    """Create a 2D pitch figure for a specific tracking frame, with optional lead-in trails and event arrows."""
    win_players: pd.DataFrame = window_data["players_window"]
    win_events: pd.DataFrame = window_data["events_window"]
    frame_df = win_players[win_players["timestamp_ms"] == selected_ts]

    fig = go.Figure()
    add_pitch_shapes(fig, attack_dir=attack_dir)

    # 1. Optional player & ball trails up to selected_ts (split across any tracking gaps)
    if show_trails:
        trail_df = win_players[
            (win_players["timestamp_ms"] <= selected_ts)
            & (win_players["timestamp_ms"] >= selected_ts - 6000)
        ]
        for pid, p_group in trail_df.groupby("player_id"):
            if pd.isna(pid):
                continue
            p_team = str(p_group["team"].iloc[0])
            if p_team == "referee":
                continue
            color = analytics.TEAM_COLORS.get(p_team, "#888888")
            for seg in analytics.split_trajectory_segments(p_group, "x", "y"):
                if len(seg) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=seg["x"],
                            y=seg["y"],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dot"),
                            opacity=0.45,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        # Ball trail across the buildup window up to selected_ts
        ball_trail = (
            win_players[win_players["timestamp_ms"] <= selected_ts]
            .drop_duplicates("timestamp_ms")[["timestamp_ms", "half", "ball_x", "ball_y"]]
            .sort_values("timestamp_ms")
        )
        for b_seg in analytics.split_trajectory_segments(ball_trail, "ball_x", "ball_y"):
            if len(b_seg) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=b_seg["ball_x"],
                        y=b_seg["ball_y"],
                        mode="lines",
                        line=dict(color="#facc15", width=2.5),
                        opacity=0.8,
                        name="Ball trajectory",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # 2. On-ball pass/shot events up to selected_ts
    past_events = win_events[win_events["timestamp_ms"] <= selected_ts]
    for _, ev in past_events.iterrows():
        if ev["event_name"] == "pass" and pd.notna(ev["x_start"]) and pd.notna(ev["receiver_x"]):
            fig.add_annotation(
                x=float(ev["receiver_x"]),
                y=float(ev["receiver_y"]),
                ax=float(ev["x_start"]),
                ay=float(ev["y_start"]),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=1.8,
                arrowcolor="#38bdf8" if ev["team"] == "home" else "#f87171",
                opacity=0.75,
            )
        elif ev["event_name"] == "shot" and pd.notna(ev["x_start"]):
            fig.add_trace(
                go.Scatter(
                    x=[float(ev["x_start"])],
                    y=[float(ev["y_start"])],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="#fde047", line=dict(color="#000", width=1)),
                    name=f"Shot #{int(ev['event_id'])}",
                    hovertemplate=(
                        f"<b>Shot #{int(ev['event_id'])} ({ev['player_id']})</b><br>"
                        f"Video time: {ev['video_time_str']}<br>"
                        f"Distance: {float(ev['shot_distance_m']):.1f} m<extra></extra>"
                    ),
                )
            )

    # 3. Players at current frame
    for team_name, label in [("away", "Away (visitors)"), ("home", "Home"), ("referee", "Referee")]:
        t_df = frame_df[frame_df["team"] == team_name]
        if t_df.empty:
            continue
        color = analytics.TEAM_COLORS[team_name]
        shirts = [
            str(pid).split("_")[-1] if pd.notna(pid) and "_" in str(pid) else ("R" if team_name == "referee" else "")
            for pid in t_df["player_id"]
        ]
        sizes = [
            19 if (highlight_player_id and str(pid) == highlight_player_id) else (11 if team_name == "referee" else 15)
            for pid in t_df["player_id"]
        ]
        border_colors = [
            "#fde047" if (highlight_player_id and str(pid) == highlight_player_id) else "#ffffff"
            for pid in t_df["player_id"]
        ]
        border_widths = [
            3.0 if (highlight_player_id and str(pid) == highlight_player_id) else 1.2
            for pid in t_df["player_id"]
        ]

        fig.add_trace(
            go.Scatter(
                x=t_df["x"],
                y=t_df["y"],
                mode="markers+text",
                text=shirts,
                textposition="middle center",
                textfont=dict(color="#ffffff", size=9, family="Arial Black"),
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(color=border_colors, width=border_widths),
                ),
                name=label,
                customdata=np.stack(
                    [
                        t_df["player_id"].fillna("referee"),
                        t_df["role"].fillna(""),
                        t_df["speed_zone"].fillna(0),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "Pitch (x, y): (%{x:.1f} m, %{y:.1f} m)<br>"
                    "Speed zone: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    # 4. Ball at current frame (explicitly handle missing ball_x/ball_y)
    if not frame_df.empty and pd.notna(frame_df["ball_x"].iloc[0]) and pd.notna(frame_df["ball_y"].iloc[0]):
        bx = float(frame_df["ball_x"].iloc[0])
        by = float(frame_df["ball_y"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=[bx],
                y=[by],
                mode="markers",
                marker=dict(
                    size=13,
                    color="#facc15",
                    symbol="circle",
                    line=dict(color="#000000", width=2),
                ),
                name="Ball (tracked)",
                hovertemplate=f"<b>Ball</b><br>x: {bx:.1f} m, y: {by:.1f} m<extra></extra>",
            )
        )
    else:
        fig.add_annotation(
            x=0,
            y=-analytics.HALF_WIDTH_M - 2.4,
            text="⚠️ Ball coordinates not located by computer vision in this frame",
            showarrow=False,
            font=dict(color="#fbbf24", size=11),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            range=[-analytics.HALF_LENGTH_M - 4.5, analytics.HALF_LENGTH_M + 4.5],
            title="Pitch length x (m, centre spot = 0)",
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-analytics.HALF_WIDTH_M - 4.5, analytics.HALF_WIDTH_M + 4.5],
            title="Pitch width y (m)",
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="#14532d",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=30),
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    return fig


def build_animated_buildup_figure(
    window_data: dict,
    attack_dir: int,
    highlight_player_id: str | None = None,
) -> go.Figure:
    """Create a Plotly Play/Pause animated pitch figure across all frames in the buildup window."""
    win_players: pd.DataFrame = window_data["players_window"]
    frame_timestamps: list[int] = window_data["frame_timestamps"]
    anchor_ms: int = window_data["anchor_timestamp_ms"]

    fig = go.Figure()
    add_pitch_shapes(fig, attack_dir=attack_dir)

    if not frame_timestamps:
        return fig

    def frame_traces(ts: int) -> list[go.Scatter]:
        f_df = win_players[win_players["timestamp_ms"] == ts]
        traces: list[go.Scatter] = []
        for team_name, label in [("away", "Away"), ("home", "Home")]:
            t_df = f_df[f_df["team"] == team_name]
            shirts = [
                str(pid).split("_")[-1] if pd.notna(pid) and "_" in str(pid) else ""
                for pid in t_df["player_id"]
            ]
            traces.append(
                go.Scatter(
                    x=t_df["x"],
                    y=t_df["y"],
                    mode="markers+text",
                    text=shirts,
                    textposition="middle center",
                    textfont=dict(color="#ffffff", size=9, family="Arial Black"),
                    marker=dict(
                        size=15,
                        color=analytics.TEAM_COLORS[team_name],
                        line=dict(color="#ffffff", width=1.2),
                    ),
                    name=label,
                )
            )
        # Ball trace
        if not f_df.empty and pd.notna(f_df["ball_x"].iloc[0]) and pd.notna(f_df["ball_y"].iloc[0]):
            bx, by = [float(f_df["ball_x"].iloc[0])], [float(f_df["ball_y"].iloc[0])]
        else:
            bx, by = [], []
        traces.append(
            go.Scatter(
                x=bx,
                y=by,
                mode="markers",
                marker=dict(size=13, color="#facc15", line=dict(color="#000000", width=2)),
                name="Ball",
            )
        )
        return traces

    # Initial traces (last frame = goal moment by default)
    init_ts = frame_timestamps[-1]
    for tr in frame_traces(init_ts):
        fig.add_trace(tr)

    frames = []
    slider_steps = []
    for ts in frame_timestamps:
        t_rel = (ts - anchor_ms) / 1000.0
        v_time = analytics.format_video_time(ts)
        frame_name = str(ts)
        frames.append(go.Frame(data=frame_traces(ts), name=frame_name))
        slider_steps.append(
            dict(
                args=[[frame_name], dict(frame=dict(duration=220, redraw=True), mode="immediate")],
                label=f"{t_rel:+.1f}s ({v_time})",
                method="animate",
            )
        )

    fig.frames = frames
    fig.update_layout(
        title="Play / Pause Buildup Animation (2.5 Hz tracking frames, single half)",
        xaxis=dict(
            range=[-analytics.HALF_LENGTH_M - 4.5, analytics.HALF_LENGTH_M + 4.5],
            title="Pitch length x (m)",
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-analytics.HALF_WIDTH_M - 4.5, analytics.HALF_WIDTH_M + 4.5],
            title="Pitch width y (m)",
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="#14532d",
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=20, r=20, t=50, b=60),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=1.13,
                xanchor="left",
                yanchor="top",
                direction="left",
                buttons=[
                    dict(
                        label="▶ Play Buildup",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=220, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=80),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=len(frame_timestamps) - 1,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Frame time: ",
                    visible=True,
                    xanchor="right",
                ),
                pad=dict(b=10, t=25),
                len=1.0,
                x=0.0,
                y=0,
                steps=slider_steps,
            )
        ],
    )
    return fig


def build_spatial_metrics_timeline_figure(
    goal_metrics_df: pd.DataFrame,
    comp_metrics_df: pd.DataFrame,
    selected_t_rel_s: float,
    goal_label: str,
    comp_label: str,
) -> go.Figure:
    """Plot Nearest Defender Distance (m) and Attacking Width (m) across the buildup window."""
    fig = go.Figure()

    # Nearest defender distance on goal buildup
    g_valid_def = goal_metrics_df.dropna(subset=["nearest_defender_m"])
    if not g_valid_def.empty:
        fig.add_trace(
            go.Scatter(
                x=g_valid_def["t_rel_s"],
                y=g_valid_def["nearest_defender_m"],
                mode="lines+markers",
                name=f"{goal_label}: Nearest defender to ball (m)",
                line=dict(color="#22c55e", width=2.5),
                marker=dict(size=5),
            )
        )

    # Nearest defender distance on comparable unsuccessful shot buildup
    c_valid_def = comp_metrics_df.dropna(subset=["nearest_defender_m"])
    if not c_valid_def.empty:
        fig.add_trace(
            go.Scatter(
                x=c_valid_def["t_rel_s"],
                y=c_valid_def["nearest_defender_m"],
                mode="lines+markers",
                name=f"{comp_label}: Nearest defender to ball (m)",
                line=dict(color="#f97316", width=2.0, dash="dash"),
                marker=dict(size=5),
            )
        )

    fig.add_vline(
        x=selected_t_rel_s,
        line_width=2,
        line_dash="dot",
        line_color="#fde047",
        annotation_text=f"Selected frame ({selected_t_rel_s:+.1f}s)",
    )

    fig.update_layout(
        title="Nearest Outfield Defender to Ball Over Buildup Window (m)",
        xaxis_title="Time relative to shot / goal moment (s)",
        yaxis_title="Distance to nearest defending outfield player (m)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_match_story_shot_map(
    shots_enriched: pd.DataFrame,
    goals: list[analytics.GoalReferenceInfo],
    half_filter: str,
) -> go.Figure:
    """Create a normalized half-pitch shot & goal map (attacking upward toward +x_att = 52.5 m)."""
    df = shots_enriched.copy()
    if half_filter == "First Half (0–0)":
        df = df[df["half"] == 1]
        goals_filtered = [g for g in goals if g.half == 1]
    elif half_filter == "Second Half (Away 4–2 Home)":
        df = df[df["half"] == 2]
        goals_filtered = [g for g in goals if g.half == 2]
    else:
        goals_filtered = goals

    fig = go.Figure()
    hl = analytics.HALF_LENGTH_M
    hw = analytics.HALF_WIDTH_M
    line_style = dict(color="#cbd5e1", width=1.8)

    # Attacking half-pitch markings (x_att from 0 to 52.5 m)
    fig.update_layout(
        shapes=[
            dict(type="rect", x0=-hw, y0=0, x1=hw, y1=hl, line=line_style, layer="below"),
            dict(
                type="rect",
                x0=-analytics.PENALTY_BOX_HALF_WIDTH_M,
                y0=hl - analytics.PENALTY_BOX_DEPTH_M,
                x1=analytics.PENALTY_BOX_HALF_WIDTH_M,
                y1=hl,
                line=line_style,
                layer="below",
            ),
            dict(
                type="rect",
                x0=-analytics.GOAL_AREA_HALF_WIDTH_M,
                y0=hl - analytics.GOAL_AREA_DEPTH_M,
                x1=analytics.GOAL_AREA_HALF_WIDTH_M,
                y1=hl,
                line=line_style,
                layer="below",
            ),
            dict(
                type="rect",
                x0=-analytics.GOAL_WIDTH_M / 2.0,
                y0=hl,
                x1=analytics.GOAL_WIDTH_M / 2.0,
                y1=hl + 1.5,
                line=dict(color="#ffffff", width=3),
                layer="below",
            ),
        ]
    )

    for team in ["away", "home"]:
        t_shots = df[(df["team"] == team) & (~df["is_goal"])]
        if not t_shots.empty:
            fig.add_trace(
                go.Scatter(
                    x=-t_shots["y_att"],  # Negative y_att puts right flank on right
                    y=t_shots["x_att"],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=analytics.TEAM_COLORS[team],
                        symbol="circle-open",
                        line=dict(width=2.5, color=analytics.TEAM_COLORS[team]),
                    ),
                    name=f"{team.capitalize()} unsuccessful shot (n={len(t_shots)})",
                    customdata=np.stack(
                        [
                            t_shots["event_id"],
                            t_shots["player_id"],
                            t_shots["video_time_str"],
                            t_shots["shot_distance_m"],
                            t_shots["nearest_defender_m"].fillna(-1),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>Shot #%{customdata[0]} (%{customdata[1]})</b><br>"
                        "Video time: %{customdata[2]}<br>"
                        "Shot distance: %{customdata[3]:.1f} m<br>"
                        "Nearest defender: %{customdata[4]:.1f} m<extra></extra>"
                    ),
                )
            )

    # All 6 goals (distinguishing the 4 with shot rows vs 2 from tracked ball)
    for g in goals_filtered:
        symbol = "star" if g.is_exact_shot_event else "diamond"
        fig.add_trace(
            go.Scatter(
                x=[-g.reference_y_att],
                y=[g.reference_x_att],
                mode="markers+text",
                text=[f"G#{g.goal_event_id}"],
                textposition="top center",
                textfont=dict(color="#fde047", size=10),
                marker=dict(
                    size=16,
                    color=analytics.TEAM_COLORS[g.team],
                    symbol=symbol,
                    line=dict(color="#fde047", width=2.2),
                ),
                name=f"Goal #{g.goal_event_id} ({g.scorer_id}, {analytics.format_video_time(g.goal_timestamp_ms)})",
                hovertemplate=(
                    f"<b>Goal #{g.goal_event_id} ({g.team.upper()} - {g.scorer_id})</b><br>"
                    f"Video time: {analytics.format_video_time(g.goal_timestamp_ms)}<br>"
                    f"Distance to goal: {g.shot_distance_m:.1f} m<br>"
                    f"Location source: {g.location_source_label}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Normalized Attacking Half Shot & Goal Map (attack_dir aligned toward top goal)",
        xaxis=dict(
            range=[-hw - 3, hw + 3],
            title="Lateral pitch position −y_att (m, attacking left → right)",
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-2, hl + 4],
            title="Forward pitch progression x_att (m, goal line = +52.5 m)",
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="#14532d",
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def main() -> None:
    data = get_cached_data()
    goals, story = get_cached_goals_and_story()
    team_stats = story["team_stats"]
    shots_enriched = story["shots_enriched"]

    st.title("⚽ Match Story & Goal Buildup Explorer")
    st.caption(
        "**Veo Football Analytics Sprint — Match 01** · Final Score: **Away 4 – 2 Home** "
        "(0–0 at Half-Time; all 6 goals in Half 2) · Pitch: 105.0 m × 67.7 m · Clock: Video time (`timestamp_ms / 1000`)"
    )

    # Top-level Match Story KPIs
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric(
            "Final Score (All 6 Goals in H2)",
            "Away 4 – 2 Home",
            delta="Away +2 goals despite near-equal volume",
        )
    with col_k2:
        st.metric(
            "Total Shots & Pass Accuracy",
            f"Away {team_stats['away']['shots']} vs Home {team_stats['home']['shots']} shots",
            delta=f"Pass %: Away {team_stats['away']['pass_completion_pct']:.1f}% | Home {team_stats['home']['pass_completion_pct']:.1f}%",
            delta_color="off",
        )
    with col_k3:
        st.metric(
            "Median Shot Distance to Goal",
            f"Away {team_stats['away']['median_shot_distance_m']:.1f} m",
            delta=f"Home median: {team_stats['home']['median_shot_distance_m']:.1f} m (+7.2 m further out)",
            delta_color="normal",
        )
    with col_k4:
        st.metric(
            "Close-Range Shots (≤ 15 m)",
            f"Away {team_stats['away']['shots_within_15m']} vs Home {team_stats['home']['shots_within_15m']}",
            delta=f"Goal mouth angle: Away {team_stats['away']['mean_shot_angle_deg']:.1f}° vs Home {team_stats['home']['mean_shot_angle_deg']:.1f}°",
        )

    # Sidebar controls
    st.sidebar.header("🎛️ Goal Buildup Controls")

    goal_options = {
        idx: (
            f"Goal {idx + 1} (#{g.goal_event_id}) · {analytics.format_video_time(g.goal_timestamp_ms)} "
            f"({g.minute:.1f}m) · {g.team.upper()} ({g.scorer_id})"
        )
        for idx, g in enumerate(goals)
    }

    selected_goal_idx = st.sidebar.selectbox(
        "1. Select Goal to Inspect (All 6 Goals)",
        options=list(goal_options.keys()),
        index=4,  # Default to Goal 5 (#1012 at 83:02 video time from demo_goal_moment.ipynb)
        format_func=lambda i: goal_options[i],
    )
    selected_goal = goals[selected_goal_idx]

    lead_in_s = st.sidebar.slider(
        "2. Buildup Lead-in Window (seconds)",
        min_value=15,
        max_value=20,
        value=18,
        step=1,
        help="Length of the tracking & event window preceding the selected goal and comparable shot.",
    )

    criterion_labels = {
        "same_team_half_location": "Same team & half + closest normalized shot location",
        "same_team_any_half_location": "Same team (any half) + closest normalized shot location",
        "closest_location_any_team": "Either team + closest normalized pitch location (x_att, y_att)",
        "closest_shot_distance": "Same team + closest distance to goal mouth (m)",
    }
    selected_criterion = st.sidebar.selectbox(
        "3. Comparable Unsuccessful Shot Rule",
        options=list(criterion_labels.keys()),
        index=0,
        format_func=lambda k: criterion_labels[k],
    )

    comparable_df = analytics.find_comparable_unsuccessful_shots(
        data["events"], selected_goal, criterion=selected_criterion
    )
    comp_options = {
        idx: (
            f"Shot #{int(row['event_id'])} · {row['video_time_str']} (H{int(row['half'])}) · "
            f"{str(row['team']).upper()} ({row['player_id']}) · {float(row['shot_distance_m']):.1f} m"
        )
        for idx, row in comparable_df.iterrows()
    }
    selected_comp_idx = st.sidebar.selectbox(
        "4. Select Ranked Comparable Unsuccessful Shot",
        options=list(comp_options.keys()),
        index=0,
        format_func=lambda i: comp_options[i],
    )
    comp_shot = comparable_df.iloc[selected_comp_idx]

    show_trails = st.sidebar.checkbox("Show 6-second player & ball trails", value=True)

    # Extract cached buildup windows for selected goal and comparable shot
    goal_window = get_cached_buildup_window(
        anchor_timestamp_ms=selected_goal.goal_timestamp_ms,
        half=selected_goal.half,
        attacking_team=selected_goal.team,
        lead_in_seconds=float(lead_in_s),
    )
    comp_window = get_cached_buildup_window(
        anchor_timestamp_ms=int(comp_shot["timestamp_ms"]),
        half=int(comp_shot["half"]),
        attacking_team=str(comp_shot["team"]),
        lead_in_seconds=float(lead_in_s),
    )

    st.markdown("---")
    st.subheader("🔍 Goal Buildup Explorer & Unsuccessful Attack Comparison")

    # Provenance banner for goal location & comparable selection
    st.info(
        f"**Selected Goal Provenance:** Goal `event_id={selected_goal.goal_event_id}` at video time "
        f"`{analytics.format_video_time(selected_goal.goal_timestamp_ms)}` (`timestamp_ms={selected_goal.goal_timestamp_ms}`, "
        f"Half {selected_goal.half}, scorer `{selected_goal.scorer_id}`, `attack_dir={selected_goal.attack_dir:+d}`). "
        f"**Location Reference:** {selected_goal.location_source_label} → Pitch `(x={selected_goal.reference_x:.1f} m, "
        f"y={selected_goal.reference_y:.1f} m)`, `{selected_goal.shot_distance_m:.1f} m` from goal centre.\n\n"
        f"**Comparable Unsuccessful Attack:** Shot `event_id={int(comp_shot['event_id'])}` at video time "
        f"`{comp_shot['video_time_str']}` (`timestamp_ms={int(comp_shot['timestamp_ms'])}`, Half {int(comp_shot['half'])}, "
        f"shooter `{comp_shot['player_id']}`). **Why selected:** {comp_shot['selection_reason']}."
    )

    # Interactive Frame Slider (relative time from -lead_in_s to 0.0s)
    frame_ts_list: list[int] = goal_window["frame_timestamps"]
    t_rel_options = [round((ts - selected_goal.goal_timestamp_ms) / 1000.0, 2) for ts in frame_ts_list]

    # Default slider to the reference shot/ball moment right before the goal
    default_rel_s = round(
        (selected_goal.reference_timestamp_ms - selected_goal.goal_timestamp_ms) / 1000.0, 2
    )
    closest_default = min(t_rel_options, key=lambda x: abs(x - default_rel_s))

    selected_rel_s = st.select_slider(
        "⏱️ Scrub Buildup Frame Timeline (seconds relative to Goal / Shot moment; 2.5 Hz tracking frames)",
        options=t_rel_options,
        value=closest_default,
        format_func=lambda s: f"{s:+.1f} s ({analytics.format_video_time(selected_goal.goal_timestamp_ms + int(s * 1000))})",
    )

    sel_frame_idx = t_rel_options.index(selected_rel_s)
    sel_goal_ts = frame_ts_list[sel_frame_idx]

    # Match the corresponding relative time in the comparable shot's buildup window
    comp_target_ms = int(comp_shot["timestamp_ms"] + selected_rel_s * 1000)
    comp_frame_df, sel_comp_ts, _ = analytics.get_nearest_frame(
        comp_window["players_window"],
        comp_target_ms,
        half=int(comp_shot["half"]),
        tolerance_ms=analytics.DEFAULT_FRAME_TOLERANCE_MS,
    )
    if sel_comp_ts is None and comp_window["frame_timestamps"]:
        sel_comp_ts = comp_window["frame_timestamps"][-1]

    # Compute spatial metrics at the selected frame and at the key shot/goal reference moment
    goal_sel_frame_df = goal_window["players_window"][
        goal_window["players_window"]["timestamp_ms"] == sel_goal_ts
    ]
    comp_sel_frame_df = comp_window["players_window"][
        comp_window["players_window"]["timestamp_ms"] == sel_comp_ts
    ]

    goal_frame_metrics = analytics.compute_frame_spatial_metrics(
        goal_sel_frame_df,
        attacking_team=selected_goal.team,
        ref_x=selected_goal.reference_x,
        ref_y=selected_goal.reference_y,
    )
    comp_frame_metrics = analytics.compute_frame_spatial_metrics(
        comp_sel_frame_df,
        attacking_team=str(comp_shot["team"]),
        ref_x=float(comp_shot["x_start"]),
        ref_y=float(comp_shot["y_start"]),
    )

    # Also compute metrics at the exact key shot/goal reference timestamp for factual summary
    goal_key_frame_df, _, _ = analytics.get_nearest_frame(
        data["players"], selected_goal.reference_timestamp_ms, selected_goal.half
    )
    goal_key_metrics = analytics.compute_frame_spatial_metrics(
        goal_key_frame_df,
        attacking_team=selected_goal.team,
        ref_x=selected_goal.reference_x,
        ref_y=selected_goal.reference_y,
    )
    comp_key_frame_df, _, _ = analytics.get_nearest_frame(
        data["players"], int(comp_shot["timestamp_ms"]), int(comp_shot["half"])
    )
    comp_key_metrics = analytics.compute_frame_spatial_metrics(
        comp_key_frame_df,
        attacking_team=str(comp_shot["team"]),
        ref_x=float(comp_shot["x_start"]),
        ref_y=float(comp_shot["y_start"]),
    )

    # Reproducible Descriptive Spatial Metrics Row (at selected slider frame)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    g_nd = goal_frame_metrics["nearest_defender_m"]
    c_nd = comp_frame_metrics["nearest_defender_m"]
    with m_col1:
        st.metric(
            f"1. Nearest Outfield Defender ({selected_rel_s:+.1f}s)",
            f"{g_nd:.1f} m" if g_nd is not None else "Ball N/A",
            delta=(
                f"{g_nd - c_nd:+.1f} m vs Comparable ({c_nd:.1f} m)"
                if (g_nd is not None and c_nd is not None)
                else "Comparable N/A"
            ),
        )
    g_aw = goal_frame_metrics["attacking_width_m"]
    c_aw = comp_frame_metrics["attacking_width_m"]
    with m_col2:
        st.metric(
            f"2. Attacking Outfield Width ({selected_rel_s:+.1f}s)",
            f"{g_aw:.1f} m" if g_aw is not None else "N/A",
            delta=(
                f"{g_aw - c_aw:+.1f} m vs Comparable ({c_aw:.1f} m)"
                if (g_aw is not None and c_aw is not None)
                else "Comparable N/A"
            ),
            delta_color="off",
        )
    g_a10 = goal_frame_metrics["attackers_near_ball_10m"]
    g_d10 = goal_frame_metrics["defenders_near_ball_10m"]
    c_a10 = comp_frame_metrics["attackers_near_ball_10m"]
    c_d10 = comp_frame_metrics["defenders_near_ball_10m"]
    with m_col3:
        st.metric(
            f"3. Numbers Near Ball ≤ 10 m ({selected_rel_s:+.1f}s)",
            f"{g_a10} Att vs {g_d10} Def" if g_a10 is not None else "Ball unlocated",
            delta=(
                f"Comparable: {c_a10} Att vs {c_d10} Def"
                if c_a10 is not None
                else "Comparable ball unlocated"
            ),
            delta_color="off",
        )
    with m_col4:
        st.metric(
            f"Penalty Box Occupation ({selected_rel_s:+.1f}s)",
            f"{goal_frame_metrics['attackers_in_box']} Att vs {goal_frame_metrics['defenders_in_box']} Def",
            delta=(
                f"Comparable: {comp_frame_metrics['attackers_in_box']} Att vs "
                f"{comp_frame_metrics['defenders_in_box']} Def"
            ),
            delta_color="off",
        )

    # Side-by-side Pitch Comparison (Goal Buildup vs Comparable Unsuccessful Shot)
    pitch_col1, pitch_col2 = st.columns(2)
    with pitch_col1:
        goal_fig = build_single_frame_pitch_figure(
            window_data=goal_window,
            selected_ts=sel_goal_ts,
            attacking_team=selected_goal.team,
            attack_dir=selected_goal.attack_dir,
            title=(
                f"GOAL #{selected_goal.goal_event_id} ({selected_goal.team.upper()} - {selected_goal.scorer_id}) · "
                f"Frame {analytics.format_video_time(sel_goal_ts)} ({selected_rel_s:+.1f}s)"
            ),
            show_trails=show_trails,
            highlight_player_id=selected_goal.scorer_id,
        )
        st.plotly_chart(goal_fig, width="stretch")

    with pitch_col2:
        comp_fig = build_single_frame_pitch_figure(
            window_data=comp_window,
            selected_ts=sel_comp_ts,
            attacking_team=str(comp_shot["team"]),
            attack_dir=int(comp_shot["attack_dir"]),
            title=(
                f"UNSUCCESSFUL SHOT #{int(comp_shot['event_id'])} ({str(comp_shot['team']).upper()} - {comp_shot['player_id']}) · "
                f"Frame {analytics.format_video_time(sel_comp_ts)} ({selected_rel_s:+.1f}s)"
            ),
            show_trails=show_trails,
            highlight_player_id=str(comp_shot["player_id"]),
        )
        st.plotly_chart(comp_fig, width="stretch")

    # Factual Plain-Language Observation Box
    factual_text = analytics.build_factual_observation(
        goal_info=selected_goal,
        goal_metrics=goal_key_metrics,
        comp_shot_row=comp_shot,
        comp_metrics=comp_key_metrics,
    )
    st.success(f"**📋 Coach Factual Spatial Observation:** {factual_text}")

    # Spatial Metrics Time-Series Chart & Event Timelines
    ts_col, ev_col = st.columns([1.15, 1.0])
    with ts_col:
        timeline_fig = build_spatial_metrics_timeline_figure(
            goal_metrics_df=goal_window["metrics_timeline"],
            comp_metrics_df=comp_window["metrics_timeline"],
            selected_t_rel_s=selected_rel_s,
            goal_label=f"Goal #{selected_goal.goal_event_id} ({selected_goal.scorer_id})",
            comp_label=f"Shot #{int(comp_shot['event_id'])} ({comp_shot['player_id']})",
        )
        st.plotly_chart(timeline_fig, width="stretch")

    with ev_col:
        st.markdown("##### 📜 Detected On-Ball Events in Lead-in Window")
        st.caption(
            "Video timestamps (`MM:SS.s`). Note: `events.csv` is a detected sample; missing event rows do not prove absence of touches."
        )
        ev_tab1, ev_tab2 = st.tabs(
            [
                f"Goal #{selected_goal.goal_event_id} Lead-in ({len(goal_window['events_window'])} events)",
                f"Shot #{int(comp_shot['event_id'])} Lead-in ({len(comp_window['events_window'])} events)",
            ]
        )
        with ev_tab1:
            g_ev = goal_window["events_window"][
                [
                    "t_rel_s",
                    "video_time_str",
                    "event_name",
                    "team",
                    "player_id",
                    "receiver_player_id",
                    "pass_distance_m",
                    "shot_distance_m",
                ]
            ].copy()
            g_ev.columns = [
                "Rel (s)",
                "Video Clock",
                "Event",
                "Team",
                "Player",
                "Receiver",
                "Pass (m)",
                "Shot Dist (m)",
            ]
            st.dataframe(g_ev, width="stretch", hide_index=True)
        with ev_tab2:
            c_ev = comp_window["events_window"][
                [
                    "t_rel_s",
                    "video_time_str",
                    "event_name",
                    "team",
                    "player_id",
                    "receiver_player_id",
                    "pass_distance_m",
                    "shot_distance_m",
                ]
            ].copy()
            c_ev.columns = [
                "Rel (s)",
                "Video Clock",
                "Event",
                "Team",
                "Player",
                "Receiver",
                "Pass (m)",
                "Shot Dist (m)",
            ]
            st.dataframe(c_ev, width="stretch", hide_index=True)

    # Play/Pause Animation Expander
    with st.expander("🎬 Interactive Play / Pause Animation of Selected Goal Buildup", expanded=True):
        anim_fig = build_animated_buildup_figure(
            window_data=goal_window,
            attack_dir=selected_goal.attack_dir,
            highlight_player_id=selected_goal.scorer_id,
        )
        st.plotly_chart(anim_fig, width="stretch")

    st.markdown("---")
    st.subheader("📊 Match Story: What May Have Decided Away 4 – 2 Home?")
    st.markdown(
        """
        On headline process totals, **Home** and **Away** were virtually indistinguishable: Home had slightly more passes
        (**439 vs 393**), identical pass completion (**64.5% vs 64.4%**), one more shot (**15 vs 14**), and more tackles (**45 vs 34**).
        Yet **Away won 4–2**, with all six goals scored in the second half.

        Inspecting the **spatial tracking and shot-location data** highlights three factual differences (descriptive patterns across 29 shots, not a causal or predictive model):
        1. **Shot Proximity & Goal-Mouth Angle:** Away's 14 shots were taken from a median distance of **13.4 m** (mean **15.5 m**, mean goal angle **25.4°**), with **8 of 14 shots within 15 m**. Home's 15 shots were taken from a median distance of **20.6 m** (mean **20.8 m**, mean goal angle **17.2°**), with **12 of 15 shots from outside 15 m**.
        2. **Second-Half Clinical Surge by `away_9`:** After a 0–0 first half (5 shots apiece), Away generated 9 shots in Half 2—including a hat-trick by **`away_9`** (video minutes **61.7**, **69.0**, and **83.0**, all struck between **11.5 m and 12.2 m**).
        3. **Second-Phase & Set-Piece Execution:** Several goals arose from fast transitions (`away_13` → `away_4` → `away_9` at 83:02), rebounds after initial shots/saves (Goals #792, #829, #914), or set-pieces (free-kick Goal #725, corner Goal #1157).
        """
    )

    story_col1, story_col2 = st.columns([1.2, 1.0])
    with story_col1:
        half_filter = st.radio(
            "Filter Normalized Attacking Shot & Goal Map by Half:",
            options=["Full Match (29 shots, 6 goals)", "First Half (0–0)", "Second Half (Away 4–2 Home)"],
            horizontal=True,
        )
        shot_map_fig = build_match_story_shot_map(shots_enriched, goals, half_filter=half_filter)
        st.plotly_chart(shot_map_fig, width="stretch")

    with story_col2:
        st.markdown("##### ⚖️ Process Parity vs Spatial Shot Quality")
        comp_table = pd.DataFrame(
            [
                {"Metric": "Goals (All in Half 2)", "Away": str(team_stats["away"]["goals"]), "Home": str(team_stats["home"]["goals"])},
                {"Metric": "Total Shots (H1 / H2)", "Away": f"{team_stats['away']['shots']} ({team_stats['away']['shots_h1']}/{team_stats['away']['shots_h2']})", "Home": f"{team_stats['home']['shots']} ({team_stats['home']['shots_h1']}/{team_stats['home']['shots_h2']})"},
                {"Metric": "Shots Inside Penalty Box", "Away": str(team_stats["away"]["shots_in_box"]), "Home": str(team_stats["home"]["shots_in_box"])},
                {"Metric": "Close-Range Shots (≤ 15.0 m)", "Away": str(team_stats["away"]["shots_within_15m"]), "Home": str(team_stats["home"]["shots_within_15m"])},
                {"Metric": "Median Shot Distance (m)", "Away": f"{team_stats['away']['median_shot_distance_m']:.1f} m", "Home": f"{team_stats['home']['median_shot_distance_m']:.1f} m"},
                {"Metric": "Mean Shot Distance (m)", "Away": f"{team_stats['away']['mean_shot_distance_m']:.1f} m", "Home": f"{team_stats['home']['mean_shot_distance_m']:.1f} m"},
                {"Metric": "Mean Goal-Mouth Angle (deg)", "Away": f"{team_stats['away']['mean_shot_angle_deg']:.1f}°", "Home": f"{team_stats['home']['mean_shot_angle_deg']:.1f}°"},
                {"Metric": "Mean Nearest Defender at Shot (m)", "Away": f"{team_stats['away']['mean_nearest_defender_at_shot_m']:.1f} m", "Home": f"{team_stats['home']['mean_nearest_defender_at_shot_m']:.1f} m"},
                {"Metric": "Passes (Completion %)", "Away": f"{team_stats['away']['passes']} ({team_stats['away']['pass_completion_pct']:.1f}%)", "Home": f"{team_stats['home']['passes']} ({team_stats['home']['pass_completion_pct']:.1f}%)"},
                {"Metric": "Progressive Passes (≥ +10 m x_att)", "Away": str(team_stats["away"]["progressive_passes_10m"]), "Home": str(team_stats["home"]["progressive_passes_10m"])},
                {"Metric": "Outfield Distance Covered (km)", "Away": f"{team_stats['away']['outfield_distance_km']:.1f} km", "Home": f"{team_stats['home']['outfield_distance_km']:.1f} km"},
                {"Metric": "Interceptions / Tackles", "Away": f"{team_stats['away']['interceptions']} / {team_stats['away']['tackles']}", "Home": f"{team_stats['home']['interceptions']} / {team_stats['home']['tackles']}"},
            ]
        )
        st.dataframe(comp_table, width="stretch", hide_index=True)

        st.markdown("##### 🥅 Complete Record of All 6 Goals (`event_name == 'goal'`)")
        goals_summary_rows = []
        for g in goals:
            goals_summary_rows.append(
                {
                    "Goal ID": f"#{g.goal_event_id}",
                    "Video Time": analytics.format_video_time(g.goal_timestamp_ms),
                    "Team": g.team.upper(),
                    "Scorer": g.scorer_id,
                    "Est. Dist (m)": f"{g.shot_distance_m:.1f} m",
                    "Coordinate Provenance": "Linked is_goal shot" if g.is_exact_shot_event else "Tracked ball (no shot row)",
                }
            )
        st.dataframe(pd.DataFrame(goals_summary_rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
