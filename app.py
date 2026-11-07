"""Streamlit dashboard for Veo Off-Ball Space Creators prototype (Port 8502)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an

st.set_page_config(
    page_title="Veo Off-Ball Space Creators",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner="Loading Veo match tracking & event datasets...")
def get_cached_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cache datasets once per session."""
    datasets = an.load_datasets(".")
    moments = an.build_attack_moments(datasets["events"])
    return (
        datasets["summary"],
        datasets["events"],
        datasets["players"],
        moments,
    )


def build_pitch_figure(
    window_info: dict,
    runs_df: pd.DataFrame,
    selected_player_id: str,
    attacking_team: str,
    attack_dir: int,
    display_mode: str,
    moment_row: pd.Series,
) -> go.Figure:
    """Create an interactive Plotly 2D football pitch showing off-ball runs and defender reaction."""
    fig = go.Figure()

    hl = an.HALF_LENGTH_M
    hw = an.HALF_WIDTH_M

    # Pitch grass background & markings
    fig.update_layout(
        plot_bgcolor="#1e3d24",
        paper_bgcolor="#0e1117",
        font={"color": "#f0f2f6"},
        margin={"l": 20, "r": 20, "t": 55, "b": 25},
        height=580,
        xaxis={
            "range": [-hl - 5.0, hl + 5.0],
            "title": "Pitch Length x (metres, centre spot = 0 m)",
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "range": [-hw - 4.0, hw + 4.0],
            "title": "Pitch Width y (metres, centre spot = 0 m)",
            "showgrid": False,
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(14,17,23,0.75)",
        },
    )

    line_style = {"color": "rgba(255,255,255,0.65)", "width": 1.8}
    shapes = [
        # Outer boundary (105 x 67.7 m)
        {
            "type": "rect",
            "x0": -hl,
            "y0": -hw,
            "x1": hl,
            "y1": hw,
            "line": line_style,
            "layer": "below",
        },
        # Halfway line
        {
            "type": "line",
            "x0": 0,
            "y0": -hw,
            "x1": 0,
            "y1": hw,
            "line": line_style,
            "layer": "below",
        },
        # Centre circle (9.15 m radius)
        {
            "type": "circle",
            "x0": -9.15,
            "y0": -9.15,
            "x1": 9.15,
            "y1": 9.15,
            "line": line_style,
            "layer": "below",
        },
    ]

    # Penalty & 6-yard boxes + goals at both ends
    for sign in (-1, 1):
        shapes.extend(
            [
                {
                    "type": "rect",
                    "x0": sign * hl,
                    "y0": -20.15,
                    "x1": sign * (hl - 16.5),
                    "y1": 20.15,
                    "line": line_style,
                    "layer": "below",
                },
                {
                    "type": "rect",
                    "x0": sign * hl,
                    "y0": -9.16,
                    "x1": sign * (hl - 5.5),
                    "y1": 9.16,
                    "line": line_style,
                    "layer": "below",
                },
                {
                    "type": "rect",
                    "x0": sign * hl,
                    "y0": -3.66,
                    "x1": sign * (hl + 1.8),
                    "y1": 3.66,
                    "line": {"color": "#ffffff", "width": 2.2},
                    "layer": "below",
                },
            ]
        )

    fig.update_layout(shapes=shapes)

    # Attacking direction indicator banner on pitch
    arrow_symbol = "➡ Attacking +x Goal" if attack_dir == 1 else "⬅ Attacking −x Goal"
    fig.add_annotation(
        x=0,
        y=hw + 2.2,
        text=(
            f"<b>{attacking_team.upper()} {arrow_symbol}</b> "
            f"(Half {int(moment_row['half'])}, Video {moment_row['video_time']}, "
            f"Window: {window_info['actual_duration_s']:.1f} s)"
        ),
        showarrow=False,
        font={"size": 13, "color": "#ffd166"},
        bgcolor="rgba(0,0,0,0.55)",
        borderpad=4,
    )

    w_df: pd.DataFrame = window_info["window_df"]
    if w_df.empty:
        return fig

    defending_team = "home" if attacking_team == "away" else "away"
    att_color = an.TEAM_COLORS[attacking_team]
    def_color = an.TEAM_COLORS[defending_team]

    # Identify the selected runner's nearest defender at start of run
    sel_row = runs_df[runs_df["player_id"] == selected_player_id]
    focal_def_id = (
        str(sel_row["nearest_def_start_id"].iloc[0])
        if not sel_row.empty
        else "Unavailable"
    )
    focal_def_end_id = (
        str(sel_row["nearest_def_end_id"].iloc[0])
        if not sel_row.empty
        else "Unavailable"
    )

    # 1. Ball trajectory (handling missing ball_x/ball_y cleanly)
    ball_df = (
        w_df.drop_duplicates("timestamp_ms")
        .sort_values("timestamp_ms")[["timestamp_ms", "t_rel_s", "ball_x", "ball_y"]]
        .dropna(subset=["ball_x", "ball_y"])
    )
    if not ball_df.empty:
        fig.add_trace(
            go.Scatter(
                x=ball_df["ball_x"],
                y=ball_df["ball_y"],
                mode="lines+markers",
                name="Ball Trajectory (Tracked Frames)",
                line={"color": "#ffd166", "width": 2.2, "dash": "dot"},
                marker={"size": 6, "color": "#ffd166", "symbol": "circle"},
                hovertemplate="Ball at %{customdata:.1f}s<br>x=%{x:.1f} m, y=%{y:.1f} m<extra></extra>",
                customdata=ball_df["t_rel_s"],
            )
        )

    # 2. Outfield Defenders (context + focal nearest defender highlighted)
    def_df = w_df[(w_df["team"] == defending_team) & (w_df["role"] == "goalkeeper")].copy()
    if not def_df.empty:
        gk_end = def_df.sort_values("timestamp_ms").groupby("player_id").last().reset_index()
        fig.add_trace(
            go.Scatter(
                x=gk_end["x"],
                y=gk_end["y"],
                mode="markers+text",
                name=f"{defending_team.upper()} Goalkeeper (Excluded from Outfield Metric)",
                marker={"size": 10, "color": "#aaaaaa", "symbol": "diamond"},
                text=gk_end["player_id"],
                textposition="bottom center",
                textfont={"size": 9, "color": "#cccccc"},
                hovertemplate="%{text} (GK)<br>x=%{x:.1f} m, y=%{y:.1f} m<extra></extra>",
            )
        )

    outfield_defs = w_df[
        (w_df["team"] == defending_team) & (w_df["role"] == "outfield")
    ].copy()

    other_def_end_x, other_def_end_y, other_def_text = [], [], []
    for did, d_grp in outfield_defs.groupby("player_id"):
        segs = an.split_player_continuous_segments(d_grp)
        if not segs:
            continue
        seg = segs[-1].sort_values("timestamp_ms")
        is_focal_def = did in (focal_def_id, focal_def_end_id)

        if is_focal_def:
            if display_mode == "Full Motion Trails":
                fig.add_trace(
                    go.Scatter(
                        x=seg["x"],
                        y=seg["y"],
                        mode="lines",
                        name=f"Defender Trail ({did})",
                        line={"color": def_color, "width": 3.0, "dash": "dash"},
                        hoverinfo="skip",
                    )
                )
            else:
                fig.add_annotation(
                    x=float(seg["x"].iloc[-1]),
                    y=float(seg["y"].iloc[-1]),
                    ax=float(seg["x"].iloc[0]),
                    ay=float(seg["y"].iloc[0]),
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2.5,
                    arrowcolor=def_color,
                )

            # Start & end markers for focal defender
            fig.add_trace(
                go.Scatter(
                    x=[float(seg["x"].iloc[0]), float(seg["x"].iloc[-1])],
                    y=[float(seg["y"].iloc[0]), float(seg["y"].iloc[-1])],
                    mode="markers+text",
                    name=f"Nearest Outfield Defender ({did})",
                    marker={
                        "size": [9, 14],
                        "color": ["rgba(255,255,255,0.4)", def_color],
                        "line": {"color": "#ffffff", "width": 1.8},
                        "symbol": ["circle-open", "square"],
                    },
                    text=[f"{did} (start)", f"{did} (end)"],
                    textposition="bottom center",
                    textfont={"size": 10, "color": "#ffb3b3"},
                    hovertemplate=f"<b>Defender {did}</b><br>x=%{{x:.1f}} m, y=%{{y:.1f}} m<extra></extra>",
                )
            )
        else:
            other_def_end_x.append(float(seg["x"].iloc[-1]))
            other_def_end_y.append(float(seg["y"].iloc[-1]))
            other_def_text.append(str(did))

    if other_def_end_x:
        fig.add_trace(
            go.Scatter(
                x=other_def_end_x,
                y=other_def_end_y,
                mode="markers+text",
                name=f"Other {defending_team.upper()} Outfield Defenders (End Pos)",
                marker={
                    "size": 10,
                    "color": def_color,
                    "opacity": 0.55,
                    "symbol": "square",
                },
                text=other_def_text,
                textposition="bottom center",
                textfont={"size": 9, "color": "rgba(255,255,255,0.65)"},
                hovertemplate="Defender %{text}<br>x=%{x:.1f} m, y=%{y:.1f} m<extra></extra>",
            )
        )

    # 3. Outfield Attackers (Teammates + Selected Runner Highlighted)
    outfield_atts = w_df[
        (w_df["team"] == attacking_team) & (w_df["role"] == "outfield")
    ].copy()

    for aid, a_grp in outfield_atts.groupby("player_id"):
        segs = an.split_player_continuous_segments(a_grp)
        if not segs:
            continue
        seg = segs[-1].sort_values("timestamp_ms")
        is_selected = aid == selected_player_id

        if is_selected:
            if display_mode == "Full Motion Trails":
                fig.add_trace(
                    go.Scatter(
                        x=seg["x"],
                        y=seg["y"],
                        mode="lines+markers",
                        name=f"Selected Runner Trail ({aid})",
                        line={"color": "#00f5d4", "width": 4.2},
                        marker={"size": 5, "color": "#00f5d4"},
                        hovertemplate=(
                            f"<b>{aid}</b> at %{{customdata:.1f}}s<br>"
                            "x=%{x:.1f} m, y=%{y:.1f} m<extra></extra>"
                        ),
                        customdata=seg["t_rel_s"],
                    )
                )
            else:
                fig.add_annotation(
                    x=float(seg["x"].iloc[-1]),
                    y=float(seg["y"].iloc[-1]),
                    ax=float(seg["x"].iloc[0]),
                    ay=float(seg["y"].iloc[0]),
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=3.8,
                    arrowcolor="#00f5d4",
                )

            # Start and End Markers for Selected Attacker
            fig.add_trace(
                go.Scatter(
                    x=[float(seg["x"].iloc[0]), float(seg["x"].iloc[-1])],
                    y=[float(seg["y"].iloc[0]), float(seg["y"].iloc[-1])],
                    mode="markers+text",
                    name=f"Selected Attacker ({aid} Start → End)",
                    marker={
                        "size": [11, 17],
                        "color": ["#ffffff", "#00f5d4"],
                        "line": {"color": "#000000", "width": 2.0},
                        "symbol": ["circle-open", "star"],
                    },
                    text=[f"{aid} (Start)", f"★ {aid} (End)"],
                    textposition="top center",
                    textfont={"size": 12, "color": "#00f5d4"},
                    hovertemplate=f"<b>Selected Attacker {aid}</b><br>x=%{{x:.1f}} m, y=%{{y:.1f}} m<extra></extra>",
                )
            )
        else:
            if display_mode == "Full Motion Trails":
                fig.add_trace(
                    go.Scatter(
                        x=seg["x"],
                        y=seg["y"],
                        mode="lines",
                        name=f"{attacking_team.upper()} Teammate Trail",
                        line={"color": att_color, "width": 1.6},
                        opacity=0.45,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            else:
                fig.add_annotation(
                    x=float(seg["x"].iloc[-1]),
                    y=float(seg["y"].iloc[-1]),
                    ax=float(seg["x"].iloc[0]),
                    ay=float(seg["y"].iloc[0]),
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.0,
                    arrowwidth=1.5,
                    arrowcolor=att_color,
                    opacity=0.55,
                )

            fig.add_trace(
                go.Scatter(
                    x=[float(seg["x"].iloc[-1])],
                    y=[float(seg["y"].iloc[-1])],
                    mode="markers+text",
                    name=f"{attacking_team.upper()} Teammate ({aid})",
                    marker={
                        "size": 11,
                        "color": att_color,
                        "line": {"color": "#ffffff", "width": 1.2},
                    },
                    text=[str(aid)],
                    textposition="top center",
                    textfont={"size": 10, "color": "#ffffff"},
                    showlegend=False,
                    hovertemplate=f"Teammate <b>{aid}</b> (End)<br>x=%{{x:.1f}} m, y=%{{y:.1f}} m<extra></extra>",
                )
            )

    return fig


def build_timeline_figure(timeline_df: pd.DataFrame, player_id: str) -> go.Figure:
    """Plot continuous time-series of forward progression, defender separation, and ball distance."""
    fig = go.Figure()
    if timeline_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=timeline_df["t_rel_s"],
            y=timeline_df["forward_gain_from_start_m"],
            mode="lines+markers",
            name="Forward Gain Δx_att (m)",
            line={"color": "#00f5d4", "width": 3.0},
            hovertemplate="t=%{x:.1f}s: Forward Gain = %{y:+.1f} m<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["t_rel_s"],
            y=timeline_df["nearest_def_dist_m"],
            mode="lines+markers",
            name="Nearest Outfield Defender Distance (m)",
            line={"color": "#ff6b6b", "width": 2.5},
            customdata=timeline_df["nearest_def_id"],
            hovertemplate=(
                "t=%{x:.1f}s: Nearest Defender = %{y:.1f} m "
                "(%{customdata})<extra></extra>"
            ),
        )
    )

    if timeline_df["distance_to_ball_m"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=timeline_df["t_rel_s"],
                y=timeline_df["distance_to_ball_m"],
                mode="lines",
                name="Distance to Ball (m)",
                line={"color": "#ffd166", "width": 2.0, "dash": "dot"},
                hovertemplate="t=%{x:.1f}s: Ball Distance = %{y:.1f} m<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Frame-by-Frame Run Dynamics for {player_id} (Same-Half Continuous Segment)",
        xaxis_title="Time Relative to Shot/Goal Anchor Frame (seconds)",
        yaxis_title="Distance / Displacement (metres)",
        height=360,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return fig


def main() -> None:
    summary_df, events_df, players_df, moments_df = get_cached_data()

    st.title("⚽ Off-Ball Space Creators — Attacking Movement & Defender Separation")
    st.markdown(
        "Identify and rank purposeful **off-ball runs** by outfield players during the build-up "
        "to real shots and goals in **Veo Sprint Match 01 (Away 4 – 2 Home)**."
    )

    # Sidebar Controls (4 meaningful interactions)
    with st.sidebar:
        st.header("🎛️ Analysis Controls")

        team_choice = st.selectbox(
            "1. Attacking Team",
            options=["away", "home"],
            index=0,
            format_func=lambda t: f"{t.upper()} ({'Won 4–2' if t == 'away' else 'Lost 2–4'})",
        )

        attack_filter = st.radio(
            "Filter Attack Type",
            options=[
                "All Shots & Goals",
                "Goals Only (All 6 Goal Rows)",
                "Shots Only (29 Shot Rows)",
            ],
            index=0,
            help=(
                "Per DATA_DICTIONARY.md: 6 `goal` rows are the complete goal record (no coordinates); "
                "only 4 of the 29 `shot` rows have `is_goal=True`."
            ),
        )

        team_moments = moments_df[moments_df["team"] == team_choice].copy()
        if attack_filter == "Goals Only (All 6 Goal Rows)":
            team_moments = team_moments[team_moments["event_name"] == "goal"].copy()
        elif attack_filter == "Shots Only (29 Shot Rows)":
            team_moments = team_moments[team_moments["event_name"] == "shot"].copy()

        # Default to the 83:02 away goal (Event #1012) when available
        moment_ids = team_moments["event_id"].tolist()
        default_idx = moment_ids.index(1012) if 1012 in moment_ids else 0

        label_map = dict(zip(team_moments["event_id"], team_moments["label"]))
        selected_event_id = st.selectbox(
            "2. Shot / Goal Attack Moment",
            options=moment_ids,
            index=default_idx,
            format_func=lambda eid: label_map.get(eid, str(eid)),
        )

        window_seconds = st.slider(
            "3. Lead-Up Tracking Window (seconds)",
            min_value=4.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="Tracking window pulled strictly within the same half and split on any frame gap.",
        )

        display_mode = st.radio(
            "Pitch Visualization Mode",
            options=["Full Motion Trails", "Before / After Vectors"],
            index=0,
        )

        with st.expander("⚙️ Heuristic Weights & Off-Ball Eligibility Rules", expanded=False):
            st.caption(
                "Adjust the explainable heuristic weights or ball-proximity threshold:"
            )
            ball_prox_m = st.slider(
                "Ball-Proximity Exclusion Radius (m)",
                min_value=1.5,
                max_value=6.0,
                value=3.0,
                step=0.5,
                help="Frames within this distance of the ball are treated as near-ball / on-ball.",
            )
            w_forward = st.slider("Forward Displacement Weight", 0.0, 1.0, 0.40, 0.05)
            w_def = st.slider("Defender Separation Gain Weight", 0.0, 1.0, 0.30, 0.05)
            w_ball = st.slider("Distance Away from Ball Weight", 0.0, 1.0, 0.15, 0.05)
            w_tempo = st.slider("Running Tempo / Effort Weight", 0.0, 1.0, 0.15, 0.05)

    moment_row = team_moments[team_moments["event_id"] == selected_event_id].iloc[0]

    # Extract continuous same-half tracking window
    window_info = an.extract_continuous_same_half_window(
        players=players_df,
        anchor_timestamp_ms=int(moment_row["timestamp_ms"]),
        half=int(moment_row["half"]),
        window_seconds=float(window_seconds),
    )

    weights = {
        "forward": w_forward,
        "defender_sep": w_def,
        "ball_dist": w_ball,
        "tempo": w_tempo,
    }

    runs_df = an.compute_off_ball_runs(
        window_info=window_info,
        events=events_df,
        attacking_team=team_choice,
        focal_player_id=str(moment_row["player_id"]),
        ball_proximity_radius_m=float(ball_prox_m),
        weights=weights,
    )

    # Top Window Summary Metrics
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric(
        "Video Timestamp",
        f"{moment_row['video_time']} (H{int(moment_row['half'])})",
        f"Nearest frame: {window_info['anchor_offset_ms']:+d} ms",
    )
    col_m2.metric(
        "Actual Available Window",
        f"{window_info['actual_duration_s']:.1f} s ({window_info['n_frames']} frames)",
        f"Requested: {window_info['requested_duration_s']:.1f} s",
    )
    col_m3.metric(
        "Ball Tracking Coverage",
        f"{window_info['ball_coverage_pct']:.0f}%",
        f"{window_info['ball_valid_frames']}/{window_info['n_frames']} frames located",
    )
    eligible_count = int(runs_df["is_eligible_off_ball"].sum()) if not runs_df.empty else 0
    col_m4.metric(
        "Eligible Off-Ball Runners",
        f"{eligible_count} outfield",
        f"From {len(runs_df)} tracked outfielders",
    )
    top_runner_str = (
        f"{runs_df.iloc[0]['player_id']} ({runs_df.iloc[0]['heuristic_score']:.1f})"
        if eligible_count > 0
        else "Unavailable"
    )
    col_m5.metric("Top Ranked Space Creator", top_runner_str, "Heuristic Score / 100")

    if window_info["ball_coverage_pct"] < 50.0:
        st.warning(
            f"**Data Availability Notice:** Ball coordinates (`ball_x`, `ball_y`) are only present in "
            f"**{window_info['ball_valid_frames']}/{window_info['n_frames']} frames "
            f"({window_info['ball_coverage_pct']:.0f}%)** during this window. "
            "Rather than fabricating 0 or NaN rankings, off-ball eligibility and composite scores "
            "that depend on ball separation are marked **Unavailable**."
        )

    if runs_df.empty:
        st.error("No outfield player tracking available for this moment.")
        return

    # 4th Control: Select Attacking Player to Inspect
    player_options = runs_df["player_id"].tolist()

    def format_player_option(pid: str) -> str:
        r = runs_df[runs_df["player_id"] == pid].iloc[0]
        if pd.notna(r["off_ball_rank"]):
            return (
                f"Rank #{int(r['off_ball_rank'])}: {pid} — "
                f"Score {r['heuristic_score']:.1f} (Δx_att {r['forward_disp_m']:+.1f} m, "
                f"ΔDefSep {r['def_sep_change_m']:+.1f} m)"
            )
        return f"[Unranked] {pid} — {r['eligibility_status']}"

    selected_player_id = st.selectbox(
        "4. Inspect Attacking Outfield Player (Pitch Highlight & Component Breakdown)",
        options=player_options,
        index=0,
        format_func=format_player_option,
    )

    sel_player_row = runs_df[runs_df["player_id"] == selected_player_id].iloc[0]

    # Main Split Layout: Pitch Visualization (Left) + Player/Defender Inspector (Right)
    left_col, right_col = st.columns([1.45, 1.0])

    with left_col:
        pitch_fig = build_pitch_figure(
            window_info=window_info,
            runs_df=runs_df,
            selected_player_id=selected_player_id,
            attacking_team=team_choice,
            attack_dir=int(moment_row["attack_dir"]),
            display_mode=display_mode,
            moment_row=moment_row,
        )
        st.plotly_chart(pitch_fig, use_container_width=True)

    with right_col:
        st.subheader(f"🔍 Runner & Defender Inspector: `{selected_player_id}`")
        st.markdown(f"**Eligibility Status:** {sel_player_row['eligibility_status']}")
        st.markdown(
            f"**Tracked Segment Duration:** `{sel_player_row['available_duration_s']:.1f} s` "
            f"(`{int(sel_player_row['n_frames'])}` continuous frames in Half {int(moment_row['half'])})"
        )

        c1, c2 = st.columns(2)
        c1.metric(
            "Forward Attacking Gain (Δx_att)",
            f"{sel_player_row['forward_disp_m']:+.1f} m",
            f"Total path covered: {sel_player_row['path_dist_m']:.1f} m",
        )
        def_sep_val = (
            f"{sel_player_row['def_sep_change_m']:+.1f} m"
            if pd.notna(sel_player_row["def_sep_change_m"])
            else "Unavailable"
        )
        c2.metric(
            "Nearest-Defender Separation Δ",
            def_sep_val,
            f"Start: {sel_player_row['nearest_def_start_m']:.1f}m ({sel_player_row['nearest_def_start_id']}) → "
            f"End: {sel_player_row['nearest_def_end_m']:.1f}m ({sel_player_row['nearest_def_end_id']})"
            if pd.notna(sel_player_row["def_sep_change_m"])
            else "Defender coords missing",
        )

        c3, c4 = st.columns(2)
        ball_dist_val = (
            f"{sel_player_row['mean_ball_dist_m']:.1f} m"
            if pd.notna(sel_player_row["mean_ball_dist_m"])
            else "Unavailable"
        )
        c3.metric(
            "Mean Distance from Ball",
            ball_dist_val,
            f"Min dist: {sel_player_row['min_ball_dist_m']:.1f} m"
            if pd.notna(sel_player_row["min_ball_dist_m"])
            else "Ball coords missing",
        )
        c4.metric(
            "Peak Speed Zone",
            f"Zone {int(sel_player_row['peak_speed_zone'])}",
            sel_player_row["peak_speed_label"],
        )

        if pd.notna(sel_player_row["initial_marker_disp_m"]):
            st.info(
                f"**Defender Co-Movement Note (Observational Only):** Initial nearest outfield defender "
                f"`{sel_player_row['nearest_def_start_id']}` displaced `{sel_player_row['initial_marker_disp_m']:.1f} m` "
                f"over this window, and end-of-window nearest defender is `{sel_player_row['nearest_def_end_id']}`. "
                "Tracking co-movement is descriptive and **does not prove causation** (defenders also react to the ball, "
                "teammates, and zonal shape)."
            )

        if sel_player_row["metrics_available"]:
            st.markdown("#### Heuristic Score Breakdown (0–100 scale)")
            comp_df = pd.DataFrame(
                {
                    "Component": [
                        f"Forward Gain ({w_forward:.0%} wt)",
                        f"Defender Sep Δ ({w_def:.0%} wt)",
                        f"Ball Separation ({w_ball:.0%} wt)",
                        f"Tempo & Effort ({w_tempo:.0%} wt)",
                    ],
                    "Sub-Score (0-100)": [
                        round(float(sel_player_row["forward_score"]), 1),
                        round(float(sel_player_row["def_sep_score"]), 1),
                        round(float(sel_player_row["ball_dist_score"]), 1),
                        round(float(sel_player_row["tempo_score"]), 1),
                    ],
                    "Raw Value": [
                        f"{sel_player_row['forward_disp_m']:+.1f} m forward",
                        f"{sel_player_row['def_sep_change_m']:+.1f} m sep change",
                        f"{sel_player_row['mean_ball_dist_m']:.1f} m mean dist",
                        f"{sel_player_row['path_dist_m']:.1f} m / Zone {int(sel_player_row['peak_speed_zone'])}",
                    ],
                }
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "Heuristic component breakdown is **Unavailable** for this player because required "
                "ball or defender tracking frames are missing in this window."
            )

    # Frame-by-frame timeline chart for selected player
    timeline_df = an.compute_player_defender_timeline(
        window_info=window_info,
        attacker_id=selected_player_id,
        attacking_team=team_choice,
    )
    if not timeline_df.empty:
        st.plotly_chart(
            build_timeline_figure(timeline_df, selected_player_id),
            use_container_width=True,
        )

    # Candidate Off-Ball Runs Ranking Table
    st.subheader("📋 Candidate Off-Ball Runs Ranking & Eligibility Table")
    st.caption(
        "*Note on Off-Ball Eligibility:* `events.csv` is a sample of 1,289 on-ball actions and **cannot prove a player "
        "had no touch**. Eligibility combines recorded events with tracking ball-proximity (`distance_to_ball_m > radius`). "
        "Goalkeepers and the referee are excluded from outfield space-creation metrics."
    )

    display_table = pd.DataFrame(
        {
            "Rank": runs_df["off_ball_rank"].apply(
                lambda r: f"#{int(r)}" if pd.notna(r) else "Unranked"
            ),
            "Player": runs_df["player_id"],
            "Heuristic Score": runs_df["heuristic_score"].apply(
                lambda s: f"{s:.1f}" if pd.notna(s) else "Unavailable"
            ),
            "Forward Gain Δx_att (m)": runs_df["forward_disp_m"].apply(
                lambda x: f"{x:+.1f} m"
            ),
            "Defender Sep Δ (m)": runs_df["def_sep_change_m"].apply(
                lambda d: f"{d:+.1f} m" if pd.notna(d) else "Unavailable"
            ),
            "Nearest Def (Start → End)": runs_df.apply(
                lambda r: (
                    f"{r['nearest_def_start_id']} ({r['nearest_def_start_m']:.1f}m) → "
                    f"{r['nearest_def_end_id']} ({r['nearest_def_end_m']:.1f}m)"
                    if pd.notna(r["def_sep_change_m"])
                    else "Unavailable"
                ),
                axis=1,
            ),
            "Mean Ball Dist (m)": runs_df["mean_ball_dist_m"].apply(
                lambda b: f"{b:.1f} m" if pd.notna(b) else "Unavailable"
            ),
            "Path Dist (m)": runs_df["path_dist_m"].apply(lambda p: f"{p:.1f} m"),
            "Avail Duration (s)": runs_df["available_duration_s"].apply(
                lambda t: f"{t:.1f} s"
            ),
            "Eligibility & Data Status": runs_df["eligibility_status"],
        }
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
