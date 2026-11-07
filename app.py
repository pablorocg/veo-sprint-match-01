"""Streamlit Dashboard: Veo Football Analytics Sprint — Passing Options Explorer.

Default local port: 8506
Launch: uv run streamlit run app.py --server.port 8506
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics

st.set_page_config(
    page_title="Veo Sprint | Passing Options Explorer",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner="Loading Veo match datasets...")
def get_cached_data() -> tuple[dict[str, pd.DataFrame], dict[int, object]]:
    datasets = analytics.load_datasets(".")
    frame_index = analytics.build_half_frame_index(datasets["players"])
    return datasets, frame_index


def format_pass_option_label(row: pd.Series) -> str:
    """Create a coach-friendly label for selecting a pass event."""
    ev_id = int(row["event_id"])
    clock = str(row["video_clock"])
    passer = str(row["player_id"]) if pd.notna(row["player_id"]) else "Unattributed"
    receiver = (
        f"→ {row['receiver_player_id']}"
        if pd.notna(row["receiver_player_id"])
        else "→ (no receiver ID)"
    )
    outcome = "✅ Completed" if row["pass_success"] else "❌ Incomplete"
    dist_str = (
        f"{float(row['pass_distance_m']):.1f} m"
        if pd.notna(row["pass_distance_m"])
        else "no end coords"
    )
    return f"#{ev_id} | {clock} (H{int(row['half'])}) | {passer} {receiver} | {dist_str} | {outcome}"


def build_pitch_figure(
    eval_res: dict,
    top_n_lanes: int = 3,
    selected_target_id: str | None = None,
) -> go.Figure:
    """Build an interactive Plotly 2D pitch figure showing actual pass and snapshot alternatives."""
    fig = go.Figure()

    hl = analytics.HALF_LENGTH_M
    hw = analytics.HALF_WIDTH_M

    # Pitch background and markings
    fig.update_layout(
        plot_bgcolor="#1e3d24",
        paper_bgcolor="#0e1117",
        font=dict(color="#f0f2f6"),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(
            range=[-hl - 5.0, hl + 5.0],
            title="Pitch Length x (metres, origin at centre spot)",
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1.0,
        ),
        yaxis=dict(
            range=[-hw - 5.0, hw + 5.0],
            title="Pitch Width y (metres)",
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(14,17,23,0.8)",
        ),
        height=620,
    )

    line_style = dict(color="rgba(255, 255, 255, 0.65)", width=2)

    # Outer boundary & halfway line
    shapes = [
        dict(type="rect", x0=-hl, y0=-hw, x1=hl, y1=hw, line=line_style, layer="below"),
        dict(type="line", x0=0, y0=-hw, x1=0, y1=hw, line=line_style, layer="below"),
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line_style, layer="below"),
    ]
    # Penalty boxes and 6-yard boxes
    for s in (-1, 1):
        shapes.append(
            dict(
                type="rect",
                x0=s * hl,
                y0=-20.15,
                x1=s * hl - s * 16.5,
                y1=20.15,
                line=line_style,
                layer="below",
            )
        )
        shapes.append(
            dict(
                type="rect",
                x0=s * hl,
                y0=-9.16,
                x1=s * hl - s * 5.5,
                y1=9.16,
                line=line_style,
                layer="below",
            )
        )
        # Goal mouth (7.32 m)
        shapes.append(
            dict(
                type="rect",
                x0=s * hl,
                y0=-3.66,
                x1=s * (hl + 1.8),
                y1=3.66,
                line=dict(color="rgba(255, 255, 255, 0.9)", width=2),
                layer="below",
            )
        )

    fig.update_layout(shapes=shapes)

    # Attacking direction banner annotation on pitch
    attack_dir = eval_res["attack_dir"]
    team = eval_res["team"]
    if attack_dir in (-1, 1) and team in ("home", "away"):
        arrow_txt = (
            f"{team.upper()} ATTACKING DIRECTION (+x ➔)"
            if attack_dir == 1
            else f"(⬅ -x) {team.upper()} ATTACKING DIRECTION"
        )
        fig.add_annotation(
            x=0,
            y=hw + 2.6,
            text=f"<b>{arrow_txt}</b>",
            showarrow=False,
            font=dict(size=13, color="#ffd166"),
            bgcolor="rgba(0,0,0,0.55)",
            borderpad=4,
        )

    passer_row = eval_res["passer_row"]
    candidates_df = eval_res["candidates_df"]
    actual_pass = eval_res["actual_pass"]

    # 1. Draw top-N heuristic alternative lanes (dashed lines from tracked passer to tracked teammates)
    if passer_row is not None and not candidates_df.empty:
        px = float(passer_row["x"])
        py = float(passer_row["y"])

        top_cands = candidates_df.head(top_n_lanes)
        for idx, cand in top_cands.iterrows():
            is_sel = selected_target_id == cand["target_player_id"]
            lane_color = "#00f5d4" if is_sel else "#ffd166"
            lane_width = 3.5 if is_sel else 2.0
            fig.add_trace(
                go.Scatter(
                    x=[px, float(cand["target_x"])],
                    y=[py, float(cand["target_y"])],
                    mode="lines",
                    line=dict(color=lane_color, width=lane_width, dash="dash"),
                    name=f"Alt #{int(cand['rank'])}: {cand['target_player_id']} ({cand['composite_score']:.0f} pts)",
                    hoverinfo="text",
                    text=(
                        f"Heuristic Option #{int(cand['rank'])} -> {cand['target_player_id']}<br>"
                        f"Score: {cand['composite_score']:.1f} / 100<br>"
                        f"Lane Clearance: {cand['lane_clearance_m']:.2f} m (Def: {cand['nearest_lane_defender_id']})<br>"
                        f"Target Separation: {cand['target_separation_m']:.2f} m<br>"
                        f"Forward Progress: {cand['forward_progress_m']:+.1f} m"
                    ),
                )
            )

        # If a specific target is selected (even outside top_n_lanes), draw its clamped defender projection
        if selected_target_id is not None:
            sel_rows = candidates_df[candidates_df["target_player_id"] == selected_target_id]
            if not sel_rows.empty:
                sel = sel_rows.iloc[0]
                if int(sel["rank"]) > top_n_lanes:
                    fig.add_trace(
                        go.Scatter(
                            x=[px, float(sel["target_x"])],
                            y=[py, float(sel["target_y"])],
                            mode="lines",
                            line=dict(color="#00f5d4", width=3.5, dash="dash"),
                            name=f"Selected Alt #{int(sel['rank'])}: {sel['target_player_id']} ({sel['composite_score']:.0f} pts)",
                        )
                    )
                # Clamped projection segment from nearest lane defender to passing lane
                if pd.notna(sel["lane_proj_x"]) and sel["nearest_lane_defender_id"] is not None:
                    opp_df = eval_res["opponents_df"]
                    def_rows = opp_df[opp_df["player_id"] == sel["nearest_lane_defender_id"]]
                    if not def_rows.empty:
                        dx = float(def_rows.iloc[0]["x"])
                        dy = float(def_rows.iloc[0]["y"])
                        fig.add_trace(
                            go.Scatter(
                                x=[dx, float(sel["lane_proj_x"])],
                                y=[dy, float(sel["lane_proj_y"])],
                                mode="lines+markers",
                                line=dict(color="#ff4d6d", width=2.5, dash="dot"),
                                marker=dict(size=7, symbol="x", color="#ff4d6d"),
                                name=f"Clamped Lane Projection ({sel['lane_clearance_m']:.2f} m)",
                                hoverinfo="text",
                                text=(
                                    f"Nearest Lane Defender: {sel['nearest_lane_defender_id']}<br>"
                                    f"Clamped Distance to Segment: {sel['lane_clearance_m']:.2f} m<br>"
                                    f"Segment Parameter t: {sel['lane_proj_t']:.2f}"
                                ),
                            )
                        )

    # 2. Draw ACTUAL event pass from events.csv (solid line/arrow)
    if actual_pass["has_start"] and actual_pass["has_end"]:
        act_color = "#2ec4b6" if actual_pass["pass_success"] else "#ff9f1c"
        outcome_lbl = "Completed" if actual_pass["pass_success"] else "Incomplete"
        fig.add_trace(
            go.Scatter(
                x=[actual_pass["x_start"], actual_pass["x_end"]],
                y=[actual_pass["y_start"], actual_pass["y_end"]],
                mode="lines+markers",
                line=dict(color=act_color, width=4.5),
                marker=dict(size=[9, 13], symbol=["circle-open", "diamond"], color=act_color),
                name=f"Actual Event Pass ({outcome_lbl})",
                hoverinfo="text",
                text=(
                    f"ACTUAL PASS (events.csv)<br>"
                    f"Start: ({actual_pass['x_start']:.1f}, {actual_pass['y_start']:.1f}) m<br>"
                    f"End (receiver_x/y): ({actual_pass['x_end']:.1f}, {actual_pass['y_end']:.1f}) m<br>"
                    f"Receiver: {actual_pass['receiver_player_id'] or 'Unattributed'}<br>"
                    f"Outcome: {outcome_lbl}"
                ),
            )
        )
    elif actual_pass["has_start"]:
        fig.add_trace(
            go.Scatter(
                x=[actual_pass["x_start"]],
                y=[actual_pass["y_start"]],
                mode="markers",
                marker=dict(size=12, symbol="circle-open", color="#ff9f1c", line=dict(width=3)),
                name="Actual Pass Start (End Coords Null)",
                hoverinfo="text",
                text=(
                    f"ACTUAL PASS START (events.csv)<br>"
                    f"({actual_pass['x_start']:.1f}, {actual_pass['y_start']:.1f}) m<br>"
                    f"End coordinates (receiver_x/y) are null on this incomplete pass."
                ),
            )
        )

    # 3. Draw Opponents
    opp_df = eval_res["opponents_df"]
    if not opp_df.empty:
        opp_team = str(opp_df["team"].iloc[0])
        opp_color = analytics.TEAM_COLORS.get(opp_team, "#d62728")
        opp_labels = [
            str(pid).split("_")[-1] if pd.notna(pid) else "?"
            for pid in opp_df["player_id"]
        ]
        opp_hover = [
            f"Opponent: {pid if pd.notna(pid) else 'Unattributed'} ({role})<br>Pos: ({x:.1f}, {y:.1f}) m"
            for pid, role, x, y in zip(opp_df["player_id"], opp_df["role"], opp_df["x"], opp_df["y"])
        ]
        fig.add_trace(
            go.Scatter(
                x=opp_df["x"],
                y=opp_df["y"],
                mode="markers+text",
                marker=dict(size=16, color=opp_color, line=dict(color="white", width=1.5)),
                text=opp_labels,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                name=f"Opponents ({opp_team})",
                hoverinfo="text",
                hovertext=opp_hover,
            )
        )

    # 4. Draw Referee (excluded from options)
    ref_df = eval_res["referee_df"]
    if not ref_df.empty:
        fig.add_trace(
            go.Scatter(
                x=ref_df["x"],
                y=ref_df["y"],
                mode="markers",
                marker=dict(size=11, color=analytics.TEAM_COLORS["referee"], symbol="square"),
                name="Referee (Excluded)",
                hoverinfo="text",
                text=[f"Referee (Excluded)<br>({x:.1f}, {y:.1f}) m" for x, y in zip(ref_df["x"], ref_df["y"])],
            )
        )

    # 5. Draw Candidate Teammates
    if not candidates_df.empty and team in ("home", "away"):
        tm_color = analytics.TEAM_COLORS.get(team, "#1f77b4")
        tm_labels = [str(pid).split("_")[-1] for pid in candidates_df["target_player_id"]]
        tm_hover = [
            (
                f"Teammate Option #{int(r.rank)}: {r.target_player_id} ({r.role})<br>"
                f"Pos: ({r.target_x:.1f}, {r.target_y:.1f}) m<br>"
                f"Heuristic Score: {r.composite_score:.1f} / 100<br>"
                f"Lane Clearance: {r.lane_clearance_m:.2f} m | Target Sep: {r.target_separation_m:.2f} m<br>"
                f"Forward Progress: {r.forward_progress_m:+.1f} m"
            )
            for r in candidates_df.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=candidates_df["target_x"],
                y=candidates_df["target_y"],
                mode="markers+text",
                marker=dict(size=18, color=tm_color, line=dict(color="#ffd166", width=2)),
                text=tm_labels,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                name=f"Teammate Candidates ({team})",
                hoverinfo="text",
                hovertext=tm_hover,
            )
        )

    # 6. Draw Tracked Passer
    if passer_row is not None and team in ("home", "away"):
        tm_color = analytics.TEAM_COLORS.get(team, "#1f77b4")
        p_num = str(passer_row["player_id"]).split("_")[-1]
        fig.add_trace(
            go.Scatter(
                x=[float(passer_row["x"])],
                y=[float(passer_row["y"])],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=tm_color,
                    symbol="star",
                    line=dict(color="white", width=2.5),
                ),
                text=[p_num],
                textposition="top center",
                textfont=dict(size=12, color="#ffd166"),
                name=f"Tracked Passer ({passer_row['player_id']})",
                hoverinfo="text",
                hovertext=[
                    f"Tracked Passer: {passer_row['player_id']}<br>"
                    f"Frame Pos: ({float(passer_row['x']):.1f}, {float(passer_row['y']):.1f}) m"
                ],
            )
        )

    # 7. Draw Tracked Ball if present
    if eval_res["ball_pos"] is not None:
        bx, by = eval_res["ball_pos"]
        fig.add_trace(
            go.Scatter(
                x=[bx],
                y=[by],
                mode="markers",
                marker=dict(size=10, color="white", line=dict(color="black", width=2)),
                name="Tracked Ball (players.parquet)",
                hoverinfo="text",
                text=[f"Tracked Ball: ({bx:.1f}, {by:.1f}) m"],
            )
        )

    return fig


def main() -> None:
    datasets, frame_index = get_cached_data()
    events_df = datasets["events"]
    players_df = datasets["players"]

    st.title("⚽ Veo Match 01: Passing Options Explorer")
    st.markdown(
        "**Compare factual on-ball pass events (`events.csv`) against snapshot-based tracked teammate alternatives (`players.parquet`).** "
        "All alternative lanes use explainable 2D geometry at the nearest same-half tracking frame—**heuristic spatial options, "
        "not expected pass completion (xPass) or ball-physics recommendations**."
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("1. Select Pass Moment")
        selected_team = st.radio(
            "Attacking Team",
            options=["home", "away"],
            format_func=lambda t: f"{t.capitalize()} ({'Blue' if t == 'home' else 'Red'})",
            horizontal=True,
        )
        selected_half = st.radio(
            "Match Half",
            options=[1, 2],
            format_func=lambda h: f"Half {h}",
            horizontal=True,
        )
        outcome_filter = st.selectbox(
            "Pass Outcome Filter",
            options=["all", "completed", "incomplete"],
            format_func=lambda o: {
                "all": "All Passes",
                "completed": "Completed Only (pass_success=True)",
                "incomplete": "Incomplete Only (pass_success=False)",
            }[o],
        )
        only_identified = st.checkbox(
            "Only passes with identified passer (player_id)",
            value=True,
            help="19 of 832 pass rows in events.csv have null player_id. Uncheck to inspect how unattributed passes are handled.",
        )

        st.divider()
        st.header("2. Frame & Candidate Rules")
        tolerance_ms = st.slider(
            "Same-Half Frame Tolerance (± ms)",
            min_value=50,
            max_value=1000,
            value=analytics.DEFAULT_FRAME_TOLERANCE_MS,
            step=50,
            help="Maximum allowed time difference between event timestamp_ms and nearest tracking frame in the same half.",
        )
        include_gk = st.checkbox(
            "Include Teammate Goalkeeper as Option",
            value=False,
            help="Outfield teammates are always included; check to allow passing back to the goalkeeper.",
        )
        top_n_lanes = st.slider(
            "Top Candidate Lanes Drawn on Pitch",
            min_value=1,
            max_value=10,
            value=3,
        )

        st.divider()
        st.header("3. Heuristic Weighting")
        st.caption(
            "Adjust how the three explainable spatial components are weighted (automatically normalized to 100%)."
        )
        w_lane = st.slider(
            "Lane Clearance Weight (Nearest Defender to Segment)",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
        )
        w_sep = st.slider(
            "Target Separation Weight (Receiver Pressure)",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
        )
        w_prog = st.slider(
            "Forward Progression Weight (Normalized by attack_dir)",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
        )

    filtered_passes = analytics.get_pass_events(
        events_df=events_df,
        team=selected_team,
        half=selected_half,
        outcome=outcome_filter,
        only_identified_passer=only_identified,
    )

    if filtered_passes.empty:
        st.warning("No pass events match the selected filters.")
        return

    # Pass event selector at top of main area
    pass_options = list(filtered_passes.index)
    selected_idx = st.selectbox(
        f"Choose a Real Pass Event ({len(filtered_passes)} matching events in Half {selected_half} for {selected_team.upper()})",
        options=pass_options,
        format_func=lambda i: format_pass_option_label(filtered_passes.loc[i]),
    )

    pass_row = filtered_passes.loc[selected_idx]
    eval_res = analytics.evaluate_pass_options(
        pass_event=pass_row,
        players_df=players_df,
        tolerance_ms=tolerance_ms,
        include_goalkeeper=include_gk,
        weights=(w_lane, w_sep, w_prog),
        frame_index=frame_index,
    )

    candidates_df = eval_res["candidates_df"]
    actual_pass = eval_res["actual_pass"]
    fl = eval_res["frame_lookup"]

    # Top summary metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Event Video Clock",
        f"{eval_res['video_clock']} (H{eval_res['half']})",
        f"{eval_res['timestamp_ms']} ms",
    )
    m2.metric(
        "Nearest Same-Half Frame",
        f"{fl['matched_timestamp_ms']} ms" if fl["matched_timestamp_ms"] else "None",
        f"{fl['offset_ms']:+d} ms offset (tol ±{tolerance_ms} ms)"
        if fl["offset_ms"] is not None
        else "No frame",
        delta_color="normal" if fl["within_tolerance"] else "inverse",
    )
    act_outcome_txt = (
        "Completed" if actual_pass["pass_success"] else "Incomplete"
    )
    act_dist_txt = (
        f"{actual_pass['pass_distance_m']:.1f} m"
        if actual_pass["pass_distance_m"] is not None
        else "End coords null"
    )
    m3.metric(
        f"Actual Pass ({eval_res['passer_id'] or 'Unattributed'})",
        f"→ {actual_pass['receiver_player_id'] or 'No ID'} ({act_outcome_txt})",
        f"Dist: {act_dist_txt} | Fwd: {actual_pass['forward_progress_m']:+.1f} m"
        if actual_pass["forward_progress_m"] is not None
        else f"Dist: {act_dist_txt}",
    )

    if not candidates_df.empty:
        top1 = candidates_df.iloc[0]
        m4.metric(
            "Top Heuristic Option (#1)",
            f"{top1['target_player_id']} ({top1['composite_score']:.1f} pts)",
            f"Lane: {top1['lane_clearance_m']:.1f}m | Sep: {top1['target_separation_m']:.1f}m | Fwd: {top1['forward_progress_m']:+.1f}m",
        )
    else:
        m4.metric("Top Heuristic Option (#1)", "No candidates", "See status warnings")

    for w_msg in eval_res["warnings"]:
        st.warning(w_msg)

    # Target inspection selector
    selected_target_id: str | None = None
    if not candidates_df.empty:
        target_choices = list(candidates_df["target_player_id"])
        default_target_idx = 0
        # Default to actual receiver if present among candidates, else rank 1
        if actual_pass["receiver_player_id"] in target_choices:
            default_target_idx = target_choices.index(actual_pass["receiver_player_id"])

        selected_target_id = st.selectbox(
            "Inspect Specific Candidate Target (highlights lane & clamped nearest-defender projection on pitch)",
            options=target_choices,
            index=default_target_idx,
            format_func=lambda tid: (
                f"#{int(candidates_df.loc[candidates_df['target_player_id'] == tid, 'rank'].iloc[0])} "
                f"{tid} — Score: {float(candidates_df.loc[candidates_df['target_player_id'] == tid, 'composite_score'].iloc[0]):.1f} pts "
                f"{'(ACTUAL EVENT RECEIVER)' if tid == actual_pass['receiver_player_id'] else ''}"
            ),
        )

    # Pitch + Inspector Columns
    col_pitch, col_info = st.columns([1.65, 1.0])

    with col_pitch:
        fig = build_pitch_figure(
            eval_res=eval_res,
            top_n_lanes=top_n_lanes,
            selected_target_id=selected_target_id,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("Factual Actual vs. Snapshot Alternative")
        st.info(
            "**Source & Timing Distinction:**\n"
            f"- **Actual Pass (`events.csv`):** Event #{eval_res['event_id']} at `{eval_res['timestamp_ms']} ms`. "
            "Start is `(x_start, y_start)`; end is `(receiver_x, receiver_y)` (`x_end`/`y_end` are null on pass rows in this dataset).\n"
            f"- **Tracked Snapshot (`players.parquet`):** Nearest Half {eval_res['half']} frame at "
            f"`{fl['matched_timestamp_ms']} ms` ({fl['offset_ms']:+d} ms offset). "
            + (
                f"Tracked passer `{eval_res['passer_id']}` is `{actual_pass['passer_event_offset_m']:.2f} m` from event start."
                if actual_pass["passer_event_offset_m"] is not None
                else "Passer position not available in tracking frame."
            )
        )

        if selected_target_id is not None and not candidates_df.empty:
            sel_row = candidates_df[candidates_df["target_player_id"] == selected_target_id].iloc[0]
            wn_lane, wn_sep, wn_prog = eval_res["weights_normalized"]
            st.markdown(f"#### Target Breakdown: `{selected_target_id}` (Rank #{int(sel_row['rank'])})")
            st.write(
                f"- **Composite Heuristic Score:** `{sel_row['composite_score']:.1f} / 100` "
                f"*(weights: {wn_lane*100:.0f}% lane, {wn_sep*100:.0f}% sep, {wn_prog*100:.0f}% prog)*\n"
                f"- **1. Clamped Lane Clearance:** `{sel_row['lane_clearance_m']:.2f} m` "
                f"(Nearest defender: `{sel_row['nearest_lane_defender_id']}`, projection parameter `t = {sel_row['lane_proj_t']:.2f}` along segment) → **{sel_row['lane_safety_score']:.1f} pts**\n"
                f"- **2. Target Separation (Pressure):** `{sel_row['target_separation_m']:.2f} m` "
                f"(Nearest defender to receiver: `{sel_row['nearest_target_defender_id']}`) → **{sel_row['target_sep_score']:.1f} pts**\n"
                f"- **3. Forward Progress (`attack_dir={eval_res['attack_dir']:+d}`):** `{sel_row['forward_progress_m']:+.2f} m` "
                f"(Pass distance: `{sel_row['pass_length_m']:.1f} m`) → **{sel_row['progression_score']:.1f} pts**"
            )

            # Bar chart of the 3 component scores for the selected candidate
            comp_fig = go.Figure(
                go.Bar(
                    x=["Lane Safety", "Target Separation", "Forward Progress", "Composite Score"],
                    y=[
                        sel_row["lane_safety_score"],
                        sel_row["target_sep_score"],
                        sel_row["progression_score"],
                        sel_row["composite_score"],
                    ],
                    marker_color=["#4cc9f0", "#4895ef", "#f72585", "#00f5d4"],
                    text=[
                        f"{sel_row['lane_safety_score']:.1f}",
                        f"{sel_row['target_sep_score']:.1f}",
                        f"{sel_row['progression_score']:.1f}",
                        f"{sel_row['composite_score']:.1f}",
                    ],
                    textposition="outside",
                )
            )
            comp_fig.update_layout(
                height=230,
                margin=dict(l=10, r=10, t=25, b=10),
                yaxis=dict(range=[0, 115], title="Score (0–100)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f2f6"),
            )
            st.plotly_chart(comp_fig, use_container_width=True)

    # Full Candidate Component Table
    st.subheader("All Snapshot Candidate Teammate Lanes & Component Scores")
    if not candidates_df.empty:
        display_df = candidates_df[
            [
                "rank",
                "target_player_id",
                "role",
                "is_actual_receiver",
                "composite_score",
                "lane_clearance_m",
                "nearest_lane_defender_id",
                "lane_proj_t",
                "lane_safety_score",
                "target_separation_m",
                "nearest_target_defender_id",
                "target_sep_score",
                "forward_progress_m",
                "pass_length_m",
                "progression_score",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "target_player_id": "Teammate ID",
                "role": "Role",
                "is_actual_receiver": "Actual Receiver?",
                "composite_score": "Heuristic Score (0-100)",
                "lane_clearance_m": "Lane Clearance (m)",
                "nearest_lane_defender_id": "Closest Lane Def",
                "lane_proj_t": "Segment Proj t (0-1)",
                "lane_safety_score": "Lane Score",
                "target_separation_m": "Target Sep (m)",
                "nearest_target_defender_id": "Closest Target Def",
                "target_sep_score": "Sep Score",
                "forward_progress_m": "Forward Gain (m)",
                "pass_length_m": "Pass Dist (m)",
                "progression_score": "Prog Score",
            }
        )
        st.dataframe(
            display_df.style.format(
                {
                    "Heuristic Score (0-100)": "{:.1f}",
                    "Lane Clearance (m)": "{:.2f}",
                    "Segment Proj t (0-1)": "{:.2f}",
                    "Lane Score": "{:.1f}",
                    "Target Sep (m)": "{:.2f}",
                    "Sep Score": "{:.1f}",
                    "Forward Gain (m)": "{:+.2f}",
                    "Pass Dist (m)": "{:.1f}",
                    "Prog Score": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No candidate rows to display for the current event and tolerance settings.")

    with st.expander("Methodology, Heuristic Assumptions & Caveats"):
        st.markdown(
            """
            - **Heuristic Geometric Lanes Only:** Candidate scores summarize static 2D positions at a single video frame (`players.parquet`). They **do not** account for player velocity vectors, body orientation, ball flight elevation/spin, or expected pass completion (`xPass`).
            - **Clamped Segment Projection (`0 ≤ t ≤ 1`):** For a passer at $A$ and candidate receiver at $B$, each opponent $P$ is projected onto segment $AB$ via $t = \\text{clip}\\left(\\frac{(P - A)\\cdot(B - A)}{\\|B - A\\|^2}, 0, 1\\right)$. The minimum Euclidean distance $\\|P - (A + t(B - A))\\|$ across all tracked opponents is the **Lane Clearance (m)** (saturating at $4.0\\text{ m} = 100\\text{ pts}$).
            - **Target Separation (Pressure):** Distance (m) from candidate teammate $B$ to their nearest tracked opponent (saturating at $10.0\\text{ m} = 100\\text{ pts}$).
            - **Normalized Forward Progress:** $\\Delta x_{\\text{att}} = (x_{\\text{target}} - x_{\\text{passer}}) \\times \\text{attack\\_dir}$ in metres, mapped linearly from $[-15\\text{ m}, +25\\text{ m}] \\to [0, 100]$.
            - **Exclusions & Missing Data:** The passer, referee (`role == 'referee'`), opponents, and unattributed tracking rows (`player_id` null) are excluded from candidate targets. Frame matching never crosses half boundaries (`half == 1` vs `half == 2`).
            """
        )


if __name__ == "__main__":
    main()
