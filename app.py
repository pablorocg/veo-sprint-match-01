"""Accessible Coach Match-Review Dashboard for Veo Football Analytics Sprint (Match 01)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analytics as an

st.set_page_config(
    page_title="Veo Coach Match Review Dashboard",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner="Loading Veo match datasets & precomputing review catalog...")
def get_cached_dashboard_data() -> dict:
    summary, events, players = an.load_datasets(".")
    team_metrics = an.compute_team_summary_metrics(summary, events, players)
    player_catalog = an.build_player_catalog(summary, players)
    moments_catalog = an.build_key_moments_catalog(events, players)
    return {
        "summary": summary,
        "events": events,
        "players": players,
        "team_metrics": team_metrics,
        "player_catalog": player_catalog,
        "moments_catalog": moments_catalog,
    }


def main() -> None:
    data = get_cached_dashboard_data()
    summary: pd.DataFrame = data["summary"]
    events: pd.DataFrame = data["events"]
    players: pd.DataFrame = data["players"]
    team_metrics: dict = data["team_metrics"]
    player_catalog: pd.DataFrame = data["player_catalog"]
    moments_catalog: pd.DataFrame = data["moments_catalog"]

    # --- HEADER ---
    st.title("⚽ Veo Amateur Coach Match-Review Dashboard")
    st.markdown(
        "**Match 01 Final Score: Away 4 – 2 Home** *(All 6 goals scored in Half 2)* · "
        "11v11 Adult (96 min in-play) · Pitch: 105.0 m × 67.7 m"
    )

    with st.expander("ℹ️ Coach's Guide: Clocks, Coverage & Data Caveats (Read First)", expanded=False):
        st.markdown(
            """
            - **Video Clock vs. Inferred Half-Elapsed Time:** `timestamp_ms` and `minute` in the source files follow the **video recording clock**, not an official referee match clock. Kick-off is at **05:51** (`351.2 s`) on the video and Half 2 starts at **59:45** (`3585.2 s`). Every moment below lists the exact **Video Timestamp** (to scrub `match_01_full.mp4`) alongside **inferred time elapsed since the start of that half** (labeled as inferred, not official match time).
            - **Recorded Event Share vs. Possession:** `events.csv` is a detected sample of 1,289 on-ball actions (`48.7%` Away vs. `51.3%` Home share of recorded on-ball events). This measures share of detected events, **not** true ball possession percentage, and missing event rows do not establish absence of touches or possession.
            - **Summary Minutes vs. Per-Half Tracking Coverage:** `player_summary.csv` reports total `minutes_played`, which hides substitutions. Use per-half tracked frame counts (`1,500 frames ≈ 10 min` at 2.5 Hz) before comparing Half 1 vs. Half 2 running rates.
            - **Goals vs. Shots Schema:** All **6 goals** are recorded as `event_name == "goal"` rows (which carry **no coordinates**). Only **4 of the 29 shots** have `is_goal == True` attached; 2 goals (`away_8` at 66:10 and `home_18` at 94:32) have no shot row in `events.csv`.
            - **Spatial Heuristics vs. Established Findings:** Descriptive summaries and single-frame defensive width/depth ("Stretched Defense" heuristic flags from `role == "outfield"`) highlight clips to review; they do **not** establish causal reasons for why a team won. About 12% of frames have unlocated ball coordinates (`ball_x`/`ball_y` null).
            """
        )

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🎯 Coach Review Controls")
    selected_team_filter = st.sidebar.radio(
        "Team Filter",
        options=["Both Teams", "away", "home"],
        format_func=lambda x: "Both Teams (Away & Home)" if x == "Both Teams" else f"{x.upper()} ({'Red' if x == 'away' else 'Blue'})",
        index=0,
    )

    selected_half_filter = st.sidebar.radio(
        "Half Filter",
        options=["Both Halves", "Half 1", "Half 2"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("🎬 Key Moment Filters")
    moment_category = st.sidebar.selectbox(
        "What should I review next?",
        options=[
            "All Goals (6)",
            "All Goals & Shots (35)",
            "Shots Only (29)",
            "Stretched Defense Moments (Heuristic)",
            "Close-Range Shots (<= 15 m)",
        ],
        index=0,
    )

    search_query = st.sidebar.text_input(
        "Search Moment (player_id, clock, tag)",
        value="",
        placeholder="e.g. away_9, 83:02, Stretched",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🏃 Player Inspector Selection")

    if selected_team_filter == "Both Teams":
        avail_players = player_catalog["player_id"].tolist()
    else:
        avail_players = player_catalog[player_catalog["team"] == selected_team_filter]["player_id"].tolist()

    default_player_idx = avail_players.index("away_9") if "away_9" in avail_players else 0
    selected_player_id = st.sidebar.selectbox(
        "Select Player for Pitch & Stats",
        options=avail_players,
        index=default_player_idx,
    )

    # --- SECTION 1: TRUSTWORTHY MATCH SUMMARY CARDS ---
    st.subheader("1. Descriptive Match Summary Cards (`player_summary.csv` & `events.csv`)")
    away_m = team_metrics["away"]
    home_m = team_metrics["home"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "Goals (event_name='goal')",
        f"Away {away_m['goals']} – {home_m['goals']} Home",
        "All 6 in Half 2",
    )
    c2.metric(
        "Total Shots (29)",
        f"Away {away_m['shots']} – {home_m['shots']} Home",
        f"Close (<=15m): {away_m['close_range_shots_le_15m']} vs {home_m['close_range_shots_le_15m']}",
    )
    c3.metric(
        "Mean Shot Distance (m)",
        f"Away {away_m['mean_shot_distance_m']} m",
        f"Home {home_m['mean_shot_distance_m']} m (5.3 m farther)",
    )
    c4.metric(
        "Pass Completion (%)",
        f"Away {away_m['pass_completion_pct']}%",
        f"Home {home_m['pass_completion_pct']}% ({away_m['passes']} vs {home_m['passes']})",
    )
    c5.metric(
        "Recorded Event Share",
        f"Away {away_m['event_share_pct']}%",
        f"Home {home_m['event_share_pct']}% (of 1,289 events)",
    )
    c6.metric(
        "Outfield Distance (km)",
        f"Away {away_m['outfield_distance_km']} km",
        f"Home {home_m['outfield_distance_km']} km (GK/Ref excl.)",
    )

    st.info(
        "💡 **Descriptive Summary Observation (What to Review Next):** Aggregate volume metrics were close between teams "
        f"(pass completion {home_m['pass_completion_pct']}% Home vs {away_m['pass_completion_pct']}% Away; share of recorded on-ball events "
        f"{home_m['event_share_pct']}% Home vs {away_m['event_share_pct']}% Away [note: share of detected events, not ball possession]; "
        f"shots {home_m['shots']} vs {away_m['shots']}; outfield distance {home_m['outfield_distance_km']} km vs {away_m['outfield_distance_km']} km; "
        f"tackles/interceptions {home_m['tackles']}T/{home_m['interceptions']}I Home vs {away_m['tackles']}T/{away_m['interceptions']}I Away). "
        "While descriptive summaries cannot establish why a team won, they point to specific differences worth reviewing on video: "
        f"Away recorded **8 shots within 15 m** (mean shot distance {away_m['mean_shot_distance_m']} m) versus **3 within 15 m** for Home "
        f"(mean {home_m['mean_shot_distance_m']} m), and substitute `away_9` scored 3 goals in 24.7 summary minutes."
    )

    st.divider()

    # --- SECTION 2: WHAT SHOULD I REVIEW NEXT? (KEY MOMENTS INSPECTOR) ---
    st.subheader("2. 'What Should I Review Next?' — Selectable Key Moments & Video Timestamps")

    filtered_moments = moments_catalog.copy()
    if selected_team_filter != "Both Teams":
        filtered_moments = filtered_moments[filtered_moments["team"] == selected_team_filter]
    if selected_half_filter == "Half 1":
        filtered_moments = filtered_moments[filtered_moments["half"] == 1]
    elif selected_half_filter == "Half 2":
        filtered_moments = filtered_moments[filtered_moments["half"] == 2]

    if moment_category == "All Goals (6)":
        filtered_moments = filtered_moments[filtered_moments["moment_type"] == "Goal"]
    elif moment_category == "Shots Only (29)":
        filtered_moments = filtered_moments[filtered_moments["moment_type"] == "Shot"]
    elif moment_category == "Stretched Defense Moments (Heuristic)":
        filtered_moments = filtered_moments[filtered_moments["is_stretched_heuristic"] == True]  # noqa: E712
    elif moment_category == "Close-Range Shots (<= 15 m)":
        filtered_moments = filtered_moments[
            (filtered_moments["moment_type"] == "Shot") & (filtered_moments["shot_distance_m"] <= 15.0)
        ]

    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            filtered_moments["label"].str.lower().str.contains(q, na=False)
            | filtered_moments["player_id"].str.lower().str.contains(q, na=False)
            | filtered_moments["tags"].str.lower().str.contains(q, na=False)
            | filtered_moments["video_clock"].str.lower().str.contains(q, na=False)
        )
        filtered_moments = filtered_moments[mask]

    if filtered_moments.empty:
        st.warning("No moments match the current sidebar filters. Resetting to all 6 Goals for display.")
        filtered_moments = moments_catalog[moments_catalog["moment_type"] == "Goal"].copy()

    col_sel, col_opt1, col_opt2, col_opt3 = st.columns([3.2, 1.2, 1.2, 1.2])
    with col_sel:
        moment_options = filtered_moments["moment_key"].tolist()
        moment_labels = dict(zip(filtered_moments["moment_key"], filtered_moments["label"]))
        selected_moment_key = st.selectbox(
            f"Select a Key Moment ({len(moment_options)} matching moments)",
            options=moment_options,
            format_func=lambda k: moment_labels.get(k, k),
        )
    with col_opt1:
        lead_seconds = st.slider("Lead-up Window (s)", min_value=5, max_value=25, value=15, step=5)
    with col_opt2:
        show_trails = st.checkbox("Show Player Trails", value=True)
    with col_opt3:
        highlight_back4 = st.checkbox("Show Deepest-4 Line", value=True)

    active_moment = filtered_moments[filtered_moments["moment_key"] == selected_moment_key].iloc[0]
    win_data = an.get_moment_window_data(events, players, active_moment, lead_seconds=lead_seconds)

    m_left, m_right = st.columns([1.65, 1.15])
    with m_left:
        fig_moment = an.plot_moment_pitch(
            active_moment,
            win_data,
            show_trails=show_trails,
            highlight_back4=highlight_back4,
        )
        st.pyplot(fig_moment, width="stretch")
        plt.close(fig_moment)

    with m_right:
        st.markdown(f"#### 📋 Moment Breakdown: `{active_moment['moment_key']}`")
        k1, k2 = st.columns(2)
        k1.metric("Video Clock (`match_01_full.mp4`)", active_moment["video_clock"], f"{active_moment['timestamp_ms']} ms")
        k2.metric("Inferred Half-Elapsed Time (Non-Official)", active_moment["inplay_clock"], f"Half {active_moment['half']}")

        k3, k4 = st.columns(2)
        shot_d_str = f"{active_moment['shot_distance_m']} m" if pd.notna(active_moment["shot_distance_m"]) else "No shot row"
        shot_a_str = f"{active_moment['shot_angle_deg']}° angle" if pd.notna(active_moment["shot_angle_deg"]) else "Goal event only"
        k3.metric("Shot Distance / Angle", shot_d_str, shot_a_str)
        k4.metric(
            "Defending Outfield Block (Heuristic)",
            f"{active_moment['outfield_width_m']}m W × {active_moment['outfield_depth_m']}m D",
            f"Deepest-4 W: {active_moment['back4_width_m']} m",
        )

        st.markdown("**Coach-Friendly Observation:**")
        st.success(active_moment["coach_observation"])

        if active_moment["missing_ball_frames"] if "missing_ball_frames" in active_moment else win_data["missing_ball_frames"] > 0:
            st.caption(
                f"⚠️ **Ball Tracking Note:** Ball position was unlocated (`ball_x`/`ball_y` null) in "
                f"{win_data['missing_ball_frames']} of {win_data['total_window_frames']} tracking frames during this {lead_seconds}s window; "
                "trajectories are split at gaps rather than interpolated across missing frames."
            )

        st.markdown(f"**On-Ball Events in {lead_seconds}s Lead-Up (`sequence_id` = {active_moment['sequence_id']}):**")
        ev_w = win_data["events_window"]
        if not ev_w.empty:
            disp_ev = ev_w[
                ["t_rel_s", "video_clock", "event_name", "team", "player_id", "receiver_player_id", "pass_distance_m", "shot_distance_m"]
            ].rename(
                columns={
                    "t_rel_s": "Rel (s)",
                    "video_clock": "Video",
                    "event_name": "Event",
                    "team": "Team",
                    "player_id": "Player",
                    "receiver_player_id": "Receiver",
                    "pass_distance_m": "Pass (m)",
                    "shot_distance_m": "Shot (m)",
                }
            )
            st.dataframe(disp_ev, width="stretch", hide_index=True)
        else:
            st.caption("No detected on-ball events in this window (note: missing event rows do not prove absence of touches).")

    with st.expander("📊 View Filtered Key Moments Table (Video Clocks, Shot Distances & Defensive Heuristics)", expanded=False):
        table_cols = [
            "moment_type",
            "video_clock",
            "inplay_clock",
            "half",
            "team",
            "player_id",
            "shot_distance_m",
            "shot_angle_deg",
            "has_shot_row",
            "outfield_width_m",
            "outfield_depth_m",
            "back4_width_m",
            "is_stretched_heuristic",
            "tags",
        ]
        st.dataframe(filtered_moments[table_cols], width="stretch", hide_index=True)

    st.divider()

    # --- SECTION 3: INTERACTIVE PLAYER PITCH & STATS INSPECTOR ---
    st.subheader(f"3. Player Performance & Spatial Profile — `{selected_player_id}`")

    p_row = player_catalog[player_catalog["player_id"] == selected_player_id].iloc[0]

    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    pc1.metric(
        "Summary Minutes Played",
        f"{p_row['minutes_played']} min",
        f"Role: {p_row['role']}",
    )
    pc2.metric(
        "Tracked Coverage (H1 / H2)",
        f"{p_row['h1_min_tracked']}m / {p_row['h2_min_tracked']}m",
        f"{int(p_row['h1_frames']):,} / {int(p_row['h2_frames']):,} frames",
    )
    pc3.metric(
        "Distance / Top Speed",
        f"{p_row['distance_km']} km",
        f"Max {p_row['top_speed_kmh']} km/h ({int(p_row['sprints'])} sprints)",
    )
    pc4.metric(
        "Passes / Completion",
        f"{int(p_row['passes_completed'])} / {int(p_row['passes'])}",
        f"{p_row['pass_completion_pct']}% ({int(p_row['touches'])} touches)",
    )
    pc5.metric(
        "Shots / Goals",
        f"{int(p_row['shots'])} Shots / {int(p_row['goals'])} Goals",
        f"Dribbles: {int(p_row['dribbles'])}",
    )
    pc6.metric(
        "Defensive Actions",
        f"{int(p_row['tackles'])}T · {int(p_row['interceptions'])}I · {int(p_row['duels'])}D",
        f"Hi-Speed Rate: {p_row['hi_speed_rate_pct']:.2f}%",
    )

    if not p_row["comparable_both_halves"]:
        st.caption(
            f"📌 **Coverage Caveat for `{selected_player_id}`:** This player has `{int(p_row['h1_frames']):,}` frames in Half 1 "
            f"(~{p_row['h1_min_tracked']} min) and `{int(p_row['h2_frames']):,}` frames in Half 2 (~{p_row['h2_min_tracked']} min). "
            "Because they were not tracked for ≥10 minutes (`1,500 frames`) in both halves, do not compare their raw H1 vs. H2 totals as fatigue."
        )
    else:
        delta_hi = p_row["h2_hi_pct"] - p_row["h1_hi_pct"]
        st.caption(
            f"✅ **Both-Halves Tracked (`≥10 min` per half):** `{selected_player_id}` high-intensity frame rate (`speed_zone >= 3`, 20+ km/h) "
            f"was **{p_row['h1_hi_pct']:.2f}%** in Half 1 vs. **{p_row['h2_hi_pct']:.2f}%** in Half 2 ({delta_hi:+.2f} percentage points)."
        )

    p_ctrl1, p_ctrl2, p_ctrl3 = st.columns([1.4, 1.4, 1.4])
    with p_ctrl1:
        player_view_mode = st.selectbox(
            "Pitch Visualization Layer",
            options=[
                "Heatmap + On-Ball Events",
                "Speed Zones + On-Ball Events",
                "Heatmap Only",
            ],
            index=0,
        )
    with p_ctrl2:
        player_half_view = st.selectbox(
            "Player Pitch Half Filter",
            options=["All", "Half 1", "Half 2"],
            index=0,
        )
    with p_ctrl3:
        normalize_dir = st.checkbox(
            "Normalize Attacking Direction (`attack_dir` → +x)",
            value=True,
            help="Multiplies (x, y) by attack_dir so both halves attack Left to Right (+x).",
        )

    p_left, p_right = st.columns([1.65, 1.15])
    with p_left:
        fig_player = an.plot_player_pitch_profile(
            players=players,
            events=events,
            player_id=selected_player_id,
            normalize_attack_dir=normalize_dir,
            half_filter=player_half_view,
            view_mode=player_view_mode,
        )
        st.pyplot(fig_player, width="stretch")
        plt.close(fig_player)

    with p_right:
        st.markdown("#### 📊 Squad Comparison Table (`player_summary.csv` + Tracked Coverage)")
        squad_df = player_catalog.copy()
        if selected_team_filter != "Both Teams":
            squad_df = squad_df[squad_df["team"] == selected_team_filter]
        squad_disp = squad_df[
            [
                "player_id",
                "team",
                "role",
                "minutes_played",
                "h1_min_tracked",
                "h2_min_tracked",
                "distance_km",
                "top_speed_kmh",
                "sprints",
                "passes",
                "pass_completion_pct",
                "shots",
                "goals",
                "tackles",
                "interceptions",
            ]
        ].sort_values(["team", "minutes_played"], ascending=[True, False])
        st.dataframe(squad_disp, width="stretch", hide_index=True, height=430)


if __name__ == "__main__":
    main()
