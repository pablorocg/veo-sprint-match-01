# Defence-to-attack implementation contract

Plan: docs/superpowers/plans/2026-11-09-defence-to-danger.md
Spec: mvp_analysis_draft.md

User priorities: rigorous, transparent, effective player ranking; source-video evidence; Sankey/alluvial diagram. Use existing environment, preserve all inputs and user modifications. No commits/pushes or new permanent dependencies. Work in current shared workspace with strict file ownership. Memtrace is unavailable (MemDB endpoint unreachable); bounded local data/source inspection is the fallback. Do not attempt reindexing or stall on that service.

## Ownership
- Core worker: transition_analysis.py and validate_transition_analysis.py only.
- Visual/media worker: transition_visuals.py and validate_transition_visuals.py only.
- Methodology reviewer: docs/superpowers/implementation/ranking_methodology_review.md only; read-only access elsewhere.
- Coordinator: defence_to_danger.ipynb, documentation, generated CSVs/media and final integration. Do not edit worker-owned files until the worker has reported completion.

Ruling: the plan's notebook-only implementation becomes a root notebook plus two small root-level helper modules. This keeps extraction/ranking and plotting/media independently testable and avoids concurrent notebook edits. No application package or test framework.

Ruling: current Orca workspace is used in place because the video and user edits are here; workers own disjoint files. No checkout/worktree/branch mutations by workers.

## Data and scope
Python 3.12, pandas, numpy, Matplotlib, pyarrow are installed via uv. Source files: events.csv, player_summary.csv, players.parquet, match_01_full.mp4. Read DATA_DICTIONARY.md first. MP4 is 6622.616 seconds, 1280x720, ~29.97fps. ffmpeg/ffprobe installed. Video seconds=timestamp_ms/1000; exact player/video validation remains pending until inspected.

Baselines: 1289 events; 33 summary rows; 6 goal rows; only4 goal-flagged shots; 177 tackles/interceptions; 138 first in sequence; 536 completed passes,523 with receiver coords/ID. No pass has usable x_end/y_end; use receiver_x/receiver_y.

Initial feasibility, not final results: at15 seconds, stopping at restarts/failed passes/new intervention, only3 shots and10 final-third entry passes. Use headline 'Who turned defence into attack?' and avoid overall MVP/causal certainty.

## Shared API (core)
extract_transitions(events: pd.DataFrame, window_s: float=15.0) -> tuple[pd.DataFrame,pd.DataFrame]
attribute_transitions(transitions: pd.DataFrame, transition_events: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]
summarise_players(metrics: pd.DataFrame, transition_events: pd.DataFrame, contributions: pd.DataFrame, summary: pd.DataFrame, progression_threshold_m: float=10.0) -> pd.DataFrame
run_analysis(events: pd.DataFrame, summary: pd.DataFrame, window_s: float=15.0, progression_threshold_m: float=10.0) -> dict

run_analysis keys: transitions, transition_events, metrics, contributions, ranking (all DataFrames).
transitions: transition_id (opening event_id, stable across windows), sequence_id, team, half, start_player_id, start_ms, end_ms, start_x_att, start_zone, stop_reason, terminal_event_id.
transition_events: original event columns plus transition_id; include opening intervention and only in-window retained subsequent events. Preserve excluded terminal metadata on transitions.
metrics: transitions columns plus n_completed_passes, has_followup, shot_count, has_shot, goal_count, has_goal, final_third_entries, has_entry, forward_metres, has_progression, outcome_category, measurable_passes, missing_endpoint_passes.
contributions: transition_id,player_id,contribution_role,event_id (unique transition_id/player_id/role). Roles: recoverer,outlet,advancer,finisher. Advancer includes entry passers and receivers; fallback greatest positive forward passer if no entry. Unknown IDs stay unknown.

Stop BEFORE a failed pass, restart(throw_in,kick_corner,kick_free_kick), second tackle/interception, opponent/sequence/half change or beyond inclusive window. Failed passes are stored in terminal metadata for failure reporting, not productive progress. Retain a goal event then end the passage. Do not reconnect. Candidate sequences start with tackle/interception; do not drop no-followup starts.
Final-third entry: completed pass x_start*attack_dir<=17.5 AND receiver_x*attack_dir>17.5. Positive forward_metres=sum(max(end_x_att-start_x_att,0)) from completed measurable passes, not a learned value model. has_progression=any completed pass gaining>=10m in passage. Missing endpoints =>unknown measurement, not zero evidence. Shot and goal indicators remain separate; no sum as chances.
start_zone: defensive_half(x_att<=0), attacking_half(0<x_att<=17.5), final_third(x_att>17.5).
outcome_category EXCLUSIVE priority: shot (shot OR standalone goal), final_third_entry, forward_progression (any>=10m completed pass), other_completed_play, no_observed_followup. Must conserve all138 starts. This is presentation categorisation, not an additive score.

Ranking columns (minimum): player_id,team,minutes_played,recovery_starts,recovery_followups,recovery_entry_passages,recovery_shot_passages,recovery_outcome_passages,outlet_passages,outlet_entry_passages,outlet_shot_passages,outlet_outcome_passages,completed_transition_passes,measurable_transition_passes,forward_metres,progressive_passes,progression_passages,entries_delivered,entries_received,entry_passages,shot_passages,pareto_tier,progression_rank.
Entry counts are direct action contributions; downstream recovery/outlet outcomes are association. Unique passages, no duplicate role inflation. Rates must carry denominators. Keep all33 players, including zero evidence. Minutes secondary; no minimum-minutes exclusion. Wilson intervals for small-sample conversion shares (descriptive, do not assume independent samples).
Primary display: role-specific evidence and Pareto tiers over entries_delivered,forward_metres,recovery_outcome_passages,outlet_outcome_passages. Smaller tier is better. Also provide progression_rank (order by forward_metres descending, entries_delivered descending then player ID), explicitly a progression-only ordering. No weighted arbitrary overall MVP score. Final argument compares direct progression, outcome association, opportunities, starting territory, robustness; permit inconclusive final choice. Reviewer may recommend refinements without silently changing API.

## Shared API (visual/media)
plot_transition_flow(metrics: pd.DataFrame) -> matplotlib.figure.Figure
- Sankey/alluvial: all candidate transitions -> team -> exclusive outcome_category. Counts and ribbon thickness represent transitions, conserve all starts, no outcomes counted twice. Support empty categories and zero flows. Labels clarify shot vs territory vs progress; colourblind-friendly.
plot_player_evidence(ranking: pd.DataFrame, player_ids: list[str]|None=None) -> Figure
- Transparent comparison with clear units and separate panels, no invented weighted score. Candidate annotations/minutes/sample sizes.
nearest_frame(players: pd.DataFrame,timestamp_ms:int,half:int)->tuple[pd.DataFrame,int]
- nearest withinhalf; earlier tie; ValueError if missing half; report absolute gap.
plot_transition_pitch(transition_events:pd.DataFrame,players:pd.DataFrame|None=None,focus_player:str|None=None)->Figure
- input events are already one transition; normalised attack direction if used, label units/video time, show event arrows and optionally one snapshot with time label. No claims from missing positions.
probe_video(path: str|Path)->dict
export_transition_clip(video_path: str|Path,output_path: str|Path,start_s:float,end_s:float,make_gif:bool=False)->dict
- subprocess argument lists (no shell). ffprobe valid source; clamp bounds; re-encode short MP4, muted; optional palettegen/paletteuse GIF10fps854width; realtime. Cache existing checked asset only when source/times/settings match (sidecar metadata), otherwise regenerate owned outputs. Do not overwrite unrelated source. Raise useful error on ffmpeg failure. Return start_s,end_s,duration_s,mp4_path,gif_path plus metadata. No full-match decode or CV model. Handle unavailable source with caller fallback rather than fake media.
- Presentation assets in clips/ remain restricted-use local artifacts.

## Verification
Core worker writes synthetic boundary/attribution/ranking checks first, watches failure, then implementation and passing checks. Run uv run python validate_transition_analysis.py, plus real-data baseline and 10/15/20-second checks. Visual worker checks conservation, missing tracking/tie, invalid video bounds, plot rendering; a tiny synthetic lavfi video is permissible in temp dir to verify extraction/GIF durations without player footage. No new framework.
Coordinator will build and execute notebook end to end, review actual figures, inspect selected real video frames, generate local exports/manifest and CSV evidence, and reconcile review findings. Report facts; never claim a video supports an action you have not seen. Please report completion with exact commands/results, important assumptions and sample findings through Orca worker_done.
