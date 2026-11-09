# Before the assist: who turned defence into danger?

Working research draft, revised after choosing the defensive-transition direction.
No player has been selected and no transition results have been calculated.

## Question and purpose

**Which player contributed most to turning defensive interventions into dangerous
attacks, and what did they do that ordinary match statistics miss?**

Look for a repeatable contribution: recovering the ball, providing an outlet, or
advancing the attack. Consider both teams and every position. The recoverer and
the player who makes the transition work may be different people.

The conclusion identifies influence in this particular phase of play. A broader
MVP claim would need additional evidence. The question should be distinctive; the
winner can be an obvious player if that is where the evidence points.

## Approaches considered

The organisers value answers that go beyond obvious match statistics. Away won
4–2 despite similar shot totals and passing accuracy. The path from defensive
action to attacking opportunity could reveal an overlooked contribution.

| Approach | Strength | Tradeoff |
|---|---|---|
| Defensive transitions | Connects defensive work with observable attacking outcomes | Recovery success and event completeness need checking |
| Escaping pressure | Could reveal an influential receiver or distributor | Needs reliable tracking matches and a pressure proxy |
| Off-ball space creation | Could reveal influence without touches | Harder to establish convincingly in two hours |

The user chose defensive transitions. Tracking will explain selected moments;
full-match off-ball modelling is outside this sprint.

## What we already checked

- 33 player-summary rows, 1,289 events, 29 shots, six goal events, and 11 saves.
- **177 tackles and interceptions**, all with start coordinates.
- **138 are the first event in their possession sequence.** These are the initial,
  explicitly bounded sample of candidate starts. The other 39 are excluded from
  the first pass; this is not a count of all actual recoveries.
- All 456 recorded sequences contain only one team's events, but some include
  restarts. Sequence membership alone cannot establish an uninterrupted transition.
- None of the 536 completed passes has a complete set of start/end coordinates
  using the x_end/y_end fields. **523 have receiver coordinates and receiver ID**;
  use those for endpoints and identify missing cases.
- Only four shots are flagged as goals. One goal sequence contains only the goal
  event, so some buildup is unobserved.

## Proposed definitions

These are analyst choices for a transparent first pass, not dataset labels.
Validate their feasibility before comparing players.

### Candidate transition

Start with a sequence whose first event is a tackle or interception. Call this a
**recorded defensive intervention** because tackle success is not explicitly
labelled. Follow subsequent events for up to **15 seconds**.

Stop earlier at the first opponent event, change of half, failed pass, restart
(throw-in, corner, or free kick), or further tackle/interception. A further
intervention makes uninterrupted control uncertain. Do not reconnect a passage
after a stopping event.

Record whether a completed pass or shot follows the opening intervention before
the passage ends. This supports subsequent attacking play; it does not prove
causal recovery success. A passage without such evidence is unresolved rather
than automatically a failed recovery. Event gaps do not prove continuous control.

### Attacking outcomes

Keep these outcomes separate:

1. **Shot:** the clearest recorded attacking outcome. Count goal events separately
   when no corresponding shot exists; a shot and its goal represent one chance.
2. **Territorial progress:** a completed pass from outside into the final third,
   using normalised coordinates (x_att <= 17.5 m to x_att > 17.5 m). This is an
   attacking opportunity proxy, not equivalent to a shot or known chance quality.

Show forward metres gained as supporting context. Record whether the opening
intervention occurred in the defensive half, the attacking half before the final
third, or the final third. Starting closer to goal makes some outcomes easier.
Missing endpoints make territorial entries unmeasurable, not automatically absent.

### Player contributions

For each passage, distinguish:

- **Recoverer:** the player credited with the opening intervention.
- **Outlet:** the first identified recipient of a completed pass.
- **Advancer:** the passer and receiver responsible for a final-third entry, or
  the passer making the greatest forward gain when coordinates are available.
- **Finisher:** the player taking the shot or scoring.

These describe observed actions, not playing positions. A player may perform
several roles. Keep the contributions separate; do not add them into a score that
gives one passage repeated credit. Involvement does not establish necessity.

## Comparing candidates

Build one transition table and summarise it by player. Include:

- candidate starts and subsequent observed attacking play;
- involvement as recoverer, outlet, and advancer;
- passages reaching the final third and passages producing a shot;
- progression, starting territory, minutes played, and recorded failures;
- unresolved passages and missing coordinate coverage.

Use counts alongside opportunity denominators. A recoverer's shot-ending share
uses their candidate starts; an outlet's share uses passages in which they
received the first completed pass. Those percentages measure different jobs and
must not be treated as directly comparable. Keep small samples visible. Match
totals are primary; per-90 rates are supporting context.

Shortlist roughly three candidates. Compare repeated contribution, attacking
outcomes, and the strongest alternative explanation. If the evidence does not
distinguish a winner, report that uncertainty.

## Explaining the football

Choose two or three passages that test the leading cases, including an
unsuccessful or ambiguous example where useful. Draw the intervention, passes,
player positions, and relevant opponents.

Use tracking to investigate pressure near the outlet, available forward options,
and whether the early passes advanced play before opponents recovered position.
Describe observable geometry and actions. Claims about space creation, pressure
escape, or prevented chances need supporting evidence; check video if available.

Match events to the nearest tracking frame and record the timestamp gap. Use
attack_dir across halves and handle missing ball coordinates. Label timestamps
as video time, not match-clock minutes.

### Video evidence and GIFs

Use the MP4 to review candidate passages before settling the conclusion. Look for
the defensive intervention, the first outlet, the forward action, and the
opponents' response. Check player identity and whether the recorded events match
what is visible. A clip can support or weaken the proposed explanation.

The supplied full-match recording uses the data's video clock:
video seconds = timestamp_ms / 1000. Confirm that the supplied file is that
recording by checking an identifiable moment in each half before batch export.
Do not introduce a clock offset unless the actual file demonstrates a mismatch.

For each selected passage, start the clip about three seconds before the opening
intervention and end about three seconds after the outcome or stopping event.
Clamp to the recording boundaries. Context beyond the analysis window is for
interpretation; it does not change which events enter the measured transition.

Keep a short MP4 as the main evidence asset and optionally export a GIF for easy
display in the notebook or presentation. FFmpeg can decode the frames directly;
there is no need to save thousands of individual images. Start GIF previews at
10 frames per second and roughly 854 pixels wide, preserving aspect ratio and
real-time playback. These are export defaults, adjustable for readability.

Select two strong examples and one rival, failed, or ambiguous example if time
allows. Store source times, transition/event IDs, player, proposed claim, observed
action, and output paths in a clip manifest. If the camera does not show the
relevant player or action clearly, mark the claim unverified.

Extracts remain subject to the dataset's restricted-use conditions. Source video
and datasets stay unchanged. Raw video and derived clips are not required for
the event-analysis cells to execute; without video, visual confirmation remains
pending and the presentation uses labelled pitch figures.

## Feasibility and checks

First inspect a small sample and count outcomes. With only 29 shots overall,
shot-ending transitions may be scarce. If so, use them as case studies and focus
the aggregate comparison on progression; adjust the headline to **Who turned
defence into attack?** rather than overstating danger.

Check whether candidate selection changes materially with 10-, 15-, and 20-second
windows. Inspect whether the sequence-start restriction misses useful examples
among the other defensive events; expand only if those starts can be identified
consistently and without overlapping attribution.

No external expected-goals model, full-match pressure model, causal
counterfactual, or arbitrary overall MVP score is required.

## Two-hour budget

| Time | Work | Output |
|---|---|---|
| 0–10 min | Audit inputs and clock conventions | Baseline and coverage |
| 10–40 min | Extract passages and check feasibility | Audited transition sample |
| 40–70 min | Attribute contributions and compare candidates | Shortlist and competing explanations |
| 70–75 min | Prepare tracking and video lookup | Frame lookup and source readiness |
| 75–85 min | Review candidate passages in the video | Visually checked claims or explicit uncertainty |
| 85–95 min | Check alternative windows and attribution | Confidence and limitations |
| 95–110 min | Export selected clips/GIFs and pitch figures | Presentation evidence |
| 110–120 min | Assemble the argument | Concise presentation |

## Proposed deliverable

A root-level notebook, defence_to_danger.ipynb, using the existing Python, pandas,
NumPy, and Matplotlib environment. Preserve datasets and anonymised IDs. Keep the
work within permitted sprint and personal-learning use. Restart the kernel and
run all cells in order after notebook edits.

When video is available, keep derived evidence in `clips/`: short MP4s, optional
GIFs, and `clip_manifest.csv`. These are generated local analysis assets, not
changes to the source datasets. During planning, FFmpeg and ffprobe were found
installed, but no MP4 was present at the repository root.

The presentation should explain:

> Player X repeatedly contributed to turning defensive interventions into attacks
> through mechanism Y. Here are the comparison, the moments, the strongest rival
> explanation, and the limits of our evidence.

Success means discovering a specific contribution that goals, tackle totals, or
pass counts would miss. A surprising name is not required.
