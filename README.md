# Veo Football Analytics Sprint: Match Dataset v1

One full 11-a-side match, captured on a Veo camera and processed by Veo's analytics
pipeline: player tracking, ball tracking, and detected on-ball events.

Prepared for the **Veo Football Analytics Sprint**, Veo Technologies, Copenhagen,
9 November 2026.

## What's in here

| File | Rows | What it is |
|---|---|---|
| `player_summary.csv` | 33 | **Start here.** One row per player: distance, speed, passes, goals |
| `events.csv` | 1,289 | On-ball events: passes, shots, goals, duels, tackles |
| `players.parquet` | 322,451 | Player tracking, ball position already joined in |
| `ball.csv` | 144,591 | The ball on its own, for trajectory and timing work |
| `DATA_DICTIONARY.md` | | Schemas, coordinate system, and what to watch for |
| `starter.ipynb` | | Loads everything, draws a pitch, builds a pass network |
| `demo_goal_moment.ipynb` | | **Worked example.** One goal, from event to picture |

**Video.** `match_01_full.mp4` is the complete recording, downloaded separately.

The data clock and the video clock are the same, so finding any moment is one sum:

```
seconds into the video = timestamp_ms / 1000
```

The goal at `timestamp_ms = 4982299` is 4982 seconds in, which is **83:02** on the video.
That is video time, not the match clock: the recording starts before kick-off (at 5:51)
and includes half time.

## Start here

Not a coder, or want numbers in the first five minutes? Open **`player_summary.csv`**
in Excel or Google Sheets. Distance covered, top speed, sprints, passes, completion
rate, shots and goals for all 33 players.

Writing code?

```bash
pip install pandas pyarrow matplotlib
jupyter notebook starter.ipynb          # the tour
jupyter notebook demo_goal_moment.ipynb # a worked example, end to end
```

`demo_goal_moment.ipynb` takes the goal at 83:02 on the video and walks from "here is a row in
`events.csv`" to a time-lapse of all 22 players in the 15 seconds before it went in.
It is the fastest way to understand how the files fit together.

Then read `DATA_DICTIONARY.md`. Three things save the most time:

- The ball is **already joined** into `players.parquet`, as `ball_x` / `ball_y`.
- Every file shares one clock, **`timestamp_ms`**. Events fall between tracking frames, so
  match to the nearest one (recipe in the dictionary).
- Teams **swap ends** at half-time, as in reality. Use the `attack_dir` column.

If you are working with an LLM, paste `DATA_DICTIONARY.md` into the context first.
It is written to be read that way.

## The match

11 v 11 adults, 96 minutes of in-play time. **Away won 4–2**, with all six goals in
the second half. Teams and players are anonymised: shirt numbers only, no names.

`home` is the home side, `away` the visitors. `away` defend the −x goal at kick-off
and they swap ends at half-time, so the team name does not tell you which goal a
side is attacking. `attack_dir` does.

Here is the whole match in ten numbers:

| | away | home |
|---|---|---|
| Goals | **4** | **2** |
| Shots | 14 | 15 |
| Passes | 393 | 439 |
| Pass completion | 64.4% | 64.5% |
| Distance covered (outfield) | 103.6 km | 101.0 km |
| Share of on-ball events | 48.7% | 51.3% |
| Tackles | 34 | 45 |
| Interceptions | 54 | 44 |

Which is the interesting part. On every process measure these two teams were
near-identical. Pass completion differs by a tenth of a point, shots by one, distance
by 2.5%. Home had marginally more of the ball. And away won by two.

**Why?** That question has no answer in the table above. It might have one in the
tracking data.

## Terms of use

This dataset is shared with Veo Football Analytics Sprint participants for use during
the event and for your own learning and experimentation afterwards.

**Please do not redistribute it.** It is one match, shared with the permission of the
club. If you want to use it for something else, just ask.

## Questions

Paddy Cox · pco@veo.co
