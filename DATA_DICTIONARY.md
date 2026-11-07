# Data dictionary: Veo Football Analytics Sprint, Match 01

One full 11-a-side match recorded by a Veo camera and processed by Veo's analytics
pipeline. Player and ball positions are derived from video by computer vision; events
are detected by models.

| | |
|---|---|
| Format | 11 v 11, adult |
| Pitch | 105.0 m long × 67.7 m wide |
| Duration | 96 minutes of in-play time |
| Score | **away 4 - 2 home** (all six goals in the second half) |

Teams are anonymised and called simply **`home`** and **`away`**. Players are
identified as `away_9`, `home_10` and so on: team plus shirt number.

`home` is the home side, `away` the visitors. At kick-off `away` defend the −x goal
and `home` the +x goal, and they swap at half-time, so the team name does not tell
you which goal a side is attacking at any given moment. Use `attack_dir` for that.

---

## 1. Coordinate system: read this first

Positions are in **metres**, origin at the **centre spot**.

| Axis | Meaning | Pitch extent |
|---|---|---|
| `x` | Along the pitch (goal to goal) | −52.5 → +52.5 |
| `y` | Across the pitch (touchline to touchline) | −33.8 → +33.8 |

Standard 2D pitch coordinates, as a broadcaster or provider would give them. The ranges
above are the pitch limits. Players and the ball are tracked a little beyond the lines, so
values can run a few metres outside them.

### Teams swap ends at half-time

These are **real pitch coordinates**, not normalised ones. Both teams change ends
between halves, exactly as they did on the day. You can see it in the goalkeepers:

| | First half | Second half |
|---|---|---|
| `away` keeper | −45.9 | **+45.5** |
| `home` keeper | +43.6 | **−46.2** |

Direction is given to you as data, in the **`attack_dir`** column on both
`events.csv` and `players.parquet`:

| `attack_dir` | Meaning |
|---|---|
| `+1` | This team is attacking the **+x** goal in this half |
| `−1` | This team is attacking the **−x** goal in this half |
| `0` | Referee |

So `away` is `+1` in the first half and `−1` in the second. `home` is the opposite.

### The two lines you will probably want

```python
# how far toward the goal you are attacking (positive = forward)
df["x_att"] = df.x * df.attack_dir

# which flank, from the team's own point of view (positive = their left)
df["y_att"] = df.y * df.attack_dir
```

With those two, everything works across both halves without special-casing: final
third is `x_att > 17.5`, progressive passes are an increase in `x_att`, and a wide
right attacker has a consistently negative `y_att` all match.

---

## 2. Which file do I want?

| I want to… | Use |
|---|---|
| Get numbers fast, in Excel | `player_summary.csv` |
| Analyse on-ball actions | `events.csv` |
| Analyse movement, space, positioning | `players.parquet` |
| Work on ball trajectories | `ball.csv` |

### Joining

Every file shares one clock: **`timestamp_ms`**, integer milliseconds.

Events happen *between* tracking frames, so a plain `==` on `timestamp_ms` will usually
find nothing. Match each event to the nearest frame instead:

```python
players = pd.read_parquet("players.parquet")   # ball is already joined in
events  = pd.read_csv("events.csv")

def frame_at(ms):
    """The tracking rows for the frame nearest to a timestamp."""
    t = players.timestamp_ms.unique()
    return players[players.timestamp_ms == t[abs(t - ms).argmin()]]

at_shot = frame_at(some_event.timestamp_ms)
```

To pull a window of time, use `>=` and `<=` on `timestamp_ms`. The notebooks do this.
`ball.csv` has many more rows than `players.parquet` has frames, so match the two the
same way.

---

## 3. `events.csv`: 1,289 on-ball events

A sample of on-ball actions. One row per event, ordered by time.

| Column | Type | Notes |
|---|---|---|
| `event_id` | int | 1..1289, in time order |
| `timestamp_ms` | int | Join key |
| `minute` | float | Minutes into the video recording. This is **not** the match clock: the recording starts before kick-off and includes half time |
| `half` | int | 1 or 2 |
| `event_name` | str | See counts below |
| `team` | str | `home` or `away` |
| `role` | str | `outfield` or `goalkeeper` |
| `player_id` | str | e.g. `away_9` |
| `attack_dir` | int | `+1` attacking +x, `−1` attacking −x, this half |
| `x_start`, `y_start` | float | Where the event happened. Null on `goal` rows. This is the action's location, so it can sit a metre or two from the ball's own position in `ball.csv` |
| `x_end`, `y_end` | float | Only some event types have an end point |
| `pass_success` | bool | Passes only |
| `receiver_player_id` | str | Passes only |
| `receiver_x`, `receiver_y` | float | Where the pass was received |
| `pass_distance_m` | float | Metres |
| `pass_angle_deg` | float | Direction of the pass, 0–360°, measured anticlockwise from +x |
| `shot_distance_m` | float | Shots only. Distance to the centre of the goal |
| `shot_angle_deg` | float | Shots only. Angle subtended by the 7.32 m goal mouth |
| `is_goal` | bool | Shots only. This shot produced a goal. See the note below |
| `sequence_id` | int | Possession spell: a run of consecutive same-team events |

`shot_distance_m` and `shot_angle_deg` are measured against the goal that team was
actually attacking, so they are already correct in both halves.

### Goals: use the `goal` rows, not `is_goal`

There are two ways a goal appears, and they do not line up one for one.

- The six **`goal` rows are the complete record**. Every goal in the match is there,
  with the scorer in `player_id`. They carry no coordinates.
- **`is_goal` marks the shot that produced a goal**, and only four shots carry it.
  Two goals have no shot row to attach to, so nothing is flagged for them.

So `events[events.is_goal]` gives you four of the six goals. For anything that counts
or lists goals, filter on `event_name == "goal"` instead:

```python
goals = events[events.event_name == "goal"]          # all six, with scorers
shots_that_scored = events[events.is_goal]           # the four with a shot attached
```

`is_goal` is still the right column when you want a goal's **location**, since the
`goal` rows have none and the shot it came from does.

### Event counts

| Event | n | | Event | n |
|---|---|---|---|---|
| `pass` | 832 | | `shot` | 29 |
| `duel` | 112 | | `kick_free_kick` | 20 |
| `interception` | 98 | | `foul` | 15 |
| `tackle` | 79 | | `save` | 11 |
| `throw_in` | 47 | | `goal` | 6 |
| `dribble` | 34 | | `kick_corner` | 6 |

---

## 4. `player_summary.csv`: 33 rows, one per player

**Start here if you want numbers without writing much code.** Opens in Excel or
Google Sheets.

| Column | Notes |
|---|---|
| `player_id`, `team`, `role` | |
| `minutes_played` | Minutes on the pitch |
| `distance_km` | Total distance covered |
| `top_speed_kmh` | Fastest recorded speed |
| `sprints` | Number of efforts above 25 km/h |
| `touches` | Total events attributed to the player |
| `passes`, `passes_completed`, `pass_completion_pct` | |
| `shots`, `goals` | |
| `dribbles`, `tackles`, `interceptions`, `duels` | |

This is broadly what Veo already gives every coach in the match report. The
interesting work is in what this table *cannot* show you.

---

## 5. `players.parquet`: 322,451 rows

Player tracking. One row per player per frame. In-play time only; warm-up and half-time are removed.

| Column | Type | Notes |
|---|---|---|
| `timestamp_ms` | int | Join key |
| `half` | int | 1 or 2 |
| `team` | str | `home`, `away`, or `referee` |
| `role` | str | `outfield`, `goalkeeper`, `referee` |
| `player_id` | str | e.g. `away_9`. **Use this as the player key** |
| `attack_dir` | int | `+1` attacking +x, `−1` attacking −x, this half |
| `x`, `y` | float | Metres |
| `speed_zone` | int | 0: 0–10, 1: 10–15, 2: 15–20, 3: 20–25, 4: 25+ km/h |
| `cumulative_distance_km` | float | Running total for that player, in kilometres |
| `ball_x`, `ball_y` | float | **Ball position, already joined onto this row.** Null where the ball was not located |

The ball is pre-joined, so you need not merge anything to ask "who was near the ball":

```python
players["distance_to_ball_m"] = np.hypot(players.x - players.ball_x,
                                         players.y - players.ball_y)
```

`cumulative_distance_km` is a true per-player running total, so
`groupby("player_id").cumulative_distance_km.max()` gives distance covered.

Squads: `away` has shirts 2–17 plus GK 1; `home` has 2–4, 6–16, 18 plus GK 1.

---

## 6. `ball.csv`

The ball's position through the match, 144,591 rows. This is the finest-grained record
of the ball, for trajectory, flight and precise timing work.

Columns: `timestamp_ms`, `half`, `x`, `y`.

For anything on the player frames you do not need this file: `players.parquet` already
carries `ball_x` and `ball_y` on every row. If you want the ball on its own, one row per
tracking frame:

```python
ball = players.drop_duplicates("timestamp_ms")[["timestamp_ms", "half", "ball_x", "ball_y"]]
```

Where the ball was not located there is no row, so gaps in this file are missing time
rather than empty values. In `players.parquet` those frames have null `ball_x` and
`ball_y` (about 12% of frames).

---

## 7. Half boundaries

| Half | Start | End |
|---|---|---|
| 1 | 351.2 s | 3210.3 s |
| 2 | 3585.0 s | 6486.2 s |

The clock is video time from the start of the recording, so the first half kicks off
around 5m51s into the footage. The first few minutes are pre-match.

`match_01_full.mp4` runs on this same clock, so no offset correction is needed:
`seconds into the video = timestamp_ms / 1000`. A shot at `timestamp_ms = 4982299`
is at 83:02 in the recording.
