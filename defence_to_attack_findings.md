# Before the assist: who turned defence into attack?

**Leading distribution candidate: away_12.** In the restricted 15-second sample,
away_12 completes 12 passes, gains 106.0 positive forward pass metres
(80.2 net metres including backward passes), and receives the first completed
pass in 8 passages. This supports a repeatable connecting role after defensive
interventions. It does not establish an overall match MVP.

## The comparison that matters

| Contribution | Strong case | Evidence in the 15-second sample |
|---|---|---|
| Passing distribution | away_12 | 106.0 positive / 80.2 net pass metres; 8 first-pass receptions |
| Final-third supply | home_12 | 2 entries delivered; 79.3 positive / 59.4 net pass metres |
| Opening interventions | home_2 | 14 candidate starts; 2 entries delivered; 4 stopping failed passes |
| Receiving and finishing | home_10 | 4 final-third entries received; 2 of the three transition shots |

The strongest alternative to away_12 is home_12: fewer passing metres, but more
direct final-third supply. Away_12 delivers no final-third entries under the
original-order baseline. That is a substantive limit on a stronger influence
claim, not a metric to hide.

## What the sample contains

There are 138 sequence-opening tackles/interceptions: 71 home and 67 away.
Only 63 passages have an observed completed pass or shot before the stopping
rule; 75 remain without that follow-up. The exclusive outcome flow is:

- 3 shot-ending passages;
- 9 final-third-entry passages without a shot;
- 23 passages with a completed forward pass of at least 10 m, without entry/shot;
- 28 with other completed play;
- 75 without observed follow-up.

There are 10 entry passages in total; one also includes a shot. Thus 12 distinct
passages reach an entry or shot. These are territorial/event outcomes, not xG or
independent estimates of causal value.

## Check the definition before declaring a winner

The notebook compares 10-, 15-, and 20-second windows, 5/10/15-m progressive-pass
thresholds, and tied event-order alternatives. The original-order 138-start
sample stays separate from the alternative sample that admits simultaneous
pass/intervention starts and guards co-detected restarts. This matters particularly
for away_12 and prevents event ordering from masquerading as football quality.

Passing metres omit ball carrying, off-ball movement, pressure and some unknown
player identities. Goalkeeper starting positions offer more passing runway, so
keepers are contextualised separately. Downstream recovery/outlet outcomes depend
on teammates and are shown with denominators. Small samples do not establish a
single overall winner.

## Show the football

- [Animated evidence slide](gif_evidence_slide.html): two looping GIFs.
- [home_12 entry sequence](clips/transition_1182.mp4): intervention, outlet and supply.
- [away_12-linked passage](clips/transition_815.mp4): distribution and later progress,
  including possible carrying omitted by pass-only metrics.
- [away_4 / away_9 counterattack](clips/transition_1006.mp4): the goal finishes at
  exactly 20 seconds, outside the 15-second baseline. This explains why Away can
  have zero baseline transition shots while still producing a decisive counterattack.
- [Static one-slide summary](defence_to_attack_one_slide.pdf).

Sampled source frames support the phase timing; exact shirt-to-dataset identity
is not independently verified throughout. See clips/clip_manifest.csv for each
claim's verification status. Clips illustrate mechanisms rather than prove
whole-match superiority. All extracts retain the dataset's restricted-use terms.

## Reproduce

Run the notebook from the repository root with the project environment:

```bash
uv run python validate_transition_analysis.py
uv run python validate_transition_visuals.py
uv run --with notebook jupyter notebook defence_to_danger.ipynb
```

Select the project kernel, restart it and run every cell. Matching media assets
are cached. Analytical cells remain usable without local video, with visual
confirmation explicitly unavailable.
