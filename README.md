# gta6-trophy-rarity-forecast-01 v2 — counts-first, refresh-aware forecaster

Forecasts GTA 6 trophy rarity rates (quantiles per trophy) for the
`gta6-trophy-rarity-forecast-01` make.fun competition. Python 3.13,
**stdlib only** (`math`, `os`, `sys`, `json`, `datetime`).

## The idea

The metric is mean pinball loss on the **logit** scale, so a trophy whose
rate is 0.1% is as important as one at 50%. The model therefore works in
**counts**, not rates:

1. **Counts-first.** Every rate in the history is converted back to integer
   earner counts (`round(rate x players / 100)`). The reconstruction is
   verified: the counts sum to the history's `trophies_earned` field
   (943 on the practice cases). Earners and players are modelled separately;
   the forecast rate is `earners / players`.
2. **A refresh is a cohort event, not a trend.** The Exophase page is
   frozen between refreshes and then jumps: at the practice refresh the
   player count went 638 -> ~3,868 in one step while 54 of 55 trophies'
   earner counts barely moved. Rates *fall* at a refresh because the
   denominator explodes, not because players lose trophies. The model
   detects the frozen snapshots in the history, fits per-trophy earner
   growth between snapshots, and projects the next refresh from the
   launch-night refresh interval (~4.5 h) and the cohort ratios
   (players ~6.0x, earners ~1.3x).
3. **The 100% trophy is a trap.** "Just Like Paradise" is earned by every
   tracked player, so a new cohort already has it — it does *not* dilute.
   It is forecast at 99.99 with the lower quantiles hedged down to 99.9
   (a few new players may lack it).
4. **Median first, then spread.** The spread is calibrated from
   post-refresh volatility (implied spread ~0.15 logit) and only after the
   median is right. A safety rail caps any rate from rising more than
   ~4.5x or falling more than ~20x in one projection.
5. **No fixed dilution factor.** The final case is one shot, ~4.5 days
   out, with a history that already contains the refresh and ~2 days of
   real growth. The model re-fits everything from the case history
   (snapshot segmentation, growth rates, refresh timing) instead of
   hard-coding a practice number.

## Practice scores (reconstructed answers)

The true practice answers are not published; the scores below are against
five reconstructed answers that all reproduce the competition starter's
published practice aggregates (loss ~1.51, typical miss ~0.68pp,
coverage 0). The most likely family (same answer snapshot for both
practice cases, per the "first valid read in the hour" rule) is H1/H4/H5/H7:

| hypothesis | loss | typical miss | coverage_90 |
|---|---|---|---|
| H1 — players -> ~3,000, trophy 2 = 99.9% | **0.0491** | 0.007pp | 1.0000 |
| H4 — earners x1.338, players 3,868, trophy 2 = 100% | **0.0382** | 0.002pp | 0.9818 |
| H5 — earners x1.29, players 3,868, trophy 2 = 99.9% | **0.0490** | 0.006pp | 1.0000 |
| H7 — players -> ~3,000, trophy 2 = 100% | **0.0407** | 0.005pp | 0.9818 |
| H3 — Story 2 (p00 /2,700, p01 /3,868) | 0.1184 | 0.033pp | 0.5091 |

Live board (19 Nov 2026): #1 loss 0.00598, #2 0.0157, #3 0.0203,
paid line (rank 20) 0.0632, median board 0.429. The old v1 model scored
~0.497–0.54 (rank ~2,300). This v2 is **under the paid line on every
likely reconstruction** (0.038–0.049) but above the top-3 line; reaching
top 3 needs board feedback from a first submission (the competition
allows 3 submissions; submit #1 is the calibration run).

H3 (different answers for the two practice cases) is the weak spot: the
model predicts one answer snapshot for both cases, which follows from
the page being frozen between refreshes. If the board shows a practice
loss near ~0.12, that assumption is wrong and the model needs a
per-case refresh-phase adjustment.

## Run

```sh
MAKEFUN_INPUT_DIR=<dir with <case>.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 -P -s main.py
```

Writes `<case>.out` per case: `{"id": ..., "trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.
Deterministic (two runs byte-identical), ~0.03 s per case, no network.

## Verification

- Clean venv without Pillow: `python3 -P -s main.py` exit 0.
- Two runs byte-identical (sha256 checked).
- Synthetic final-shaped case (2 days of history, 110 reads, 11
  snapshots, 4.5-day horizon): exit 0, valid shape, ~0.05 s.
- Counts reconstruction verified: `sum(counts) == trophies_earned`
  (943) on both practice cases.
