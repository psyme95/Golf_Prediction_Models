# Recommended Betting Strategies

Decision record from the walk-forward backtest (2020–2026, 5 test years, both
tours), after an exhaustive sweep of 948 strategy evaluations. All figures are
net of 3% Betfair commission charged on net market P&L.

**Status: paper testing. Nothing here has been staked with real money.**

---

## Tier 1 — Core: Top20 lay

**Lay every player whose `Normalised_Model_Odds` exceeds the available lay
odds, in the Top20 market, on both tours. No other filter.**

| | PGA | Euro |
|---|---|---|
| Sharpe (per event) | 0.497 | 0.466 |
| Years profitable | 5/5 | 5/5 |
| Bets per event | 81 | 82 |
| Loss rate | 18.7% | 17.2% |
| Mean P&L per event (per £1 stake) | £7.98 | £8.25 |
| Worst event (per £1 stake) | −£38 | −£40 |
| Liability per event (per £1 stake) | £651 | £763 |
| Return on liability | 1.23% | 1.08% |

Why this one: highest risk-adjusted return of all 472 strategies tested on both
tours; profits earned against a genuine 18% loss rate rather than the absence
of rare events; most capital-efficient of the lay strategies; and it sits on a
plateau — the eight next-best variants (`Odds>=1.5`, `Rating<75`, `Odds>=2`,
`Odds>=3`, `Rating<70`, `Odds<50`, …) all score 0.437–0.464, so the result does
not depend on a precise threshold.

Optional risk control: an odds ceiling of 50 costs almost no Sharpe
(0.497 → 0.490 on PGA) and caps any single bet's liability at 47× stake instead
of 109×. Worth taking on a small bankroll.

## Tier 2 — Add when capital allows: Top10 lay

**Same rule in the Top10 market. No rating filter.**

| | PGA | Euro |
|---|---|---|
| Sharpe | 0.378 | 0.320 |
| Years profitable | 5/5 | 5/5 |
| Bets per event | 83 | 68 |
| Mean P&L per event (per £1 stake) | £13.26 | £7.62 |
| Liability per event (per £1 stake) | £2,747 | £1,449 |
| Return on liability | 0.48% | 0.53% |

*A `rating < 75` filter was originally specified here and removed on
2026-08-11 after testing — see "Filters tested and rejected" below.*

Adding this to Tier 1 raises combined Sharpe to 0.520 on PGA under equal-
liability weighting (vs 0.491 for Top20 alone) and 0.469 on Euro under equal
stakes. It roughly triples absolute return — but consumes ~5× the capital, so
return on liability more than halves.

**Size it below Top20, not equal to it.** Fund Tier 1 fully first.

## Tier 3 — Optional, experimental: Top10 back

**Back players rated 65+ in the Top10 market where `Normalised_Model_Odds` is
below the available lay odds.**

| | PGA | Euro |
|---|---|---|
| Sharpe | 0.169 | 0.108 |
| Years profitable | 4/5 | 4/5 |
| Mean P&L per event (per £1 stake) | £1.51 | £1.28 |
| Capital per event (per £1 stake) | £17 | £21 |
| **Return on capital** | **8.7%** | **6.2%** |

The only back strategy of ~230 tested to survive screening. Low Sharpe, but
backing ties up only the stake, making it by far the most capital-efficient
option. Adding it to Tier 1 leaves PGA Sharpe near-flat (0.491 → 0.486) while
lifting return on capital (1.20% → 1.39%); on Euro it lowers Sharpe
(0.460 → 0.410). Weaker year-consistency than Tiers 1–2.

**The rating floor here is load-bearing, unlike Tier 2's.** Unfiltered Top10
backing loses money on both tours (Sharpe −0.018 PGA, −0.055 Euro; totals −£78
and −£438). Adding the floor turns it profitable: bootstrap Sharpe difference
+0.186 on PGA (95% CI [+0.012, +0.380], significant) and +0.163 on Euro
(CI [−0.020, +0.353], 96% — just short of significant).

The *direction* is well supported; the exact threshold is not. PGA peaks at 65
(Sharpe 0.169) while Euro peaks at 60 (0.148) and is flat across 55–70. Read it
as "a floor somewhere around 60–65", not a tuned constant.

Treat as experimental — worth paper-testing alongside, not core.

---

## Explicitly excluded

| Market / side | Reason |
|---|---|
| **Winner — all strategies** | Zero of ~118 variants survive. Every profitable one has a 0.0–0.3% loss rate: it lays extreme longshots, collecting pennies against rare catastrophic losses. Euro's version was hit once in 5 years and its Sharpe collapsed from ~1.5 to 0.31; PGA's simply hasn't been hit yet. |
| **Top5 — all strategies** | Zero survivors. Best variants fail on tail structure (0.8–1.1% loss rate), year-consistency (2–3 of 5 PGA years), or concentration (one bet loss = up to 79% of total profit). Top5 lay loses money outright on PGA. |
| **All backing except Top10 rating≥65** | Negative or near-zero Sharpe on at least one tour. |
| **Tight edge filters** | Raising the edge threshold improves ROI per bet but cuts bets per event from ~80 to ~8, destroying within-event diversification. Sharpe falls ~0.50 → ~0.24. |

## Filters tested and rejected

**`rating < 75` on the Top10/Top20 lays — rejected 2026-08-11.** It topped a
min-Sharpe ranking of ~470 strategies, but that was a multiple-comparisons
artifact. The filter removes only 2.5–6% of bets, and those bets are close to
break-even (PGA Top10: 396 bets worth −£6 over five years). Bootstrapped Sharpe
differences straddle zero on all four tour/market combinations, and the point
estimate is *negative* on three of them:

| | Observed ΔSharpe | 95% CI | P(helps) |
|---|---|---|---|
| Top10 PGA | −0.002 | [−0.009, +0.006] | 33% |
| Top10 Euro | +0.019 | [−0.004, +0.042] | 94% |
| Top20 PGA | −0.010 | [−0.025, +0.004] | 8% |
| Top20 Euro | −0.004 | [−0.027, +0.019] | 38% |

Any *meaningful* rating ceiling is actively harmful — Top20 PGA falls 0.497 →
0.443 (<70) → 0.417 (<65) → 0.321 (<60). Rating ceilings do not belong on the
lay strategies at all.

**Minimum field size of 120 — considered and rejected 2026-08-11.** Small PGA
fields are genuinely weaker (£4.01/event vs £9.63; 29% of PGA events), because
in a 40-player field half the field finishes top-20 and there is little
mispricing to exploit. Excluding them would lift PGA Tier 1 Sharpe 0.497 →
0.541. Rejected on opportunity cost: with only ~2 events a week, dropping
29% of them for a marginal Sharpe gain is not worth the lost volume, and those
events remain profitable. Euro has too few small fields (12 in five years) to
test the effect.

Other event characteristics showed no reliable pattern: field strength, best-
player rating, favourite odds and median odds all wobbled without direction or
disagreed between tours. Ties at the cut looked meaningful on PGA but reversed
on Euro — and in any case placer counts and tie counts are only known after the
event, so they cannot serve as filters.

## Capital requirements

Per £1 of lay stake, running Tier 1 on both tours: **~£1,400 of liability in a
typical week, ~£3,300 at peak.** Expected return ≈ £567/year per £1 of stake
(PGA ~37 events/yr, Euro ~33).

Scale the stake to the bankroll — Sharpe is scale-invariant, so the strategy's
quality is unchanged at any size. Sub-£2 staking makes small starts practical:
at £0.25 stake, Tier 1 on both tours needs roughly £350 typical / £825 peak.

If capital is tight, **drop the second tour before diluting the strategy** —
running one tour keeps quality intact, whereas tightening filters to save
capital costs real Sharpe.

## Health warnings

1. **Every number above is in-sample strategy selection.** The models were
   trained walk-forward (out-of-sample predictions), but the choice
   of *which* strategy to run was made by inspecting these same results. Only
   forward paper testing is a true out-of-sample test.
2. **Execution is unmodelled.** ~80 lay bets per event per market assumes fills
   at pre-event snapshot prices across the whole field, including outsiders
   where place-market liquidity is thinnest. If only the liquid half fills, it
   is a different, untested strategy.
3. **2026 is a partial year**, so the fifth year of every consistency count
   carries less weight than the others.
4. **Back and lay are netted separately** for commission. Running both in the
   same market would net together in reality — slightly favourable versus these
   figures.

## Next step

Paper-trade Tier 1 (optionally Tiers 2–3 alongside) for 6–8 weeks, recording
actual fill prices against `Normalised_Model_Odds`, before staking real money.
