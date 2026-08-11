# Recommended Betting Strategies — Lay Only

Decision record from the walk-forward backtest (2020–2026, 5 test years, both
tours), after an exhaustive sweep of 948 strategy evaluations. All figures are
net of 3% Betfair commission charged on net market P&L, expressed **per £1 of
lay stake** (fixed-stake staking).

Back betting was investigated across ~230 variants and **abandoned** — see
"Why lay only" below.

**Status: paper testing. Nothing here has been staked with real money.**

---

## Tier 1 — Core: Top20 lay

**Lay every player whose `Normalised_Model_Odds` exceeds the available lay
odds, in the Top20 market, on both tours. No filter.**

| | PGA | Euro |
|---|---|---|
| Sharpe (per event) | 0.497 | 0.466 |
| Years profitable | 5/5 | 5/5 |
| Bets per event | 81 | 82 |
| Loss rate | 18.7% | 17.2% |
| Mean P&L per event | £7.98 | £8.25 |
| Worst event | −£38 | −£40 |
| Liability per event | £651 | £763 |
| Return on liability | 1.23% | 1.08% |

## Tier 2 — Add when capital allows: Top10 lay

**Same rule in the Top10 market. No filter.**

| | PGA | Euro |
|---|---|---|
| Sharpe | 0.378 | 0.320 |
| Years profitable | 5/5 | 5/5 |
| Bets per event | 83 | 68 |
| Mean P&L per event | £13.26 | £7.62 |
| Liability per event | £2,747 | £1,449 |
| Return on liability | 0.48% | 0.53% |

Higher absolute return than Tier 1 but ~4× the capital and lower Sharpe, so
**fund Tier 1 fully first**. Combined with Tier 1 it lifts PGA Sharpe to 0.520
under equal-liability weighting (vs 0.491 for Tier 1 alone).

---

## Evidence

### 1. Year by year — every year profitable, on both tours

Total P&L per £1 of stake (2026 is a partial year):

| Test year | PGA Top20 | Euro Top20 | PGA Top10 | Euro Top10 |
|---|---|---|---|---|
| 2022 | +£466 | +£361 | +£764 | +£143 |
| 2023 | +£433 | +£436 | +£867 | +£640 |
| 2024 | +£331 | +£169 | +£692 | +£61 |
| 2025 | +£234 | +£287 | +£177 | +£199 |
| 2026 (partial) | +£28 | +£92 | +£33 | +£223 |
| **Total** | **+£1,492** | **+£1,345** | **+£2,533** | **+£1,265** |

20 of 20 tour-market-years positive.

### 2. Neighbourhood — the result does not depend on a precise setting

Every single-filter variant around each baseline, both bases, ≥200 bets on both
tours:

| | Top20 | Top10 |
|---|---|---|
| Variants tested | 54 | 55 |
| Profitable on **both** tours | **48** | **48** |
| Positive Sharpe on both tours | 48 | 48 |
| ≥4/5 years positive on both tours | 42 | 40 |
| 5/5 years positive on both tours | 29 | 21 |

For Top20, the nine highest-Sharpe variants are **all 5/5 years on both tours**
(`none`, `odds≥1.5`, `rating<75`, `odds≥2`, `odds≥3`, `rating<70`, `odds<50`,
`edge≤0.952`, `rating<65`) with Sharpe 0.386–0.466. Top10 shows the same: its
top six are all 5/5 on both tours, Sharpe 0.318–0.339.

This is a plateau, not a spike. Noise does not usually produce one.

### 3. The failures are coherent

The ~6 variants that fail per market are not random — they are the ones that
lay **only favourites or only short prices** (`rating≥70`, `rating≥75`,
`odds<3`, `odds<5`). That matches the favourite–longshot bias: short-priced
players are fairly or generously priced, so laying them earns nothing. The
strategies make their money laying mid-priced and outsider players. What works
and what fails are explained by the same mechanism.

---

## ⚠ Edge decay on PGA

Per-event P&L by year (per £1 stake) — the totals above hide a trend:

| Test year | PGA Top20 | PGA Top10 | Euro Top20 | Euro Top10 |
|---|---|---|---|---|
| 2022 | £11.08 | £17.77 | £9.03 | £3.57 |
| 2023 | £10.83 | £21.14 | £11.47 | £16.84 |
| 2024 | £7.53 | £15.72 | £4.71 | £1.64 |
| 2025 | £5.71 | £4.33 | £8.19 | £5.52 |
| 2026 | £1.41 | £1.49 | £6.54 | £14.89 |

**PGA declines monotonically on both markets**; Euro is noisy with no trend.
Still profitable every year, but the recent rate is well below the five-year
average. Possible causes: PGA place markets becoming more efficient, model
degradation, or chance — five points is not enough to distinguish them.

**Plan on the recent rate, not the average.** For PGA, £2–6 per event per £1
stake is a fairer expectation than the headline £7.98. If paper testing comes
in near the recent numbers rather than the average, that is the trend
continuing, not a failure of the strategy.

---

## Why lay only

Backing was tested across ~230 variants and abandoned:

- Zero-margin backing loses 8–22% ROI in every market on both tours.
- Raising the edge threshold does not rescue it — PGA stays negative at every
  threshold tested.
- Only one back variant (Top10, rating ≥ 65) survived screening at all, with
  roughly a third of the lay Sharpe. Dropped 2026-08-11 to keep the operation
  to a single, well-evidenced mechanism.

---

## Markets excluded

| Market | Reason |
|---|---|
| **Winner** | Zero of ~118 variants survive. Every profitable one has a 0.0–0.3% loss rate: it lays extreme longshots, collecting pennies against rare catastrophic losses. Euro's version was hit once in five years and its Sharpe collapsed from ~1.5 to 0.31; PGA's simply hasn't been hit yet. |
| **Top5** | Zero survivors. Best variants fail on tail structure (0.8–1.1% loss rate), year-consistency (2–3 of 5 PGA years), or concentration (one bet loss = up to 79% of total profit). Top5 lay loses money outright on PGA. |

## Filters tested and rejected

**`rating < 75`** — rejected 2026-08-11. Removes only 2.5–6% of bets, worth
about nothing (PGA Top10: 396 bets totalling −£6 over five years). Bootstrapped
Sharpe differences straddle zero on all four tour/market combinations, negative
on three:

| | Observed ΔSharpe | 95% CI | P(helps) |
|---|---|---|---|
| Top10 PGA | −0.002 | [−0.009, +0.006] | 33% |
| Top10 Euro | +0.019 | [−0.004, +0.042] | 94% |
| Top20 PGA | −0.010 | [−0.025, +0.004] | 8% |
| Top20 Euro | −0.004 | [−0.027, +0.019] | 38% |

Any *meaningful* rating ceiling is actively harmful (Top20 PGA: 0.497 → 0.443
at <70 → 0.417 at <65 → 0.321 at <60).

**Minimum field size of 120** — considered and rejected. Small PGA fields are
genuinely weaker (£4.01/event vs £9.63, 29% of PGA events) because in a
40-player field half the field finishes top-20, leaving little to exploit.
Excluding them would lift PGA Tier 1 Sharpe to 0.541. Rejected on opportunity
cost: with only ~2 events a week, losing 29% of them is not worth a marginal
Sharpe gain when they remain profitable.

**Edge thresholds** — raising the minimum edge improves ROI per bet but cuts
bets per event from ~80 to ~8, destroying within-event diversification. Sharpe
falls ~0.50 → ~0.24.

Other event characteristics showed no reliable pattern: field strength,
best-player rating, favourite odds and median odds all wobbled without
direction or disagreed between tours. Ties at the cut looked meaningful on PGA
but reversed on Euro — and placer counts and tie counts are only known after
the event, so they cannot serve as filters anyway.

---

## Capital requirements

Per £1 of lay stake, Tier 1 on both tours: **~£1,400 liability in a typical
week, ~£3,300 at peak.** Scale the stake to the bankroll — Sharpe is
scale-invariant, so strategy quality is unchanged at any size. At £0.25 stake
that is roughly £350 typical / £825 peak.

If capital is tight, **drop the second tour before diluting the strategy** —
running one tour keeps quality intact, whereas tightening filters to save
capital costs real Sharpe.

## Health warnings

1. **In-sample strategy selection.** The models were trained walk-forward
   (out-of-sample predictions), but the choice of *which* strategy to run came
   from inspecting these same results. Only forward paper testing is a true
   out-of-sample test. The plateau and mechanism evidence above are the
   defence against this, not a guarantee.
2. **Execution is unmodelled.** ~80 lay bets per event assumes fills at
   pre-event snapshot prices across the whole field, including outsiders where
   place-market liquidity is thinnest. If only the liquid half fills, it is a
   different, untested strategy.
3. **PGA edge decay** (above) — plan on recent-year rates.
4. **2026 is a partial year**, so the fifth point of every series is thinner
   than the others.

## Next step

Paper-trade Tier 1 (Tier 2 alongside if capital allows) for 6–8 weeks,
recording actual fill prices against `Normalised_Model_Odds`, before staking
real money.
