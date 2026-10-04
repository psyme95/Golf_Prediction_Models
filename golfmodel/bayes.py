"""Bayesian field model: one simulated finishing order for all four markets.

The ensemble in modeling.py trains a separate classifier per market, so nothing
forces P(win) <= P(top5) <= P(top10) <= P(top20) and the walk-forward
predictions break this ordering for around 1.8% of PGA rows. This model
predicts the score instead and reads every market off the same simulated
field, so the ordering holds by construction.

With rel = score - event field mean (strokes per round, lower is better):

    rel_i = x_i . beta + e_i
    beta  ~ N(beta_hat, sigma^2 (X'X + lambda I)^-1)     conjugate posterior
    e_i   ~ empirical residuals for x_i's skill band

Each simulation draws beta from its posterior, adds a residual and ranks the
field, so both parameter uncertainty and round-to-round variation reach the
probabilities. Residuals are resampled within bands of predicted skill because
they are heteroscedastic (better players are more consistent) and right-skewed,
and resampling keeps both without assuming a distribution.

A per-player latent skill term (Kalman-filtered over score history) was also
tried and dropped, as it did not improve out-of-sample RMSE. The provider's
rating and form features already summarise a richer score history.
"""

import numpy as np
import pandas as pd

from .config import BASE_MODEL_VARS, CROSS_MARKET_VARS, MARKETS, RANDOM_SEED

# Markets in ascending cut order. Winner is cut 1; for the place markets the
# cut equals market_size.
MARKET_CUTS = {name: (1 if name == "Winner" else m["market_size"])
               for name, m in MARKETS.items()}

N_SIMS = 20_000
SIM_CHUNK = 4_000        # bounds peak memory; results are unchanged

# Golf is scored in whole strokes and `score` is a per-round average, so one
# stroke over a 4-round event moves it by 0.25, the granularity visible in the
# data. Rounding simulated scores onto this grid reproduces golf's tie
# convention, which is not cosmetic: with continuous draws exactly 20 players
# have posn <= 20 and the field sums to exactly 20, but ties (six players tied
# 18th all get posn 18) push the real figure to 22.5. Continuous ranking would
# systematically under-predict against the binary top_k targets.
SCORE_GRID = 0.25

# Missed-cut players average two rounds, so their granularity is really 0.5, but
# they are outside every market priced here so one grid is used for all.

N_SKILL_BANDS = 10       # residual pools, by predicted score
RIDGE = 1.0


def _prior_vars(prior: str) -> list[str]:
    """Feature columns. 'market' adds the de-vigged cross-market prices, which
    shrinks the model toward the book, and the documented live edge is
    disagreement with the book, so 'features' (market-blind) is the default."""
    if prior == "market":
        return BASE_MODEL_VARS + CROSS_MARKET_VARS
    return list(BASE_MODEL_VARS)


def event_rel(df: pd.DataFrame) -> pd.Series:
    """Score relative to the event's own field mean, in strokes per round.

    Removes course difficulty and field strength, which is what makes scores
    comparable across events: event mean score varies by 0.18 sd on PGA and
    0.75 on Euro.
    """
    score = pd.to_numeric(df["score"], errors="coerce")
    return score - score.groupby(df["eventID"]).transform("mean")


def _design(df: pd.DataFrame, pkg: dict) -> np.ndarray:
    """Standardised feature matrix with an intercept.

    Missing values become 0 after standardisation, i.e. "average here", rather
    than dropping the row: the field has to stay whole or the finishing order
    is wrong for everyone else too.
    """
    raw = df.reindex(columns=pkg["cols"]).apply(pd.to_numeric, errors="coerce")
    Z = ((raw - pkg["mean"]) / pkg["sd"]).fillna(0.0).to_numpy(dtype=float)
    return np.column_stack([np.ones(len(df)), Z])


def _centre(mu: np.ndarray) -> np.ndarray:
    """Impose the per-event sum-to-zero constraint.

    rel is mean-zero within every event by construction, so the model carries
    that constraint too. Enforcing it absorbs any per-event offset, drift in
    the feature distribution between training and test years, for instance,
    which otherwise biases the PGA 2022 predictive mean by +0.40 strokes. A
    constant offset cannot change a ranking, so this does not move market
    probabilities directly; it keeps the residuals honest.
    """
    if mu.ndim == 1:
        finite = np.isfinite(mu)
        return mu - mu[finite].mean() if finite.any() else mu
    return mu - mu.mean(axis=0, keepdims=True)


def fit_model(train_df: pd.DataFrame, prior: str = "features") -> dict:
    """Conjugate Bayesian linear regression of relative score on features, plus
    empirical residual pools by predicted-skill band.

    A ridge-penalised normal prior on the coefficients (intercept unpenalised)
    gives the closed-form posterior
    beta | y ~ N(beta_hat, sigma^2 (X'X + lambda I)^-1). No sampler needed.
    """
    cols = _prior_vars(prior)
    raw = train_df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    pkg = {"cols": cols, "mean": raw.mean(),
           "sd": raw.std().replace(0.0, 1.0).fillna(1.0)}

    rel = event_rel(train_df).to_numpy(dtype=float)
    X_all = _design(train_df, pkg)
    ok = np.isfinite(rel) & np.isfinite(X_all).all(axis=1)
    X, y = X_all[ok], rel[ok]

    lam = np.eye(X.shape[1]) * RIDGE
    lam[0, 0] = 0.0                              # never penalise the intercept
    A = X.T @ X + lam
    beta_hat = np.linalg.solve(A, X.T @ y)

    resid = y - X @ beta_hat
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)

    # Posterior covariance of beta, as a Cholesky factor: a draw is
    # beta_hat + chol @ z. Symmetrised first against round-off asymmetry.
    cov = sigma2 * np.linalg.inv(A)
    pkg["beta_cov_chol"] = np.linalg.cholesky((cov + cov.T) / 2.0)
    pkg["beta"] = beta_hat
    pkg["sigma2"] = sigma2

    # Residual pools by predicted skill. Banding on the point-estimate fit
    # captures the measured 1.27-1.31x spread ratio between best and worst
    # deciles; resampling carries the skew and fat tails along with it.
    mu_hat = X @ beta_hat
    edges = np.quantile(mu_hat, np.linspace(0, 1, N_SKILL_BANDS + 1)[1:-1])
    band = np.searchsorted(edges, mu_hat)
    pkg["band_edges"] = edges
    pkg["resid_pools"] = [resid[band == b] if (band == b).any() else resid
                          for b in range(N_SKILL_BANDS)]
    return pkg


def _count_within_cuts(S: np.ndarray, cuts: list[int]) -> np.ndarray:
    """For each player and cut, simulated fields finishing posn <= cut.

    Place markets count every tied player, matching the binary `posn <= k`
    target; the Winner market splits the credit, because a tie there goes to a
    playoff rather than a dead heat.
    """
    n = S.shape[0]
    order = np.argsort(S, axis=0, kind="stable")
    ranked = np.take_along_axis(S, order, axis=0)

    # posn = 1 + players strictly better = 1 + index of the first player sharing
    # this score. A running max of the run-start index gives that in one pass.
    # Counting distinct better *scores* instead would collapse the field.
    is_new = np.ones(ranked.shape, dtype=bool)
    np.not_equal(ranked[1:], ranked[:-1], out=is_new[1:])
    idx = np.arange(n, dtype=np.int32)[:, None]
    run_start = np.maximum.accumulate(np.where(is_new, idx, np.int32(0)), axis=0)

    posn = np.empty_like(run_start)
    np.put_along_axis(posn, order, run_start + 1, axis=0)

    cols = []
    for c in cuts:
        inside = posn <= c
        if c == 1:
            # The Winner market has no dead heat, a tie at the top is
            # settled by a playoff, so exactly one player wins. Splitting the
            # credit among those tied reproduces that in expectation. Counting
            # all of them, as the place markets correctly do, inflates the
            # per-event sum to 1.27 against an actual 1.00.
            tied = inside.sum(axis=0, keepdims=True)
            cols.append(np.where(inside, 1.0 / np.maximum(tied, 1), 0.0).sum(axis=1))
        else:
            cols.append(inside.sum(axis=1))
    return np.column_stack(cols)


def posterior_predictive(df: pd.DataFrame, pkg: dict, cuts: list[int],
                         n_sims: int = N_SIMS,
                         seed: int = RANDOM_SEED) -> np.ndarray:
    """P(posn <= cut) for every player and cut, from one simulated field.

    Returns (n_players, len(cuts)). Every cut is read off the same finishing
    order, so the columns are monotone by construction.
    """
    rng = np.random.default_rng(seed)
    Z = _design(df, pkg)
    n, p = Z.shape

    mu_hat = _centre(Z @ pkg["beta"])
    band = np.searchsorted(pkg["band_edges"], mu_hat)
    rows_by_band = [np.flatnonzero(band == b) for b in range(N_SKILL_BANDS)]
    pools = pkg["resid_pools"]

    counts = np.zeros((n, len(cuts)), dtype=float)
    done = 0
    while done < n_sims:
        k = min(SIM_CHUNK, n_sims - done)

        # Posterior draws of beta carry parameter uncertainty, not just noise.
        B = pkg["beta"][:, None] + pkg["beta_cov_chol"] @ rng.standard_normal((p, k))
        S = _centre(Z @ B)

        # Each player's residual comes from his own skill band.
        for rows, pool in zip(rows_by_band, pools):
            if rows.size:
                S[rows] += rng.choice(pool, size=(rows.size, k))

        np.round(S / SCORE_GRID, out=S)
        S *= SCORE_GRID
        counts += _count_within_cuts(S, cuts)
        done += k

    return counts / n_sims


def predict_event_bayes(event_full: pd.DataFrame, market_name: str, pkg: dict,
                        cache: dict, seed: int = RANDOM_SEED,
                        n_sims: int = N_SIMS) -> pd.DataFrame | None:
    """One market's probabilities for one event, in predict_event's frame.

    The simulation is shared across markets through `cache`: all four markets
    come off the same field, which is the point of the model.
    """
    market = MARKETS[market_name]
    odds_col = market["odds_col"]
    if odds_col not in event_full.columns:
        return None

    df = event_full.copy()
    df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")

    key = (str(df["eventID"].iloc[0]), len(df))
    if cache.get("key") != key:
        probs = posterior_predictive(df, pkg, list(MARKET_CUTS.values()),
                                     n_sims=n_sims, seed=seed)
        cache.clear()
        cache["key"] = key
        cache["probs"] = pd.DataFrame(probs, index=df.index,
                                      columns=list(MARKET_CUTS))

    proba = cache["probs"][market_name]

    # Match predict_event: a row without a market price cannot be bet on.
    keep = proba.notna() & df[odds_col].notna()
    if not keep.any():
        return None
    result = df[keep].copy()
    proba = proba[keep].to_numpy(dtype=float)

    prob_sum = proba.sum()
    market_size = market["market_size"]
    norm_prob = (proba / prob_sum) * market_size if prob_sum > 0 else proba

    result["Model_Score"] = proba.round(5)
    result["Probability"] = proba.round(6)
    result["Normalised_Probability"] = np.round(norm_prob, 6)
    result["Normalised_Model_Odds"] = np.round(1.0 / np.clip(norm_prob, 1e-8, None), 2)
    return result


def make_predict_fn(pkg: dict, seed: int = RANDOM_SEED, n_sims: int = N_SIMS):
    """predict_fn for backtest.backtest_events, closing over the fitted model."""
    cache: dict = {}

    def predict_fn(event_full: pd.DataFrame, market_name: str):
        return predict_event_bayes(event_full, market_name, pkg, cache,
                                   seed=seed, n_sims=n_sims)

    return predict_fn
