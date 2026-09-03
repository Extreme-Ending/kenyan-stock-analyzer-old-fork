"""
Accuracy backtest module.

Joins data/history/daily_snapshots.csv (written by history_tracker.py)
against itself N trading days forward to answer a question the tool has
never been able to answer about itself: did its own "bullish"/"strong buy"
calls actually perform? See IMPROVEMENTS.txt item 2.

Every stat here is descriptive, computed purely from this tool's own
recorded history -- never a live network call. A horizon with too little
accumulated history to say anything degrades to an explicit
'insufficient_history' status rather than raising or showing a misleading
number computed from one or two data points (fail-safe, per CLAUDE.md).
"""

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

# Trading-day horizons to backtest. These are counted in recorded PIPELINE
# RUNS per symbol, not calendar days -- a symbol only has a row for a day
# the pipeline actually ran, so "5 days forward" means "5 runs forward" for
# that symbol specifically (correct as long as the pipeline runs ~daily).
DEFAULT_HORIZONS = (5, 20, 60)

# Overall-score buckets for the "does a higher score predict a better
# forward return" breakdown. Upper bound of the last bucket is 101 so a
# score of exactly 100 falls inside it (buckets are checked as [lo, hi)).
SCORE_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]

# Sub-score column names as recorded by history_tracker.py.
FACTOR_COLUMNS = {
    "value": "score_value",
    "quality": "score_quality",
    "momentum": "score_momentum",
    "dividend": "score_dividend",
    "liquidity": "score_liquidity",
}


def _forward_return_for_horizon(history_df, horizon):
    """
    Add a `fwd_return_{horizon}` column: percent price change from each row
    to that same symbol's row `horizon` recorded runs later.

    Args:
        history_df: DataFrame from HistoryTracker.load_history().
        horizon: how many runs forward to look, per symbol.

    Returns:
        A copy of history_df with the new column added (NaN wherever no
        row exists that far ahead yet -- e.g. the most recent `horizon`
        rows for every symbol, or any symbol with less history than that).
    """
    out = history_df.sort_values(["symbol", "date"]).copy()
    col = f"fwd_return_{horizon}"
    future_price = out.groupby("symbol")["price"].shift(-horizon)
    out[col] = (future_price / out["price"] - 1) * 100
    return out


def hit_rate_by_signal(df_with_returns, horizon):
    """
    Hit rate per technical signal at the given horizon: % of 'bullish'
    calls with a positive forward return, % of 'bearish' calls with a
    negative one, % of 'neutral' calls that stayed roughly flat (+/-2%).

    Args:
        df_with_returns: output of _forward_return_for_horizon.
        horizon: the horizon whose fwd_return_{horizon} column to use.

    Returns:
        dict: {signal: {'hit_rate': pct, 'n': count}}. Empty if the
        forward-return column isn't present or no signal has any data yet.
    """
    col = f"fwd_return_{horizon}"
    if col not in df_with_returns.columns:
        return {}
    valid = df_with_returns.dropna(subset=[col, "overall_signal"])
    result = {}
    for signal in ("bullish", "bearish", "neutral"):
        rows = valid[valid["overall_signal"] == signal]
        if rows.empty:
            continue
        if signal == "bullish":
            hits = (rows[col] > 0).sum()
        elif signal == "bearish":
            hits = (rows[col] < 0).sum()
        else:
            hits = (rows[col].abs() <= 2).sum()
        result[signal] = {"hit_rate": round(100 * hits / len(rows), 1), "n": int(len(rows))}
    return result


def avg_return_by_score_bucket(df_with_returns, horizon):
    """
    Average forward return per overall-score bucket at the given horizon
    -- the core "does a higher score predict a better return" check.

    Args:
        df_with_returns: output of _forward_return_for_horizon.
        horizon: the horizon whose fwd_return_{horizon} column to use.

    Returns:
        dict: {"0-20": {'avg_return': pct, 'n': count}, ...} for buckets
        with at least one row. Empty if no data yet.
    """
    col = f"fwd_return_{horizon}"
    if col not in df_with_returns.columns:
        return {}
    valid = df_with_returns.dropna(subset=[col, "score"])
    result = {}
    for lo, hi in SCORE_BUCKETS:
        rows = valid[(valid["score"] >= lo) & (valid["score"] < hi)]
        if rows.empty:
            continue
        label = f"{lo}-{min(hi, 100)}"
        result[label] = {"avg_return": round(rows[col].mean(), 2), "n": int(len(rows))}
    return result


def factor_correlation(df_with_returns, horizon):
    """
    Pearson correlation between each sub-score (value/quality/momentum/
    dividend/liquidity) and forward return at the given horizon -- which
    factors actually track future performance, versus which don't.

    Args:
        df_with_returns: output of _forward_return_for_horizon.
        horizon: the horizon whose fwd_return_{horizon} column to use.

    Returns:
        dict: {factor: {'correlation': r, 'n': count}} for factors with
        at least 3 paired data points (fewer makes r meaningless). Empty
        if no factor has enough data yet.
    """
    col = f"fwd_return_{horizon}"
    if col not in df_with_returns.columns:
        return {}
    result = {}
    for factor, score_col in FACTOR_COLUMNS.items():
        if score_col not in df_with_returns.columns:
            continue
        valid = df_with_returns.dropna(subset=[col, score_col])
        if len(valid) < 3:
            continue
        r = valid[score_col].corr(valid[col])
        if pd.isna(r):
            continue
        result[factor] = {"correlation": round(r, 3), "n": int(len(valid))}
    return result


def run_backtest(history_df, horizons=DEFAULT_HORIZONS):
    """
    Run the full accuracy backtest over recorded history.

    Args:
        history_df: DataFrame from HistoryTracker.load_history() (or any
            DataFrame with the same columns) -- None/empty is handled.
        horizons: trading-run-ahead horizons to test.

    Returns:
        dict: {
            'days_recorded': int,
            'horizons': {
                5: {'status': 'ok', 'n_pairs': int, 'hit_rate': {...},
                    'score_buckets': {...}, 'factor_correlation': {...}}
                    | {'status': 'insufficient_history'},
                ...
            }
        }
        Never raises -- a horizon without enough recorded history to pair
        against reports 'insufficient_history' rather than an empty or
        misleading stat block.
    """
    if history_df is None or history_df.empty:
        return {"days_recorded": 0,
               "horizons": {h: {"status": "insufficient_history"} for h in horizons}}

    days_recorded = int(history_df["date"].nunique())
    result = {"days_recorded": days_recorded, "horizons": {}}

    for horizon in horizons:
        if days_recorded < horizon + 1:
            result["horizons"][horizon] = {"status": "insufficient_history"}
            continue
        try:
            df_h = _forward_return_for_horizon(history_df, horizon)
            n_pairs = int(df_h[f"fwd_return_{horizon}"].notna().sum())
            if n_pairs == 0:
                result["horizons"][horizon] = {"status": "insufficient_history"}
                continue
            result["horizons"][horizon] = {
                "status": "ok",
                "n_pairs": n_pairs,
                "hit_rate": hit_rate_by_signal(df_h, horizon),
                "score_buckets": avg_return_by_score_bucket(df_h, horizon),
                "factor_correlation": factor_correlation(df_h, horizon),
            }
        except Exception as e:
            logger.warning(f"Backtest for {horizon}-run horizon failed: {e}")
            result["horizons"][horizon] = {"status": "insufficient_history"}

    return result


# ---- Test ----
if __name__ == "__main__":
    import sys
    from logger import setup_logging
    from history_tracker import HistoryTracker

    setup_logging()
    ht = HistoryTracker(data_dir="../data")
    history = ht.load_history()
    print(f"{history['date'].nunique() if not history.empty else 0} day(s) recorded")
    summary = run_backtest(history)
    print(f"days_recorded={summary['days_recorded']}")
    for h, stats in summary["horizons"].items():
        print(f"  {h}-run horizon: {stats}")
    sys.exit(0)
