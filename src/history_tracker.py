"""
History tracker.

Appends a daily snapshot of every stock (price, change, signals, overall
score, and each factor sub-score) to a persistent CSV so the tool can, over
time, be audited for accuracy — "how did the signals we showed actually
play out?" — and so price history from our own feed accumulates
independently of any single provider. src/backtest.py reads this file back
to compute forward-return hit rates (see IMPROVEMENTS.txt item 2).

Idempotent per day: re-running on the same date replaces that day's rows
rather than duplicating them. Fails safe: any error is logged and the pipeline
continues.
"""

import os
from datetime import datetime

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

HISTORY_SUBDIR = "history"
HISTORY_FILE = "daily_snapshots.csv"


class HistoryTracker:
    def __init__(self, data_dir="data"):
        self.dir = os.path.join(data_dir, HISTORY_SUBDIR)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, HISTORY_FILE)

    def record_snapshot(self, analysis_results, fundamentals_data=None,
                        scores=None, validations=None, date=None):
        """
        Append today's snapshot. Returns number of rows written, or 0 on error.
        """
        try:
            date = date or datetime.now().strftime("%Y-%m-%d")
            fundamentals_data = fundamentals_data or {}
            scores = scores or {}
            validations = validations or {}

            rows = []
            for symbol, result in analysis_results.items():
                if not result:
                    continue
                latest = result.get("latest", {})
                signals = result.get("signals", {})
                fund = fundamentals_data.get(symbol, {})
                sc = scores.get(symbol, {})
                val = validations.get(symbol, {})
                rows.append({
                    "date": date,
                    "symbol": symbol,
                    "price": latest.get("close"),
                    "change_pct": result.get("daily_change_pct"),
                    "rsi": latest.get("rsi"),
                    "overall_signal": signals.get("overall"),
                    "tech_rating": fund.get("tech_rating"),
                    "score": sc.get("overall"),
                    # Sub-scores, recorded so the backtest module (see
                    # IMPROVEMENTS.txt item 2) can correlate each factor
                    # against forward returns, not just the overall blend.
                    "score_value": sc.get("value"),
                    "score_quality": sc.get("quality"),
                    "score_momentum": sc.get("momentum"),
                    "score_dividend": sc.get("dividend"),
                    "score_liquidity": sc.get("liquidity"),
                    "reference_price": val.get("reference_price"),
                    "price_status": val.get("status"),
                })

            if not rows:
                return 0

            new_df = pd.DataFrame(rows)

            # Merge with existing, replacing today's rows
            if os.path.exists(self.path):
                try:
                    old = pd.read_csv(self.path)
                    old = old[old["date"] != date]
                    combined = pd.concat([old, new_df], ignore_index=True)
                except Exception as e:
                    logger.debug(f"History merge fallback: {e}")
                    combined = new_df
            else:
                combined = new_df

            combined.to_csv(self.path, index=False)
            logger.info(f"History snapshot: {len(new_df)} rows for {date} "
                        f"({len(combined)} total)")
            return len(new_df)
        except Exception as e:
            logger.warning(f"History snapshot failed: {e}")
            return 0

    def load_history(self):
        """Return the full history DataFrame, or empty if none."""
        if os.path.exists(self.path):
            try:
                return pd.read_csv(self.path)
            except Exception as e:
                logger.debug(f"History load error: {e}")
        return pd.DataFrame()

    def days_recorded(self):
        df = self.load_history()
        return df["date"].nunique() if not df.empty else 0


# ---- Test ----
if __name__ == "__main__":
    from logger import setup_logging
    setup_logging()
    ht = HistoryTracker(data_dir="../data")
    fake = {
        "SCOM": {"latest": {"close": 35.0, "rsi": 55.0},
                 "signals": {"overall": "bullish"}, "daily_change_pct": 1.2},
    }
    n = ht.record_snapshot(fake, {"SCOM": {"tech_rating": 0.18}},
                           {"SCOM": {"overall": 68, "value": 55, "quality": 70,
                                    "momentum": 75, "dividend": 60, "liquidity": 80}},
                           {"SCOM": {"reference_price": 35.05, "status": "ok"}})
    print(f"wrote {n} rows; days recorded={ht.days_recorded()}")
    print(ht.load_history().tail())
