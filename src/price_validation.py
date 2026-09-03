"""
Price validation module.

Cross-checks the prices shown on the dashboard (sourced from TradingView)
against an INDEPENDENT second source and assesses data freshness. Primary
reference: afx.kwayisi.org, which publishes the Nairobi Securities Exchange
board. Fallback: the official NSE daily price list PDF (reusing
DataAcquisition's existing OCR-based PDF parser) when afx is unreachable —
see IMPROVEMENTS.txt item 5.

The goal is accuracy: rather than trusting a single feed, every stock's price
is compared against a second source and flagged when they disagree, and each
price is tagged with when the stock last actually traded (thinly-traded NSE
counters may not trade for days, making a "last price" misleading). The
disagreement threshold itself scales with how liquid the stock is: a flat
threshold flags far more false "mismatch"es on thinly-traded counters
(naturally wider spreads between two sources) than on heavily-traded ones.

Everything here fails safe: if BOTH sources are unreachable, validation
degrades to "unverified" and the rest of the pipeline is unaffected.
"""

import os
import re
import json
import requests
from datetime import datetime

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

AFX_URL = "https://afx.kwayisi.org/nse/"
AFX_SOURCE = "afx.kwayisi.org"
PDF_SOURCE = "NSE official PDF"

# Liquidity-scaled disagreement threshold (IMPROVEMENTS.txt item 5). A
# thinly-traded counter can go days between trades, so its "last price"
# from two independent sources can legitimately diverge more than a
# heavily-traded counter's without either source being wrong. Tiers are by
# average daily value traded (KES); checked top-down, first match wins.
# None/no volume data gets the most tolerant tier — fail-safe, since we
# can't tell how liquid the stock is without it.
LIQUIDITY_THRESHOLD_TIERS = [
    (10_000_000, 1.0),  # >= 10M KES/day -> base threshold, unscaled
    (1_000_000, 2.0),   # >= 1M          -> 2x
    (100_000, 3.0),     # >= 100K        -> 3x
    (0, 5.0),           # thinly traded / unknown -> 5x
]


class PriceValidator:
    """Validate dashboard prices against an independent source + freshness."""

    def __init__(self, cache_dir="data", disagree_threshold_pct=1.0):
        """
        Args:
            cache_dir: Directory for the daily reference-price cache.
            disagree_threshold_pct: Base percent difference above which the
                two sources are considered to disagree, before liquidity
                scaling (see LIQUIDITY_THRESHOLD_TIERS).
        """
        self.cache_dir = cache_dir
        self.disagree_threshold_pct = disagree_threshold_pct
        os.makedirs(self.cache_dir, exist_ok=True)
        self._reference = None  # {ticker: {'price': float, 'volume': int}}
        self._reference_source = None  # set once fetch_reference_prices() runs
        self._clean_old_cache()

    def _clean_old_cache(self):
        today = datetime.now().strftime("%Y%m%d")
        for fname in os.listdir(self.cache_dir):
            if fname.startswith("reference_prices_") and today not in fname:
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                except OSError:
                    pass

    # ---- Reference price acquisition ----

    def _cache_path(self):
        today = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.cache_dir, f"reference_prices_{today}.json")

    def fetch_reference_prices(self, force_refresh=False):
        """
        Fetch the independent NSE price board: afx.kwayisi.org first, then
        the official NSE PDF price list if afx is unreachable (see
        IMPROVEMENTS.txt item 5) — only one source needs to answer for
        validation to work, instead of the whole run degrading to
        "unverified" the moment afx has a bad day.

        Always fetched fresh once per run (cheap for afx; the PDF fallback
        only triggers, and only downloads/OCRs once, when afx fails), then
        held in memory for the rest of the run. We deliberately do NOT reuse
        a same-day disk cache: NSE prices settle after the close, so an
        earlier intraday snapshot would be stale — every run must reflect
        the current official price.

        Returns:
            dict: {ticker: {'price': float, 'volume': int, 'change': float}},
            or {} if both sources fail. self._reference_source names which
            source actually answered ('afx.kwayisi.org', 'NSE official PDF',
            or None if neither did).
        """
        if self._reference is not None:
            return self._reference

        path = self._cache_path()

        self._reference = self._fetch_afx_reference()
        self._reference_source = AFX_SOURCE if self._reference else None

        if not self._reference:
            logger.warning(f"{AFX_SOURCE} unavailable — falling back to {PDF_SOURCE}")
            self._reference = self._fetch_pdf_reference()
            self._reference_source = PDF_SOURCE if self._reference else None

        if not self._reference:
            logger.warning(
                "Both reference price sources unavailable — skipping validation "
                "and price anchoring. TradingView values will be used as-is."
            )
            self._reference = {}
            return self._reference

        logger.info(
            f"Reference prices from {self._reference_source}: "
            f"{len(self._reference)} stocks"
        )
        try:
            with open(path, "w") as f:
                json.dump(self._reference, f)
        except Exception as e:
            logger.debug(f"Reference cache write error: {e}")

        return self._reference

    def _fetch_afx_reference(self):
        """Primary reference source. Returns {} on any failure (fail-safe)."""
        from utils import http_get
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = http_get(AFX_URL, headers=headers)
        if resp is None or resp.status_code != 200:
            return {}
        try:
            return self._parse_afx(resp.text)
        except Exception as e:
            logger.warning(f"{AFX_SOURCE} parse failed: {e}")
            return {}

    def _fetch_pdf_reference(self):
        """
        Fallback reference source: the official NSE daily price list PDF,
        via DataAcquisition's existing OCR-based parser. Reshapes its
        {symbol: {open, high, low, close, volume}} into this module's
        {ticker: {'price', 'volume', 'change'}} shape ('change' isn't in
        the PDF, left None — downstream code already treats a missing
        change as optional). Returns {} on any failure (fail-safe).
        """
        try:
            from data_acquisition import DataAcquisition
            board = DataAcquisition(cache_dir=self.cache_dir).get_pdf_price_board()
        except Exception as e:
            logger.warning(f"{PDF_SOURCE} fallback failed: {e}")
            return {}
        return {
            ticker: {"price": row.get("close"), "volume": row.get("volume"), "change": None}
            for ticker, row in (board or {}).items()
            if row.get("close")
        }

    @staticmethod
    def _parse_afx(html):
        """
        Parse the afx.kwayisi.org NSE board. Its rows use unclosed tags:
            <tr><td><a>TICKER</a><td><a>Name</a><td>VOLUME<td>PRICE<td ...>CHG
        """
        strip = lambda s: re.sub(r"<[^>]+>", "", s).replace("&amp;", "&").strip()
        out = {}
        for row in html.split("<tr>"):
            if "/nse/" not in row:
                continue
            segs = row.split("<td")
            if len(segs) < 5:
                continue
            cells = [
                strip(s.split(">", 1)[1]) if ">" in s else "" for s in segs[1:]
            ]
            ticker = cells[0]
            if not re.fullmatch(r"[A-Z0-9]{2,6}", ticker):
                continue
            try:
                volume = int(cells[2].replace(",", ""))
                price = float(cells[3].replace(",", ""))
            except (ValueError, IndexError):
                continue
            # The last cell is the day's absolute price change (may be +/-).
            change = None
            if len(cells) > 4:
                try:
                    change = float(cells[4].replace(",", "").replace("+", ""))
                except ValueError:
                    change = None
            out[ticker] = {"price": price, "volume": volume, "change": change}
        return out

    # ---- Validation ----

    @staticmethod
    def _avg_value_traded(history_df):
        """
        Average daily KES value traded (close * volume) over the most
        recent up-to-20 actually-traded bars — the liquidity proxy for
        LIQUIDITY_THRESHOLD_TIERS.

        Returns:
            float, or None if there's no usable volume data (history_df is
            missing, empty, lacks a 'volume' column, or every bar has zero
            volume).
        """
        if history_df is None or history_df.empty or "volume" not in history_df.columns:
            return None
        traded = history_df[history_df["volume"] > 0]
        if traded.empty:
            return None
        recent = traded.tail(20)
        value_traded = (recent["close"] * recent["volume"]).mean()
        return float(value_traded) if pd.notna(value_traded) else None

    @staticmethod
    def _liquidity_multiplier(avg_value_traded):
        """
        Scale factor for disagree_threshold_pct based on average daily
        value traded (see LIQUIDITY_THRESHOLD_TIERS). None (no volume data)
        gets the most tolerant tier — fail-safe, since we can't tell how
        liquid the stock is without it.
        """
        if avg_value_traded is None:
            return LIQUIDITY_THRESHOLD_TIERS[-1][1]
        for floor, multiplier in LIQUIDITY_THRESHOLD_TIERS:
            if avg_value_traded >= floor:
                return multiplier
        return LIQUIDITY_THRESHOLD_TIERS[-1][1]

    def validate(self, symbol, dashboard_price, history_df=None):
        """
        Validate one stock's price and assess freshness.

        Args:
            symbol: Ticker symbol.
            dashboard_price: The price the dashboard shows (TradingView).
            history_df: Optional OHLCV DataFrame to derive last-traded date
                and average value traded (used to scale the disagreement
                threshold — see LIQUIDITY_THRESHOLD_TIERS).

        Returns:
            dict with keys:
                reference_price, reference_source, pct_diff, agree,
                threshold_pct, liquidity_multiplier, last_traded_date,
                days_stale, is_stale, status, note
        """
        ref = self.fetch_reference_prices()
        ref_row = ref.get(symbol.upper()) if ref else None
        reference_price = ref_row["price"] if ref_row else None

        avg_value_traded = self._avg_value_traded(history_df)
        liquidity_multiplier = self._liquidity_multiplier(avg_value_traded)
        threshold_pct = round(self.disagree_threshold_pct * liquidity_multiplier, 2)

        pct_diff = None
        agree = None
        if reference_price and dashboard_price and reference_price > 0:
            pct_diff = (dashboard_price - reference_price) / reference_price * 100
            agree = abs(pct_diff) <= threshold_pct

        # Freshness from history (last bar with real volume)
        last_traded_date = None
        days_stale = None
        if history_df is not None and not history_df.empty:
            try:
                traded = history_df
                if "volume" in history_df.columns:
                    traded = history_df[history_df["volume"] > 0]
                if not traded.empty:
                    last_idx = traded.index[-1]
                    last_traded_date = (
                        last_idx.strftime("%Y-%m-%d")
                        if hasattr(last_idx, "strftime")
                        else str(last_idx)
                    )
                    if hasattr(last_idx, "date"):
                        days_stale = (datetime.now().date() - last_idx.date()).days
            except Exception as e:
                logger.debug(f"Freshness calc error for {symbol}: {e}")

        is_stale = bool(days_stale and days_stale > 1)

        # Status precedence: unverified -> mismatch -> stale -> ok
        if reference_price is None:
            status = "unverified"
            note = "No independent source to compare against"
        elif agree is False:
            status = "mismatch"
            note = (
                f"TradingView ({dashboard_price:g}) differs from the "
                f"{self._reference_source} price ({reference_price:g}) by "
                f"{pct_diff:+.1f}% (threshold {threshold_pct:g}%) — price uncertain"
            )
        elif is_stale:
            status = "stale"
            note = f"Last traded {days_stale} days ago ({last_traded_date})"
        else:
            status = "ok"
            note = f"Matches {self._reference_source} within {threshold_pct:g}%"

        return {
            "reference_price": reference_price,
            "reference_source": self._reference_source,
            "pct_diff": round(pct_diff, 2) if pct_diff is not None else None,
            "agree": agree,
            "threshold_pct": threshold_pct,
            "liquidity_multiplier": liquidity_multiplier,
            "last_traded_date": last_traded_date,
            "days_stale": days_stale,
            "is_stale": is_stale,
            "status": status,
            "note": note,
        }


def apply_official_close(analysis_results, reference, logger=None):
    """
    Anchor each stock's DISPLAYED price and daily change to the NSE official
    closing price (from the reference board), which after market close is a
    stable, settled value — unlike TradingView's ~15-min-delayed feed.

    - Sets latest['close'] to the official close.
    - Recomputes daily_change_pct from the official day's change when available.
    - Keeps the original TradingView close as latest['tv_close'] so the two can
      still be cross-checked, and tags latest['price_source'].

    Fails safe per stock: if there is no official price, the TradingView value
    is left untouched. Returns the number of stocks anchored.
    """
    anchored = 0
    for sym, result in (analysis_results or {}).items():
        if not result:
            continue
        latest = result.get('latest')
        if latest is None:
            continue
        row = (reference or {}).get(sym.upper())
        if not row or not row.get('price'):
            latest['price_source'] = 'TradingView'
            continue
        off = row['price']
        latest['tv_close'] = latest.get('close')
        latest['close'] = off
        # Generic label: `reference` may now come from either afx.kwayisi.org
        # or the NSE PDF fallback (see fetch_reference_prices) -- the actual
        # source per stock is in validation['reference_source'] instead.
        latest['price_source'] = 'NSE official'
        chg = row.get('change')
        if chg is not None and (off - chg) != 0:
            result['daily_change_pct'] = chg / (off - chg) * 100.0
        anchored += 1
    if logger:
        logger.info(f"  Anchored {anchored} prices to the NSE official close")
    return anchored


# ---- Test ----
if __name__ == "__main__":
    from logger import setup_logging
    setup_logging()
    logger = get_logger(__name__)

    pv = PriceValidator()
    ref = pv.fetch_reference_prices()
    print(f"Reference prices: {len(ref)} stocks")
    for sym, price in [("SCOM", 35.0), ("KCB", 81.0), ("HAFR", 1.18)]:
        print(f"\n{sym} @ {price}:")
        print(" ", pv.validate(sym, price))
