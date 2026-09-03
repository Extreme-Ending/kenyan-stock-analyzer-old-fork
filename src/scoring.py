"""
Transparent factor-scoring & screening module.

Combines the metrics the pipeline already gathers (valuation, quality,
momentum, dividend, liquidity) into a transparent 0-100 score PER FACTOR and
an overall blend. Every input and every point is exposed in `reasons`, so the
score is a screen you can inspect and tune — never a black box.

This is a mechanical screen of public metrics, NOT investment advice.

Also produces per-stock alerts (oversold, near 52-week low, strong signal,
high sustainable yield, illiquid, price-source mismatch) for the dashboard.
"""

from logger import get_logger

logger = get_logger(__name__)

# Default factor weights (must sum to 1.0). Tunable via Config.
DEFAULT_WEIGHTS = {
    "value": 0.25,
    "quality": 0.25,
    "momentum": 0.20,
    "dividend": 0.15,
    "liquidity": 0.15,
}

# Below this many peers, a sector median is too noisy to score against
# (e.g. a 1-2 stock "sector" bucket) -- fall back to the absolute score alone.
MIN_SECTOR_PEERS = 3


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def _sector_median(fund, sector_medians, key):
    """
    Look up this stock's sector median for `key`.

    Args:
        fund: This stock's fundamentals dict (used only for its 'sector').
        sector_medians: {sector: {metric: median, 'count': n}} from
            market_context.compute_sector_medians(), or None/empty.
        key: Metric name, e.g. 'pe_ratio'.

    Returns:
        The median, or None if unavailable or the sector has fewer than
        MIN_SECTOR_PEERS stocks with data for this metric.
    """
    if not sector_medians:
        return None
    bucket = sector_medians.get(fund.get("sector") or "Unknown")
    if not bucket or bucket.get("count", 0) < MIN_SECTOR_PEERS:
        return None
    return bucket.get(key)


def _relative_verdict(ratio, lower_is_better):
    """Plain-English verdict for a value/median ratio (mirrors
    market_context.valuation_vs_sector's thresholds, for consistent language
    with the fundamentals page)."""
    if lower_is_better:
        if ratio < 0.8:
            return "cheaper than sector"
        if ratio > 1.25:
            return "pricier than sector"
    else:
        if ratio > 1.25:
            return "above sector"
        if ratio < 0.8:
            return "below sector"
    return "in line with sector"


def _blend_with_sector(absolute_score, value, median, lower_is_better):
    """
    Blend an absolute sub-score 50/50 with a sector-relative score (50 = in
    line with the sector median, scaled linearly above/below).

    Fail-safe: if `median` is unavailable (thin sector, or sector_medians not
    passed to score_stock), returns the absolute score unchanged -- a missing
    sector comparison never blocks or skews scoring, it just isn't added.

    Returns:
        (score, extra_reason_string_or_None)
    """
    if absolute_score is None or value is None or not median or median <= 0:
        return absolute_score, None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return absolute_score, None
    ratio = value / median
    delta = (ratio - 1) * 50
    relative_score = _clamp(50 - delta if lower_is_better else 50 + delta)
    blended = (absolute_score + relative_score) / 2
    return blended, f"sector: {_relative_verdict(ratio, lower_is_better)}"


def _score_value(fund, sector_medians=None):
    """
    Lower P/E and P/B score higher, blended with how the stock's P/E and P/B
    compare to its own sector's median (when enough sector peers exist --
    see MIN_SECTOR_PEERS) so a naturally-low-P/E sector (e.g. banking) isn't
    scored on the same absolute curve as a naturally-higher-P/E one.

    Args:
        fund: This stock's fundamentals dict.
        sector_medians: {sector: {metric: median, 'count': n}} from
            market_context.compute_sector_medians(), or None to score on the
            absolute curve alone.

    Returns:
        (score, reasons)
    """
    parts, reasons = [], []
    pe = fund.get("pe_ratio")
    if pe and pe > 0:
        s = _clamp(100 - (pe - 8) * 4)  # pe 8 -> 100, pe 33 -> 0
        s, extra = _blend_with_sector(s, pe, _sector_median(fund, sector_medians, "pe_ratio"), True)
        parts.append(s)
        reasons.append(f"P/E {pe:.1f}" + (f" ({extra})" if extra else ""))
    pb = fund.get("price_to_book")
    if pb and pb > 0:
        s = _clamp(100 - (pb - 1) * 30)  # pb 1 -> 100, pb ~4.3 -> 0
        s, extra = _blend_with_sector(s, pb, _sector_median(fund, sector_medians, "price_to_book"), True)
        parts.append(s)
        reasons.append(f"P/B {pb:.2f}" + (f" ({extra})" if extra else ""))
    peg = fund.get("peg_ratio")
    if peg and peg > 0:
        s = _clamp(100 - (peg - 0.5) * 50)  # peg 0.5 -> 100, peg 2.5 -> 0 (no sector PEG median available)
        parts.append(s)
        reasons.append(f"PEG {peg:.2f}")
    if not parts:
        return None, ["no valuation data"]
    return round(sum(parts) / len(parts)), reasons


def _score_quality(fund, sector_medians=None):
    """
    Higher ROE/margins, lower leverage score higher. ROE is blended with how
    it compares to the stock's sector median (see _score_value's docstring
    for why -- same sector-relative rationale applies to ROE).

    Args:
        fund: This stock's fundamentals dict.
        sector_medians: {sector: {metric: median, 'count': n}}, or None to
            score ROE on the absolute curve alone.

    Returns:
        (score, reasons)
    """
    parts, reasons = [], []
    roe = fund.get("roe")
    if roe is not None:
        s = _clamp(roe * 4)  # roe 25% -> 100
        s, extra = _blend_with_sector(s, roe, _sector_median(fund, sector_medians, "roe"), False)
        parts.append(s)
        reasons.append(f"ROE {roe:.1f}%" + (f" ({extra})" if extra else ""))
    nm = fund.get("net_margin")
    if nm is not None:
        parts.append(_clamp(nm * 3.3))  # ~30% -> 100
        reasons.append(f"net margin {nm:.1f}%")
    de = fund.get("debt_to_equity")
    if de is not None and de >= 0:
        parts.append(_clamp(100 - de * 40))  # de 0 -> 100, de 2.5 -> 0
        reasons.append(f"D/E {de:.2f}")
    cr = fund.get("current_ratio")
    if cr is not None and cr > 0:
        parts.append(_clamp(cr * 50))  # cr 2 -> 100
        reasons.append(f"current ratio {cr:.2f}")
    if not parts:
        return None, ["no quality data"]
    return round(sum(parts) / len(parts)), reasons


def _score_momentum(analysis_result, fund):
    """Trend/MACD/RSI and 3-month performance."""
    parts, reasons = [], []
    signals = (analysis_result or {}).get("signals", {})
    latest = (analysis_result or {}).get("latest", {})

    overall = signals.get("overall")
    if overall == "bullish":
        parts.append(75); reasons.append("technical: bullish")
    elif overall == "bearish":
        parts.append(25); reasons.append("technical: bearish")
    elif overall == "neutral":
        parts.append(50); reasons.append("technical: neutral")

    rsi = latest.get("rsi")
    if rsi is not None:
        # Reward healthy uptrend (50-65); penalise overbought/oversold extremes
        if rsi > 70:
            parts.append(35)
        elif rsi < 30:
            parts.append(45)  # oversold: possible bounce, not strong momentum
        else:
            parts.append(_clamp(50 + (rsi - 50) * 2))
        reasons.append(f"RSI {rsi:.0f}")

    perf = fund.get("perf_3m")
    if perf is not None:
        parts.append(_clamp(50 + perf * 2))  # +25% -> 100
        reasons.append(f"3M {perf:+.1f}%")

    if not parts:
        return None, ["no momentum data"]
    return round(sum(parts) / len(parts)), reasons


def _score_dividend(fund, sector_medians=None):
    """
    Reward yield (blended with how it compares to the stock's sector median,
    see _score_value's docstring), but only if the payout looks sustainable.

    Args:
        fund: This stock's fundamentals dict.
        sector_medians: {sector: {metric: median, 'count': n}}, or None to
            score yield on the absolute curve alone.

    Returns:
        (score, reasons)
    """
    parts, reasons = [], []
    dy = fund.get("dividend_yield")
    if dy is not None:
        s = _clamp(dy * 12.5)  # 8% -> 100
        s, extra = _blend_with_sector(s, dy, _sector_median(fund, sector_medians, "dividend_yield"), False)
        parts.append(s)
        reasons.append(f"yield {dy:.1f}%" + (f" ({extra})" if extra else ""))
    payout = fund.get("dividend_payout_ratio")
    if payout is not None and payout > 0:
        # 40-70% is healthy; >100% is unsustainable
        if payout > 100:
            parts.append(20)
        elif payout > 80:
            parts.append(55)
        else:
            parts.append(85)
        reasons.append(f"payout {payout:.0f}%")
    if not parts:
        return None, ["no dividend"]
    return round(sum(parts) / len(parts)), reasons


def _score_liquidity(fund):
    """Higher traded value = easier to enter/exit. KES value traded per day."""
    vt = fund.get("value_traded")
    if vt is None or vt <= 0:
        return None, ["no liquidity data"]
    # 100M KES/day -> ~100; 1M -> ~30
    import math
    s = _clamp((math.log10(vt) - 6) * 33)  # 1e6 ->0, 1e9 ->99
    return round(s), [f"traded KES {vt/1e6:.1f}M"]


# Thresholds for horizon classification (see IMPROVEMENTS.txt item 8).
_HORIZON_STRONG = 65
_HORIZON_WEAK = 40


def _short_term_trigger(analysis_result):
    """
    Name the specific technical trigger behind a SHORT-TERM call, with a
    holding period tied to that trigger type rather than one fixed guess.
    """
    signals = (analysis_result or {}).get("signals", {})
    latest = (analysis_result or {}).get("latest", {})
    rsi = latest.get("rsi")
    if signals.get("macd") == "bullish_cross":
        return "1-4 weeks", "fresh MACD bullish crossover"
    if signals.get("ma_crossover") == "golden_cross":
        return "1-3 months", "golden cross (50/200-day MA)"
    if rsi is not None and rsi < 30:
        return "days", "RSI oversold bounce"
    return "1-4 weeks", "bullish technical momentum"


def _classify_horizon(scores, analysis_result):
    """
    Classify the time horizon a stock's call is actionable over, from which
    sub-scores are actually driving it (see IMPROVEMENTS.txt item 8). A
    momentum-only call (fresh MACD cross, RSI bounce) is a days-to-weeks
    trade; a fundamentals-only call (cheap + high quality + sustainable
    yield, all agreeing) is a months-to-years thesis. Showing both as the
    same undifferentiated "Buy" is misleading.

    Args:
        scores: {value, quality, momentum, dividend, liquidity} sub-scores
            (0-100 or None), as computed in score_stock().
        analysis_result: dict from AnalysisEngine.analyze_multiple_stocks,
            used only to name the specific technical trigger for the
            suggested short-term holding period.

    Returns:
        dict: {label: 'SHORT-TERM'|'LONG-TERM'|'MIXED'|'UNCLEAR',
               period: suggested holding-period string or None,
               drivers: list of sub-score names behind the label,
               reason: one-sentence plain-English explanation}
    """
    momentum = scores.get("momentum")
    fundamentals = {k: scores[k] for k in ("value", "quality", "dividend") if scores.get(k) is not None}
    fundamental_avg = sum(fundamentals.values()) / len(fundamentals) if fundamentals else None

    # "Agree" = not just a high average, but no single fundamental factor
    # dragging the thesis down (e.g. cheap but low-quality shouldn't count).
    fundamentals_agree = (
        fundamental_avg is not None and fundamental_avg >= _HORIZON_STRONG
        and all(v >= 50 for v in fundamentals.values())
    )
    momentum_strong = momentum is not None and momentum >= _HORIZON_STRONG
    momentum_weak = momentum is not None and momentum < _HORIZON_WEAK
    fundamentals_weak = fundamental_avg is not None and fundamental_avg < _HORIZON_WEAK

    if momentum is None and fundamental_avg is None:
        return {"label": "UNCLEAR", "period": None, "drivers": [],
                "reason": "Not enough momentum or fundamental data to classify."}

    # Explicit conflicts -- surface these rather than forcing a single label.
    if fundamentals_agree and momentum_weak:
        return {"label": "MIXED", "period": None,
                "drivers": list(fundamentals.keys()) + ["momentum"],
                "reason": "Fundamentals look attractive but the technical trend is bearish -- conflicting signals."}
    if momentum_strong and fundamentals_weak:
        return {"label": "MIXED", "period": None,
                "drivers": ["momentum"] + list(fundamentals.keys()),
                "reason": "Technically bullish but fundamentals are weak -- a momentum trade, not a fundamentals-backed one."}

    if momentum_strong and fundamentals_agree:
        return {"label": "LONG-TERM", "period": "6-12+ months (with a near-term tailwind)",
                "drivers": ["momentum"] + list(fundamentals.keys()),
                "reason": "Fundamentals and momentum agree -- a long-term thesis with a favorable entry point."}

    if momentum_strong:
        period, trigger = _short_term_trigger(analysis_result)
        return {"label": "SHORT-TERM", "period": period, "drivers": ["momentum"],
                "reason": f"Momentum-driven ({trigger}); fundamentals aren't confirming a longer thesis."}

    if fundamentals_agree:
        return {"label": "LONG-TERM", "period": "6-12+ months",
                "drivers": list(fundamentals.keys()),
                "reason": "Value, quality and dividend sub-scores agree the stock is attractively priced and sound."}

    return {"label": "MIXED", "period": None, "drivers": [],
            "reason": "No single factor group dominates strongly enough to call a horizon."}


def score_stock(symbol, analysis_result, fund, weights=None, sector_medians=None):
    """
    Produce a transparent factor score for one stock.

    Args:
        symbol: Stock ticker.
        analysis_result: dict from AnalysisEngine.analyze_multiple_stocks.
        fund: This stock's fundamentals dict.
        weights: Optional override for DEFAULT_WEIGHTS.
        sector_medians: Optional {sector: {metric: median, 'count': n}} from
            market_context.compute_sector_medians(). When given, P/E, P/B,
            dividend yield, and ROE are each blended 50/50 with how the
            stock compares to its own sector's median (see
            _score_value/_score_quality/_score_dividend), instead of being
            scored on one fixed absolute curve regardless of sector. Falls
            back to the absolute curve alone when omitted or when a sector
            has too few peers (MIN_SECTOR_PEERS) to be meaningful.

    Returns dict:
        {overall, value, quality, momentum, dividend, liquidity, reasons,
         horizon}
    Sub-scores are 0-100 or None when data is missing. `overall` is the
    weighted blend of the available sub-scores (weights renormalised).
    `horizon` classifies the time horizon this call is actionable over
    (SHORT-TERM/LONG-TERM/MIXED/UNCLEAR) from which sub-scores are driving
    it -- see _classify_horizon's docstring (IMPROVEMENTS.txt item 8).
    """
    weights = weights or DEFAULT_WEIGHTS
    fund = fund or {}

    subs = {
        "value": _score_value(fund, sector_medians),
        "quality": _score_quality(fund, sector_medians),
        "momentum": _score_momentum(analysis_result, fund),
        "dividend": _score_dividend(fund, sector_medians),
        "liquidity": _score_liquidity(fund),
    }

    scores = {k: v[0] for k, v in subs.items()}
    reasons = {k: v[1] for k, v in subs.items()}

    # Weighted blend over available sub-scores only
    num = 0.0
    den = 0.0
    for k, s in scores.items():
        if s is not None:
            w = weights.get(k, 0)
            num += s * w
            den += w
    overall = round(num / den) if den > 0 else None

    return {
        "symbol": symbol,
        "overall": overall,
        **scores,
        "reasons": reasons,
        "horizon": _classify_horizon(scores, analysis_result),
    }


def generate_alerts(symbol, analysis_result, fund, validation=None):
    """
    Produce a list of short, transparent alert strings for one stock.
    Each alert states the fact that triggered it.
    """
    alerts = []
    latest = (analysis_result or {}).get("latest", {})
    signals = (analysis_result or {}).get("signals", {})
    fund = fund or {}

    rsi = latest.get("rsi")
    if rsi is not None:
        if rsi < 30:
            alerts.append(f"🟢 Oversold (RSI {rsi:.0f})")
        elif rsi > 70:
            alerts.append(f"🔴 Overbought (RSI {rsi:.0f})")

    # 52-week proximity
    close = latest.get("close")
    hi = fund.get("price_52w_high")
    lo = fund.get("price_52w_low")
    if close and hi and hi > 0 and close >= hi * 0.98:
        alerts.append("🔺 Near 52-week high")
    if close and lo and lo > 0 and close <= lo * 1.03:
        alerts.append("🔻 Near 52-week low")

    # Strong technical signal
    tr = fund.get("tech_rating")
    if tr is not None:
        if tr >= 0.5:
            alerts.append("⭐ TradingView: Strong Buy signal")
        elif tr <= -0.5:
            alerts.append("⚠️ TradingView: Strong Sell signal")

    # Fresh MACD cross
    if signals.get("macd") == "bullish_cross":
        alerts.append("📈 MACD bullish crossover today")
    elif signals.get("macd") == "bearish_cross":
        alerts.append("📉 MACD bearish crossover today")

    # High sustainable dividend yield
    dy = fund.get("dividend_yield")
    payout = fund.get("dividend_payout_ratio")
    if dy and dy >= 8 and (payout is None or payout <= 100):
        alerts.append(f"💰 High dividend yield ({dy:.1f}%)")

    # Upcoming dividend ex-date
    if fund.get("dividend_ex_date_is_upcoming") and fund.get("dividend_ex_date"):
        alerts.append(f"📅 Ex-dividend {fund['dividend_ex_date']}")

    # Illiquid warning
    vt = fund.get("value_traded")
    if vt is not None and vt < 1_000_000:
        alerts.append("💧 Thinly traded (hard to exit)")

    # Price-source disagreement
    if validation and validation.get("status") == "mismatch":
        alerts.append(f"❗ Price unverified — {validation.get('note', '')}")
    elif validation and validation.get("status") == "stale":
        alerts.append(f"🕒 {validation.get('note', '')}")

    return alerts


# ---- Test ----
if __name__ == "__main__":
    import json, glob
    files = glob.glob("../data/fundamentals_*.json")
    if files:
        data = json.load(open(files[0]))
        for sym in ["SCOM", "KCB", "EQTY", "HAFR"]:
            f = data.get(sym, {})
            sc = score_stock(sym, {}, f)
            print(f"\n{sym}: overall={sc['overall']}  "
                  f"V={sc['value']} Q={sc['quality']} M={sc['momentum']} "
                  f"D={sc['dividend']} L={sc['liquidity']}")
            print("   alerts:", generate_alerts(sym, {}, f))
