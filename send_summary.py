#!/usr/bin/env python3
"""
Daily summary notifier.

Runs a lean version of the pipeline (data → analysis → fundamentals →
price validation → scoring), builds a compact one-page PDF summary of key
metrics, and delivers it on one of three intraday checkpoints (see
ROADMAP.txt section 1): --checkpoint open (email, full summary + PDF),
mid (Telegram, short intraday pulse), or close (Telegram, end-of-day
summary + PDF). Intended to run unattended on a schedule (e.g. GitHub
Actions, three times a day), independent of any laptop.

Does NOT generate the full dashboard or per-stock reports — it is a subset.
Run manually to test:  python send_summary.py --checkpoint open
"""

import os
import sys
from datetime import datetime

# Ensure Homebrew libs are findable for WeasyPrint on macOS (no-op on Linux)
if sys.platform == 'darwin':
    hb = '/opt/homebrew/lib'
    if os.path.isdir(hb):
        os.environ['DYLD_LIBRARY_PATH'] = f"{hb}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from dotenv import load_dotenv
load_dotenv()

from config import Config
from logger import get_logger, setup_logging
from data_acquisition import DataAcquisition
from analysis_engine import AnalysisEngine
from fundamental_analysis import FundamentalAnalysis
from report_generator import ReportGenerator
from sector_analysis import SectorAnalyzer

config = Config()
setup_logging(config)
logger = get_logger(__name__)


def market_closed_today():
    """
    Return (closed: bool, reason: str|None) for the NSE today, evaluated in
    Nairobi time. The NSE does not trade on weekends or Kenyan public
    holidays (New Year, Easter, Labour Day, Madaraka, Mashujaa, Jamhuri,
    Christmas, Boxing Day, Eid, etc.).
    """
    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("Africa/Nairobi")).date()
    except Exception:
        today = datetime.now().date()

    if today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return True, "weekend"

    try:
        import holidays
        ke = holidays.Kenya(years=today.year)
        if today in ke:
            return True, ke.get(today)
    except Exception as e:
        # If the holiday check is unavailable, don't block the report.
        logger.warning(f"Holiday check unavailable ({e}); proceeding anyway.")

    return False, None


VALID_CHECKPOINTS = ('open', 'mid', 'close')


def get_checkpoint() -> str:
    """
    Read --checkpoint <open|mid|close> from argv. Defaults to 'open' (email,
    full summary) so existing callers that don't pass --checkpoint keep
    today's behavior unchanged.

    Returns:
        One of VALID_CHECKPOINTS.

    Raises:
        SystemExit: if --checkpoint is given an unrecognized value.
    """
    if '--checkpoint' not in sys.argv:
        return 'open'
    i = sys.argv.index('--checkpoint')
    if i + 1 >= len(sys.argv):
        sys.exit("--checkpoint requires a value: open, mid, or close")
    value = sys.argv[i + 1]
    if value not in VALID_CHECKPOINTS:
        sys.exit(f"--checkpoint must be one of {VALID_CHECKPOINTS}, got '{value}'")
    return value


def main():
    checkpoint = get_checkpoint()

    logger.info("=" * 60)
    logger.info(f"NSE SUMMARY [{checkpoint.upper()}] — {datetime.now():%Y-%m-%d %H:%M}")
    logger.info("=" * 60)

    # Skip weekends and Kenyan public holidays (NSE is closed — no new data).
    # Pass --ignore-calendar to force a run anyway (e.g. manual testing).
    if '--ignore-calendar' not in sys.argv:
        closed, reason = market_closed_today()
        if closed:
            logger.info(f"NSE is closed today ({reason}) — skipping [{checkpoint}] summary.")
            print(f"Skipped: NSE closed today ({reason}). No notification sent.")
            return

    force = '--force-refresh' in sys.argv

    # New day → wipe stale data cache so the first run fetches fresh (does NOT
    # touch the reports folder here, to avoid clobbering a local dashboard).
    from utils import enforce_daily_cache
    enforce_daily_cache(config.cache_dir)

    data_acq = DataAcquisition(data_sources=config.data_sources, cache_dir=config.cache_dir)
    engine = AnalysisEngine(config=config)
    # clean_old=False so building the summary does not wipe an existing
    # dashboard in the same reports folder (matters only on a shared machine;
    # on the GitHub runner the folder is already cleared each run).
    report_gen = ReportGenerator(template_dir=config.template_dir,
                                 output_dir=config.report_directory, clean_old=False)

    # ---- Data + analysis ----
    logger.info("Fetching stock data...")
    stock_data = data_acq.fetch_all_stocks(period='6mo', interval='1d', force_refresh=force)
    if not stock_data:
        logger.error("No stock data — aborting summary")
        sys.exit(1)
    analysis_results = engine.analyze_multiple_stocks(stock_data)

    # ---- Anchor prices to the NSE official close + cross-check ----
    validations = {}
    if config.enable_price_validation or config.enable_official_close:
        try:
            from price_validation import PriceValidator, apply_official_close
            pv = PriceValidator(cache_dir=config.cache_dir,
                                disagree_threshold_pct=config.price_disagree_threshold_pct)
            reference = pv.fetch_reference_prices()
            if config.enable_price_validation:
                for sym, r in analysis_results.items():
                    if r:
                        validations[sym] = pv.validate(sym, r.get('latest', {}).get('close'), stock_data.get(sym))
            if config.enable_official_close:
                apply_official_close(analysis_results, reference, logger)
        except Exception as e:
            logger.warning(f"Official-close/validation skipped: {e}")

    breadth = engine.calculate_market_breadth(analysis_results)
    sector_data = SectorAnalyzer().analyze_sectors(stock_data, analysis_results)

    # ---- Fundamentals ----
    fund_analyzer = FundamentalAnalysis(cache_dir=config.cache_dir)
    fundamentals_data = fund_analyzer.fetch_all_fundamentals(force_refresh=force)

    # ---- Validate dividends against the authoritative NSE calendar ----
    try:
        from dividend_calendar import apply_dividend_calendar
        apply_dividend_calendar(fundamentals_data, cache_dir=config.cache_dir, logger=logger)
    except Exception as e:
        logger.warning(f"Dividend validation skipped: {e}")

    # ---- Export upcoming earnings as an iCalendar (.ics) for email attach ----
    ics_path = None
    try:
        from earnings_calendar import write_ics
        ics_path = write_ics(
            fundamentals_data,
            os.path.join(config.report_directory, "earnings.ics"),
        )
    except Exception as e:
        logger.warning(f"Earnings ICS export skipped: {e}")

    # ---- Context, scoring, alerts ----
    usd_kes = None
    sector_medians = {}
    try:
        from market_context import compute_sector_medians, fetch_usd_kes
        sector_medians = compute_sector_medians(fundamentals_data)
        if config.enable_fx:
            usd_kes = fetch_usd_kes()
    except Exception as e:
        logger.warning(f"Market context skipped: {e}")

    scores, alerts = {}, {}
    try:
        from scoring import score_stock, generate_alerts
        for sym, r in analysis_results.items():
            if r:
                f = fundamentals_data.get(sym, {})
                scores[sym] = score_stock(sym, r, f, sector_medians=sector_medians)
                a = generate_alerts(sym, r, f, validations.get(sym))
                if a:
                    alerts[sym] = a
    except Exception as e:
        logger.warning(f"Scoring skipped: {e}")

    # ---- Build the summary PDF ----
    logger.info("Building summary PDF...")
    result = report_gen.generate_summary(
        analysis_results, fundamentals_data=fundamentals_data, validations=validations,
        scores=scores, alerts=alerts, breadth=breadth, sector_data=sector_data,
        usd_kes=usd_kes, watchlist=config.stock_symbols, report_type='both',
    )
    # result is (html, pdf) for report_type='both', or a single path
    pdf_path = None
    if isinstance(result, tuple):
        for p in result:
            if p and p.endswith('.pdf'):
                pdf_path = p
    elif isinstance(result, str) and result.endswith('.pdf'):
        pdf_path = result

    # ---- Deliver on the requested checkpoint ----
    # open  -> email (full summary + PDF + .ics)
    # mid   -> Telegram (short intraday pulse, text only)
    # close -> Telegram (end-of-day summary + PDF)
    if checkpoint == 'open':
        ok = send_open_email(config, analysis_results, sector_data, breadth, pdf_path, ics_path, result)
    else:
        ok = send_telegram_update(config, checkpoint, analysis_results, sector_data, breadth, alerts, pdf_path, result)

    if not ok:
        sys.exit(1)


def send_open_email(config, analysis_results, sector_data, breadth, pdf_path, ics_path, built_result):
    """
    Send the market-open summary by email (full body + PDF + .ics attachments).

    Args:
        config: Config object.
        analysis_results: dict from AnalysisEngine.analyze_multiple_stocks.
        sector_data: dict from SectorAnalyzer.analyze_sectors.
        breadth: dict from AnalysisEngine.calculate_market_breadth.
        pdf_path: Path to the generated summary PDF, or None.
        ics_path: Path to the generated earnings .ics, or None.
        built_result: Whatever report_gen.generate_summary returned, used
            only for the "built but not sent" log/print message.

    Returns:
        bool: True on success, or if email is intentionally disabled
        (config.enable_email_notifications is false — that's a valid
        "built but not sent" outcome, not a failure).
    """
    if not config.enable_email_notifications:
        logger.warning("ENABLE_EMAIL_NOTIFICATIONS is false — summary built but not emailed.")
        print(f"Summary built: {built_result}")
        return True

    from email_notifier import EmailNotifier
    notifier = EmailNotifier(config)
    body = notifier.generate_email_body(analysis_results, sector_data, breadth)
    attachments = [pdf_path] if pdf_path else []
    if ics_path and os.path.exists(ics_path):
        attachments.append(ics_path)
    subject = f"NSE Market Open Summary — {datetime.now():%Y-%m-%d}"
    ok = notifier.send_report(subject, body, attachments=attachments)
    if ok:
        logger.info(f"Summary emailed (attachment: {pdf_path})")
        print("Summary emailed successfully.")
    else:
        logger.error("Failed to email summary")
    return ok


def send_telegram_update(config, checkpoint, analysis_results, sector_data, breadth, alerts, pdf_path, built_result):
    """
    Send a mid-day or close-of-market update via Telegram.

    'mid' sends a short text-only intraday pulse; 'close' sends the
    end-of-day summary text plus the PDF as a document attachment.

    Args:
        config: Config object.
        checkpoint: 'mid' or 'close'.
        analysis_results: dict from AnalysisEngine.analyze_multiple_stocks.
        sector_data: dict from SectorAnalyzer.analyze_sectors.
        breadth: dict from AnalysisEngine.calculate_market_breadth.
        alerts: dict of symbol -> list of alert strings.
        pdf_path: Path to the generated summary PDF, or None.
        built_result: Whatever report_gen.generate_summary returned, used
            only for the "built but not sent" log/print message.

    Returns:
        bool: True on success, or if Telegram is intentionally disabled
        (config.enable_telegram_notifications is false).
    """
    if not config.enable_telegram_notifications:
        logger.warning("ENABLE_TELEGRAM_NOTIFICATIONS is false — summary built but not sent.")
        print(f"Summary built: {built_result}")
        return True

    from telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier(config)

    # 'mid' -> text-only intraday pulse (no breadth/sectors, keeps it short).
    # 'close' -> full text summary plus the PDF as a document attachment.
    if checkpoint == 'mid':
        text = notifier.generate_summary_text(analysis_results, checkpoint=checkpoint, alerts=alerts)
        ok = notifier.send_message(text)
    else:  # close
        text = notifier.generate_summary_text(analysis_results, sector_data=sector_data,
                                              breadth=breadth, checkpoint=checkpoint, alerts=alerts)
        ok = notifier.send_message(text)
        if pdf_path and os.path.exists(pdf_path):
            ok = notifier.send_document(pdf_path, caption="NSE Daily Summary PDF") and ok

    if ok:
        logger.info(f"Telegram [{checkpoint}] update sent")
        print(f"Telegram [{checkpoint}] update sent successfully.")
    else:
        logger.error(f"Failed to send Telegram [{checkpoint}] update")
    return ok


if __name__ == "__main__":
    main()
