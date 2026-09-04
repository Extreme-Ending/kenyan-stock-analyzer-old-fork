"""
Telegram notification module for the Kenyan Stock Analyzer.

Sends compact market summaries as Telegram messages via the Bot API, with
optional document (PDF) delivery. Free, official Bot API — no business
verification needed (see ROADMAP.txt section 1b for setup via @BotFather).
"""

import requests
from utils import DEFAULT_HTTP_TIMEOUT, now_eat
from logger import get_logger

logger = get_logger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

# Telegram's sendDocument caption limit (characters).
_CAPTION_LIMIT = 1024

_CHECKPOINT_LABELS = {
    'open': 'Market Open',
    'mid': 'Mid-Day',
    'close': 'Market Close',
}


class TelegramNotifier:
    """Sends notifications to a Telegram chat via the Bot API."""

    def __init__(self, config):
        """
        Args:
            config: Config object with telegram_bot_token/telegram_chat_ids
                settings.
        """
        self.config = config
        self.bot_token = config.telegram_bot_token
        self.chat_ids = config.telegram_chat_ids
        self._validate()

    def _validate(self):
        """Check that required settings are present."""
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not configured")
        if not self.chat_ids:
            logger.warning("TELEGRAM_CHAT_ID not configured")

    def send_message(self, text, parse_mode='HTML'):
        """
        Send a text message to every configured Telegram chat.

        Args:
            text: Message body. When parse_mode='HTML', only Telegram's HTML
                subset is honored (b, i, a, code, pre) -- not full HTML like
                the email body.
            parse_mode: Telegram parse mode, default 'HTML'.

        Returns:
            bool: True if sent to at least one chat, False otherwise
                (including when Telegram is not configured, or every send
                failed). A failure on one chat_id does not stop delivery to
                the others -- this degrades the channel per-recipient, it
                never raises to the caller.
        """
        if not self.bot_token or not self.chat_ids:
            logger.error("Telegram not configured — cannot send")
            return False

        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        sent = 0
        for chat_id in self.chat_ids:
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True,
            }
            try:
                response = requests.post(url, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
                if response.status_code == 200 and response.json().get('ok'):
                    sent += 1
                else:
                    logger.error(f"Telegram sendMessage to {chat_id} failed: "
                                f"{response.status_code} {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send Telegram message to {chat_id}: {e}")

        logger.info(f"Telegram message sent to {sent}/{len(self.chat_ids)} chat(s)")
        return sent > 0

    def send_document(self, filepath, caption=None):
        """
        Send a local file (e.g. the PDF summary) to every configured chat.

        Args:
            filepath: Path to the local file to send.
            caption: Optional caption text, truncated to Telegram's limit.

        Returns:
            bool: True if sent to at least one chat, False otherwise
                (missing config, missing file, or every request failing all
                degrade to False rather than raising).
        """
        if not self.bot_token or not self.chat_ids:
            logger.error("Telegram not configured — cannot send")
            return False

        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendDocument"
        sent = 0
        for chat_id in self.chat_ids:
            data = {'chat_id': chat_id}
            if caption:
                data['caption'] = caption[:_CAPTION_LIMIT]
            try:
                with open(filepath, 'rb') as f:
                    files = {'document': f}
                    response = requests.post(url, data=data, files=files, timeout=DEFAULT_HTTP_TIMEOUT)
                if response.status_code == 200 and response.json().get('ok'):
                    sent += 1
                else:
                    logger.error(f"Telegram sendDocument to {chat_id} failed: "
                                f"{response.status_code} {response.text}")
            except (OSError, requests.exceptions.RequestException) as e:
                logger.error(f"Failed to send Telegram document to {chat_id}: {e}")

        logger.info(f"Telegram document sent to {sent}/{len(self.chat_ids)} chat(s): {filepath}")
        return sent > 0

    def generate_summary_text(self, analysis_results, sector_data=None,
                              breadth=None, checkpoint='mid', alerts=None):
        """
        Build a compact market summary in Telegram's HTML subset.

        Args:
            analysis_results: dict from AnalysisEngine.
            sector_data: dict from SectorAnalyzer.
            breadth: dict from AnalysisEngine.calculate_market_breadth.
            checkpoint: one of 'open', 'mid', 'close' -- controls the
                heading text only; the intraday vs. end-of-day content
                selection is decided by the caller.
            alerts: optional dict of symbol -> list of alert strings, from
                scoring.generate_alerts. When given, the single highest
                alert per symbol is included so the message stays compact.

        Returns:
            str: Telegram-HTML formatted message text.
        """
        now = now_eat().strftime('%Y-%m-%d %H:%M EAT')
        total = len(analysis_results)

        bullish = sum(
            1 for r in analysis_results.values()
            if r and r.get('signals', {}).get('overall') == 'bullish'
        )
        bearish = sum(
            1 for r in analysis_results.values()
            if r and r.get('signals', {}).get('overall') == 'bearish'
        )

        changes = []
        for sym, r in analysis_results.items():
            if r and r.get('daily_change_pct') is not None:
                changes.append((sym, r['daily_change_pct']))
        changes.sort(key=lambda x: x[1], reverse=True)
        top_gainers = changes[:3]
        top_losers = changes[-3:][::-1] if len(changes) >= 3 else []

        label = _CHECKPOINT_LABELS.get(checkpoint, 'Update')
        lines = [
            f"<b>NSE {label} Update</b>",
            now,
            "",
            f"Stocks: {total} | Bullish: {bullish} | Bearish: {bearish}",
        ]

        if breadth:
            parts = []
            for key, breadth_label in [
                ('pct_above_sma50', 'Above SMA50'),
                ('pct_bullish_macd', 'Bullish MACD'),
                ('pct_rsi_above_50', 'RSI>50'),
            ]:
                if key in breadth:
                    parts.append(f"{breadth_label}: {breadth[key]}%")
            if parts:
                lines.append(" | ".join(parts))

        if top_gainers or top_losers:
            lines.append("")
            lines.append("<b>Top Movers</b>")
            for sym, chg in top_gainers + top_losers:
                arrow = "▲" if chg > 0 else "▼"
                lines.append(f"{arrow} {sym}: {chg:+.2f}%")

        if sector_data:
            lines.append("")
            lines.append("<b>Sectors</b>")
            for name, data in sector_data.items():
                lines.append(f"{name}: {data['avg_change_pct']:+.2f}% ({data['count']} stocks)")

        if alerts:
            lines.append("")
            lines.append("<b>Alerts</b>")
            for sym, sym_alerts in alerts.items():
                if sym_alerts:
                    lines.append(f"⚠ {sym}: {sym_alerts[0]}")

        return "\n".join(lines)
