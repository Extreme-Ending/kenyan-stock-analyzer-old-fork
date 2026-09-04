"""
Email notification module for the Kenyan Stock Analyzer.

Sends HTML emails with market summaries and optional PDF attachments.
Supports Gmail (app passwords) and generic SMTP servers.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.utils import format_datetime
from logger import get_logger
from utils import now_eat

logger = get_logger(__name__)


class EmailNotifier:
    """Sends email notifications with report attachments."""

    def __init__(self, config):
        """
        Args:
            config: Config object with SMTP settings.
        """
        self.config = config
        self.smtp_host = config.smtp_host
        self.smtp_port = config.smtp_port
        self.user = config.email_user
        self.password = config.email_password
        self.recipients = config.email_recipients

        self._validate()

    def _validate(self):
        """Check that required settings are present."""
        if not self.user:
            logger.warning("EMAIL_USER not configured")
        if not self.password:
            logger.warning("EMAIL_PASSWORD not configured")
        if not self.recipients:
            logger.warning("EMAIL_RECIPIENTS not configured")

    def send_report(self, subject, html_body, attachments=None):
        """
        Send an email with HTML body and optional attachments.

        Args:
            subject: Email subject line.
            html_body: HTML string for the email body.
            attachments: List of file paths to attach.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not self.user or not self.password or not self.recipients:
            logger.error("Email not configured — cannot send")
            return False

        msg = MIMEMultipart('mixed')
        msg['Subject'] = subject
        msg['From'] = self.user
        msg['To'] = ', '.join(self.recipients)
        msg['Date'] = format_datetime(now_eat())

        # Attach HTML body
        html_part = MIMEText(html_body, 'html', 'utf-8')
        msg.attach(html_part)

        # Attach files. .ics files get a proper text/calendar MIME so Gmail
        # renders the "Add to calendar" button on the message.
        if attachments:
            for filepath in attachments:
                try:
                    filename = filepath.split('/')[-1]
                    lower = filename.lower()
                    with open(filepath, 'rb') as f:
                        payload = f.read()
                    if lower.endswith('.ics'):
                        part = MIMEBase('text', 'calendar', method='PUBLISH', name=filename)
                    elif lower.endswith('.pdf'):
                        part = MIMEBase('application', 'pdf', name=filename)
                    else:
                        part = MIMEBase('application', 'octet-stream')
                    part.set_payload(payload)
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename="{filename}"'
                    )
                    msg.attach(part)
                except Exception as e:
                    logger.error(f"Failed to attach {filepath}: {e}")

        # Send
        try:
            context = ssl.create_default_context()
            if self.smtp_port == 587:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.starttls(context=context)
                    server.login(self.user, self.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30, context=context) as server:
                    server.login(self.user, self.password)
                    server.send_message(msg)

            logger.info(f"Email sent to {len(self.recipients)} recipients")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "SMTP authentication failed. If using Gmail, ensure you're "
                "using an App Password (not your regular password). "
                "See: https://myaccount.google.com/apppasswords"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def generate_email_body(self, analysis_results, sector_data=None,
                            breadth=None):
        """
        Generate a compact HTML email body with market summary.

        Args:
            analysis_results: dict from AnalysisEngine.
            sector_data: dict from SectorAnalyzer.
            breadth: dict from AnalysisEngine.calculate_market_breadth.

        Returns:
            HTML string suitable for email clients.
        """
        now = now_eat().strftime('%Y-%m-%d %H:%M EAT')
        total = len(analysis_results)

        # Count signals
        bullish = sum(
            1 for r in analysis_results.values()
            if r and r.get('signals', {}).get('overall') == 'bullish'
        )
        bearish = sum(
            1 for r in analysis_results.values()
            if r and r.get('signals', {}).get('overall') == 'bearish'
        )

        # Top gainers/losers
        changes = []
        for sym, r in analysis_results.items():
            if r and r.get('daily_change_pct') is not None:
                changes.append((sym, r['daily_change_pct']))
        changes.sort(key=lambda x: x[1], reverse=True)
        top_gainers = changes[:3]
        top_losers = changes[-3:][::-1] if len(changes) >= 3 else []

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #1e293b; padding: 20px; }}
        h1 {{ color: #1e293b; font-size: 1.3rem; }}
        h2 {{ font-size: 1.1rem; margin-top: 20px; border-bottom: 2px solid #3b82f6; display: inline-block; }}
        .header {{ background: linear-gradient(135deg, #1e293b, #334155); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .header h1 {{ color: white; }}
        .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
        .stat {{ background: #f1f5f9; padding: 12px 16px; border-radius: 8px; text-align: center; flex: 1; min-width: 100px; }}
        .stat .big {{ font-size: 1.5rem; font-weight: 700; color: #3b82f6; }}
        .stat .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; }}
        .bullish {{ color: #22c55e; }}
        .bearish {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 8px 0; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; color: #64748b; font-size: 0.75rem; text-transform: uppercase; }}
        .footer {{ margin-top: 24px; padding-top: 12px; border-top: 1px solid #e2e8f0; font-size: 0.75rem; color: #94a3b8; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NSE Daily Market Report</h1>
        <p>{now}</p>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="big">{total}</div>
            <div class="label">Stocks</div>
        </div>
        <div class="stat">
            <div class="big bullish">{bullish}</div>
            <div class="label">Bullish</div>
        </div>
        <div class="stat">
            <div class="big bearish">{bearish}</div>
            <div class="label">Bearish</div>
        </div>
        <div class="stat">
            <div class="big">{len(sector_data) if sector_data else 0}</div>
            <div class="label">Sectors</div>
        </div>
    </div>
"""
        # Market breadth
        if breadth:
            html += """
    <h2>Market Breadth</h2>
    <div class="stats">
"""
            for key, label in [
                ('pct_above_sma50', 'Above SMA50'),
                ('pct_bullish_macd', 'Bullish MACD'),
                ('pct_rsi_above_50', 'RSI > 50'),
            ]:
                if key in breadth:
                    html += f"""
        <div class="stat">
            <div class="big">{breadth[key]}%</div>
            <div class="label">{label}</div>
        </div>"""
            html += "\n    </div>\n"

        # Top gainers/losers
        if top_gainers or top_losers:
            html += """
    <h2>Top Movers</h2>
    <table>
        <tr><th>Symbol</th><th>Change</th><th>Direction</th></tr>
"""
            for sym, chg in top_gainers + top_losers:
                direction = "▲" if chg > 0 else "▼"
                cls = "bullish" if chg > 0 else "bearish"
                html += (
                    f'        <tr><td>{sym}</td>'
                    f'<td class="{cls}">{chg:+.2f}%</td>'
                    f'<td class="{cls}">{direction}</td></tr>\n'
                )
            html += "    </table>\n"

        # Sector performance
        if sector_data:
            html += """
    <h2>Sector Performance</h2>
    <table>
        <tr><th>Sector</th><th>Stocks</th><th>Avg Change</th><th>Bullish %</th></tr>
"""
            for name, data in sector_data.items():
                cls = "bullish" if data['avg_change_pct'] >= 0 else "bearish"
                html += (
                    f'        <tr><td>{name}</td>'
                    f'<td>{data["count"]}</td>'
                    f'<td class="{cls}">{data["avg_change_pct"]:+.2f}%</td>'
                    f'<td>{data["bullish_ratio"]}%</td></tr>\n'
                )
            html += "    </table>\n"

        html += f"""
    <div class="footer">
        Generated by Kenyan Stock Analyzer — {now}<br>
        For full reports, check the reports directory.
    </div>
</body>
</html>"""

        return html