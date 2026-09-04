#!/usr/bin/env python3
"""
Telegram bot poller.

GitHub Actions can't run an always-on server to receive Telegram's webhook
in real time, so this polls Telegram's getUpdates API on a short interval
instead (see .github/workflows/telegram-poller.yml, every 5 minutes --
GitHub's minimum cron granularity). No local state is needed to track
which updates have already been seen: passing offset=<last update_id>+1
back to Telegram marks them consumed server-side, for good, regardless of
which process polls next (see TelegramNotifier.get_updates's docstring).

Two behaviors, by sender:
  - The configured owner chat(s) (config.telegram_chat_ids) sending
    exactly "/refresh" -> triggers a full data refresh. This script sends
    the fresh summary back to the owner only (never broadcast) and writes
    a `refresh_requested` GitHub Actions output so the workflow can
    conditionally run the heavier dashboard rebuild + Pages redeploy in a
    separate job.
  - Anyone else's first message -> added to the persisted subscriber list
    (src/telegram_subscribers.py) and immediately sent the current summary
    + PDF as a one-time welcome. They then also receive every future
    scheduled broadcast (send_summary.py's mid/close checkpoints already
    read this same subscriber list) -- but never the owner's on-demand
    /refresh sends, which stay private to avoid spamming subscribers every
    time the owner tests something.

Run manually to test: python telegram_poller.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from dotenv import load_dotenv
load_dotenv()

from config import Config
from logger import get_logger, setup_logging
from telegram_notifier import TelegramNotifier

config = Config()
setup_logging(config)
logger = get_logger(__name__)

REFRESH_COMMAND = '/refresh'


def _write_github_output(name, value):
    """
    Append a `name=value` line to $GITHUB_OUTPUT so a later workflow step
    can read it via `steps.<id>.outputs.<name>`. No-op (just logs) outside
    GitHub Actions, e.g. when run manually -- never raises.
    """
    path = os.environ.get('GITHUB_OUTPUT')
    if not path:
        logger.debug(f"GITHUB_OUTPUT not set (not running in Actions) -- {name}={value}")
        return
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"{name}={value}\n")
    except OSError as e:
        logger.warning(f"Could not write GitHub Actions output {name}: {e}")


def partition_updates(updates, owner_ids, subscribers_path=None):
    """
    Sort a batch of Telegram updates into an owner refresh request and any
    newly-seen subscriber chat IDs. Pulled out of main() so this routing
    logic -- the one part of the poller with real branching to get wrong
    (misrouting a stranger's message as an owner refresh, for instance) --
    is testable with synthetic updates, without hitting the real Bot API.

    Args:
        updates: Raw list[dict] from TelegramNotifier.get_updates().
        owner_ids: Set of chat IDs (as str) allowed to trigger /refresh.
        subscribers_path: Override for telegram_subscribers' storage path
            (tests use a temp file; production uses the module default).

    Returns:
        tuple[bool, list[str]]: (refresh_requested, new_subscriber_chat_ids)
    """
    from telegram_subscribers import add_subscriber, DEFAULT_SUBSCRIBERS_PATH
    path = subscribers_path or DEFAULT_SUBSCRIBERS_PATH

    refresh_requested = False
    new_subscribers = []
    for update in updates:
        message = update.get('message') or update.get('edited_message')
        if not message or 'chat' not in message:
            continue
        chat_id = str(message['chat']['id'])
        text = (message.get('text') or '').strip()

        if chat_id in owner_ids and text.lower() == REFRESH_COMMAND:
            logger.info(f"Refresh requested by owner chat {chat_id}")
            refresh_requested = True
        elif chat_id not in owner_ids:
            if add_subscriber(chat_id, path=path):
                logger.info(f"New Telegram subscriber: {chat_id}")
                new_subscribers.append(chat_id)

    return refresh_requested, new_subscribers


def main():
    # print() alongside logger throughout this function -- matches
    # send_summary.py's existing convention for CLI entry-point status
    # output (see CLAUDE.md's Logging section), and unlike the logger
    # calls, is guaranteed to show up in the GitHub Actions step log
    # regardless of console-handler behavior in that environment. This
    # poller runs unattended every 5 minutes; when a real /refresh comes
    # in, the Actions log needs to be readable without guessing.
    if not config.enable_telegram_notifications:
        logger.warning("ENABLE_TELEGRAM_NOTIFICATIONS is false — poller has nothing to do.")
        print("Telegram notifications disabled — nothing to poll.")
        _write_github_output('refresh_requested', 'false')
        return

    notifier = TelegramNotifier(config)
    owner_ids = {str(c) for c in config.telegram_chat_ids}

    updates = notifier.get_updates()
    if not updates:
        logger.info("No new Telegram messages.")
        print("No new Telegram messages.")
        _write_github_output('refresh_requested', 'false')
        return

    print(f"Received {len(updates)} update(s) since last poll.")
    refresh_requested, new_subscribers = partition_updates(updates, owner_ids)
    print(f"refresh_requested={refresh_requested}, new_subscribers={new_subscribers}")

    # Mark every update up to the highest update_id seen as consumed, so
    # the next poll (5 minutes from now, by a totally separate process)
    # never sees these again. The result is discarded -- this call exists
    # purely for its offset side effect.
    max_update_id = max(u['update_id'] for u in updates)
    notifier.get_updates(offset=max_update_id + 1, timeout=0)

    if not refresh_requested and not new_subscribers:
        _write_github_output('refresh_requested', 'false')
        return

    # Both a refresh and a welcome need the same freshly-built summary --
    # build it once and reuse it for whichever recipients need it.
    print("Building summary artifacts...")
    from send_summary import build_summary
    artifacts = build_summary(force_refresh=refresh_requested)
    text = notifier.generate_summary_text(
        artifacts['analysis_results'], sector_data=artifacts['sector_data'],
        breadth=artifacts['breadth'], checkpoint='close', alerts=artifacts['alerts'],
    )
    pdf_path = artifacts['pdf_path']

    if refresh_requested:
        for chat_id in owner_ids:
            ok = notifier.send_message_to(chat_id, f"<b>Refresh complete.</b>\n\n{text}")
            if pdf_path and os.path.exists(pdf_path):
                ok = notifier.send_document_to(chat_id, pdf_path, caption="NSE Refresh Summary") and ok
            print(f"Refresh reply to owner {chat_id}: {'sent' if ok else 'FAILED'}")

    welcome_text = (
        "<b>Welcome to the NSE Stock Analyzer bot!</b>\n"
        "You'll now receive market updates automatically. Here's the latest:\n\n"
        f"{text}"
    )
    for chat_id in new_subscribers:
        ok = notifier.send_message_to(chat_id, welcome_text)
        if pdf_path and os.path.exists(pdf_path):
            ok = notifier.send_document_to(chat_id, pdf_path, caption="NSE Daily Summary PDF") and ok
        print(f"Welcome message to new subscriber {chat_id}: {'sent' if ok else 'FAILED'}")

    _write_github_output('refresh_requested', 'true' if refresh_requested else 'false')


if __name__ == "__main__":
    main()
