"""
Telegram subscriber list persistence for the Kenyan Stock Analyzer.

Anyone who messages the bot is added here (see telegram_poller.py) and
from then on included in every scheduled Telegram broadcast
(send_summary.py's mid/close checkpoints), in addition to the fixed
TELEGRAM_CHAT_ID owner config.

Persisted as JSON under data/history/ -- the one data/ subdirectory NOT
gitignored (see CLAUDE.md's Project Structure) -- since GitHub Actions
runners are ephemeral and this list must survive across scheduled runs.
telegram_poller.py's workflow commits changes to this file back to the
repo; nothing else writes to it.
"""

import json
import os
from typing import List

from logger import get_logger

logger = get_logger(__name__)

DEFAULT_SUBSCRIBERS_PATH = os.path.join('data', 'history', 'telegram_subscribers.json')


def load_subscribers(path: str = DEFAULT_SUBSCRIBERS_PATH) -> List[str]:
    """
    Load the persisted list of subscriber chat IDs.

    Args:
        path: Path to the JSON file.

    Returns:
        list[str]: Chat IDs, or an empty list if the file is missing,
        empty, or malformed -- a corrupt subscriber file must never crash
        the notification pipeline, it just means no dynamic subscribers
        get a broadcast this run.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read subscriber list {path}: {e}")
        return []
    if not isinstance(data, list):
        logger.warning(f"{path} did not contain a JSON list; ignoring")
        return []
    return [str(c) for c in data]


def add_subscriber(chat_id, path: str = DEFAULT_SUBSCRIBERS_PATH) -> bool:
    """
    Add a chat ID to the persisted subscriber list if not already present.

    Args:
        chat_id: Telegram chat ID to add (any type accepting str()).
        path: Path to the JSON file.

    Returns:
        bool: True if this was a new subscriber (file was written), False
        if already subscribed (no write) or if the write failed -- lets
        the caller decide whether to send a one-time welcome message.
    """
    chat_id = str(chat_id)
    subscribers = load_subscribers(path)
    if chat_id in subscribers:
        return False
    subscribers.append(chat_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(subscribers, f, indent=2)
    except OSError as e:
        logger.error(f"Could not persist new subscriber {chat_id}: {e}")
        return False
    return True
