#!/bin/bash
# NSE Stock Analyzer — Notification Entry Point
# Sends the open/mid/close summary via send_summary.py. All settings
# (ENABLE_EMAIL_NOTIFICATIONS, ENABLE_TELEGRAM_NOTIFICATIONS,
# TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, EMAIL_*, ...) come from .env —
# nothing to configure here.
#
# Usage: ./send_notification.sh open|mid|close [extra send_summary.py args]
#   ./send_notification.sh open
#   ./send_notification.sh mid --ignore-calendar
#   ./send_notification.sh close --force-refresh

set -e
cd "$(dirname "$0")"

CHECKPOINT="$1"
if [[ "$CHECKPOINT" != "open" && "$CHECKPOINT" != "mid" && "$CHECKPOINT" != "close" ]]; then
  echo "Usage: $0 open|mid|close [extra send_summary.py args]" >&2
  exit 1
fi
shift

./venv/bin/python send_summary.py --checkpoint "$CHECKPOINT" "$@"
