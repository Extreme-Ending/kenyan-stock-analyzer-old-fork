#!/bin/bash
# Kenyan Stock Analyzer — Single Entry Point
# Usage: ./run.sh

set -e
cd "$(dirname "$0")"

# Activate venv
source venv/bin/activate

# Fix WeasyPrint on macOS (no-op elsewhere)
if [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
fi

# Fix SSL for TradingView. Ask certifi for its own bundle path instead of
# hardcoding a Python version's site-packages dir -- the old hardcoded
# python3.14 path breaks every HTTPS fetch with "[Errno 2] No such file or
# directory" on any venv built with a different Python version.
CERT_FILE="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
if [ -n "$CERT_FILE" ]; then
  export SSL_CERT_FILE="$CERT_FILE"
fi

# Run the pipeline
python main.py \
  --report-type both \
  --export-excel \
  --detailed \
  "$@"

# Open the dashboard (best-effort; prints the path if there's no desktop opener)
echo ""
echo "Opening dashboard..."
if command -v open >/dev/null 2>&1; then
  open reports/index.html
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open reports/index.html
else
  echo "Dashboard ready: $(pwd)/reports/index.html"
fi