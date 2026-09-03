#!/bin/bash
# Stage real changes and commit -- never .env/.env.local, and never pure
# file-permission noise (this sandbox has a habit of chmod'ing the whole
# tree, which git tracks as a "modified" file with a 0/0 diff).
#
# Usage: ./safe_commit.sh "commit message"

set -e
cd "$(dirname "$0")"

# This checkout is sometimes owned by a different user than the one running
# git (sandbox artifact) -- git refuses to operate at all ("dubious
# ownership") unless told per-invocation to trust it. -c is a one-off
# override, not a persisted config change.
git() { command git -c safe.directory='*' "$@"; }

MSG="$1"
if [ -z "$MSG" ]; then
  echo "Usage: $0 \"commit message\"" >&2
  exit 1
fi

# Never let .env slip in, even if something upstream force-adds it later.
git reset -q -- .env .env.local 2>/dev/null || true

echo "Staging real changes (skipping pure permission-mode noise)..."
while IFS= read -r -d '' entry; do
  file="${entry:3}"
  # Untracked files always show a real diff when staged -- just add them.
  if ! git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
    git add -- "$file"
    continue
  fi
  # Tracked files: skip if the only change is the mode bit (0/0 in numstat).
  numstat="$(git diff --numstat -- "$file")"
  if [[ "$numstat" == 0$'\t'0$'\t'* ]]; then
    continue
  fi
  git add -- "$file"
done < <(git status --porcelain -z -- . ':!.env' ':!.env.local')

echo
echo "Staged for commit:"
git status --short --untracked-files=no | grep -E '^[MARC]' || echo "  (nothing staged -- exiting)"
echo

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "$MSG"
