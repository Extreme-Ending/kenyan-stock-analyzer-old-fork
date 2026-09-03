#!/bin/bash
# Interactive commit+push: prompts for a commit message, stages real changes
# (never .env/.env.local, never pure file-permission noise -- this sandbox
# has a habit of chmod'ing the whole tree, which git tracks as a "modified"
# file with a 0/0 diff), shows what's staged, commits, then asks before
# pushing.
#
# Usage: ./safe_commit.sh

set -e
cd "$(dirname "$0")"

# This checkout is sometimes owned by a different user than the one running
# git (sandbox artifact) -- git refuses to operate at all ("dubious
# ownership") unless told per-invocation to trust it. -c is a one-off
# override, not a persisted config change.
git() { command git -c safe.directory='*' "$@"; }

# Never let .env slip in, even if something upstream force-adds it later.
git reset -q -- .env .env.local 2>/dev/null || true

echo "Staging real changes (skipping .env and pure permission-mode noise)..."
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
staged="$(git status --short --untracked-files=no | grep -E '^[MARC]' || true)"
if [ -z "$staged" ]; then
  echo "  (nothing staged -- nothing to commit)"
  exit 0
fi
echo "$staged"
echo

read -r -p "Commit message: " MSG
if [ -z "$MSG" ]; then
  echo "Empty message -- aborting (changes remain staged, nothing committed)." >&2
  exit 1
fi

git commit -m "$MSG"
echo

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# One-time nudge: without a credential helper, every push re-prompts for a
# username/PAT. Never set this automatically (git config changes are the
# user's call, not a script's) -- just make the fix impossible to miss.
if [ -z "$(git config --get credential.helper 2>/dev/null)" ]; then
  echo "Note: no git credential helper is configured, so the push below may"
  echo "ask for your username/token again. To stop that happening every"
  echo "time, run this once (caches it in ~/.git-credentials):"
  echo "  git config --global credential.helper store"
  echo
fi

read -r -p "Push '$BRANCH' to origin now? [y/N] " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git push
  else
    git push -u origin "$BRANCH"
  fi
else
  echo "Committed locally only (not pushed)."
fi
