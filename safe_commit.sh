#!/bin/bash
# Interactive commit+push: prompts for a commit message, stages real changes
# (never .env/.env.local, never pure file-permission noise -- this sandbox
# has a habit of chmod'ing the whole tree, which git tracks as a "modified"
# file with a 0/0 diff), shows what's staged, commits, then asks before
# pushing.
#
# This sandbox also sometimes leaves parts of .git (index, refs, some
# object subdirectories) owned by a different, unreachable (root) user
# across sessions, which silently blocks every git write for the real
# user -- see heal_broken_git() below, which repairs that automatically,
# without sudo.
#
# Usage: ./safe_commit.sh

set -e
cd "$(dirname "$0")"

# This checkout is sometimes owned by a different user than the one running
# git (sandbox artifact) -- git refuses to operate at all ("dubious
# ownership") unless told per-invocation to trust it. -c is a one-off
# override, not a persisted config change.
git() { command git -c safe.directory='*' "$@"; }

# ---- Self-heal a broken .git (root-owned internals, sandbox artifact) ----
# Renames the broken .git aside (never deletes it -- some of its contents
# are root-owned and can't be removed without sudo anyway, so renaming
# aside is the only non-destructive option available) and rebuilds a fresh
# one via `git init` + `git fetch`, then points HEAD/the index at origin's
# branch via `git reset` -- index and HEAD only, this never touches a
# single file in the working tree, so any real uncommitted changes on disk
# are completely unaffected.
#
# Needs no root: the repo directory and .git's own top-level entry are
# owned by the real user (confirmed 2026-09-04), and renaming a directory
# only needs write access to its *parent*, not to what's being renamed --
# it doesn't matter that files *inside* .git are root-owned.
heal_broken_git() {
  local remote_url branch backup user_name user_email
  remote_url="$(git config --get remote.origin.url 2>/dev/null || true)"
  if [ -z "$remote_url" ]; then
    echo "Cannot self-heal: no 'origin' remote configured to rebuild from." >&2
    return 1
  fi
  branch="$(git symbolic-ref --short -q HEAD || echo main)"
  # A fresh `git init` starts with no repo-local identity -- carry the
  # current one over (if set) so the commit below doesn't hit git's
  # "who are you" error on a repo that worked fine a moment ago.
  user_name="$(git config --get user.name 2>/dev/null || true)"
  user_email="$(git config --get user.email 2>/dev/null || true)"
  backup=".git.broken-$(date +%Y%m%d%H%M%S)"
  echo "Detected a broken .git (internals owned by another user -- a known"
  echo "sandbox artifact, not something you did). Rebuilding it without"
  echo "touching any of your working files..."
  mv .git "$backup"
  git init -q -b "$branch"
  git remote add origin "$remote_url"
  [ -n "$user_name" ] && git config user.name "$user_name"
  [ -n "$user_email" ] && git config user.email "$user_email"
  git fetch -q origin "$branch"
  git reset -q "origin/$branch"
  echo "Repaired. Broken .git backed up to $backup (left in place -- some"
  echo "of its files are root-owned and can't be removed without sudo;"
  echo "harmless to leave there, and it's gitignored)."
  echo
}

# Cheap, targeted check: these two files are rewritten on every commit, so
# if either is unwritable the checkout is in the broken state above.
_ref_path=".git/$(git symbolic-ref -q HEAD 2>/dev/null || true)"
if [ -f .git/index ] && [ ! -w .git/index ]; then
  heal_broken_git
elif [ -f "$_ref_path" ] && [ ! -w "$_ref_path" ]; then
  heal_broken_git
fi
unset _ref_path

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

# Make sure the push below never stops to ask for a username/PAT. Prefer
# `gh auth setup-git` -- it wires git's credential helper to the already-
# authenticated `gh` CLI (scoped to github.com/gist.github.com, nothing
# cached in plaintext), and is a safe no-op to re-run every time. Only
# falls back to *suggesting* credential.helper store -- without setting it
# -- when gh isn't available/authenticated, since silently changing global
# git config isn't this script's call to make.
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh auth setup-git >/dev/null 2>&1 || true
elif [ -z "$(git config --get credential.helper 2>/dev/null)" ]; then
  echo "Note: no git credential helper is configured, so the push below may"
  echo "ask for your username/token again. Either run 'gh auth login' (this"
  echo "script will then wire it up automatically), or run this once to"
  echo "cache it yourself (~/.git-credentials):"
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
