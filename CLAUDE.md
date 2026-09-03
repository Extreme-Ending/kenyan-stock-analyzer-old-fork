# CLAUDE.md
============================================================
STEERING DOCUMENT

Priority: High

This document guides architectural, code-quality, and testing
decisions for the Kenyan Stock Analyzer.

All contributors, including AI coding agents, should review
this document before implementing significant changes.

============================================================

============================================================
STEERING DOCUMENTS
============================================================

Before implementing significant changes, contributors and AI
coding agents should review:

1. README.md            — what the tool does, how to run it
2. IMPROVEMENTS.txt      — tracked data-analysis & accuracy work
3. ROADMAP.txt           — broader feature roadmap (notifications,
                            reliability, UX, testing, ops)

If implementation conflicts with these documents, seek
clarification before proceeding. When a change closes or
supersedes a tracked item in IMPROVEMENTS.txt or ROADMAP.txt,
update that file in the same change.

---

# Project Philosophy

This is a financial data tool. A wrong number here can cost
someone real money. Prioritize:

1. Accuracy over speed — cross-check, don't guess
2. Transparency over black-box output — every score/signal
   must expose WHY it fired (see src/scoring.py's `reasons`
   pattern — follow it for any new scoring/signal logic)
3. Fail-safe degradation over crashing — one bad data source
   degrades that feature to "unverified"/empty, it never takes
   down the whole run (see "Error Handling" below)
4. Readability over cleverness
5. Small incremental changes over unnecessary rewrites
6. Correctness over premature optimization

When uncertain, choose the safest, most transparent, most
maintainable option.

---

# Architecture Rules

The pipeline is a sequence of independent stages, each owned by
one src/ module:

  data_acquisition -> analysis_engine / fundamental_analysis
    -> price_validation / dividend_calendar / earnings_calendar
    -> market_context / sector_analysis -> scoring
    -> history_tracker -> report_generator

Every module has a single responsibility. Never combine:

- scraping / fetching external data
- analysis / scoring logic
- HTML / report rendering

into one module. A new external data source gets its own
module under src/, following the existing pattern (see
price_validation.py or dividend_calendar.py as references —
fetch, parse, fail-safe fallback, all in one focused file).

Pipeline stages should remain callable/testable independently
of the full main.py run.

---

# Python Version

- Python 3.10+ (README baseline; dev venv here runs 3.12)
- UTF-8
- Prefer standard library before adding a dependency

---

# Dependency Management

- pip + requirements.txt
- Before adding a dependency: justify why it's needed, prefer
  actively maintained libraries, avoid unnecessary packages
- Never install outside the project venv

---

# Project Structure

Keep the existing layout:

stockapp/
  main.py                 entry point — orchestrates the pipeline
  send_summary.py          lean pipeline for the daily email/notification job
  scheduler.py              optional local scheduler
  src/                     one module per pipeline stage/data source
  templates/                Jinja2 templates
  reports/                  generated output (gitignored)
  data/                     cache + history (gitignored except history/)
  logs/                     application logs (gitignored)
  test.py                   test suite (see Testing below)
  .github/workflows/        CI + scheduled jobs

Business logic never belongs inside CLI entry points
(main.py, send_summary.py, scheduler.py) — those orchestrate
calls into src/, they don't contain analysis logic themselves.

---

# Code Style

New or modified public functions should have:

- type hints
- a docstring (Google-style: description, Args, Returns, Raises)
- one responsibility

This project does not yet have type hints/docstrings on every
existing function — do not do a drive-by rewrite of untouched
code just to add them. Bring code up to standard when you're
already modifying it, not opportunistically elsewhere.

Functions should generally stay under ~50 lines. Longer
orchestration functions (e.g. main.py's pipeline driver) are
fine when they're coordinating well-defined steps, not doing
the work themselves.

Prefer composition and pure functions over inheritance.

---

# Naming

- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private members: _leading_underscore
- Avoid abbreviations

---

# Imports

Order: standard library, then third-party, then local modules.
Never use wildcard imports.

---

# Type Safety & Data Contracts

Existing modules pass plain dicts between pipeline stages
(e.g. `analysis_results`, `fundamentals_data`, `scores`). That
is the established convention here — don't force a project-wide
rewrite to Pydantic/dataclasses.

For NEW structured data shapes, prefer a TypedDict or dataclass
over an untyped dict so the shape is self-documenting. If you
keep using a plain dict, document its keys/types in the
function's docstring.

---

# Documentation

Every module states its purpose and responsibilities in a
module-level docstring — this repo already does this well
(see price_validation.py, history_tracker.py). Keep it up.

Every public function documents description, arguments,
returns, and any failure modes (what it returns/logs when a
data source is unavailable, since that's a normal, expected
path here — not an edge case).

---

# Logging

Use `src/logger.py`'s `get_logger(__name__)`.

`print()` is reserved for:
- each module's existing `if __name__ == "__main__":` self-test
  block (established convention — keep it)
- CLI status output in main.py / run.sh

Never use print() inside pipeline logic.

Log context, operation, and failures. Never log secrets, API
keys, or tokens (there shouldn't be any hardcoded, but be
careful with response bodies from scraped sources in debug logs).

---

# Error Handling — "Fail Safe" is the core design principle

Every external call (TradingView, afx.kwayisi.org,
mystocks.co.ke, NSE PDF, FX API, news, oil/CBK/T-bills/bonds
pages) MUST degrade gracefully on failure:

- Catch specific exceptions, not bare `except:`
- Return None / empty dict / "unverified" status and log a
  WARNING — never let one source's failure raise out of the
  pipeline and abort the whole run
- Preserve exception chaining (`raise ... from e`) on the rare
  path where re-raising is actually correct (i.e. a bug in our
  own code, not an external source being down)

This is not optional — it's why the tool currently survives
CBK/T-bills/bonds pages being down without crashing. Match this
pattern for every new data source.

---

# Security

- Never hardcode credentials, API keys, or secrets
- Secrets live in `.env` (gitignored, local only) or GitHub
  Actions secrets (scheduled jobs) — never in code, README, or
  commit history
- Always use HTTPS
- Always set request timeouts — reuse `utils.http_get()` rather
  than calling `requests` directly, so every external call gets
  the same timeout/retry behavior for free
- Validate/sanitize anything parsed from scraped HTML/PDF before
  it flows into analysis (see IMPROVEMENTS.txt #6, #7)
- Never use eval(), exec(), pickle.loads(), or shell=True

---

# Performance

- Reuse `utils.py`'s `retry` decorator and `http_get()` helper
  instead of writing new HTTP/retry logic
- Cache external calls per run-day where the existing pattern
  already does this (price_validation.py, data_acquisition.py)
- Avoid premature optimization; profile before optimizing

---

# Testing

Framework: `unittest` (see `test.py`). Migrating to pytest
incrementally is fine; a big-bang rewrite is not.

- Tests must never require live network access — follow the
  existing `make_sample_data()` pattern (reproducible synthetic
  OHLCV via a fixed seed) rather than hitting real APIs
- Every new feature needs unit tests, including failure-mode
  tests (what happens when a source is unavailable — see Error
  Handling above; that path should be tested, not just the
  happy path)
- Bug fixes require a regression test
- `scoring.py`, `price_validation.py`, and the dividend/earnings
  calendar modules currently lack coverage — extend, don't
  ignore, when touching that code (tracked in IMPROVEMENTS.txt)

---

# Determinism

Given identical input data, analysis and scoring functions must
produce identical output. Live data fetches are inherently
non-deterministic (that's expected) — but everything downstream
of a fetched DataFrame (indicators, scores, signals) must be a
pure function of that data.

---

# Formatting

No linter is currently wired into this repo. If one is added,
prefer `ruff` (check + format), wire it into CI, and apply it
repo-wide as its own change — not silently mixed into an
unrelated feature PR.

---

# Git

- Never modify unrelated files
- Prefer small, focused commits
- Commit format: `type(scope): description`
  e.g. `fix(price_validation): scale disagree threshold by liquidity`
- This repo receives external PRs (see merge history from
  contributors) — keep changes reviewable and scoped

---

# AI Coding Behaviour

Before writing code:

1. Read the target module's existing docstring and patterns —
   this codebase is unusually well-documented per-module; use it
2. Check `src/utils.py` before adding new HTTP/retry/support-
   resistance logic — it may already exist
3. Minimize changes; don't refactor unrelated code in the same
   change
4. Preserve the fail-safe pattern for anything touching an
   external source
5. When a request is ambiguous about scope, ask a concise
   clarifying question rather than guessing

Never rewrite working code without justification. Never
introduce a breaking change (e.g. changing a dict shape another
module depends on) without checking every caller.

---

# Definition of Done

A task is complete only when:

✓ Code runs (`python main.py` / relevant entry point)
✓ Existing tests still pass (`python test.py`)
✓ New tests added for new behavior, including failure modes
✓ Type hints + docstring on new/changed public functions
✓ Logging added for new failure points
✓ Fail-safe degradation preserved for any external-source code
✓ No secrets committed
✓ No unrelated files touched
✓ IMPROVEMENTS.txt / ROADMAP.txt updated if this closes or
  changes a tracked item

---

# Forbidden Practices

Do NOT:

- wildcard imports
- mutable default arguments
- hardcode secrets or commit `.env`
- global mutable state
- bare `except:` / swallowed exceptions
- disable SSL verification
- use eval(), exec(), pickle.loads(), or shell=True
- duplicate logic that already exists in src/utils.py
- remove tests without approval
- modify unrelated code in the same change
- optimize without evidence of a real bottleneck
- skip the fail-safe pattern for external-source code
- add a dependency without explaining why

Always leave the repository cleaner than you found it.
