"""
src/api_client.py
=================
HTTP client for the football-data.org v4 REST API.

Responsibilities
----------------
* Enforce the FREE_PLUS_THREE rate limit (10 requests / 60 s) via a
  thread-safe sliding-window RateLimiter decorator so callers never need
  to manage sleeps manually and can never trigger a 429.
* Expose three clean data-fetching methods used by the scanner and models:
    - get_team_matches()     → recent FINISHED results for form / Poisson fit
    - get_upcoming_matches() → SCHEDULED fixtures with embedded odds
    - get_competition_teams()→ full squad list (used for batch pre-caching)
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Sliding-window rate limiter implemented as a callable decorator.

    Algorithm
    ---------
    Maintain a list of timestamps for the most recent `max_calls` requests.
    Before each new request:
      1. Purge timestamps older than `period` seconds.
      2. If the window is full, sleep until the oldest timestamp exits it,
         then clear the window and proceed.
      3. Append the current timestamp and execute the wrapped function.

    This guarantees ≤ max_calls requests inside any rolling `period`-second
    window regardless of call frequency, preventing 429 responses from the
    football-data.org API.

    Example
    -------
        limiter = RateLimiter(max_calls=10, period=60)

        @limiter
        def fetch(url): ...
    """

    def __init__(self, max_calls: int, period: int) -> None:
        self.max_calls = max_calls
        self.period    = period
        self._window: list[float] = []   # timestamps of recent calls

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.monotonic()

            # Drop timestamps outside the rolling window
            self._window = [t for t in self._window if now - t < self.period]

            if len(self._window) >= self.max_calls:
                # Sleep until the oldest call is more than `period` seconds old
                wait = self.period - (now - self._window[0]) + 0.2
                logger.info(
                    "[RateLimiter] Budget exhausted — sleeping %.1fs before next call",
                    wait,
                )
                time.sleep(wait)
                self._window.clear()

            self._window.append(time.monotonic())
            return func(*args, **kwargs)

        return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class FootballDataClient:
    """
    Thin wrapper around the football-data.org v4 REST API.

    All network calls are rate-limited automatically — callers simply call
    the public helpers and never worry about sleep() or 429 errors.

    Parameters
    ----------
    api_key  : Your football-data.org API key.
    base_url : Base URL (default https://api.football-data.org/v4).
    max_calls: Maximum API calls per `period` seconds (default 10).
    period   : Rate-limit window in seconds (default 60).
    """

    def __init__(
        self,
        api_key:   str,
        base_url:  str   = "https://api.football-data.org/v4",
        max_calls: int   = 10,
        period:    int   = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._limiter  = RateLimiter(max_calls=max_calls, period=period)

        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Token": api_key,
            "Accept":       "application/json",
        })

        # Wrap the internal _get at instantiation time so each client
        # instance has its own independent rate-limit counter.
        self._get = self._limiter(self.__get_raw)

    # ── Private ───────────────────────────────────────────────────────────────

    def __get_raw(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Execute one HTTP GET; return parsed JSON or {} on error."""
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            msgs = {
                429: "429 Too Many Requests — rate limiter needs tightening",
                 403: "403 Forbidden — check API key and subscription tier",
                404: f"404 Not Found — endpoint '{endpoint}' may be invalid",
            }
            logger.error(msgs.get(code, f"HTTP {code} on {endpoint}: {exc}"))
        except requests.RequestException as exc:
            logger.error("Network error on %s: %s", endpoint, exc)
        return {}

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_team_matches(self, team_id: int, limit: int = 10) -> list[dict]:
        """
        Return up to `limit` most-recent FINISHED matches for a team.
        Used by models.py to fit per-team Poisson attack/defence rates.
        """
        data = self._get(
            f"/teams/{team_id}/matches",
            params={"status": "FINISHED", "limit": limit},
        )
        return data.get("matches", [])

    def get_upcoming_matches(
        self,
        competition_code: str,
        days_ahead: int = 7,
    ) -> list[dict]:
        """
        Return SCHEDULED matches for a competition within the next `days_ahead`
        days.  The API embeds 1X2 odds in the response when available.
        """
        from datetime import datetime, timedelta, timezone

        today    = datetime.now(timezone.utc)
        date_to  = today + timedelta(days=days_ahead)
        data = self._get(
            f"/competitions/{competition_code}/matches",
            params={
                "status":   "SCHEDULED",
                "dateFrom": today.strftime("%Y-%m-%d"),
                "dateTo":   date_to.strftime("%Y-%m-%d"),
            },
        )
        return data.get("matches", [])

    def get_competition_teams(self, competition_code: str) -> list[dict]:
        """Return all teams currently in a competition (used for pre-caching)."""
        data = self._get(f"/competitions/{competition_code}/teams")
        return data.get("teams", [])
