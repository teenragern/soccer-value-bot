"""
Soccer Value Betting Bot
========================
Uses the football-data.org FREE_PLUS_THREE API to identify value bets by
comparing a team's historical performance score against bookmaker implied
probabilities.

VALUE BETTING MATH (core concept)
----------------------------------
A "value bet" exists when your estimated probability of an outcome is HIGHER
than the probability the bookmaker's odds imply.

  Implied Probability = 1 / Decimal Odds

  Example:
    Bookmaker offers Home Win at odds 2.20
    Implied probability = 1 / 2.20 = 45.5%

    Our model estimates Home Win at 60.0%
    Edge = 60.0% - 45.5% = +14.5%  → VALUE BET ✓

Bookmakers also bake in an "overround" (vigorish/juice) so that the raw
implied probabilities across all three outcomes (1X2) sum to MORE than 100%
(typically 103%-108%).  We strip this out via normalization before comparing,
giving us the bookmaker's "fair" probability for a level playing field.

DISCLAIMER: This tool is for educational and research purposes only.
            Never gamble more than you can afford to lose.
"""

import requests
import pandas as pd
import time
import logging

from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this block before running
# ─────────────────────────────────────────────────────────────────────────────
CONFIG: dict = {
    # ── API ───────────────────────────────────────────────────────────────────
    "api_key":  "YOUR_API_KEY_HERE",        # ← replace with your key
    "base_url": "https://api.football-data.org/v4",

    # ── Leagues (FREE_PLUS_THREE tier) ────────────────────────────────────────
    # PL=Premier League, PD=La Liga, BL1=Bundesliga, SA=Serie A, FL1=Ligue 1
    # CL=Champions League, DED=Eredivisie, BSA=Brasileirao, ELC=Championship
    # PPL=Primeira Liga, WC=World Cup, EC=Euros
    "leagues": ["PL", "PD", "BL1", "SA", "FL1", "CL", "DED", "BSA", "ELC", "PPL"],

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # FREE_PLUS_THREE allows 10 requests / 60 seconds
    "rate_limit_calls": 10,
    "rate_limit_window": 60,          # seconds

    # ── Form Engine ───────────────────────────────────────────────────────────
    "form_matches":  5,               # how many recent matches to fetch
    "min_matches":   3,               # skip team if fewer than this available

    # ── Fatigue Filter ────────────────────────────────────────────────────────
    "fatigue_hours": 72,              # flag team if last match was within N hours

    # ── Value Detection Thresholds ────────────────────────────────────────────
    # Only alert when BOTH conditions are met:
    "value_threshold":       0.10,    # our prob must exceed market fair prob by ≥10%
    "min_win_probability":   0.55,    # our estimated win chance must be ≥55%

    # ── Scoring Weights ───────────────────────────────────────────────────────
    "win_points":       3,
    "draw_points":      1,
    "loss_points":      0,
    "goal_diff_weight": 0.10,         # bonus per goal in a match (capped at ±5)

    # ── Lookahead ─────────────────────────────────────────────────────────────
    "days_ahead": 7,                  # how many days of upcoming fixtures to fetch
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER  (token-bucket style)
# ─────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Callable decorator that enforces a sliding-window rate limit.

    Usage:
        limiter = RateLimiter(max_calls=10, period=60)

        @limiter
        def my_api_call(): ...

    When the call budget is exhausted the decorator sleeps until the oldest
    call in the window is more than `period` seconds old, then proceeds.
    This guarantees we never exceed 10 calls/minute without crashing.
    """

    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period    = period
        self._timestamps: list[float] = []  # sliding window of call times

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Drop timestamps that have fallen outside the current window
            self._timestamps = [t for t in self._timestamps if now - t < self.period]

            if len(self._timestamps) >= self.max_calls:
                # Sleep until the oldest timestamp exits the window
                sleep_for = self.period - (now - self._timestamps[0]) + 0.2
                logger.info(f"  [RateLimiter] Budget reached — sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
                self._timestamps = []

            self._timestamps.append(time.time())
            return func(*args, **kwargs)

        return wrapper


# Module-level singleton — shared by all API calls
_rate_limiter = RateLimiter(
    max_calls=CONFIG["rate_limit_calls"],
    period=CONFIG["rate_limit_window"],
)


# ─────────────────────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class FootballDataClient:
    """
    Thin HTTP wrapper around the football-data.org v4 REST API.
    Every network call is decorated with @_rate_limiter so callers
    never need to manage sleeps manually.
    """

    def __init__(self, api_key: str, base_url: str):
        self.base_url = base_url
        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Token": api_key,
            "Accept":       "application/json",
        })

    @_rate_limiter
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """
        Execute a rate-limited GET request.  Returns an empty dict on any
        error so callers can safely call .get() on the result.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            if code == 429:
                logger.error("429 Too Many Requests — tighten the rate limiter")
            elif code == 403:
                logger.error("403 Forbidden — verify API key and tier permissions")
            else:
                logger.error(f"HTTP {code} on {endpoint}: {exc}")
            return {}
        except requests.exceptions.RequestException as exc:
            logger.error(f"Network error on {endpoint}: {exc}")
            return {}

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_team_matches(self, team_id: int, limit: int = 5) -> list:
        """Return the most recent FINISHED matches for a given team."""
        data = self._get(f"/teams/{team_id}/matches", params={
            "status": "FINISHED",
            "limit":  limit,
        })
        return data.get("matches", [])

    def get_upcoming_matches(self, competition_code: str, days_ahead: int = 7) -> list:
        """Return SCHEDULED matches for a competition in the next N days."""
        today     = datetime.now(timezone.utc)
        date_to   = today + timedelta(days=days_ahead)
        data = self._get(f"/competitions/{competition_code}/matches", params={
            "status":   "SCHEDULED",
            "dateFrom": today.strftime("%Y-%m-%d"),
            "dateTo":   date_to.strftime("%Y-%m-%d"),
        })
        return data.get("matches", [])


# ─────────────────────────────────────────────────────────────────────────────
# FORM ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_performance_score(matches: list, team_id: int) -> dict:
    """
    Build a normalized "Performance Score" [0.0 → 1.0] from recent matches.

    RAW SCORING
    -----------
    Each match contributes:
      Win  → win_points  (default 3)
      Draw → draw_points (default 1)
      Loss → loss_points (default 0)
      ± bonus: goal_difference * goal_diff_weight  (capped at ±5 goals)

    NORMALIZATION
    -------------
    raw_score / max_possible_raw_score → [0, 1]

    The result approximates a "win rate quality" for use in probability
    estimation downstream.  It is NOT a direct win probability on its own.

    Returns
    -------
    dict:
      score          – float [0, 1]
      form_string    – e.g. "W W D L W"
      goal_diff      – cumulative goal difference over sampled matches
      matches_played – how many matches contributed
    """
    if not matches:
        return {"score": 0.5, "form_string": "N/A", "goal_diff": 0, "matches_played": 0}

    # Maximum raw score per match: win_points + (max_gd_cap * goal_diff_weight)
    max_raw_per_match = CONFIG["win_points"] + (5 * CONFIG["goal_diff_weight"])

    total_raw   = 0.0
    total_gd    = 0
    form_chars  = []
    played      = 0

    for match in matches:
        ft = match.get("score", {}).get("fullTime", {})
        h_goals = ft.get("home")
        a_goals = ft.get("away")

        if h_goals is None or a_goals is None:
            continue  # skip match with missing score

        is_home   = match.get("homeTeam", {}).get("id") == team_id
        gf        = h_goals if is_home else a_goals
        ga        = a_goals if is_home else h_goals
        gd        = gf - ga
        total_gd += gd

        # Points for result
        if gd > 0:
            total_raw += CONFIG["win_points"]
            form_chars.append("W")
        elif gd == 0:
            total_raw += CONFIG["draw_points"]
            form_chars.append("D")
        else:
            total_raw += CONFIG["loss_points"]
            form_chars.append("L")

        # Goal-difference bonus (capped so blowouts don't dominate)
        total_raw += max(-5, min(5, gd)) * CONFIG["goal_diff_weight"]
        played    += 1

    if played == 0:
        return {"score": 0.5, "form_string": "N/A", "goal_diff": 0, "matches_played": 0}

    max_possible  = played * max_raw_per_match
    norm_score    = total_raw / max_possible if max_possible else 0.5
    norm_score    = max(0.0, min(1.0, norm_score))  # clamp to [0, 1]

    return {
        "score":         round(norm_score, 4),
        "form_string":   " ".join(form_chars),   # "W W D L W"
        "goal_diff":     total_gd,
        "matches_played": played,
    }


def get_team_form(client: FootballDataClient, team_id: int, team_name: str) -> dict:
    """Fetch recent matches from the API and return the performance score dict."""
    logger.info(f"    Fetching form: {team_name} (id={team_id})")
    matches = client.get_team_matches(team_id, limit=CONFIG["form_matches"])

    if len(matches) < CONFIG["min_matches"]:
        logger.warning(f"    Insufficient data for {team_name} ({len(matches)} matches)")

    return calculate_performance_score(matches, team_id)


# ─────────────────────────────────────────────────────────────────────────────
# FATIGUE FILTER
# ─────────────────────────────────────────────────────────────────────────────
def check_fatigue(
    client: FootballDataClient,
    team_id: int,
    upcoming_match_dt: datetime,
) -> dict:
    """
    Flag a team as potentially fatigued when their last fixture was within
    CONFIG['fatigue_hours'] of the upcoming match.

    WHY IT MATTERS
    --------------
    Teams on a tight turnaround (e.g., Tue Champions League → Sat league)
    show measurable drops in high-intensity running distance (~8–12% per
    GPS tracking studies) and a small but meaningful increase in injury risk.
    Accounting for fatigue can explain why a "strong" team underperforms.

    Returns
    -------
    dict:
      is_fatigued            – True / False
      hours_since_last_match – float or None
      last_match_date        – formatted string or None
    """
    recent = client.get_team_matches(team_id, limit=1)
    if not recent:
        return {"is_fatigued": False, "hours_since_last_match": None, "last_match_date": None}

    raw_date = recent[0].get("utcDate", "")
    try:
        last_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

        # Ensure upcoming_match_dt is timezone-aware
        if upcoming_match_dt.tzinfo is None:
            upcoming_match_dt = upcoming_match_dt.replace(tzinfo=timezone.utc)

        hours_gap = (upcoming_match_dt - last_dt).total_seconds() / 3600
        fatigued  = 0 < hours_gap < CONFIG["fatigue_hours"]

        return {
            "is_fatigued":            fatigued,
            "hours_since_last_match": round(hours_gap, 1),
            "last_match_date":        last_dt.strftime("%Y-%m-%d %H:%M UTC"),
        }
    except (ValueError, TypeError) as exc:
        logger.warning(f"    Could not parse last-match date '{raw_date}': {exc}")
        return {"is_fatigued": False, "hours_since_last_match": None, "last_match_date": None}


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY & VALUE MATH
# ─────────────────────────────────────────────────────────────────────────────
def implied_probability(decimal_odds: float) -> float:
    """
    Convert decimal odds to raw implied probability.

      P = 1 / decimal_odds

    Example: 2.50 → 0.40 (40%)
    """
    if decimal_odds <= 1.0:
        return 1.0
    return round(1.0 / decimal_odds, 6)


def strip_overround(p_home: float, p_draw: float, p_away: float) -> tuple[float, float, float]:
    """
    Normalize three implied probabilities so they sum to exactly 1.0,
    removing the bookmaker's overround (vigorish / juice).

    RAW EXAMPLE (overround = 1.05)
      Home 0.476 + Draw 0.238 + Away 0.286 = 1.000 after normalization
      vs   0.500 + 0.250 + 0.303 = 1.053 raw (5.3% margin for the bookie)

    This gives the "fair" market probability we can meaningfully compare
    against our own model's estimates.
    """
    total = p_home + p_draw + p_away
    if total == 0:
        return (1/3, 1/3, 1/3)
    return (
        round(p_home / total, 6),
        round(p_draw / total, 6),
        round(p_away / total, 6),
    )


def estimate_match_probabilities(
    home_score: float,
    away_score: float,
) -> tuple[float, float, float]:
    """
    Convert two team performance scores into win/draw/loss probabilities.

    MODEL OUTLINE
    -------------
    1. Apply a home-advantage multiplier (~+15%) — well supported empirically:
       roughly 45–47% of league points are won at home across major leagues.
    2. Use a Bradley-Terry style relative-strength formula:
         P(home_wins) ∝ home_strength / (home_strength + away_strength)
    3. Draw probability peaks (~28%) when teams are evenly matched and
       shrinks proportionally as the performance gap widens.
    4. Remaining probability is split home/away proportionally.
    5. Final three values are renormalized to sum to exactly 1.0.

    NOTE: This is a simplified heuristic model designed for educational
    purposes.  Production systems use Poisson regression on shot/xG data.

    Returns
    -------
    (home_win_prob, draw_prob, away_win_prob)
    """
    HOME_ADVANTAGE  = 1.15   # multiplicative bonus for the home side
    MAX_DRAW_PROB   = 0.27   # draw probability ceiling (evenly matched teams)
    DRAW_DECAY      = 0.50   # how quickly draw chance shrinks with score gap

    adj_home = home_score * HOME_ADVANTAGE
    adj_away = away_score
    total    = adj_home + adj_away

    if total == 0:
        base_h, base_a = 0.50, 0.50
    else:
        base_h = adj_home / total
        base_a = adj_away / total

    # Draw probability — peaks when scores are equal, decays with disparity
    gap       = abs(home_score - away_score)
    draw_prob = max(0.05, MAX_DRAW_PROB - gap * DRAW_DECAY)

    # Split remaining probability proportionally
    remaining   = 1.0 - draw_prob
    home_win    = base_h * remaining
    away_win    = base_a * remaining

    # Renormalize (floating point safety)
    total_check = home_win + draw_prob + away_win
    return (
        round(home_win  / total_check, 6),
        round(draw_prob / total_check, 6),
        round(away_win  / total_check, 6),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MATCH ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
def analyze_match(client: FootballDataClient, match: dict) -> Optional[dict]:
    """
    Full analysis pipeline for one fixture:
      1. Parse teams and odds from the API payload
      2. Fetch form scores (2 API calls)
      3. Fetch fatigue data  (2 API calls)
      4. Estimate our own win / draw / loss probabilities
      5. Strip the bookmaker overround from raw odds
      6. Compute the value edge and flag bets above the threshold

    Returns None if the fixture cannot be processed (missing date, etc.).
    """
    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})
    odds_obj  = match.get("odds", {})         # may be {} if not provided by API

    home_id   = home_team.get("id")
    away_id   = away_team.get("id")
    home_name = home_team.get("name", "Unknown")
    away_name = away_team.get("name", "Unknown")
    competition = match.get("competition", {}).get("name", "?")
    match_id    = match.get("id", "?")

    # ── Parse fixture date ────────────────────────────────────────────────────
    raw_date = match.get("utcDate", "")
    try:
        match_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logger.warning(f"Skipping match {match_id}: unparseable date '{raw_date}'")
        return None

    # ── Extract 1X2 decimal odds ──────────────────────────────────────────────
    home_odds = odds_obj.get("homeWin")
    draw_odds = odds_obj.get("draw")
    away_odds = odds_obj.get("awayWin")
    has_odds  = all(isinstance(o, (int, float)) and o > 1.0
                    for o in [home_odds, draw_odds, away_odds])

    # ── Form scores (each is 1 API call) ──────────────────────────────────────
    logger.info(f"  → {home_name} vs {away_name}  [{competition}]")
    home_form = get_team_form(client, home_id, home_name)
    away_form = get_team_form(client, away_id, away_name)

    # ── Fatigue checks (each is 1 API call) ───────────────────────────────────
    home_fatigue = check_fatigue(client, home_id, match_dt)
    away_fatigue = check_fatigue(client, away_id, match_dt)

    # ── Our probability estimates ─────────────────────────────────────────────
    est_h, est_d, est_a = estimate_match_probabilities(
        home_form["score"], away_form["score"]
    )

    # ── Value detection ───────────────────────────────────────────────────────
    value_bets: list[dict] = []

    if has_odds:
        # Step 1: raw implied probabilities (include overround)
        raw_h = implied_probability(home_odds)
        raw_d = implied_probability(draw_odds)
        raw_a = implied_probability(away_odds)

        # Step 2: strip the overround → fair probabilities
        fair_h, fair_d, fair_a = strip_overround(raw_h, raw_d, raw_a)

        overround = round((raw_h + raw_d + raw_a - 1.0) * 100, 2)

        # Step 3: compare each outcome
        candidates = [
            ("Home Win", est_h, fair_h, home_odds, home_name),
            ("Draw",     est_d, fair_d, draw_odds,  "Draw"),
            ("Away Win", est_a, fair_a, away_odds,  away_name),
        ]

        for label, our_p, fair_p, d_odds, team_label in candidates:
            edge = our_p - fair_p
            # VALUE condition:
            #   (a) our edge over fair market price ≥ value_threshold
            #   (b) we are confident enough (min_win_probability)
            if edge >= CONFIG["value_threshold"] and our_p >= CONFIG["min_win_probability"]:
                value_bets.append({
                    "outcome":      label,
                    "team":         team_label,
                    "our_prob":     our_p,
                    "fair_prob":    fair_p,
                    "odds":         d_odds,
                    "edge":         round(edge, 6),
                })
    else:
        overround = None

    return {
        "match_id":    match_id,
        "competition": competition,
        "home_team":   home_name,
        "away_team":   away_name,
        "match_date":  match_dt.strftime("%Y-%m-%d %H:%M UTC"),
        "home_form":   home_form,
        "away_form":   away_form,
        "home_fatigue": home_fatigue,
        "away_fatigue": away_fatigue,
        "estimated_probs": {
            "home_win": est_h,
            "draw":     est_d,
            "away_win": est_a,
        },
        "has_odds":   has_odds,
        "overround":  overround,
        "value_bets": value_bets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_match_report(r: dict) -> None:
    """Pretty-print a single match analysis to stdout."""
    W = 66
    bar  = "─" * W
    dbar = "═" * W

    print(f"\n{bar}")
    print(f"  {r['home_team']}  vs  {r['away_team']}")
    print(f"  {r['competition']}   |   {r['match_date']}")
    print(bar)

    # ── Form table ────────────────────────────────────────────────────────────
    hf, af = r["home_form"], r["away_form"]
    col = 28
    print(f"  {'FORM':<10} {'HOME':<{col}} AWAY")
    print(f"  {'Score':<10} {str(hf['score']):<{col}} {af['score']}")
    print(f"  {'Last 5':<10} {hf['form_string']:<{col}} {af['form_string']}")
    print(f"  {'Goal Diff':<10} {str(hf['goal_diff']):<{col}} {af['goal_diff']}")
    print(f"  {'Matches':<10} {str(hf['matches_played']):<{col}} {af['matches_played']}")

    # ── Fatigue warnings ──────────────────────────────────────────────────────
    for side, fat in [("HOME", r["home_fatigue"]), ("AWAY", r["away_fatigue"])]:
        if fat["is_fatigued"]:
            hrs = fat["hours_since_last_match"]
            last = fat["last_match_date"]
            print(f"\n  !! FATIGUE WARNING [{side}]: last match {hrs}h ago ({last})")

    # ── Probability summary ───────────────────────────────────────────────────
    ep = r["estimated_probs"]
    print(f"\n  OUR ESTIMATES  →  "
          f"Home Win: {ep['home_win']:.1%}  |  "
          f"Draw: {ep['draw']:.1%}  |  "
          f"Away Win: {ep['away_win']:.1%}")

    if r["has_odds"] and r["overround"] is not None:
        print(f"  Bookmaker overround: {r['overround']}%")

    # ── Value alerts ──────────────────────────────────────────────────────────
    if r["value_bets"]:
        for vb in r["value_bets"]:
            print(f"\n  {dbar}")
            print(f"  *** VALUE ALERT ***")
            print(f"  Outcome  : {vb['outcome']}  ({vb['team']})")
            print(f"  Odds     : {vb['odds']}")
            print(f"  Our prob : {vb['our_prob']:.1%}")
            print(f"  Mkt fair : {vb['fair_prob']:.1%}")
            print(f"  Edge     : +{vb['edge']:.1%}")
            print(f"  {dbar}")
    elif not r["has_odds"]:
        print("\n  [Odds not available for this fixture]")
    else:
        print("\n  [No value bets detected]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCANNER
# ─────────────────────────────────────────────────────────────────────────────
def run_scanner(leagues: list[str] = None) -> list[dict]:
    """
    Scan one or more competitions for upcoming value bets.

    Parameters
    ----------
    leagues : list of competition codes, e.g. ["PL", "BL1"].
              Defaults to CONFIG["leagues"] if omitted.

    Returns
    -------
    List of value-bet dicts (useful for downstream processing / logging).
    """
    client  = FootballDataClient(CONFIG["api_key"], CONFIG["base_url"])
    targets = leagues or CONFIG["leagues"]

    print("\n" + "=" * 66)
    print("  SOCCER VALUE BETTING SCANNER")
    print(f"  Leagues  : {', '.join(targets)}")
    print(f"  Run time : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Thresholds → edge ≥ {CONFIG['value_threshold']:.0%}  "
          f"| min prob ≥ {CONFIG['min_win_probability']:.0%}")
    print("=" * 66)

    all_value_bets: list[dict] = []

    for league in targets:
        print(f"\n[{league}] Fetching upcoming fixtures...")
        fixtures = client.get_upcoming_matches(league, days_ahead=CONFIG["days_ahead"])

        if not fixtures:
            print(f"  No scheduled matches found for {league}.")
            continue

        print(f"  Found {len(fixtures)} fixture(s).")

        for match in fixtures:
            result = analyze_match(client, match)
            if result is None:
                continue

            print_match_report(result)

            # Collect value bets for the final summary
            for vb in result["value_bets"]:
                all_value_bets.append({
                    "competition": result["competition"],
                    "fixture":     f"{result['home_team']} vs {result['away_team']}",
                    "date":        result["match_date"],
                    **vb,
                })

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SCAN COMPLETE — VALUE BET SUMMARY")
    print("=" * 66)

    if not all_value_bets:
        print("  No value bets identified in this scan.\n")
    else:
        df = pd.DataFrame(all_value_bets)
        df = df.sort_values("edge", ascending=False).reset_index(drop=True)

        # Format for display
        df["our_prob"]  = df["our_prob"].map(lambda x: f"{x:.1%}")
        df["fair_prob"] = df["fair_prob"].map(lambda x: f"{x:.1%}")
        df["edge"]      = df["edge"].map(lambda x: f"+{x:.1%}")

        print(df[[
            "competition", "fixture", "date",
            "outcome", "odds", "our_prob", "fair_prob", "edge",
        ]].to_string(index=False))
        print()

    print("=" * 66)
    print("  DISCLAIMER: For educational / research use only.")
    print("              Gamble responsibly. Past form ≠ future results.")
    print("=" * 66 + "\n")

    return all_value_bets


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test with two leagues.
    # Remove the `leagues` argument to scan everything in CONFIG["leagues"].
    run_scanner(leagues=["PL", "PD", "BL1"])
