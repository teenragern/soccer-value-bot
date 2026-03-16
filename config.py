"""
config.py — Central configuration for the Soccer Value Betting Bot.
Edit this file before running the scanner.
"""

CONFIG: dict = {
    # ── API ───────────────────────────────────────────────────────────────────
    "api_key":  "YOUR_API_KEY_HERE",        # ← paste your football-data.org key
    "base_url": "https://api.football-data.org/v4",

    # ── FREE_PLUS_THREE league codes ──────────────────────────────────────────
    # WC  = FIFA World Cup          CL  = UEFA Champions League
    # BL1 = Bundesliga              DED = Eredivisie
    # BSA = Brasileirao Série A     PD  = La Liga (Primera Division)
    # FL1 = Ligue 1                 ELC = Championship
    # PPL = Primeira Liga           EC  = UEFA European Championship
    # SA  = Serie A                 PL  = Premier League
    "leagues": ["PL", "PD", "BL1", "SA", "FL1", "CL", "DED", "BSA", "ELC", "PPL"],

    # ── Rate limiting ─────────────────────────────────────────────────────────
    "rate_limit_calls":  10,   # max requests allowed…
    "rate_limit_window": 60,   # …within this many seconds

    # ── Fixture window ────────────────────────────────────────────────────────
    "days_ahead": 7,           # how far ahead to look for upcoming matches

    # ── Form Engine ───────────────────────────────────────────────────────────
    "form_matches": 10,        # recent matches fetched for attack/defence stats
    "min_matches":   5,        # minimum to trust a team's Poisson parameters

    # ── Fatigue Filter ────────────────────────────────────────────────────────
    "fatigue_hours": 72,       # flag team if last match was within N hours

    # ── Value Detection ───────────────────────────────────────────────────────
    # A "value bet" fires when BOTH conditions are met:
    "value_threshold":     0.08,   # our prob must exceed fair market prob by ≥ 8 pp
    "min_win_probability": 0.50,   # we must estimate at least 50% chance of winning

    # ── Poisson model ─────────────────────────────────────────────────────────
    "home_advantage":  1.15,   # multiplicative boost to home attack strength
    "poisson_max_goals": 10,   # upper bound for goal grid (10 × 10 matrix)
    # League-wide averages used as fallback when a team has < min_matches
    "league_avg_scored":    1.4,
    "league_avg_conceded":  1.4,
}
