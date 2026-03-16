"""
src/models.py
=============
Probabilistic match modelling via the Dixon-Coles adjusted Poisson model.

WHY POISSON?
------------
In association football the number of goals scored by each side can be
modelled as independent Poisson random variables.  If team A scores at
an average rate of λ_A goals per match and team B at λ_B, then:

    P(A scores k goals) = e^(-λ_A) * λ_A^k / k!

The joint scoreline probability P(A=i, B=j) is the product of the two
independent Poisson PMFs — except for low-scoring draws (0-0, 1-1) which
occur *slightly* more often than the independence assumption predicts.
The Dixon-Coles (1997) ρ correction adjusts for this.

PARAMETER ESTIMATION
--------------------
For each team we fit two parameters from their last N matches:

    attack_rate  = (goals scored by team)    / N matches
    defence_rate = (goals conceded by team)  / N matches

The expected goals (xG proxy) for a fixture are then:

    λ_home = attack_home  × (1/defence_away) × league_avg_scored × home_adv
    λ_away = attack_away  × (1/defence_home) × league_avg_scored

OUTCOME PROBABILITIES
---------------------
We iterate over all scorelines (i, j) up to `max_goals` and sum:

    P(home win) = Σ P(i>j)
    P(draw)     = Σ P(i=j)
    P(away win) = Σ P(i<j)

VALUE DETECTION
---------------
After building the outcome probabilities we compare them against the
bookmaker's implied (overround-stripped) probabilities.

    Edge = P_model(outcome) − P_fair_market(outcome)

If edge ≥ value_threshold AND P_model ≥ min_win_probability → VALUE ALERT.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamStats:
    """Attack and defence parameters estimated from recent matches."""
    team_id:      int
    team_name:    str
    attack_rate:  float          # avg goals scored per match
    defence_rate: float          # avg goals conceded per match
    matches_used: int
    form_string:  str = ""       # e.g. "W W D L W W"
    goal_diff:    int = 0


@dataclass
class FatigueInfo:
    """Result of the fatigue check for one team."""
    is_fatigued:            bool
    hours_since_last_match: float | None = None
    last_match_date:        str   | None = None


@dataclass
class ValueBet:
    """A single detected value-bet opportunity."""
    outcome:   str           # "Home Win" | "Draw" | "Away Win"
    team:      str
    our_prob:  float         # model probability
    fair_prob: float         # overround-stripped bookmaker probability
    odds:      float         # raw decimal odds
    edge:      float         # our_prob − fair_prob


@dataclass
class MatchAnalysis:
    """Full analysis output for one fixture."""
    match_id:    int | str
    competition: str
    home_team:   str
    away_team:   str
    match_date:  str

    home_stats:   TeamStats | None
    away_stats:   TeamStats | None
    home_fatigue: FatigueInfo
    away_fatigue: FatigueInfo

    lambda_home: float = 0.0     # expected goals — home side
    lambda_away: float = 0.0     # expected goals — away side

    prob_home_win: float = 0.0
    prob_draw:     float = 0.0
    prob_away_win: float = 0.0

    has_odds:   bool  = False
    overround:  float = 0.0      # bookmaker margin as a percentage

    value_bets: list[ValueBet] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# TEAM STATISTICS  (Poisson parameter estimation)
# ─────────────────────────────────────────────────────────────────────────────

def build_team_stats(
    matches:   list[dict],
    team_id:   int,
    team_name: str,
    league_avg_scored:   float = 1.4,
    league_avg_conceded: float = 1.4,
    min_matches: int = 5,
) -> TeamStats:
    """
    Estimate a team's Poisson attack and defence rates from match history.

    For each finished match we record:
        goals_for     (from the team's perspective)
        goals_against

    We then compute:
        attack_rate  = mean(goals_for)
        defence_rate = mean(goals_against)

    If the team has fewer than `min_matches` of data we fall back to
    league-average values to avoid fitting on noise.

    Parameters
    ----------
    matches              : List of match dicts from the API.
    team_id              : The team's numerical ID.
    team_name            : Human-readable name for logging.
    league_avg_scored    : Fallback when sample is too small.
    league_avg_conceded  : Fallback when sample is too small.
    min_matches          : Minimum sample size for reliable estimates.
    """
    goals_for: list[int]     = []
    goals_against: list[int] = []
    form_chars: list[str]    = []
    total_gd = 0

    for match in matches:
        ft = match.get("score", {}).get("fullTime", {})
        h  = ft.get("home")
        a  = ft.get("away")

        if h is None or a is None:
            continue   # skip match with missing score

        is_home = match.get("homeTeam", {}).get("id") == team_id
        gf = h if is_home else a
        ga = a if is_home else h
        gd = gf - ga

        goals_for.append(gf)
        goals_against.append(ga)
        total_gd += gd

        form_chars.append("W" if gd > 0 else ("D" if gd == 0 else "L"))

    n = len(goals_for)

    if n < min_matches:
        logger.warning(
            "    %s: only %d matches — using league-average Poisson parameters", team_name, n
        )
        attack  = league_avg_scored
        defence = league_avg_conceded
    else:
        attack  = sum(goals_for)     / n
        defence = sum(goals_against) / n

    return TeamStats(
        team_id      = team_id,
        team_name    = team_name,
        attack_rate  = round(attack,  4),
        defence_rate = round(defence, 4),
        matches_used = n,
        form_string  = " ".join(form_chars),
        goal_diff    = total_gd,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FATIGUE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_fatigue(
    last_match: dict | None,
    upcoming_dt: datetime,
    fatigue_hours: int = 72,
) -> FatigueInfo:
    """
    Return a FatigueInfo for one team given their most recent match dict.

    A team is flagged as fatigued when:
        0 < (upcoming_match_dt − last_match_dt).total_hours < fatigue_hours

    This catches short turnarounds such as:
        Tuesday Champions League → Saturday Premier League  (~84 h, borderline)
        Thursday Europa League   → Sunday league            (~60 h, flagged)

    Parameters
    ----------
    last_match   : The most-recent FINISHED match dict from the API, or None.
    upcoming_dt  : The datetime of the fixture being analysed.
    fatigue_hours: Flag threshold in hours.
    """
    if not last_match:
        return FatigueInfo(is_fatigued=False)

    raw = last_match.get("utcDate", "")
    try:
        last_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if upcoming_dt.tzinfo is None:
            upcoming_dt = upcoming_dt.replace(tzinfo=timezone.utc)

        hours_gap = (upcoming_dt - last_dt).total_seconds() / 3600
        fatigued  = 0 < hours_gap < fatigue_hours

        return FatigueInfo(
            is_fatigued            = fatigued,
            hours_since_last_match = round(hours_gap, 1),
            last_match_date        = last_dt.strftime("%Y-%m-%d %H:%M UTC"),
        )
    except (ValueError, TypeError) as exc:
        logger.warning("    Could not parse last-match date '%s': %s", raw, exc)
        return FatigueInfo(is_fatigued=False)


# ─────────────────────────────────────────────────────────────────────────────
# POISSON GOAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_pmf(lam: float, k: int) -> float:
    """P(X = k) for X ~ Poisson(λ) computed in log-space for numerical safety."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # log P = k * ln(λ) - λ - ln(k!)
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _dc_correction(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon-Coles (1997) low-score correction factor τ(i, j, λ_h, λ_a, ρ).

    The independent Poisson model under-predicts 0-0 and 1-1 draws and
    over-predicts 1-0 and 0-1 scorelines.  The τ factor re-weights these
    four cells to match empirical frequencies:

        τ(0,0) = 1 − λ_h * λ_a * ρ
        τ(1,0) = 1 + λ_a * ρ
        τ(0,1) = 1 + λ_h * ρ
        τ(1,1) = 1 − ρ
        τ(i,j) = 1  otherwise

    ρ (rho) is a small negative constant typically fitted at ~ −0.13 on
    European league data.  We use a fixed conservative value here.
    """
    RHO = -0.13   # industry standard default; tune on historical data for accuracy

    if i == 0 and j == 0:
        return 1 - lam_h * lam_a * RHO
    if i == 1 and j == 0:
        return 1 + lam_a * RHO
    if i == 0 and j == 1:
        return 1 + lam_h * RHO
    if i == 1 and j == 1:
        return 1 - RHO
    return 1.0


def compute_scoreline_matrix(
    lam_home: float,
    lam_away: float,
    max_goals: int = 10,
    use_dc_correction: bool = True,
) -> list[list[float]]:
    """
    Build an (max_goals+1) × (max_goals+1) matrix of scoreline probabilities.

    matrix[i][j] = P(home scores i goals, away scores j goals)

    The Dixon-Coles correction is applied to the four low-score cells.
    All cells are renormalized to sum to 1.0 after the correction.

    Parameters
    ----------
    lam_home           : Expected goals for the home side (λ_home).
    lam_away           : Expected goals for the away side (λ_away).
    max_goals          : Truncation point.  Goals > max_goals are ignored.
    use_dc_correction  : Whether to apply the Dixon-Coles ρ correction.
    """
    N = max_goals + 1
    matrix: list[list[float]] = [[0.0] * N for _ in range(N)]

    for i in range(N):         # home goals
        for j in range(N):     # away goals
            p = _poisson_pmf(lam_home, i) * _poisson_pmf(lam_away, j)
            if use_dc_correction:
                p *= _dc_correction(i, j, lam_home, lam_away, rho=-0.13)
            matrix[i][j] = p

    # Renormalize (DC correction slightly perturbs the total)
    total = sum(matrix[i][j] for i in range(N) for j in range(N))
    if total > 0:
        for i in range(N):
            for j in range(N):
                matrix[i][j] /= total

    return matrix


def outcome_probabilities(matrix: list[list[float]]) -> tuple[float, float, float]:
    """
    Aggregate a scoreline matrix into 1X2 outcome probabilities.

    Returns
    -------
    (P_home_win, P_draw, P_away_win)
    """
    p_home = p_draw = p_away = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    return round(p_home, 6), round(p_draw, 6), round(p_away, 6)


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED GOALS  (λ calculation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda(
    attack:      float,
    defence_opp: float,
    league_avg:  float,
    home_adv:    float = 1.0,
) -> float:
    """
    Compute the expected goals λ for one side of a fixture.

    Formula (Dixon-Coles attack/defence parameterisation):

        λ = attack_rate × (1 / defence_opp) × league_avg × home_advantage

    Intuition:
        A strong attack (high attack_rate) against a weak defence
        (high defence_opp → low 1/defence_opp when defence_opp > 1) scores more.

    Parameters
    ----------
    attack       : Team's average goals-scored per match.
    defence_opp  : Opponent's average goals-conceded per match.
    league_avg   : League-wide average goals per match (scaling constant).
    home_adv     : Multiplicative home-advantage factor (1.15 for home, 1.0 for away).
    """
    if defence_opp <= 0 or league_avg <= 0:
        return attack * home_adv   # safe fallback

    # The (1 / defence_opp) term measures how "leaky" the opposition defence is.
    # Multiplying by league_avg keeps λ on a realistic goals-per-game scale.
    return round(attack * (1.0 / defence_opp) * league_avg * home_adv, 4)


# ─────────────────────────────────────────────────────────────────────────────
# ODDS / VALUE MATH
# ─────────────────────────────────────────────────────────────────────────────

def implied_probability(decimal_odds: float) -> float:
    """
    Raw implied probability from decimal odds.

        P = 1 / decimal_odds

    Example: 2.50 → 0.40 (40%).
    The raw value includes the bookmaker overround and should be normalized
    before being compared against model probabilities.
    """
    return round(1.0 / decimal_odds, 6) if decimal_odds > 1.0 else 1.0


def strip_overround(
    p_home: float, p_draw: float, p_away: float
) -> tuple[float, float, float]:
    """
    Normalize three implied probabilities to sum to 1.0, removing the
    bookmaker's overround (also called the "vig" or "juice").

    EXAMPLE
    -------
    Raw implied probs:
        Home 0.500 + Draw 0.250 + Away 0.303 = 1.053  (5.3% margin)

    After normalization:
        Home 0.475 + Draw 0.237 + Away 0.288 = 1.000  (fair prices)

    These fair prices are what we compare against our Poisson estimates.
    """
    total = p_home + p_draw + p_away
    if total == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (
        round(p_home / total, 6),
        round(p_draw / total, 6),
        round(p_away / total, 6),
    )


def detect_value_bets(
    our_probs:    tuple[float, float, float],   # (home, draw, away) from model
    fair_probs:   tuple[float, float, float],   # overround-stripped market probs
    odds:         tuple[float, float, float],   # (home_odds, draw_odds, away_odds)
    team_names:   tuple[str, str],              # (home_name, away_name)
    value_threshold:     float = 0.08,
    min_win_probability: float = 0.50,
) -> list[ValueBet]:
    """
    Compare model probabilities against fair market probabilities and return
    a list of ValueBet objects that exceed both detection thresholds.

    VALUE CONDITION
    ---------------
    A bet has value when:
        (1)  edge = our_prob − fair_prob  ≥  value_threshold
        (2)  our_prob                     ≥  min_win_probability

    Condition (1) ensures the edge is large enough to overcome variance.
    Condition (2) avoids flagging low-confidence long-shots.

    Parameters
    ----------
    our_probs            : (P_home_win, P_draw, P_away_win) from the Poisson model.
    fair_probs           : Overround-stripped market probabilities.
    odds                 : Raw decimal odds (home, draw, away).
    team_names           : (home_team_name, away_team_name).
    value_threshold      : Minimum edge in probability points.
    min_win_probability  : Minimum confidence required to flag a bet.
    """
    labels = ["Home Win", "Draw", "Away Win"]
    teams  = [team_names[0], "Draw", team_names[1]]

    results: list[ValueBet] = []
    for label, team, our_p, fair_p, d_odds in zip(
        labels, teams, our_probs, fair_probs, odds
    ):
        edge = our_p - fair_p
        if edge >= value_threshold and our_p >= min_win_probability:
            results.append(
                ValueBet(
                    outcome  = label,
                    team     = team,
                    our_prob = round(our_p,   4),
                    fair_prob= round(fair_p,  4),
                    odds     = d_odds,
                    edge     = round(edge,    4),
                )
            )

    return results
