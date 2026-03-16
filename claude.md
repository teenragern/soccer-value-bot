# Soccer Betting Bot Project Specs

## Tech Stack
- Python 3.x
- API: football-data.org (FREE_PLUS_THREE tier)
- Competitions: PL, BL1, SA, PD, FL1, CL, ELC, DED, PPL, BSA
- Tools: pandas, requests, scipy (for Poisson)

## API Constraints
- Rate Limit: 10 requests per minute. Use a wrapper to handle 429 errors.
- Authentication: Use the X-Auth-Token header.

## Core Logic Requirements
1. **The Engine**: Compare 1X2 market odds against a 5-match form-based "Expected Win %".
2. **Poisson Model**: Implement a Poisson Distribution to predict exact score probabilities based on Attacking/Defensive strength from league standings.
3. **Value Threshold**: Only flag "Value Bets" where (Your Probability * Decimal Odds) > 1.15 (15% edge).
4. **Fatigue Filter**: Reduce the expected strength of any team that played a match < 72 hours ago.

## Antigravity Integration
- If a UI is needed, generate standard HTML/JS for the Antigravity Browser Agent to preview.
- All logs should be written to `bot_log.json` for Antigravity's parallel monitoring agents.