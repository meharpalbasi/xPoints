# xPoints

Expected-points pipeline for Fantasy Premier League with a validated official
FPL baseline and an experimental XGBoost candidate.

## Production status

The scheduled production feed currently runs in explicit **baseline mode** and
publishes official FPL `ep_next` values with source/version provenance. The
XGBoost path is not promoted automatically: it can be requested manually, but
must pass player-parity, fixture, provenance, validity, and non-zero output
gates or the workflow publishes the baseline instead.

## What It Does

xPoints predicts how many FPL points each player will score in the upcoming gameweek(s). Predictions are based on:

- Historical performance (goals, assists, clean sheets, bonus points)
- Recent form (rolling averages over last 3-5 gameweeks)
- Fixture difficulty
- Minutes played trends
- Position-specific patterns

## Output Files

| File | Description |
|------|-------------|
| `predictions.json` | Player predictions for upcoming gameweeks (consumed by [fplanaly.st](https://fplanaly.st)) |
| `fpl_full_history_metrics.json` | Extended player metrics and historical data |

## How It Works

1. **Data Collection** — Fetches player data, fixtures, and per-gameweek history from the official FPL API
2. **Feature Engineering** — Creates rolling averages, form metrics, and fixture difficulty ratings
3. **Candidate Training** — Trains an experimental XGBoost regressor on current-season history
4. **Validation** — Blocks incomplete, stale-schema, invalid, or all-zero candidates
5. **Publication** — Atomically publishes the requested source, falling back to `ep_next` when a model candidate fails

## Automation

Predictions update daily via GitHub Actions:

- **`daily_update.yml`** — Runs at 16:30 UTC in baseline mode and generates a validated `predictions.json`
- **`daily_metrics.yml`** — Generates extended metrics file

## Scripts

| Script | Purpose |
|--------|---------|
| `script.py` | Main prediction pipeline (fetch → train → predict → output) |
| `baseline.py` | Production `ep_next` baseline and immutable GW snapshot writer |
| `prediction_safety.py` | Shared fixture derivation, output validation, and atomic publication |
| `script2.py` | Extended metrics generation with additional feature engineering |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run predictions
python script.py

# Run extended metrics
python script2.py
```

## Integration

The `predictions.json` output is consumed by:
- [fplanaly.st](https://fplanaly.st) — FPL analytics web app
- [FPL Analyst](https://github.com/meharpalbasi/fpl-analyst-mobile) — iOS companion app

## Data Source

All data comes from the official Fantasy Premier League API at `fantasy.premierleague.com`.

---

Built by [Meharpal Basi](https://meharpalbasi.com)
