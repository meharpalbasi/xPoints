# xPoints

Expected points predictor for Fantasy Premier League using XGBoost machine learning.

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
3. **Model Training** — Trains an XGBoost regressor on historical data using time-series cross-validation
4. **Prediction** — Generates expected points for the next gameweek

## Automation

Predictions update daily via GitHub Actions:

- **`daily_update.yml`** — Runs at 16:30 UTC, generates `predictions.json`
- **`daily_metrics.yml`** — Generates extended metrics file

## Scripts

| Script | Purpose |
|--------|---------|
| `script.py` | Main prediction pipeline (fetch → train → predict → output) |
| `script2.py` | Extended metrics generation with additional feature engineering |

## Usage

```bash
# Install dependencies
pip install requests pandas numpy tqdm scikit-learn xgboost

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
