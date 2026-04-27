# xPoints v2 — Rebuild Plan

**Goal:** Transform xPoints from a single XGBoost model into a position-specific ensemble system rivaling OpenFPL/FPL Review accuracy.

**Reference:** [OpenFPL paper](https://arxiv.org/html/2508.09992v1) | [OpenFPL repo](https://github.com/daniegr/OpenFPL) | [AIrsenal](https://github.com/alan-turing-institute/AIrsenal)

**Benchmark targets:**
| Metric | Current (est.) | Target | Ceiling |
|--------|---------------|--------|---------|
| RMSE | ~3.5 | 2.9-3.0 | 2.8 |
| MAE | ~2.5 | 2.0-2.1 | 1.96 |
| R² | <0.05 | 0.10-0.15 | ~0.15 |

---

## Phase 1: Quick Wins (PR #1)
**Branch:** `feat/quick-wins`
**Impact:** Immediate accuracy + diagnostics improvement. No architecture changes.

### 1.1 Remove StandardScaler
- XGBoost is tree-based, invariant to monotonic transforms. Remove `scaler` entirely.
- Removes unnecessary complexity + a potential source of bugs.

### 1.2 Add more rolling features
Current: goals, assists, CS, xG, xA, minutes (6 stats × 3 windows = 18 features)
Add:
- `saves` — major GK points source, currently ignored
- `bonus` — strong signal for high-return players
- `bps` (bonus points system raw score) — predicts bonus allocation
- `yellow_cards` — negative points, predictable for some players
- `ict_index` components already merged but not rolled — roll `influence`, `creativity`, `threat`

### 1.3 Feature importance logging
- After training, save `best_xgb.feature_importances_` to `feature_importance.json`
- Print top 15 features to stdout
- Critical for understanding what's driving predictions

### 1.4 Better evaluation metrics
- Add MAE, R², Spearman rank correlation alongside MSE
- Add per-position breakdown (GK/DEF/MID/FWD)
- Add per-return-category evaluation (Zeros/Blanks/Tickers/Haulers per OpenFPL)
- Save evaluation results to `evaluation.json`

### 1.5 Expand validation set
- Current: 3 GWs (~1,500 obs). Too thin.
- Change to: 6 GWs minimum
- Better: implement walk-forward validation (train on GW1-N, test on N+1, slide forward)

### 1.6 Fix TimeSeriesSplit usage
- Current `TimeSeriesSplit` in GridSearchCV splits by row index, not gameweek
- Players from the same GW can appear in both train and validation
- Fix: custom CV splitter that splits by `round` column

---

## Phase 2: Position-Specific Models (PR #2)
**Branch:** `feat/position-models`
**Impact:** Biggest single accuracy improvement. Different positions score differently.

### 2.1 Split into 4 model pipelines
- GK model: emphasise saves, CS, goals conceded, opponent xG
- DEF model: emphasise CS, goals conceded, BPS, opponent attacking stats
- MID model: emphasise xG, xA, goals, assists, creativity, key passes
- FWD model: emphasise xG, goals, shots, threat, opponent defensive stats

### 2.2 Position-specific feature sets
Per OpenFPL Table 2:
- GK features: 196 total (player: FPL points, minutes, assists, goals conceded, saves, penalties saved, cards, xGChain, xGBuildup, key passes, xA + team/opponent metrics)
- DEF/MID/FWD features: 206 total (adds goals scored, penalties missed, shots, xG, BPS)
- Each feature × 5 time horizons (1, 3, 5, 10, 38 matches)

### 2.3 Ensemble: XGBoost + Random Forest
- Train both XGBoost and Random Forest per position
- Final prediction = median of all models (OpenFPL uses 50 models total)
- Start simple: 1 XGB + 1 RF per position = 8 models, median per position

### 2.4 Sample weighting
- Use `KBinsDiscretizer` to bin target points by entropy
- Weight high-return samples higher so model doesn't just predict 2 for everyone
- Per OpenFPL: 2 bins for GK, 3 for DEF, 4 for MID, 3 for FWD

---

## Phase 3: New Data Sources (PR #3)
**Branch:** `feat/understat-data`
**Impact:** Richer features = better predictions. This is what separates good models from average.

### 3.1 Understat integration
```bash
pip install understat  # async
# OR
pip install understatapi  # sync, easier
```
**Player-level:** xG, xA, npxG, xGChain, xGBuildup, key passes, shots
**Team-level:** xG, xGA, Deep, PPDA (att + def)

### 3.2 Opponent features (THREE-LAYER DESIGN)
Per OpenFPL, features should be:
- **Player (Xp):** Individual rolling stats
- **Team (Xt):** Team goals scored/conceded, xG, xGA, PPDA, Deep, league rank
- **Opponent (Xo):** SAME stats but for the opposition team

Currently xPoints has Xp only. Adding Xt + Xo is critical — the model needs to know that scoring against Burnley's defence is easier than Man City's.

### 3.3 Odds-derived features
- Source: football-data.co.uk CSVs (free, reliable)
- Features: implied_win_prob, implied_goals, implied_clean_sheet
- Key insight from research: odds features often outperform raw stats because markets aggregate all available information

### 3.4 Multi-horizon windows
- Current: [3, 5, 8]
- OpenFPL: [1, 3, 5, 10, 38]
- Add 1-match (captures immediate form) and 38-match (season baseline)
- Each feature × 5 windows = much richer feature space

### 3.5 Venue-specific historical points
- Track player's average points at home vs away separately
- "Relevant FPL points" per OpenFPL = historical points at the venue of the upcoming match
- Some players are dramatically better at home

---

## Phase 4: Advanced (Future PRs)
**These are stretch goals — do after Phases 1-3 are solid.**

### 4.1 xMins model
- Predict expected minutes separately (biggest gap vs FPL Review)
- Use: recent minutes pattern, availability %, fixture congestion, team news
- Apply as: `final_prediction = xPoints_if_plays × P(plays_60+_mins)`

### 4.2 Component prediction
- Instead of predicting total_points directly, predict:
  - P(goal), P(assist), P(clean_sheet), E(bonus), E(minutes)
- Aggregate: `xPoints = P(goal)×points_per_goal + P(assist)×3 + ...`
- This is what FPL Review does — more interpretable and robust

### 4.3 CatBoost as 3rd ensemble member
- Strong with categorical features (position, team, opponent)
- Easy addition once ensemble framework is in place

### 4.4 FBref advanced stats
- SCA/GCA (shot/goal creating actions) — predict assists + bonus
- Progressive passes, carries into final third
- Pressures (proxy for BPS)
- Rate-limited: 3s between requests, cache aggressively

### 4.5 Set piece taker identification
- Source: setpiecetakers.com (scraping required)
- Penalty taker = ~0.76 xG per penalty. Massive edge.

### 4.6 DGW fix
- Current: `xPoints * fixture_count` (naive, wrong)
- Better: predict each fixture independently with its own opponent features
- Apply rotation discount based on minutes model
- Sum per-fixture predictions

---

## File Structure (proposed)

```
xPoints/
├── script.py              → DEPRECATED (keep for reference)
├── src/
│   ├── data/
│   │   ├── fpl_api.py          # FPL API data fetching
│   │   ├── understat_api.py    # Understat data fetching
│   │   ├── odds_data.py        # Football-data.co.uk odds
│   │   └── features.py         # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py            # Training pipeline (all positions)
│   │   ├── predict.py          # Prediction pipeline
│   │   ├── ensemble.py         # Ensemble averaging logic
│   │   └── evaluate.py         # Evaluation metrics + diagnostics
│   ├── utils/
│   │   ├── validation.py       # Walk-forward CV, custom splitters
│   │   └── config.py           # Feature sets per position, hyperparams
│   └── __init__.py
├── models/                 # Saved model artifacts
├── data/                   # Cached data files
├── predictions.json        # Output
├── evaluation.json         # Model diagnostics
├── feature_importance.json # Feature rankings
├── requirements.txt
└── PLAN.md                 # This file
```

---

## Implementation Order

| Phase | PR | Est. Effort | Dependencies |
|-------|------------|-------------|--------------|
| 1 | `feat/quick-wins` | 2-3 hours | None |
| 2 | `feat/position-models` | 4-6 hours | Phase 1 merged |
| 3 | `feat/understat-data` | 3-4 hours | Phase 2 merged |
| 4.x | Individual PRs | Varies | Phase 3 merged |

**Phase 1 can start immediately. Each phase builds on the last.**
