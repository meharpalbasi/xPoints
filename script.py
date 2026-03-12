###############################################################################
# PHASE 1 CHANGES (feat/quick-wins):
# - Removed StandardScaler (XGBoost doesn't need it)
# - Added rolling features: saves, bonus, bps, yellow_cards, influence,
#   creativity, threat
# - Added feature importance logging (feature_importance.json + top 15 print)
# - Added better evaluation: MAE, R², Spearman rank correlation, per-position
#   breakdown. Saved to evaluation.json.
# - Increased test set from 3 GWs to 6 GWs
# - Fixed TimeSeriesSplit: custom CV splitter that splits by 'round' column
###############################################################################

import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr

###############################################################################
# CUSTOM CV SPLITTER — splits by 'round' column, not row index
###############################################################################
class GameweekTimeSeriesSplit:
    """Custom CV splitter that splits by the 'round' column.
    
    Ensures no data leakage: all rows from a given gameweek are entirely
    in train or validation, never split across both.
    """
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
    
    def split(self, X, y=None, groups=None):
        """groups must be the 'round' values aligned with X."""
        if groups is None:
            raise ValueError("GameweekTimeSeriesSplit requires groups (round values)")
        unique_rounds = sorted(groups.unique())
        n_rounds = len(unique_rounds)
        # Need at least n_splits + 1 rounds
        if n_rounds < self.n_splits + 1:
            raise ValueError(f"Need at least {self.n_splits + 1} rounds, got {n_rounds}")
        
        # Each fold: train on rounds[:i+1], validate on rounds[i+1]
        # We use the last n_splits rounds as validation sets
        for i in range(self.n_splits):
            val_round_idx = n_rounds - self.n_splits + i
            train_rounds = unique_rounds[:val_round_idx]
            val_rounds = [unique_rounds[val_round_idx]]
            
            train_mask = groups.isin(train_rounds)
            val_mask = groups.isin(val_rounds)
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


###############################################################################
# 1) FETCH GLOBAL DATA (PLAYERS, TEAMS, FIXTURES)
###############################################################################
bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(bootstrap_url)
data = response.json()

players_df = pd.DataFrame(data["elements"])
teams_df = pd.DataFrame(data["teams"])
events_df = pd.DataFrame(data["events"])

fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
fixtures_data = requests.get(fixtures_url).json()
fixtures_df = pd.DataFrame(fixtures_data)

# Convert 'deadline_time' to a proper datetime (UTC).
events_df["deadline_time"] = pd.to_datetime(events_df["deadline_time"], utc=True)

# Detect "next" gameweek that hasn't started yet (deadline_time > now).
now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
future_events = events_df[(events_df["finished"] == False) &
                          (events_df["deadline_time"] > now_utc)]

if not future_events.empty:
    next_gw = future_events["id"].min()
else:
    # Fallback if no future deadlines
    next_gw = events_df[events_df["finished"] == False]["id"].min()

print(f"Detected Next Gameweek: {next_gw}")

###############################################################################
# 2) FETCH PER-GW HISTORY FOR EACH PLAYER
###############################################################################
player_ids = players_df["id"].tolist()

def fetch_player_history(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    j = resp.json()
    history = j.get("history", [])
    if len(history) == 0:
        return None
    df = pd.DataFrame(history)
    df["player_id"] = player_id
    return df

all_histories = []
for pid in tqdm(player_ids, desc="Fetching player histories"):
    hdf = fetch_player_history(pid)
    if hdf is not None:
        all_histories.append(hdf)

if not all_histories:
    raise ValueError("No historical data found. Check the API or season status.")

full_history_df = pd.concat(all_histories, ignore_index=True)

###############################################################################
# 2.1) FETCH PER-PLAYER UPCOMING FIXTURES
###############################################################################
def fetch_player_fixtures(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    fixtures = data.get("fixtures", [])
    if not fixtures:
        return None
    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df["player_id"] = player_id
    return fixtures_df

all_upcoming_fixtures = []
for pid in tqdm(player_ids, desc="Fetching player upcoming fixtures"):
    fdf = fetch_player_fixtures(pid)
    if fdf is not None:
        all_upcoming_fixtures.append(fdf)

upcoming_fixtures_df = pd.concat(all_upcoming_fixtures, ignore_index=True) if all_upcoming_fixtures else None

###############################################################################
# 2.2) ADD MORE FEATURES FROM players_df
###############################################################################
players_df["chance_of_playing_next_round"] = pd.to_numeric(
    players_df["chance_of_playing_next_round"], errors="coerce"
).fillna(100.0)

status_map = {"a": 4.0, "d": 2.0, "i": 0.0, "s": 0.0, "u": 1.0}
players_df["status_numeric"] = players_df["status"].map(status_map).fillna(1.0)
players_df["form"] = pd.to_numeric(players_df["form"], errors="coerce").fillna(0.0)

# Rename static ICT columns to avoid overwriting per-GW history values
# (history already has influence, creativity, threat per gameweek)
players_merge_cols = players_df[["id", "element_type", "team", "web_name", "selected_by_percent",
                "influence", "creativity", "threat", "ict_index", "status_numeric",
                "chance_of_playing_next_round", "form"]].rename(columns={
    "influence": "influence_season",
    "creativity": "creativity_season",
    "threat": "threat_season",
    "ict_index": "ict_index_season",
})

full_history_df = full_history_df.merge(
    players_merge_cols,
    left_on="player_id", right_on="id", how="left"
)

# Ensure per-GW influence/creativity/threat are numeric (API returns strings)
for col in ["influence", "creativity", "threat"]:
    full_history_df[col] = pd.to_numeric(full_history_df[col], errors="coerce").fillna(0.0)
full_history_df.sort_values(["player_id", "round"], inplace=True)

###############################################################################
# 3) MULTI-ROLLING WINDOWS FOR PLAYER STATS
###############################################################################
def create_multi_rolling_features(df, group_col="player_id", windows=[3, 5, 8]):
    # Original lag features
    df["goals_scored_lag"] = df.groupby(group_col)["goals_scored"].shift(1)
    df["assists_lag"] = df.groupby(group_col)["assists"].shift(1)
    df["clean_sheets_lag"] = df.groupby(group_col)["clean_sheets"].shift(1)
    df["xG_lag"] = df.groupby(group_col)["expected_goals"].shift(1)
    df["xA_lag"] = df.groupby(group_col)["expected_assists"].shift(1)
    df["minutes_lag"] = df.groupby(group_col)["minutes"].shift(1)

    # New lag features (Phase 1)
    df["saves_lag"] = df.groupby(group_col)["saves"].shift(1)
    df["bonus_lag"] = df.groupby(group_col)["bonus"].shift(1)
    df["bps_lag"] = df.groupby(group_col)["bps"].shift(1)
    df["yellow_cards_lag"] = df.groupby(group_col)["yellow_cards"].shift(1)
    df["influence_lag"] = df.groupby(group_col)["influence"].shift(1)
    df["creativity_lag"] = df.groupby(group_col)["creativity"].shift(1)
    df["threat_lag"] = df.groupby(group_col)["threat"].shift(1)

    for w in windows:
        # Original rolling features
        df[f"goals_scored_rolling_{w}"] = (
            df.groupby(group_col)["goals_scored_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"assists_rolling_{w}"] = (
            df.groupby(group_col)["assists_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"cs_rolling_{w}"] = (
            df.groupby(group_col)["clean_sheets_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"xG_rolling_{w}"] = (
            df.groupby(group_col)["xG_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"xA_rolling_{w}"] = (
            df.groupby(group_col)["xA_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"minutes_rolling_{w}"] = (
            df.groupby(group_col)["minutes_lag"].rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

        # New rolling features (Phase 1)
        df[f"saves_rolling_{w}"] = (
            df.groupby(group_col)["saves_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"bonus_rolling_{w}"] = (
            df.groupby(group_col)["bonus_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"bps_rolling_{w}"] = (
            df.groupby(group_col)["bps_lag"].rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"yellow_cards_rolling_{w}"] = (
            df.groupby(group_col)["yellow_cards_lag"].rolling(w, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        df[f"influence_rolling_{w}"] = (
            df.groupby(group_col)["influence_lag"].rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"creativity_rolling_{w}"] = (
            df.groupby(group_col)["creativity_lag"].rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"threat_rolling_{w}"] = (
            df.groupby(group_col)["threat_lag"].rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
    return df

full_history_df = create_multi_rolling_features(full_history_df, windows=[3, 5, 8])

###############################################################################
# 4) MERGE FIXTURE DIFFICULTY + HOME/AWAY INDICATOR (GLOBAL)
###############################################################################
fixtures_df = fixtures_df.rename(columns={"id": "fixture_id"})

def get_difficulty(row):
    fix_id = row["fixture"]
    was_home = row["was_home"]
    match_info = fixtures_df.loc[fixtures_df["fixture_id"] == fix_id]
    if match_info.empty:
        return np.nan
    return match_info["team_h_difficulty"].values[0] if was_home else match_info["team_a_difficulty"].values[0]

full_history_df["fixture_difficulty"] = full_history_df.apply(get_difficulty, axis=1)
full_history_df["home_dummy"] = full_history_df["was_home"].astype(int)

###############################################################################
# 5) PREPARE DATA FOR MODELING
###############################################################################
full_history_df["target_points"] = full_history_df["total_points"]

feature_cols = []
for w in [3, 5, 8]:
    feature_cols += [
        f"goals_scored_rolling_{w}",
        f"assists_rolling_{w}",
        f"cs_rolling_{w}",
        f"xG_rolling_{w}",
        f"xA_rolling_{w}",
        f"minutes_rolling_{w}",
        f"saves_rolling_{w}",
        f"bonus_rolling_{w}",
        f"bps_rolling_{w}",
        f"yellow_cards_rolling_{w}",
        f"influence_rolling_{w}",
        f"creativity_rolling_{w}",
        f"threat_rolling_{w}",
    ]
feature_cols += [
    "element_type",
    "fixture_difficulty",
    "selected_by_percent",
    "home_dummy",
    "status_numeric",
    "chance_of_playing_next_round",
    "form"
]

target_col = "target_points"

# Drop rows missing any feature cols or target
model_df = full_history_df.dropna(subset=feature_cols + [target_col]).copy()
model_df["selected_by_percent"] = pd.to_numeric(model_df["selected_by_percent"], errors="coerce").fillna(0.0)

X = model_df[feature_cols]
y = model_df[target_col]

###############################################################################
# 5.1) TIME-BASED SPLIT
###############################################################################
all_rounds = sorted(model_df["round"].unique())
if len(all_rounds) < 7:
    raise ValueError("Not enough rounds to do a time-based split (need at least 7).")

test_rounds = all_rounds[-6:]
train_rounds = all_rounds[:-6]

train_df = model_df[model_df["round"].isin(train_rounds)]
test_df = model_df[model_df["round"].isin(test_rounds)]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# No scaler needed — XGBoost is tree-based, invariant to monotonic transforms

###############################################################################
# 6) HYPERPARAMETER TUNING WITH GRIDSEARCH USING XGBOOST
###############################################################################
# Custom CV splitter that respects gameweek boundaries
gw_cv = GameweekTimeSeriesSplit(n_splits=3)
train_rounds_series = train_df["round"].reset_index(drop=True)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42, objective="reg:squarederror")

# Reset index so cv indices align with the DataFrame
X_train_reset = X_train.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=gw_cv.split(X_train_reset, y_train_reset, groups=train_rounds_series),
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train_reset, y_train_reset)

print("Best params:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

y_pred_test = best_xgb.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
test_spearman, test_spearman_p = spearmanr(y_test, y_pred_test)

print(f"\n=== Evaluation on last 6 GWs ===")
print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {np.sqrt(test_mse):.4f}")
print(f"MAE:  {test_mae:.4f}")
print(f"R²:   {test_r2:.4f}")
print(f"Spearman ρ: {test_spearman:.4f} (p={test_spearman_p:.2e})")

###############################################################################
# 6.1) PER-POSITION EVALUATION
###############################################################################
position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
test_df_eval = test_df.copy()
test_df_eval["y_pred"] = y_pred_test

eval_results = {
    "overall": {
        "mse": round(test_mse, 4),
        "rmse": round(np.sqrt(test_mse), 4),
        "mae": round(test_mae, 4),
        "r2": round(test_r2, 4),
        "spearman_rho": round(test_spearman, 4),
        "spearman_p": float(test_spearman_p),
        "n_samples": len(y_test),
        "test_gameweeks": [int(r) for r in test_rounds],
    },
    "per_position": {}
}

print(f"\n=== Per-Position Breakdown ===")
for pos_id, pos_name in position_map.items():
    pos_mask = test_df_eval["element_type"] == pos_id
    if pos_mask.sum() == 0:
        continue
    pos_y = test_df_eval.loc[pos_mask, target_col]
    pos_pred = test_df_eval.loc[pos_mask, "y_pred"]
    pos_mse = mean_squared_error(pos_y, pos_pred)
    pos_mae = mean_absolute_error(pos_y, pos_pred)
    pos_r2 = r2_score(pos_y, pos_pred)
    pos_sp, pos_sp_p = spearmanr(pos_y, pos_pred)
    
    print(f"  {pos_name}: MAE={pos_mae:.3f}  R²={pos_r2:.4f}  Spearman={pos_sp:.3f}  (n={pos_mask.sum()})")
    eval_results["per_position"][pos_name] = {
        "mse": round(pos_mse, 4),
        "rmse": round(np.sqrt(pos_mse), 4),
        "mae": round(pos_mae, 4),
        "r2": round(pos_r2, 4),
        "spearman_rho": round(pos_sp, 4),
        "n_samples": int(pos_mask.sum()),
    }

with open("evaluation.json", "w") as f:
    json.dump(eval_results, f, indent=2)
print("\nEvaluation saved to evaluation.json")

###############################################################################
# 6.2) FEATURE IMPORTANCE LOGGING
###############################################################################
importances = best_xgb.feature_importances_
feat_imp = dict(zip(feature_cols, [round(float(v), 6) for v in importances]))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

with open("feature_importance.json", "w") as f:
    json.dump(feat_imp_sorted, f, indent=2)

print(f"\n=== Top 15 Features ===")
for i, (feat, imp) in enumerate(feat_imp_sorted.items()):
    if i >= 15:
        break
    print(f"  {i+1:2d}. {feat:35s} {imp:.4f}")
print("Feature importances saved to feature_importance.json")

###############################################################################
# 7) BUILD PREDICTION DATAFRAME FOR NEXT GW WITH DGW/BGW SUPPORT
###############################################################################
# For each player, get the row of their *latest* round in full_history_df
latest_per_player = (
    full_history_df.groupby("player_id")
    .apply(lambda grp: grp.loc[grp["round"].idxmax()])
    .reset_index(drop=True)
)

pred_rows = []
for _, p_row in latest_per_player.iterrows():
    row_data = {f: np.nan for f in feature_cols}
    row_data["player_id"] = p_row["player_id"]
    row_data["web_name"] = p_row["web_name"]
    row_data["team"] = p_row["team"]
    
    for w in [3, 5, 8]:
        row_data[f"goals_scored_rolling_{w}"] = p_row[f"goals_scored_rolling_{w}"]
        row_data[f"assists_rolling_{w}"] = p_row[f"assists_rolling_{w}"]
        row_data[f"cs_rolling_{w}"] = p_row[f"cs_rolling_{w}"]
        row_data[f"xG_rolling_{w}"] = p_row[f"xG_rolling_{w}"]
        row_data[f"xA_rolling_{w}"] = p_row[f"xA_rolling_{w}"]
        row_data[f"minutes_rolling_{w}"] = p_row[f"minutes_rolling_{w}"]
        row_data[f"saves_rolling_{w}"] = p_row[f"saves_rolling_{w}"]
        row_data[f"bonus_rolling_{w}"] = p_row[f"bonus_rolling_{w}"]
        row_data[f"bps_rolling_{w}"] = p_row[f"bps_rolling_{w}"]
        row_data[f"yellow_cards_rolling_{w}"] = p_row[f"yellow_cards_rolling_{w}"]
        row_data[f"influence_rolling_{w}"] = p_row[f"influence_rolling_{w}"]
        row_data[f"creativity_rolling_{w}"] = p_row[f"creativity_rolling_{w}"]
        row_data[f"threat_rolling_{w}"] = p_row[f"threat_rolling_{w}"]
    
    row_data["element_type"] = p_row["element_type"]
    row_data["selected_by_percent"] = p_row["selected_by_percent"]
    row_data["status_numeric"] = p_row["status_numeric"]
    row_data["chance_of_playing_next_round"] = p_row["chance_of_playing_next_round"]
    row_data["form"] = p_row["form"]
    
    pred_rows.append(row_data)

# Corrected line - removed .Scope
pred_df = pd.DataFrame(pred_rows)

# Handle DGWs and BGWs
if upcoming_fixtures_df is None:
    print("No upcoming fixture data found. Skipping DGW/BGW adjustments.")
else:
    # Filter fixtures for next_gw and calculate fixture count and aggregated features
    player_next_fixtures = upcoming_fixtures_df[upcoming_fixtures_df["event"] == next_gw]
    
    fixture_count = player_next_fixtures.groupby("player_id").size().reset_index(name="fixture_count")
    avg_difficulty = player_next_fixtures.groupby("player_id")["difficulty"].mean().reset_index(name="avg_difficulty")
    home_proportion = player_next_fixtures.groupby("player_id")["is_home"].mean().reset_index(name="home_proportion")
    
    # Merge the individual DataFrames into one with player_id as a column
    dgw_bgw_df = fixture_count.merge(avg_difficulty, on="player_id", how="outer") \
                              .merge(home_proportion, on="player_id", how="outer")
    
    # Merge DGW/BGW info into pred_df
    pred_df = pred_df.merge(dgw_bgw_df, on="player_id", how="left")
    
    # Fill missing values: BGW players get 0 fixtures, default difficulty/home to neutral
    pred_df["fixture_count"] = pred_df["fixture_count"].fillna(0)
    pred_df["avg_difficulty"] = pred_df["avg_difficulty"].fillna(3.0)   # Neutral difficulty
    pred_df["home_proportion"] = pred_df["home_proportion"].fillna(0.5) # Neutral home/away
    
    # Overwrite fixture_difficulty and home_dummy with DGW/BGW-adjusted values
    pred_df["fixture_difficulty"] = pred_df["avg_difficulty"]
    pred_df["home_dummy"] = pred_df["home_proportion"]

# Prepare data for prediction
X_next = pred_df[feature_cols].copy()
X_next["selected_by_percent"] = pd.to_numeric(X_next["selected_by_percent"], errors="coerce").fillna(0.0)

# Predict with XGBoost and adjust for fixture count (DGW scaling)
pred_df["xPoints_raw"] = best_xgb.predict(X_next)
pred_df["xPoints"] = pred_df["xPoints_raw"] * pred_df["fixture_count"]

# Force xPoints = 0 for BGW players or 0% chance of playing
pred_df.loc[(pred_df["fixture_count"] == 0) | (pred_df["chance_of_playing_next_round"] == 0), "xPoints"] = 0

# Map team ID -> name for display
team_map = dict(zip(teams_df["id"], teams_df["name"]))
pred_df["team_name"] = pred_df["team"].map(team_map)

pred_df.sort_values("xPoints", ascending=False, inplace=True)

# Show top 30
print("\n=== Next Gameweek xPoints (DGW/BGW Adjusted) ===")
print(pred_df[["web_name", "team_name", "fixture_count", "chance_of_playing_next_round", "xPoints"]].head(30))

# If you want to save results to JSON
pred_df.to_json("predictions.json", orient="records", indent=2)
