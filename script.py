import requests
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from prediction_safety import (
    POSITIONS,
    build_player_fixture_rows,
    season_from_events,
    write_predictions,
)

MODEL_VERSION = "xgboost-current-season-1"

###############################################################################
# 1) FETCH GLOBAL DATA (PLAYERS, TEAMS, FIXTURES)
###############################################################################
bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(bootstrap_url, timeout=30)
response.raise_for_status()
data = response.json()

players_df = pd.DataFrame(data["elements"])
teams_df = pd.DataFrame(data["teams"])
events_df = pd.DataFrame(data["events"])

fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
fixtures_response = requests.get(fixtures_url, timeout=30)
fixtures_response.raise_for_status()
fixtures_data = fixtures_response.json()
fixtures_df = pd.DataFrame(fixtures_data)

# Convert 'deadline_time' to a proper datetime (UTC).
events_df["deadline_time"] = pd.to_datetime(events_df["deadline_time"], utc=True)

# Detect "next" gameweek that hasn't started yet (deadline_time > now).
now_utc = datetime.datetime.now(datetime.timezone.utc)
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
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
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
# 2.2) ADD MORE FEATURES FROM players_df
###############################################################################
players_df["chance_of_playing_next_round"] = pd.to_numeric(
    players_df["chance_of_playing_next_round"], errors="coerce"
).fillna(100.0)

status_map = {"a": 4.0, "d": 2.0, "i": 0.0, "s": 0.0, "u": 1.0}
players_df["status_numeric"] = players_df["status"].map(status_map).fillna(1.0)
players_df["form"] = pd.to_numeric(players_df["form"], errors="coerce").fillna(0.0)

full_history_df = full_history_df.merge(
    players_df[["id", "code", "element_type", "team", "web_name", "selected_by_percent",
                "influence", "creativity", "threat", "ict_index", "status_numeric",
                "chance_of_playing_next_round", "form"]],
    left_on="player_id", right_on="id", how="left"
)
full_history_df.sort_values(["player_id", "round"], inplace=True)

###############################################################################
# 3) MULTI-ROLLING WINDOWS FOR PLAYER STATS
###############################################################################
def create_multi_rolling_features(df, group_col="player_id", windows=[3, 5, 8]):
    df["goals_scored_lag"] = df.groupby(group_col)["goals_scored"].shift(1)
    df["assists_lag"] = df.groupby(group_col)["assists"].shift(1)
    df["clean_sheets_lag"] = df.groupby(group_col)["clean_sheets"].shift(1)
    df["xG_lag"] = df.groupby(group_col)["expected_goals"].shift(1)
    df["xA_lag"] = df.groupby(group_col)["expected_assists"].shift(1)
    df["minutes_lag"] = df.groupby(group_col)["minutes"].shift(1)

    for w in windows:
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
        f"minutes_rolling_{w}"
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
if len(all_rounds) < 4:
    raise ValueError("Not enough rounds to do a time-based split in this example.")

test_rounds = all_rounds[-3:]
train_rounds = all_rounds[:-3]

train_df = model_df[model_df["round"].isin(train_rounds)]
test_df = model_df[model_df["round"].isin(test_rounds)]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

###############################################################################
# 6) HYPERPARAMETER TUNING WITH GRIDSEARCH USING XGBOOST
###############################################################################
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42, objective="reg:squarederror")

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("Best params:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

y_pred_test = best_xgb.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Time-based split MSE on last 3 GWs: {test_mse:.4f}")

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
    row_data["player_code"] = p_row["code"]
    row_data["web_name"] = p_row["web_name"]
    row_data["team"] = p_row["team"]
    
    for w in [3, 5, 8]:
        row_data[f"goals_scored_rolling_{w}"] = p_row[f"goals_scored_rolling_{w}"]
        row_data[f"assists_rolling_{w}"] = p_row[f"assists_rolling_{w}"]
        row_data[f"cs_rolling_{w}"] = p_row[f"cs_rolling_{w}"]
        row_data[f"xG_rolling_{w}"] = p_row[f"xG_rolling_{w}"]
        row_data[f"xA_rolling_{w}"] = p_row[f"xA_rolling_{w}"]
        row_data[f"minutes_rolling_{w}"] = p_row[f"minutes_rolling_{w}"]
    
    row_data["element_type"] = p_row["element_type"]
    row_data["selected_by_percent"] = p_row["selected_by_percent"]
    row_data["status_numeric"] = p_row["status_numeric"]
    row_data["chance_of_playing_next_round"] = p_row["chance_of_playing_next_round"]
    row_data["form"] = p_row["form"]
    
    pred_rows.append(row_data)

# Corrected line - removed .Scope
pred_df = pd.DataFrame(pred_rows)

# Derive DGW/BGW features from the already validated global fixture feed.
# This removes hundreds of duplicate element-summary calls and, critically,
# means an upstream failure cannot be mistaken for a confirmed blank gameweek.
fixture_features_df = pd.DataFrame(build_player_fixture_rows(
    players_df[["id", "team"]].to_dict(orient="records"),
    fixtures_data,
    next_gw,
))
pred_df = pred_df.merge(fixture_features_df, on="player_id", how="left")

# Prepare data for prediction
X_next = pred_df[feature_cols].copy()
X_next["selected_by_percent"] = pd.to_numeric(X_next["selected_by_percent"], errors="coerce").fillna(0.0)
X_next_scaled = scaler.transform(X_next)

# Predict with XGBoost and adjust for fixture count (DGW scaling)
pred_df["xPoints_raw"] = best_xgb.predict(X_next_scaled)
pred_df["xPoints"] = pred_df["xPoints_raw"] * pred_df["fixture_count"]
pred_df["xPoints"] = pred_df["xPoints"].clip(lower=0)

# Force xPoints = 0 for BGW players or 0% chance of playing
pred_df.loc[(pred_df["fixture_count"] == 0) | (pred_df["chance_of_playing_next_round"] == 0), "xPoints"] = 0

# Map team ID -> name for display
team_map = dict(zip(teams_df["id"], teams_df["name"]))
pred_df["team_name"] = pred_df["team"].map(team_map)
pred_df["position"] = pred_df["element_type"].map(POSITIONS)
pred_df["generated_at"] = datetime.datetime.now(
    datetime.timezone.utc
).strftime("%Y-%m-%dT%H:%M:%SZ")
pred_df["season"] = season_from_events(data["events"])
pred_df["gameweek"] = int(next_gw)
pred_df["source"] = "model"
pred_df["model_version"] = MODEL_VERSION

pred_df.sort_values("xPoints", ascending=False, inplace=True)

# Show top 30
print("\n=== Next Gameweek xPoints (DGW/BGW Adjusted) ===")
print(pred_df[["web_name", "team_name", "fixture_count", "chance_of_playing_next_round", "xPoints"]].head(30))

rows = json.loads(pred_df.to_json(orient="records"))
official_player_ids = players_df[
    players_df["element_type"].isin(POSITIONS)
]["id"].tolist()
write_predictions("predictions.json", rows, official_player_ids, int(next_gw))
