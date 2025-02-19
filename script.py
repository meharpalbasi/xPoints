import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
    # fallback if no future deadlines
    next_gw = events_df[events_df["finished"] == False]["id"].min()

print(f"Detected Next Gameweek: {next_gw}")

###############################################################################
# 2) FETCH PER-GW HISTORY FOR EACH PLAYER
###############################################################################
player_ids = players_df["id"].tolist()

def fetch_player_history(player_id):
    """Pull the past 'history' data for a single player."""
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
    """Pull the *future* 'fixtures' data for a single player."""
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
#     (e.g. 'status', 'chance_of_playing_next_round', 'form')
###############################################################################
players_df["chance_of_playing_next_round"] = pd.to_numeric(
    players_df["chance_of_playing_next_round"], errors="coerce"
).fillna(100.0)  # If missing, assume 100% for demonstration

# Simple numeric mapping for 'status' column
status_map = {
    "a": 4.0,  # fully available
    "d": 2.0,  # doubtful
    "i": 0.0,  # injured
    "s": 0.0,  # suspended
    "u": 1.0,  # unavailable
}
players_df["status_numeric"] = players_df["status"].map(status_map).fillna(1.0)

# Convert 'form' to numeric
players_df["form"] = pd.to_numeric(players_df["form"], errors="coerce").fillna(0.0)

# Now merge these new columns into full_history_df
full_history_df = full_history_df.merge(
    players_df[[
        "id", "element_type", "team", "web_name",
        "selected_by_percent", "influence", "creativity", "threat", "ict_index",
        "status_numeric", "chance_of_playing_next_round", "form"
    ]],
    left_on="player_id",
    right_on="id",
    how="left",
)
full_history_df.sort_values(["player_id", "round"], inplace=True)

###############################################################################
# 3) MULTI-ROLLING WINDOWS FOR PLAYER STATS
###############################################################################
def create_multi_rolling_features(df, group_col="player_id", windows=[3, 5, 8]):
    df["goals_scored_lag"] = df.groupby(group_col)["goals_scored"].shift(1)
    df["assists_lag"]      = df.groupby(group_col)["assists"].shift(1)
    df["clean_sheets_lag"] = df.groupby(group_col)["clean_sheets"].shift(1)
    df["xG_lag"]           = df.groupby(group_col)["expected_goals"].shift(1)
    df["xA_lag"]           = df.groupby(group_col)["expected_assists"].shift(1)
    df["minutes_lag"]      = df.groupby(group_col)["minutes"].shift(1)

    for w in windows:
        df[f"goals_scored_rolling_{w}"] = (
            df.groupby(group_col)["goals_scored_lag"]
              .rolling(w, min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
        df[f"assists_rolling_{w}"] = (
            df.groupby(group_col)["assists_lag"]
              .rolling(w, min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
        df[f"cs_rolling_{w}"] = (
            df.groupby(group_col)["clean_sheets_lag"]
              .rolling(w, min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
        df[f"xG_rolling_{w}"] = (
            df.groupby(group_col)["xG_lag"]
              .rolling(w, min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
        df[f"xA_rolling_{w}"] = (
            df.groupby(group_col)["xA_lag"]
              .rolling(w, min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
        # average minutes over the window
        df[f"minutes_rolling_{w}"] = (
            df.groupby(group_col)["minutes_lag"]
              .rolling(w, min_periods=1).mean()
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
    if was_home:
        return match_info["team_h_difficulty"].values[0]
    else:
        return match_info["team_a_difficulty"].values[0]

full_history_df["fixture_difficulty"] = full_history_df.apply(get_difficulty, axis=1)
full_history_df["home_dummy"] = full_history_df["was_home"].astype(int)

###############################################################################
# 4.1) TEAM-LEVEL (OPPONENT) ROLLING STATS
###############################################################################
team_gw_home = fixtures_df[["event", "team_h", "team_h_score", "team_a_score"]].copy()
team_gw_home.rename(columns={
    "team_h": "team",
    "team_h_score": "goals_scored",
    "team_a_score": "goals_conceded",
    "event": "round"
}, inplace=True)

team_gw_away = fixtures_df[["event", "team_a", "team_h_score", "team_a_score"]].copy()
team_gw_away.rename(columns={
    "team_a": "team",
    "team_h_score": "goals_conceded",
    "team_a_score": "goals_scored",
    "event": "round"
}, inplace=True)

team_gw_stats = pd.concat([team_gw_home, team_gw_away], ignore_index=True)
team_gw_stats.sort_values(["team", "round"], inplace=True)

# Rolling stats for each team
for w in [3, 5, 8]:
    team_gw_stats[f"gc_rolling_{w}"] = (
        team_gw_stats.groupby("team")["goals_conceded"]
                     .rolling(w, min_periods=1).mean()
                     .reset_index(level=0, drop=True)
    )
    team_gw_stats[f"gf_rolling_{w}"] = (
        team_gw_stats.groupby("team")["goals_scored"]
                     .rolling(w, min_periods=1).mean()
                     .reset_index(level=0, drop=True)
    )

# Merge Opponent's stats into player DF
full_history_df = full_history_df.merge(
    team_gw_stats[[
        "team", "round",
        "gc_rolling_3", "gc_rolling_5", "gc_rolling_8",
        "gf_rolling_3", "gf_rolling_5", "gf_rolling_8"
    ]],
    left_on=["opponent_team", "round"],
    right_on=["team", "round"],
    how="left",
    suffixes=("", "_opp")
)

full_history_df.rename(columns={
    "gc_rolling_3": "opp_gc_rolling_3",
    "gc_rolling_5": "opp_gc_rolling_5",
    "gc_rolling_8": "opp_gc_rolling_8",
    "gf_rolling_3": "opp_gf_rolling_3",
    "gf_rolling_5": "opp_gf_rolling_5",
    "gf_rolling_8": "opp_gf_rolling_8"
}, inplace=True)

###############################################################################
# 5) PREPARE DATA FOR MODELING
###############################################################################
full_history_df["target_points"] = full_history_df["total_points"]

# We will define a broad feature list
feature_cols = []
# (A) player's rolling stats
for w in [3, 5, 8]:
    feature_cols += [
        f"goals_scored_rolling_{w}",
        f"assists_rolling_{w}",
        f"cs_rolling_{w}",
        f"xG_rolling_{w}",
        f"xA_rolling_{w}",
        f"minutes_rolling_{w}",
    ]
# (B) opponent rolling stats
for w in [3, 5, 8]:
    feature_cols += [
        f"opp_gc_rolling_{w}",
        f"opp_gf_rolling_{w}",
    ]
# (C) Additional
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

model_df = full_history_df.dropna(subset=feature_cols + [target_col]).copy()

# Convert columns to numeric as needed
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
test_df  = model_df[model_df["round"].isin(test_rounds)]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test  = test_df[feature_cols]
y_test  = test_df[target_col]

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

###############################################################################
# 6) HYPERPARAMETER TUNING WITH GRIDSEARCH
###############################################################################
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("Best params:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

y_pred_test = best_rf.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Time-based split MSE on last 3 GWs: {test_mse:.4f}")

###############################################################################
# 7) BUILD PREDICTION DATAFRAME FOR NEXT GW
###############################################################################
# We'll get each player's latest row from full_history_df
latest_per_player = (
    full_history_df.groupby("player_id")
    .apply(lambda grp: grp.loc[grp["round"].idxmax()])
    .reset_index(drop=True)
)

# We'll create a 'pred_df' for next GW by copying relevant rolling stats
pred_rows = []
for _, p_row in latest_per_player.iterrows():
    row_data = {}
    for f in feature_cols:
        row_data[f] = np.nan  # default
    
    row_data["player_id"] = p_row["player_id"]
    row_data["web_name"]  = p_row["web_name"]
    row_data["team"]      = p_row["team"]
    
    # Populate the rolling stats from the player's row
    for w in [3, 5, 8]:
        row_data[f"goals_scored_rolling_{w}"] = p_row[f"goals_scored_rolling_{w}"]
        row_data[f"assists_rolling_{w}"]      = p_row[f"assists_rolling_{w}"]
        row_data[f"cs_rolling_{w}"]           = p_row[f"cs_rolling_{w}"]
        row_data[f"xG_rolling_{w}"]           = p_row[f"xG_rolling_{w}"]
        row_data[f"xA_rolling_{w}"]           = p_row[f"xA_rolling_{w}"]
        row_data[f"minutes_rolling_{w}"]      = p_row[f"minutes_rolling_{w}"]
    
    # Additional features from player
    row_data["element_type"]                 = p_row["element_type"]
    row_data["selected_by_percent"]          = p_row["selected_by_percent"]
    row_data["status_numeric"]               = p_row["status_numeric"]
    row_data["chance_of_playing_next_round"] = p_row["chance_of_playing_next_round"]
    row_data["form"]                         = p_row["form"]
    
    # We will set fixture_difficulty and home_dummy to NaN for now;
    # we will overwrite them from the *player-specific fixture* data below.
    row_data["fixture_difficulty"] = np.nan
    row_data["home_dummy"]         = np.nan
    
    pred_rows.append(row_data)

pred_df = pd.DataFrame(pred_rows)

# If we have no upcoming_fixtures_df, bail early
if upcoming_fixtures_df is None:
    print("No upcoming fixture data found. Using global approach or skipping.")
else:
    # Step: filter upcoming_fixtures_df to only next_gw
    player_next_fixture = (
        upcoming_fixtures_df[ upcoming_fixtures_df["event"] == next_gw ]
        .groupby("player_id")
        .first()  # If a player has multiple fixtures in this GW (DGW), handle accordingly
        .reset_index()
    )
    
    # Merge the columns from the player's *next fixture* into pred_df
    pred_df = pred_df.merge(
        player_next_fixture[["player_id", "is_home", "difficulty"]],
        on="player_id", how="left"
    )
    
    # Overwrite the fixture difficulty and home_dummy with the direct values
    pred_df["fixture_difficulty"] = pred_df["difficulty"]
    pred_df["home_dummy"]         = pred_df["is_home"].astype(float)
    
# Now drop rows that have no fixture_difficulty (i.e. no next fixture)
pred_df.dropna(subset=["fixture_difficulty"], inplace=True)

# Prepare data for predict
X_next = pred_df[feature_cols].copy()
X_next["selected_by_percent"] = pd.to_numeric(X_next["selected_by_percent"], errors="coerce").fillna(0.0)

# Scale
X_next_scaled = scaler.transform(X_next)

# Predict with best RF
pred_df["xPoints"] = best_rf.predict(X_next_scaled)

###############################################################################
# OPTIONAL FIX: Force xPoints = 0 if chance_of_playing_next_round == 0
###############################################################################
pred_df.loc[pred_df["chance_of_playing_next_round"] == 0, "xPoints"] = 0

# Map team ID -> name
team_map = dict(zip(teams_df["id"], teams_df["name"]))
pred_df["team_name"] = pred_df["team"].map(team_map)

pred_df.sort_values("xPoints", ascending=False, inplace=True)

# Show top 30
print("\n=== Next Gameweek xPoints (with per-player fixture data) ===")
print(pred_df[["web_name", "team_name", "chance_of_playing_next_round", "xPoints"]])
pred_df.to_json("predictions.json", orient="records", indent=2)
