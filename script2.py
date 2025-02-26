import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_bootstrap_data():
    """Fetch bootstrap-static data and return players_df, teams_df, events_df."""
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(bootstrap_url)
    data = response.json()
    
    players_df = pd.DataFrame(data["elements"])
    teams_df   = pd.DataFrame(data["teams"])
    events_df  = pd.DataFrame(data["events"])
    
    return players_df, teams_df, events_df

def get_fixtures_data():
    """Fetch the fixtures data."""
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(fixtures_url)
    fixtures_data = response.json()
    fixtures_df = pd.DataFrame(fixtures_data)
    return fixtures_df

def get_player_history(player_id):
    """
    Fetch per-GW history for a single player.
    Returns a DataFrame with columns like 'round', 'minutes', 'goals_scored', 'total_points', etc.
    """
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    
    j = resp.json()
    history = j.get("history", [])
    if not history:
        return None
    
    df = pd.DataFrame(history)
    df["player_id"] = player_id
    return df

def create_full_history_df(players_df):
    """
    For each player in `players_df`, fetch their per-GW history and concatenate.
    Returns a DataFrame with combined data for all players & GWs.
    """
    all_histories = []
    for pid in tqdm(players_df["id"], desc="Fetching player histories"):
        hdf = get_player_history(pid)
        if hdf is not None:
            all_histories.append(hdf)
            
    if not all_histories:
        raise ValueError("No player history found from the API.")
    
    # Combine into one big DF
    full_history_df = pd.concat(all_histories, ignore_index=True)
    return full_history_df

def create_extended_metrics(df, windows=[3, 5, 8], ewm_halflife=3):
    """
    Extends the custom_metrics function with additional advanced FPL stats
    using only data already available in the DataFrame.
    """
    # Ensure correct ordering by player_id, round
    df = df.sort_values(["player_id", "round"]).copy()
    
    # Convert string columns to numeric
    numeric_columns = ["minutes", "expected_goals", "expected_assists", "goals_scored", 
                       "assists", "total_points", "bonus", "now_cost", 
                       "selected_by_percent", "chance_of_playing_next_round"]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    ###########################################################################
    # 1. Expected Goal Involvement (xGI) Metrics
    ###########################################################################
    # Combined xG and xA
    df["xGI"] = df["expected_goals"] + df["expected_assists"]
    df["xGI_per_90"] = np.where(df["minutes"] > 0, 90 * df["xGI"] / df["minutes"], 0)
    
    # Rolling xGI
    for w in windows:
        df[f"xGI_rolling_{w}"] = df.groupby("player_id")["xGI"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )
    
    ###########################################################################
    # 2. xG Overperformance/Underperformance
    ###########################################################################
    df["xG_delta"] = df["goals_scored"] - df["expected_goals"]
    df["xA_delta"] = df["assists"] - df["expected_assists"]
    
    for w in windows:
        df[f"xG_delta_rolling_{w}"] = df.groupby("player_id")["xG_delta"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"xA_delta_rolling_{w}"] = df.groupby("player_id")["xA_delta"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
    
    ###########################################################################
    # 3. Home/Away Performance Split
    ###########################################################################
    home_points = df[df["was_home"] == True].groupby("player_id")["total_points"]
    away_points = df[df["was_home"] == False].groupby("player_id")["total_points"]
    
    for w in [3, 5]:
        home_roll = home_points.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df.loc[df["was_home"] == True, f"home_points_avg_{w}"] = home_roll
        
        away_roll = away_points.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df.loc[df["was_home"] == False, f"away_points_avg_{w}"] = away_roll
    
    for w in [3, 5]:
        df[f"home_points_avg_{w}"] = df.groupby("player_id")[f"home_points_avg_{w}"].fillna(method="ffill")
        df[f"away_points_avg_{w}"] = df.groupby("player_id")[f"away_points_avg_{w}"].fillna(method="ffill")
        df[f"home_points_avg_{w}"] = df.groupby("player_id")[f"home_points_avg_{w}"].fillna(df[f"away_points_avg_{w}"])
        df[f"away_points_avg_{w}"] = df.groupby("player_id")[f"away_points_avg_{w}"].fillna(df[f"home_points_avg_{w}"])
        df[f"home_away_ratio_{w}"] = np.where(
            df[f"away_points_avg_{w}"] > 0,
            df[f"home_points_avg_{w}"] / df[f"away_points_avg_{w}"],
            df[f"home_points_avg_{w}"]
        )
    
    ###########################################################################
    # 4. Advanced Form Indicators
    ###########################################################################
    def weighted_points(series):
        if len(series) < 3:
            return series.mean()
        weights = np.array([0.5, 0.3, 0.2])
        return np.sum(series.iloc[-3:].values * weights)
    
    df["momentum_score"] = df.groupby("player_id")["total_points"].transform(
        lambda s: s.rolling(3, min_periods=1).apply(weighted_points, raw=False)
    )
    
    df["points_lag1"] = df.groupby("player_id")["total_points"].shift(1)
    df["points_lag2"] = df.groupby("player_id")["total_points"].shift(2)
    df["points_lag3"] = df.groupby("player_id")["total_points"].shift(3)
    
    df["form_acceleration"] = (df["total_points"] - df["points_lag1"]) - (df["points_lag1"] - df["points_lag2"])
    
    ###########################################################################
    # 5. Advanced Bonus Point Metrics
    ###########################################################################
    df["bonus_per_point"] = np.where(
        df["total_points"] - df["bonus"] > 0,
        df["bonus"] / (df["total_points"] - df["bonus"]),
        0
    )
    
    df["bonus_efficiency_rolling_5"] = df.groupby("player_id")["bonus_per_point"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    
    ###########################################################################
    # 6. Risk Assessment Metrics
    ###########################################################################
    df["minutes_volatility"] = df.groupby("player_id")["minutes"].transform(
        lambda s: s.rolling(5, min_periods=2).std()
    )
    
    if "chance_of_playing_next_round" in df.columns:
        df["chance_of_playing_lag"] = df.groupby("player_id")["chance_of_playing_next_round"].shift(1)
        df["chance_of_playing_change"] = df["chance_of_playing_next_round"] - df["chance_of_playing_lag"]
    
    ###########################################################################
    # 7. Ownership and Price Metrics
    ###########################################################################
    if "selected_by_percent" in df.columns:
        df["selected_by_percent_change"] = df.groupby("player_id")["selected_by_percent"].diff()
        df["ownership_momentum"] = df.groupby("player_id")["selected_by_percent_change"].transform(
            lambda s: s.rolling(3, min_periods=1).sum()
        )
    
    df["price_change"] = df.groupby("player_id")["now_cost"].diff()
    df["points_change"] = df.groupby("player_id")["total_points"].diff()
    
    df["price_elasticity"] = np.where(
        (df["price_change"] != 0) & (~df["price_change"].isna()),
        df["points_change"] / df["price_change"],
        0
    )
    
    ###########################################################################
    # 8. Involvement Metrics
    ###########################################################################
    df["goal_involvement"] = df["goals_scored"] + df["assists"]
    
    for w in windows:
        df[f"goal_involvement_rolling_{w}"] = df.groupby("player_id")["goal_involvement"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )
    
    df["goal_involvement_per_90"] = np.where(
        df["minutes"] > 0,
        90 * df["goal_involvement"] / df["minutes"],
        0
    )
    
    ###########################################################################
    # 9. Efficiency Metrics
    ###########################################################################
    df["points_per_minute"] = np.where(
        df["minutes"] > 0,
        df["total_points"] / df["minutes"],
        0
    )
    
    df["xG_conversion"] = np.where(
        df["expected_goals"] > 0,
        df["goals_scored"] / df["expected_goals"],
        0
    )
    
    df["xA_conversion"] = np.where(
        df["expected_assists"] > 0,
        df["assists"] / df["expected_assists"],
        0
    )
    
    for w in windows:
        df[f"xG_sum_{w}"] = df.groupby("player_id")["expected_goals"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )
        df[f"goals_sum_{w}"] = df.groupby("player_id")["goals_scored"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )
        
        df[f"xG_conversion_rolling_{w}"] = np.where(
            df[f"xG_sum_{w}"] > 0,
            df[f"goals_sum_{w}"] / df[f"xG_sum_{w}"],
            0
        )
    
    ###########################################################################
    # 10. Consistency vs. Explosiveness
    ###########################################################################
    df["points_cv_5"] = np.where(
        df.groupby("player_id")["total_points"].transform(
            lambda s: s.rolling(5, min_periods=2).mean()
        ) > 0,
        df["points_std_5"] / df.groupby("player_id")["total_points"].transform(
            lambda s: s.rolling(5, min_periods=2).mean()
        ),
        0
    )
    
    df["big_haul"] = (df["total_points"] >= 10).astype(int)
    df["big_haul_rate_5"] = df.groupby("player_id")["big_haul"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    
    return df

def create_custom_metrics(df, windows=[3, 5, 8], ewm_halflife=3):
    """
    Given a merged FPL DataFrame `df` with columns:
      player_id, round, minutes, total_points, goals_scored, assists,
      now_cost, expected_goals, expected_assists, ...
    This function returns the same DataFrame with new advanced metric columns.
    """
    df = df.sort_values(["player_id", "round"]).copy()
    
    numeric_columns = ["minutes", "expected_goals", "expected_assists", "goals_scored", 
                       "assists", "total_points", "bonus", "now_cost", 
                       "selected_by_percent", "chance_of_playing_next_round"]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ###########################################################################
    # 1) Rolling Points per 90
    ###########################################################################
    for w in windows:
        col_points_roll = f"points_rolling_{w}"
        col_minutes_roll = f"minutes_rolling_{w}"
        col_points90 = f"points_per_90_rolling_{w}"

        df[col_points_roll] = df.groupby("player_id")["total_points"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )
        df[col_minutes_roll] = df.groupby("player_id")["minutes"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )

        df[col_points90] = np.where(
            df[col_minutes_roll] > 0,
            90 * df[col_points_roll] / df[col_minutes_roll],
            0
        )

    ###########################################################################
    # 2) Exponential Weighted Recent Form (EWM of total_points)
    ###########################################################################
    def ewm_points(series, halflife):
        return series.shift(1).ewm(halflife=halflife).mean()

    df["points_ewma"] = df.groupby("player_id")["total_points"].transform(
        lambda s: ewm_points(s, ewm_halflife)
    )

    ###########################################################################
    # 3) Rolling xG / xA
    ###########################################################################
    for w in windows:
        df[f"xG_rolling_{w}"] = df.groupby("player_id")["expected_goals"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )
        df[f"xA_rolling_{w}"] = df.groupby("player_id")["expected_assists"].transform(
            lambda s: s.rolling(w, min_periods=1).sum()
        )

    ###########################################################################
    # 4) Points per million
    ###########################################################################
    df["points_per_million"] = np.where(
        df["now_cost"] > 0,
        df["total_points"] / (df["now_cost"] / 10.0),
        0
    )

    ###########################################################################
    # 5) Hot Streak Indicator (>=5 points for 3 consecutive GWs)
    ###########################################################################
    df["points_ge_5"] = (df["total_points"] >= 5).astype(int)
    def rolling_hot_streak(series):
        roll3 = series.rolling(3, min_periods=3).sum()
        return (roll3 == 3).astype(int)

    df["hot_streak_3"] = df.groupby("player_id")["points_ge_5"].transform(rolling_hot_streak)

    ###########################################################################
    # 6) Consistency: Std Dev of Points (last 5 GWs)
    ###########################################################################
    def rolling_std_5(series):
        return series.rolling(5, min_periods=2).std()

    df["points_std_5"] = df.groupby("player_id")["total_points"].transform(rolling_std_5)

    ###########################################################################
    # 7) Rotation Risk: Start Rate
    ###########################################################################
    df["started"] = (df["minutes"] >= 60).astype(int)
    def rolling_start_rate_5(series):
        return series.rolling(5, min_periods=1).mean()

    df["start_rate_5"] = df.groupby("player_id")["started"].transform(rolling_start_rate_5)

    ###########################################################################
    # 8) Ceiling vs. Floor: Double Returns vs. Blanks
    ###########################################################################
    df["double_return"] = ((df["goals_scored"] + df["assists"]) >= 2).astype(int)
    df["blank"] = (df["total_points"] <= 2).astype(int)

    def rolling_sum_5(series):
        return series.rolling(5, min_periods=1).sum()

    df["double_return_5"] = df.groupby("player_id")["double_return"].transform(rolling_sum_5)
    df["blank_5"] = df.groupby("player_id")["blank"].transform(rolling_sum_5)

    return df

def create_all_metrics(df, windows=[3, 5, 8], ewm_halflife=3):
    """
    Combines both the original custom_metrics and the new extended_metrics
    into a single function.
    """
    df = create_custom_metrics(df, windows, ewm_halflife)
    df = create_extended_metrics(df, windows, ewm_halflife)
    return df

if __name__ == "__main__":
    # 1) Fetch the main FPL data
    players_df, teams_df, events_df = get_bootstrap_data()
    fixtures_df = get_fixtures_data()
    
    # 2) Combine all player histories into one DataFrame
    full_history_df = create_full_history_df(players_df)
    
    # 3) Merge additional columns from players_df (customize as needed)
    extra_cols = [
        "id", "web_name", "element_type", "team", "selected_by_percent",
        "now_cost", "form", "status", "chance_of_playing_next_round",
    ]
    merged_df = full_history_df.merge(
        players_df[extra_cols],
        left_on="player_id",
        right_on="id",
        how="left"
    )
    
    # 4) Sort the merged data
    merged_df.sort_values(["player_id", "round"], inplace=True)
    
    # 5) Create all metrics using the merged data
    df_with_all_metrics = create_all_metrics(merged_df)
    
    # 6) Convert the final DataFrame to JSON and save it
    json_data = df_with_all_metrics.to_json(orient="records")
    with open("fpl_full_history_metrics.json", "w") as f:
        f.write(json_data)
    
    print("JSON data saved to fpl_full_history_metrics.json")
