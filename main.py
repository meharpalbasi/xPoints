#!/usr/bin/env python3
"""xPoints v2 — Main orchestrator: fetch → features → train → predict → save.

Usage:
    python main.py
    python main.py --evaluate   # also run walk-forward evaluation
"""

import argparse
import json

import pandas as pd

from src.data.fpl_api import (
    detect_next_gameweek,
    enrich_players_df,
    fetch_all_player_data,
    fetch_bootstrap,
    fetch_fixtures,
    get_dataframes,
    merge_player_metadata,
)
from src.data.features import build_feature_matrix
from src.models.evaluate import walk_forward_evaluate
from src.models.predict import predict_next_gameweek
from src.models.train import train_all_positions


def main() -> None:
    """Run the full xPoints pipeline."""
    parser = argparse.ArgumentParser(description="xPoints v2 prediction pipeline")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run walk-forward evaluation"
    )
    args = parser.parse_args()

    # ── 1. Fetch data ────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Fetching FPL data")
    print("=" * 60)

    bootstrap = fetch_bootstrap()
    players_df, teams_df, events_df = get_dataframes(bootstrap)
    fixtures_df = fetch_fixtures()
    next_gw = detect_next_gameweek(events_df)
    print(f"Detected next gameweek: GW{next_gw}")

    players_df = enrich_players_df(players_df)
    player_ids = players_df["id"].tolist()

    history_df, upcoming_df = fetch_all_player_data(player_ids)
    history_df = merge_player_metadata(history_df, players_df)
    print(f"Fetched history: {len(history_df)} rows for {history_df['player_id'].nunique()} players")

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering")
    print("=" * 60)

    feature_df = build_feature_matrix(history_df, fixtures_df)
    print(f"Feature matrix: {feature_df.shape[0]} rows × {feature_df.shape[1]} columns")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        n = (feature_df["position"] == pos).sum()
        print(f"  {pos}: {n} rows")

    # ── 3. Train models ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training position-specific ensemble models")
    print("=" * 60)

    artifacts = train_all_positions(feature_df)
    print(f"\nTrained models for: {list(artifacts.keys())}")

    # ── 4. Evaluate (optional) ───────────────────────────────────────────
    if args.evaluate:
        print("\n" + "=" * 60)
        print("STEP 4: Post-hoc evaluation (in-sample — see warning)")
        print("=" * 60)
        print("⚠️  Models were trained on all data; these metrics are NOT")
        print("   out-of-sample. Useful for sanity-checking, not for")
        print("   reporting generalisation performance.")

        eval_results = walk_forward_evaluate(feature_df, artifacts)

        # Print summary
        overall = eval_results.get("overall", {})
        if overall:
            print(f"\n  Overall — RMSE: {overall['rmse']}, MAE: {overall['mae']}, "
                  f"R²: {overall['r2']}, Spearman: {overall['spearman_corr']}")

        for pos, metrics in eval_results.get("per_position", {}).items():
            print(f"  {pos:3s} (n={metrics['n']:>5d}) — RMSE: {metrics['rmse']}, "
                  f"MAE: {metrics['mae']}, R²: {metrics['r2']}")
            for cat, cat_m in metrics.get("categories", {}).items():
                if cat_m.get("n", 0) > 0:
                    print(f"        {cat:8s} (n={cat_m['n']:>5d}): MAE={cat_m['mae']}, RMSE={cat_m['rmse']}")

        # Save evaluation
        with open("evaluation.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        print("\nSaved evaluation.json")

    # ── 5. Predict next gameweek ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 5: Predicting GW{next_gw}")
    print("=" * 60)

    pred_df = predict_next_gameweek(
        feature_df, players_df, teams_df, upcoming_df, next_gw, artifacts
    )

    # Display top 30
    display_cols = ["web_name", "team_name", "position", "xPoints"]
    if "fixture_count" in pred_df.columns:
        display_cols.insert(3, "fixture_count")
    print(f"\nTop 30 predicted players for GW{next_gw}:")
    print(pred_df[display_cols].head(30).to_string(index=False))

    # ── 6. Save predictions ──────────────────────────────────────────────
    output_cols = [
        "player_id", "web_name", "team_name", "position",
        "xPoints_raw", "xPoints",
    ]
    if "fixture_count" in pred_df.columns:
        output_cols.insert(5, "fixture_count")

    available_out = [c for c in output_cols if c in pred_df.columns]
    pred_df[available_out].to_json("predictions.json", orient="records", indent=2)
    print(f"\nSaved predictions.json ({len(pred_df)} players)")


if __name__ == "__main__":
    main()
