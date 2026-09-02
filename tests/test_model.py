import importlib.util
import unittest

HAS_ML_DEPS = all(importlib.util.find_spec(m) for m in ("pandas", "sklearn", "xgboost"))
if HAS_ML_DEPS:
    import numpy as np
    import pandas as pd
    from model import apply_availability, assemble_history, build_rows, prediction_frame, train_and_predict


def season_rows(season, code, element, points):
    rows = []
    for gw, pts in enumerate(points, start=1):
        r = {"element": element, "code": code, "GW": gw, "season": season, "season_gw": season * 100 + gw,
             "total_points": pts, "minutes": 90 if gw % 3 else 20, "value": 50, "price": 5.0, "position": "MID",
             "position_id": 3, "team": 1, "name": f"p{code}", "fixture_count": 1, "home_share": 1.0}
        for s in ("expected_goals", "expected_assists", "bps", "bonus", "defensive_contribution", "starts",
                  "goals_scored", "assists", "clean_sheets", "saves", "expected_goals_conceded"):
            r[s] = 1.0
        rows.append(r)
    return pd.DataFrame(rows)


@unittest.skipUnless(HAS_ML_DEPS, "model deps not installed")
class CrossSeasonTests(unittest.TestCase):
    def test_same_player_different_element_ids_is_one_history_ordered_by_season(self):
        hist = assemble_history([season_rows(2025, code=777, element=12, points=[2, 6, 4]),
                                 season_rows(2026, code=777, element=431, points=[8])])
        self.assertEqual(hist["code"].nunique(), 1)
        self.assertEqual(hist["season_gw"].tolist(), [202501, 202502, 202503, 202601])

    def test_prediction_row_features_come_only_from_played_matches(self):
        from backtest import build_features
        hist = assemble_history([season_rows(2025, 777, 12, [2, 6, 4]), season_rows(2026, 777, 431, [8])])
        pred = season_rows(2026, 777, 431, [np.nan]); pred["GW"] = 2; pred["season_gw"] = 202602
        for s in ("total_points", "minutes", "expected_goals"):
            pred[s] = np.nan
        frame = pd.concat([hist, pred], ignore_index=True).sort_values(["code", "season_gw"]).reset_index(drop=True)
        f = build_features(frame, key="code")
        target = frame.index[frame["season_gw"] == 202602][0]
        self.assertAlmostEqual(f.loc[target, "total_points_r1"], 8.0)        # newest completed match, once
        self.assertAlmostEqual(f.loc[target, "total_points_r3"], (6 + 4 + 8) / 3)
        self.assertEqual(f.loc[target, "games_played_std"], 4)

    def test_train_and_predict_returns_one_prediction_per_target_row(self):
        hist = assemble_history([season_rows(2025, c, c, list(np.random.default_rng(c).integers(0, 9, 12)))
                                 for c in range(1, 25)])
        pred = pd.concat([season_rows(2026, c, c + 500, [np.nan]) for c in range(1, 25)], ignore_index=True)
        for s in ("total_points", "minutes", "expected_goals"):
            pred[s] = np.nan
        targets, preds, features, n_train, comp = train_and_predict(
            hist, pred, params=dict(objective="reg:tweedie", tweedie_variance_power=1.3, n_estimators=20,
                                    max_depth=2, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0,
                                    random_state=0, n_jobs=1))
        self.assertEqual(len(preds), 24)
        self.assertEqual(n_train, 24 * 12)
        self.assertTrue((preds >= 0).all())                                  # Tweedie: non-negative
        self.assertTrue(((comp["p_start60"] >= 0) & (comp["p_start60"] <= 1)).all())
        np.testing.assert_allclose(preds, comp["p_start60"] * comp["xp_if_start"])   # the product, exactly
        self.assertEqual(set(comp), {"p_start60", "xp_if_start", "cond_head", "price_expect"})


@unittest.skipUnless(HAS_ML_DEPS, "model deps not installed")
class AvailabilityAndRowsTests(unittest.TestCase):
    def test_apply_availability_policy(self):
        self.assertEqual(apply_availability(4.0, "a", None, 1), 4.0)
        self.assertEqual(apply_availability(4.0, "i", None, 1), 0.0)
        self.assertEqual(apply_availability(4.0, "s", 25, 1), 0.0)
        self.assertAlmostEqual(apply_availability(4.0, "d", 25, 1), 1.0)
        self.assertEqual(apply_availability(4.0, "d", None, 1), 4.0)        # doubtful, no % -> unscaled
        self.assertEqual(apply_availability(4.0, "a", None, 0), 0.0)         # blank gameweek

    def test_rows_carry_publication_schema_and_price_tiebreak(self):
        from prediction_safety import REQUIRED_FIELDS, validation_problems
        bootstrap = {"teams": [{"id": 1, "name": "Arsenal"}],
                     "elements": [{"id": 10, "code": 1, "web_name": "A", "team": 1, "element_type": 3, "now_cost": 80,
                                   "status": "a", "ep_next": "3.5"},
                                  {"id": 11, "code": 2, "web_name": "B", "team": 1, "element_type": 3, "now_cost": 120,
                                   "status": "d", "chance_of_playing_next_round": 50, "ep_next": "4.0"}]}
        fixtures = [{"event": 3, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 3}]
        bootstrap["elements"].append({"id": 12, "code": 3, "web_name": "C", "team": 2, "element_type": 2,
                                      "now_cost": 45, "status": "a", "ep_next": "1.0"})
        bootstrap["teams"].append({"id": 2, "name": "Chelsea"})
        targets = prediction_frame(bootstrap, fixtures, 3, 2026)
        rows = build_rows(bootstrap, targets, [4.0, 4.0, 1.0], 3, "2026-09-01T00:00:00Z", "2026/27")
        self.assertEqual(validation_problems(rows, [10, 11, 12], 3), [])
        self.assertTrue(REQUIRED_FIELDS <= rows[0].keys())
        self.assertEqual([r["player_id"] for r in rows], [10, 11, 12])   # B halved to 2.0 by doubt -> A first
        self.assertEqual(rows[1]["xPoints"], 2.0)
        self.assertEqual(rows[1]["xPoints_raw"], 4.0)
        self.assertEqual({r["source"] for r in rows}, {"xpoints_model"})
