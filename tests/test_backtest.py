import importlib.util
import unittest

HAS_ML_DEPS = all(importlib.util.find_spec(m) for m in ("pandas", "sklearn", "xgboost"))
if HAS_ML_DEPS:
    import pandas as pd
    from backtest import build_features, load_player_gameweeks


@unittest.skipUnless(HAS_ML_DEPS, "backtest harness deps (pandas/sklearn/xgboost) not installed")
class _NeedsML(unittest.TestCase):
    pass


def frame(points_by_gw, minutes_by_gw=None):
    rows = []
    for gw, pts in enumerate(points_by_gw, start=1):
        rows.append({"element": 1, "GW": gw, "total_points": pts,
                     "minutes": (minutes_by_gw or [90] * len(points_by_gw))[gw - 1],
                     "expected_goals": 0.1, "expected_assists": 0.1, "bps": 10, "bonus": 0,
                     "defensive_contribution": 5, "starts": 1, "goals_scored": 0, "assists": 0,
                     "clean_sheets": 0, "saves": 0, "expected_goals_conceded": 1.0,
                     "home_share": 1.0, "value": 50, "position": "MID", "team": 1, "name": "p1",
                     "fixture_count": 1, "position_id": 3, "price": 5.0})
    return pd.DataFrame(rows)


class AsOfFeatureTests(_NeedsML):
    def test_features_at_gw_g_never_see_gw_g(self):
        """Changing ONLY the target gameweek's own row must leave its features unchanged."""
        base = frame([2, 6, 1, 9, 3])
        tampered = frame([2, 6, 1, 20, 3])   # GW4's own points changed
        fb, ft = build_features(base), build_features(tampered)
        gw4 = base.index[base["GW"] == 4][0]
        pd.testing.assert_series_equal(fb.loc[gw4], ft.loc[gw4])
        # ...but GW5's features DO see GW4 (it is history by then)
        gw5 = base.index[base["GW"] == 5][0]
        self.assertNotEqual(fb.loc[gw5, "total_points_r1"], ft.loc[gw5, "total_points_r1"])

    def test_rolling_windows_and_expanding_mean(self):
        f = build_features(frame([2, 6, 1, 9, 3]))
        self.assertTrue(pd.isna(f.loc[0, "total_points_r3"]))            # nothing before GW1
        self.assertAlmostEqual(f.loc[3, "total_points_r3"], (2 + 6 + 1) / 3)  # GW4 sees GW1-3
        self.assertAlmostEqual(f.loc[4, "total_points_std"], (2 + 6 + 1 + 9) / 4)
        self.assertEqual(f.loc[4, "games_played_std"], 4)

    def test_dgw_rows_are_summed_to_player_gameweek_grain(self):
        import tempfile, os
        df = pd.DataFrame([
            {"name": "a", "position": "DEF", "team": 1, "element": 7, "GW": 3, "fixture": 100, "minutes": 90,
             "total_points": 6, "value": 45, "was_home": "True", "opponent_team": 2},
            {"name": "a", "position": "DEF", "team": 1, "element": 7, "GW": 3, "fixture": 101, "minutes": 60,
             "total_points": 2, "value": 45, "was_home": "False", "opponent_team": 3},
        ])
        for c in ("expected_goals", "expected_assists", "bps", "bonus", "defensive_contribution", "starts",
                  "goals_scored", "assists", "clean_sheets", "saves", "expected_goals_conceded"):
            df[c] = 1
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as t:
            df.to_csv(t.name, index=False); path = t.name
        try:
            pg = load_player_gameweeks(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(pg), 1)
        self.assertEqual(pg.loc[0, "total_points"], 8)
        self.assertEqual(pg.loc[0, "minutes"], 150)
        self.assertEqual(pg.loc[0, "fixture_count"], 2)
        self.assertAlmostEqual(pg.loc[0, "home_share"], 0.5)
