import datetime as dt
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
if HAS_PANDAS:
    import numpy as np
    import pandas as pd
    from backtest import MARKET_FEATURES, build_features, market_features
    from fpl_history import COLUMNS, cache_is_fresh, load_or_fetch, player_rows, season_rows

BOOTSTRAP = {
    "events": [{"id": 1, "finished": True}, {"id": 2, "finished": True}, {"id": 3, "finished": False}],
    "teams": [{"id": 1, "name": "Arsenal"}],
    "elements": [{"id": 10, "code": 500, "web_name": "A", "team": 1, "element_type": 3},
                 {"id": 11, "code": 501, "web_name": "B", "team": 1, "element_type": 5}],  # 5: manager, skipped
}
HISTORY = {10: {"history": [
    {"round": 1, "fixture": 1, "opponent_team": 2, "was_home": True, "minutes": 90, "total_points": 6,
     "starts": 1, "expected_goals": "0.40", "value": 55, "selected": 1000, "transfers_in": 50, "transfers_out": 10,
     "transfers_balance": 40, "team_h_score": 2, "team_a_score": 0, "kickoff_time": "2026-08-15T14:00:00Z"},
    {"round": 2, "fixture": 12, "opponent_team": 3, "was_home": False, "minutes": 30, "total_points": 1,
     "starts": 0, "expected_goals": "0.10", "value": 56, "selected": 1200, "transfers_in": 300, "transfers_out": 20,
     "transfers_balance": 280, "team_h_score": 1, "team_a_score": 1, "kickoff_time": "2026-08-22T14:00:00Z"},
    {"round": 3, "fixture": 25, "opponent_team": 4, "was_home": True, "minutes": 0, "total_points": 0,
     "starts": 0, "expected_goals": "0.00", "value": 56, "selected": 1300, "transfers_in": 0, "transfers_out": 0,
     "transfers_balance": 0, "team_h_score": None, "team_a_score": None, "kickoff_time": "2026-08-29T14:00:00Z"},
]}}


def fake_fetch(url):
    return HISTORY[int(url.rstrip("/").split("/")[-1])]


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class LiveSeasonTests(unittest.TestCase):
    def test_only_finished_gameweeks_become_history(self):
        rows = player_rows(BOOTSTRAP["elements"][0], HISTORY[10]["history"], {1, 2}, {1: "Arsenal"})
        self.assertEqual([r["GW"] for r in rows], [1, 2])         # GW3's pre-kick-off row is not history
        self.assertEqual(rows[0]["position"], "MID")
        self.assertEqual(rows[0]["selected"], 1000)
        self.assertEqual(rows[1]["was_home"], False)

    def test_season_rows_have_the_merged_gw_shape_and_skip_non_players(self):
        df = season_rows(BOOTSTRAP, fetch=fake_fetch, workers=2)
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["element"]), {10})

    def test_cache_is_fresh_only_when_recent_and_covering_the_newest_finished_gameweek(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fpl_history_2026-27.csv"
            pd.DataFrame({"GW": [1, 2], "element": [10, 10]}).to_csv(path, index=False)
            now = dt.datetime.now(dt.timezone.utc)
            self.assertTrue(cache_is_fresh(path, newest_finished=2, now=now))
            self.assertFalse(cache_is_fresh(path, newest_finished=3, now=now))      # GW3 is final, cache lacks it
            self.assertFalse(cache_is_fresh(path, newest_finished=2, now=now + dt.timedelta(hours=7)))
            self.assertFalse(cache_is_fresh(Path(d) / "missing.csv", newest_finished=1, now=now))

    def test_load_or_fetch_writes_the_cache_and_then_reads_it(self):
        calls = []

        def counting(url):
            calls.append(url)
            return fake_fetch(url)

        with tempfile.TemporaryDirectory() as d:
            first = load_or_fetch("2026-27", BOOTSTRAP, cache_dir=d, fetch=counting)
            second = load_or_fetch("2026-27", BOOTSTRAP, cache_dir=d, fetch=counting)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(first), len(second), 2)


def market_frame():
    rows = []
    for gw, (sel, t_in, t_out, price) in enumerate([(1000, 50, 10, 5.0), (1200, 300, 20, 5.1), (1300, 0, 0, 5.1)], start=1):
        rows.append({"element": 1, "GW": gw, "total_points": 2, "minutes": 90, "expected_goals": 0.1,
                     "expected_assists": 0.1, "bps": 10, "bonus": 0, "defensive_contribution": 5, "starts": 1,
                     "goals_scored": 0, "assists": 0, "clean_sheets": 0, "saves": 0, "expected_goals_conceded": 1.0,
                     "home_share": 1.0, "value": price * 10, "position": "MID", "team": 1, "name": "p1",
                     "fixture_count": 1, "position_id": 3, "price": price,
                     "selected": sel, "transfers_in": t_in, "transfers_out": t_out, "transfers_balance": t_in - t_out})
        rows.append({**rows[-1], "element": 2, "selected": sel * 4, "transfers_in": 0, "transfers_out": t_out * 3,
                     "transfers_balance": -t_out * 3, "price": 8.0})
    return pd.DataFrame(rows).sort_values(["element", "GW"]).reset_index(drop=True)


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class MarketFeatureTests(unittest.TestCase):
    def test_shares_ranks_and_price_moves_are_deadline_known_per_row(self):
        pg = market_frame()
        f = market_features(pg)
        p1_gw2 = pg.index[(pg["element"] == 1) & (pg["GW"] == 2)][0]
        self.assertAlmostEqual(f.loc[p1_gw2, "net_transfer_share"], 280 / 1200)
        self.assertAlmostEqual(f.loc[p1_gw2, "transfers_out_share"], 20 / 1220)
        self.assertAlmostEqual(f.loc[p1_gw2, "ownership_pct_rank"], 0.5)        # the less-owned of two
        self.assertAlmostEqual(f.loc[p1_gw2, "price_change"], 0.1)
        self.assertTrue(np.isnan(f.loc[pg.index[(pg["element"] == 1) & (pg["GW"] == 1)][0], "price_change"]))

    def test_a_gameweek_without_any_transfer_data_reads_as_unknown(self):
        f = market_features(market_frame())
        pg = market_frame()
        gw3 = pg.index[pg["GW"] == 3]
        self.assertTrue(f.loc[gw3, "net_transfer_share"].isna().all())
        self.assertFalse(f.loc[gw3, "ownership_pct_rank"].isna().any())        # ownership is still known

    def test_build_features_appends_the_block_only_on_request(self):
        pg = market_frame()
        base, with_market = build_features(pg), build_features(pg, market=True)
        self.assertEqual([c for c in with_market.columns if c not in base.columns], MARKET_FEATURES)
