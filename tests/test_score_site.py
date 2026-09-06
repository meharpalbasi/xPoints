import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from score_site import build_scorecard, grade, merge_gameweek, score_projection_rows, score_xmins_rows
from site_snapshot import shape_projections, shape_xmins, write_archive


def live(**stats):
    return {int(k): {"minutes": m, "total_points": p} for k, (m, p) in stats.items()}


class ProjectionScoringTests(unittest.TestCase):
    def test_populations_missing_rows_and_blank_projection(self):
        rows = [
            {"player_id": 1, "xp": {"4": 6.0, "5": 5.0}},
            {"player_id": 2, "xp": {"4": 2.0, "5": 2.5}},
            {"player_id": 3, "xp": {"4": 1.0, "5": 0.5}},
            {"player_id": 4, "xp": {"5": 3.0}},   # no projection for 4: a blank, counts as zero
            {"player_id": 9, "xp": {"4": 4.0}},   # absent from the live file
        ]
        stats = live(**{"1": (90, 9), "2": (70, 2), "3": (15, 1), "4": (0, 0)})
        result = score_projection_rows(rows, stats, 4)
        self.assertEqual(result["missing_from_live"], 1)
        self.assertEqual(result["n"], {"all": 4, "played": 3, "starters": 2})
        self.assertAlmostEqual(result["metrics"]["all"]["mae"], (3 + 0 + 0 + 0) / 4)
        self.assertAlmostEqual(result["metrics"]["all"]["zero_mae"], (9 + 2 + 1 + 0) / 4)
        self.assertIn("spearman", result["metrics"]["all"])
        self.assertEqual(result["metrics"]["starters"], {}, "fewer than three starters gives no metrics")

    def test_rank_metrics_need_enough_rows(self):
        rows = [{"player_id": i, "xp": {"4": float(i)}} for i in range(1, 25)]
        stats = {i: {"minutes": 90, "total_points": i} for i in range(1, 25)}
        result = score_projection_rows(rows, stats, 4)
        self.assertAlmostEqual(result["metrics"]["starters"]["spearman"], 1.0)
        self.assertAlmostEqual(result["metrics"]["starters"]["precision_at_20"], 1.0)
        self.assertEqual(result["metrics"]["starters"]["mae"], 0.0)


class MinutesScoringTests(unittest.TestCase):
    def test_brier_base_rate_and_calibration(self):
        rows = [
            {"player_id": 1, "p60": 0.9, "xMins": 85, "pAppear": 0.95},
            {"player_id": 2, "p60": 0.1, "xMins": 10, "pAppear": 0.5},
            {"player_id": 3, "p60": 0.5, "xMins": 45, "pAppear": 0.6},
            {"player_id": 4, "p60": None, "xMins": 30},  # no probability: skipped
            {"player_id": 5, "p60": 0.7, "xMins": 70},   # not in the live file
        ]
        stats = live(**{"1": (90, 6), "2": (0, 0), "3": (60, 2), "4": (20, 1)})
        r = score_xmins_rows(rows, stats)
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["missing_from_live"], 1)
        self.assertAlmostEqual(r["base_rate_60"], 2 / 3, places=4)
        self.assertAlmostEqual(r["brier_60"], (0.1 ** 2 + 0.1 ** 2 + 0.5 ** 2) / 3, places=4)
        self.assertAlmostEqual(r["brier_base_rate"], ((1 / 3) ** 2 + (2 / 3) ** 2 + (1 / 3) ** 2) / 3, places=4)
        self.assertGreater(r["brier_skill"], 0)
        self.assertAlmostEqual(r["minutes_mae"], (5 + 10 + 15) / 3, places=4)
        self.assertAlmostEqual(r["minutes_mae_played"], (5 + 15) / 2, places=4)
        self.assertEqual(len(r["calibration"]), 5)
        top = r["calibration"][-1]
        self.assertEqual(top["n"], 1)
        self.assertAlmostEqual(top["actual"], 1.0)
        self.assertIsNotNone(r["brier_appear"])

    def test_empty(self):
        self.assertEqual(score_xmins_rows([], {})["n"], 0)


class FilesTests(unittest.TestCase):
    def test_merge_keeps_existing_entries(self):
        existing = {"gameweek": 4, "projections": {"3": {"horizon_week": 1, "n": {"all": 1}}}, "xmins": {"n": 5}}
        merged = merge_gameweek(existing, {"projections": {"3": {"horizon_week": 1, "n": {"all": 999}}, "4": {"horizon_week": 0}}, "xmins": {"n": 6}})
        self.assertEqual(merged["projections"]["3"]["n"]["all"], 1, "an entry already written never changes")
        self.assertEqual(merged["projections"]["4"], {"horizon_week": 0})
        self.assertEqual(merged["xmins"]["n"], 5)

    def test_scorecard_by_horizon_week_and_minutes(self):
        scores = {
            4: {"projections": {"4": {"horizon_week": 0, "n": {"all": 600, "starters": 200}, "metrics": {"all": {"mae": 1.5, "zero_mae": 1.6}, "starters": {"spearman": 0.5, "spearman_ci95": [0.4, 0.6], "precision_at_20": 0.3}}}},
                "xmins": {"n": 600, "brier_60": 0.15, "brier_base_rate": 0.22, "brier_skill": 0.3, "minutes_mae": 20.0, "minutes_mae_played": 15.0, "base_rate_60": 0.33, "calibration": [{"from": 0, "to": 0.2, "n": 10, "predicted": 0.1, "actual": 0.1}]}},
            5: {"projections": {"4": {"horizon_week": 1, "n": {"all": 600, "starters": 210}, "metrics": {"all": {"mae": 1.7, "zero_mae": 1.6}, "starters": {"spearman": 0.4, "spearman_ci95": [0.3, 0.5], "precision_at_20": 0.25}}},
                                 "5": {"horizon_week": 0, "n": {"all": 600, "starters": 210}, "metrics": {"all": {"mae": 1.4, "zero_mae": 1.6}, "starters": {"spearman": 0.6, "spearman_ci95": [0.5, 0.7], "precision_at_20": 0.35}}}}},
        }
        card = build_scorecard(scores)
        self.assertEqual(card["summary"]["gameweeks_graded"], 2)
        self.assertEqual(card["summary"]["by_horizon_week"]["0"]["gameweeks"], 2)
        self.assertAlmostEqual(card["summary"]["by_horizon_week"]["0"]["mean_mae_all"], 1.45)
        self.assertAlmostEqual(card["summary"]["by_horizon_week"]["1"]["mean_spearman_starters"], 0.4)
        self.assertEqual(card["summary"]["minutes"]["gameweeks"], 1)
        self.assertAlmostEqual(card["summary"]["minutes"]["mean_brier_60"], 0.15)
        self.assertEqual(card["summary"]["minutes"]["latest_gameweek"], 4)
        self.assertEqual(len(card["summary"]["minutes"]["latest_calibration"]), 1)
        self.assertEqual(len(card["gameweeks"]), 3)
        self.assertNotIn("\u2014", json.dumps(card))

    def test_grade_scores_only_final_gameweeks_and_never_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            snap = {"gameweek": 4, "horizon": [4, 5, 6], "deadline": "2026-09-12T12:30:00Z", "archived_at": "2026-09-12T11:07:00Z", "model": "proj-1", "feed": {},
                    "rows": [{"player_id": i, "xp": {"4": float(i), "5": float(i) / 2, "6": 1.0}} for i in range(1, 30)]}
            (d / "p4.json").write_text(json.dumps(snap))
            xm = {"gameweek": 4, "archived_at": "2026-09-12T11:07:00Z", "model": "xmins-1", "rows": [{"player_id": i, "p60": 0.5 + (i % 2) * 0.4, "xMins": 60.0, "pAppear": 0.9} for i in range(1, 30)]}
            (d / "x4.json").write_text(json.dumps(xm))
            bootstrap = {"events": [{"id": 4, "finished": True, "data_checked": True, "deadline_time": "2026-09-12T12:30:00Z"}, {"id": 5, "finished": True, "data_checked": False}, {"id": 6, "finished": False}]}
            calls = []

            def live_for(gw):
                calls.append(gw)
                return {i: {"minutes": 90 if i % 2 else 0, "total_points": i} for i in range(1, 30)}

            updated = grade(bootstrap, {4: d / "p4.json"}, {4: d / "x4.json"}, {}, live_for, now=dt.datetime(2026, 9, 15, tzinfo=dt.timezone.utc))
            self.assertEqual(list(updated), [4], "gameweek 5 is finished but not data_checked, 6 is not finished")
            self.assertEqual(calls, [4], "the live file is fetched once and shared")
            entry = updated[4]["projections"]["4"]
            self.assertEqual(entry["horizon_week"], 0)
            self.assertEqual(entry["snapshot_deadline"], "2026-09-12T12:30:00Z")
            self.assertAlmostEqual(entry["metrics"]["all"]["mae"], 0.0)
            self.assertEqual(updated[4]["xmins"]["n"], 29)
            again = grade(bootstrap, {4: d / "p4.json"}, {4: d / "x4.json"}, updated, live_for, now=dt.datetime(2026, 9, 16, tzinfo=dt.timezone.utc))
            self.assertEqual(again, {}, "nothing new to grade the second time")


class SnapshotTests(unittest.TestCase):
    def table(self, n=450):
        players = {str(i): {"code": i * 7, "web_name": f"p{i}", "team": 1 + i % 20, "element_type": 1 + i % 4, "rate": 3.5, "rateSource": "feed", "xp": {"4": 2.0, "5": 1.5}} for i in range(1, n + 1)}
        return {"gameweek": 4, "horizon": [4, 5], "model": "proj-1", "minutes": {"model": "xmins-1"}, "feed": {"available": True, "gameweek": 4, "label": "feed", "generatedAt": "2026-09-12T10:00:00Z"}, "generatedAt": "2026-09-12T11:00:00Z", "players": players}

    def test_shape_projections_validates_and_orders(self):
        payload, problems = shape_projections(self.table(), 4)
        self.assertEqual(problems, [])
        self.assertEqual(payload["gameweek"], 4)
        self.assertEqual(payload["rows"][0]["player_id"], 1)
        self.assertEqual(payload["rows"][0]["xp"], {"4": 2.0, "5": 1.5})
        self.assertEqual(payload["feed"]["gameweek"], 4)
        _, wrong = shape_projections(self.table(), 5)
        self.assertTrue(any("target gameweek" in p for p in wrong))
        _, few = shape_projections(self.table(10), 4)
        self.assertTrue(any("only 10" in p for p in few))
        zeros = self.table()
        for p in zeros["players"].values():
            p["xp"] = {"4": 0, "5": 0}
        _, flat = shape_projections(zeros, 4)
        self.assertTrue(any("above zero" in p for p in flat))

    def test_shape_xmins_validates(self):
        rows = [{"id": i, "code": i, "web_name": f"p{i}", "team": 1, "element_type": 3, "status": "a", "xMins": 70.0, "pStart": 0.8, "p60": 0.7, "pAppear": 0.9, "fixtures": 1, "availability": 1, "evidence": 3, "role": "nailed"} for i in range(1, 451)]
        payload, problems = shape_xmins({"gameweek": 4, "model": "xmins-1", "gameweeksUsed": [3, 2, 1], "generatedAt": "x", "players": rows}, 4)
        self.assertEqual(problems, [])
        self.assertEqual(payload["rows"][0]["p60"], 0.7)
        _, wrong = shape_xmins({"gameweek": 3, "players": rows}, 4)
        self.assertTrue(any("target gameweek" in p for p in wrong))

    def test_write_archive_freezes_after_the_deadline_and_skips_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"gameweek": 4, "rows": [{"player_id": 1, "xp": {"4": 2.0}}]}
            before = dt.datetime(2026, 9, 12, 11, 0, tzinfo=dt.timezone.utc)
            after = dt.datetime(2026, 9, 12, 13, 0, tzinfo=dt.timezone.utc)
            self.assertTrue(write_archive("projections", 4, payload, "2026-09-12T12:30:00Z", out_dir=tmp, now=before))
            written = json.loads((Path(tmp) / "projections" / "gw4.json").read_text())
            self.assertEqual(written["archived_at"], "2026-09-12T11:00:00Z")
            self.assertFalse(write_archive("projections", 4, payload, "2026-09-12T12:30:00Z", out_dir=tmp, now=before), "unchanged rows are not rewritten")
            changed = {"gameweek": 4, "rows": [{"player_id": 1, "xp": {"4": 3.0}}]}
            self.assertFalse(write_archive("projections", 4, changed, "2026-09-12T12:30:00Z", out_dir=tmp, now=after), "frozen after the deadline")
            self.assertEqual(json.loads((Path(tmp) / "projections" / "gw4.json").read_text())["rows"][0]["xp"]["4"], 2.0)


if __name__ == "__main__":
    unittest.main()
