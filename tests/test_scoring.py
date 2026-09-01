import unittest

from score import build_scorecard, checked_events, score_gameweek


def pred(pid, xp, cost=None, blend=None, pos="MID"):
    r = {"player_id": pid, "web_name": f"p{pid}", "position": pos, "xPoints": xp,
         "generated_at": "2026-08-21T16:58:19Z", "source": "fpl_ep_next",
         "model_version": "ep-next-baseline-1"}
    if cost is not None:
        r["now_cost"] = cost
    if blend is not None:
        r["blend_rank"] = blend
    return r


def live(**stats):
    return {pid: {"minutes": m, "total_points": p} for pid, (m, p) in stats.items()}


class ScoreGameweekTests(unittest.TestCase):
    def test_populations_and_baselines_on_identical_rows(self):
        rows = [pred(1, 4.0), pred(2, 3.0), pred(3, 1.0), pred(4, 0.5), pred(9, 2.0)]
        stats = live(**{"1": (90, 8), "2": (75, 2), "3": (10, 1), "4": (0, 0)})
        stats = {int(k): v for k, v in stats.items()}
        result, joined = score_gameweek(rows, stats)

        self.assertEqual(result["missing_from_live"], 1)          # player 9 absent
        self.assertEqual(result["n"], {"all": 4, "played": 3, "starters": 2})
        self.assertEqual(result["predictors_available"], ["ep_next", "zero"])
        all_m = result["metrics"]["all"]
        self.assertAlmostEqual(all_m["zero"]["mae"], (8 + 2 + 1 + 0) / 4)
        self.assertAlmostEqual(all_m["ep_next"]["mae"], (4 + 1 + 0 + 0.5) / 4)
        self.assertNotIn("spearman", all_m["zero"])                # rank metrics meaningless for zero
        self.assertEqual(all_m["ep_next"]["captain_regret"], 0.0)  # top pick (1) was the top scorer
        self.assertNotIn("captain_regret", result["metrics"]["starters"]["ep_next"])

    def test_price_and_blend_predictors_only_when_archive_carries_them(self):
        stats = {1: {"minutes": 90, "total_points": 6}, 2: {"minutes": 90, "total_points": 2},
                 3: {"minutes": 90, "total_points": 9}}
        without, _ = score_gameweek([pred(1, 4.0), pred(2, 3.0), pred(3, 2.0)], stats)
        self.assertEqual(without["predictors_available"], ["ep_next", "zero"])
        with_all, _ = score_gameweek(
            [pred(1, 4.0, cost=80, blend=2), pred(2, 3.0, cost=50, blend=3), pred(3, 2.0, cost=120, blend=1)],
            stats,
        )
        self.assertEqual(with_all["predictors_available"], ["blend", "ep_next", "price", "zero"])
        # blend ranked player 3 first, and player 3 scored most: perfect rank correlation
        self.assertAlmostEqual(with_all["metrics"]["all"]["blend"]["spearman"], 1.0)

    def test_checked_events_requires_finished_and_data_checked(self):
        events = [{"id": 1, "finished": True, "data_checked": True},
                  {"id": 2, "finished": True, "data_checked": False},
                  {"id": 3, "finished": False, "data_checked": False}]
        self.assertEqual(list(checked_events(events)), [1])

    def test_scorecard_uses_prior_sd_until_three_gameweeks(self):
        scores = {1: {"deadline": "d", "prediction": {"generated_at": "g", "source": "s"},
                      "n": {"all": 10, "played": 5, "starters": 3},
                      "metrics": {"all": {"ep_next": {"mae": 1.5}, "zero": {"mae": 1.4}},
                                  "starters": {"ep_next": {"spearman": 0.1, "spearman_ci95": [0, 0.2],
                                                           "precision_at_20": 0.2}}}}}
        card = build_scorecard(scores)
        self.assertEqual(card["summary"]["scored_gameweeks"], 1)
        self.assertEqual(card["gameweeks"][0]["zero_mae_all"], 1.4)
        self.assertIn("prior", card["summary"]["power"]["sd_source"])
        self.assertEqual(card["summary"]["power"]["per_gameweek_sd_used"], 0.091)
        self.assertEqual(card["summary"]["power"]["gameweeks_needed_power_80"], 27)
