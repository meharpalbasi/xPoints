import unittest

from score import (availability_factor, build_scorecard, checked_events, model_probabilities,
                   promotion_gate, score_gameweek)


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


def gw_row(gw, model_sp, feed_sp, model_p20=0.2, feed_p20=0.2):
    return {"gameweek": gw, "model_spearman_starters": model_sp, "ep_next_spearman_starters": feed_sp,
            "model_precision_at_20_starters": model_p20, "ep_next_precision_at_20_starters": feed_p20}


class PromotionGateTests(unittest.TestCase):
    def test_not_ready_until_six_paired_gameweeks(self):
        gate = promotion_gate([gw_row(3, 0.2, 0.1)])
        self.assertFalse(gate["ready"])
        self.assertEqual(gate["gameweeks_in_window"], 1)
        self.assertIn("1 of 6 paired gameweeks graded", gate["reasons"])
        self.assertIsNone(gate["starter_spearman_diff"]["se"])     # one gameweek has no spread

    def test_ready_only_when_ahead_with_interval_clear_of_zero_and_precision_level(self):
        rows = [gw_row(g, 0.20 + 0.01 * (g % 2), 0.10) for g in range(1, 7)]
        gate = promotion_gate(rows)
        self.assertTrue(gate["ready"])
        self.assertEqual(gate["reasons"], [])
        self.assertGreater(gate["starter_spearman_diff"]["ci95"][0], 0)
        # Same means, wildly varying differences: the interval includes zero
        noisy = [gw_row(g, 0.10 + (0.3 if g % 2 else -0.1), 0.10) for g in range(1, 7)]
        self.assertIn("paired 95% interval for starter rank match includes zero", promotion_gate(noisy)["reasons"])
        # Ahead on ranking but behind on precision at 20
        behind = [gw_row(g, 0.20, 0.10, model_p20=0.1, feed_p20=0.2) for g in range(1, 7)]
        self.assertIn("starter precision at 20 below ep_next", promotion_gate(behind)["reasons"])

    def test_window_is_the_last_six_paired_rows_only(self):
        rows = [gw_row(g, -0.5, 0.1) for g in range(1, 5)] + [gw_row(g, 0.21 + 0.01 * (g % 2), 0.1) for g in range(5, 11)]
        rows.insert(2, {"gameweek": 99, "model_spearman_starters": None, "ep_next_spearman_starters": 0.1})
        gate = promotion_gate(rows)
        self.assertEqual(gate["gameweeks"], [5, 6, 7, 8, 9, 10])
        self.assertTrue(gate["ready"])

    def test_scorecard_carries_the_gate(self):
        scores = {1: {"deadline": "d", "prediction": {"generated_at": "g", "source": "s"},
                      "n": {"all": 10, "played": 5, "starters": 3},
                      "metrics": {"all": {"ep_next": {"mae": 1.5}, "zero": {"mae": 1.4}},
                                  "starters": {"ep_next": {"spearman": 0.1, "spearman_ci95": [0, 0.2],
                                                           "precision_at_20": 0.2}}}}}
        card = build_scorecard(scores)
        self.assertFalse(card["summary"]["promotion_ready"])
        self.assertEqual(card["summary"]["promotion"]["gameweeks_in_window"], 0)


class ModelProbabilityTests(unittest.TestCase):
    def test_availability_factor_matches_the_model_rule(self):
        self.assertEqual(availability_factor("a", None, 1), 1.0)
        self.assertEqual(availability_factor("i", None, 1), 0.0)
        self.assertEqual(availability_factor("d", 25, 1), 0.25)
        self.assertEqual(availability_factor("d", None, 1), 1.0)
        self.assertEqual(availability_factor("a", None, 0), 0.0)

    def test_model_probabilities_apply_availability_and_skip_rows_without_components(self):
        rows = [{"player_id": 1, "p_start60": 0.8, "status": "a", "fixture_count": 1},
                {"player_id": 2, "p_start60": 0.8, "status": "d", "chance_of_playing_next_round": 50, "fixture_count": 1},
                {"player_id": 3, "xPoints": 2.0}]
        self.assertEqual(model_probabilities(rows), {1: 0.8, 2: 0.4})

    def test_p60_brier_is_graded_beside_the_base_rate_on_identical_rows(self):
        preds = [pred(1, 5.0), pred(2, 3.0), pred(3, 1.0), pred(4, 0.5)]
        stats = live(**{"1": (90, 8), "2": (90, 2), "3": (10, 1), "4": (0, 0)})
        stats = {int(k): v for k, v in stats.items()}
        result, _ = score_gameweek(preds, stats, probs={"model_p60": {1: 0.9, 2: 0.6, 3: 0.2, 4: 0.1}})
        m = result["metrics"]["all"]["model_p60"]
        self.assertEqual(m["n"], 4)
        self.assertAlmostEqual(m["base_rate"], 0.5)
        self.assertAlmostEqual(m["brier"], (0.01 + 0.16 + 0.04 + 0.01) / 4)
        self.assertAlmostEqual(m["base_rate_brier"], 0.25)
        self.assertEqual(result["metrics"]["starters"]["model_p60"]["n"], 2)
        self.assertNotIn("model_p60", result["predictors_available"])
