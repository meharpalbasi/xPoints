import json
import tempfile
import unittest
from pathlib import Path

from prediction_safety import (
    PredictionValidationError,
    build_player_fixture_rows,
    validation_problems,
    write_predictions,
)


def prediction(player_id, xpoints=4.0, fixture_count=1):
    return {
        "player_id": player_id,
        "player_code": 1000 + player_id,
        "team": 1 if player_id == 1 else 2,
        "element_type": 3,
        "position": "MID",
        "xPoints": xpoints,
        "fixture_count": fixture_count,
        "generated_at": "2026-08-20T17:00:07Z",
        "season": "2026/27",
        "gameweek": 1,
        "source": "model",
        "model_version": "candidate-1",
    }


class FixtureFeatureTests(unittest.TestCase):
    def test_builds_dgw_and_confirmed_bgw_rows_from_global_fixtures(self):
        players = [
            {"id": 1, "team": 1},
            {"id": 2, "team": 2},
            {"id": 3, "team": 3},
            {"id": 4, "team": 4},
        ]
        fixtures = [
            {"event": 1, "team_h": 1, "team_a": 2,
             "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"event": 1, "team_h": 3, "team_a": 1,
             "team_h_difficulty": 3, "team_a_difficulty": 3},
        ]

        rows = {row["player_id"]: row for row in build_player_fixture_rows(
            players, fixtures, 1
        )}

        self.assertEqual(rows[1]["fixture_count"], 2)
        self.assertEqual(rows[1]["avg_difficulty"], 2.5)
        self.assertEqual(rows[1]["home_proportion"], 0.5)
        self.assertEqual(rows[2]["fixture_count"], 1)
        self.assertEqual(rows[3]["fixture_count"], 1)
        self.assertEqual(rows[4]["fixture_count"], 0)
        self.assertEqual(rows[4]["fixture_difficulty"], 0.0)

    def test_empty_target_gameweek_is_unknown_not_a_global_bgw(self):
        with self.assertRaises(PredictionValidationError):
            build_player_fixture_rows([{"id": 1, "team": 1}], [], 1)


class PublicationTests(unittest.TestCase):
    def test_rejects_incomplete_and_all_zero_candidates(self):
        problems = validation_problems(
            [prediction(1, xpoints=0)],
            expected_player_ids=[1, 2],
            target_gameweek=1,
        )
        self.assertTrue(any("missing 1 official player" in problem for problem in problems))
        self.assertTrue(any("positive xPoints" in problem for problem in problems))

    def test_invalid_candidate_does_not_replace_known_good_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text('{"known": "good"}', encoding="utf-8")

            with self.assertRaises(PredictionValidationError):
                write_predictions(
                    path,
                    [prediction(1, xpoints=0)],
                    expected_player_ids=[1],
                    target_gameweek=1,
                )

            self.assertEqual(json.loads(path.read_text()), {"known": "good"})

    def test_valid_candidate_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            rows = [prediction(1), prediction(2)]
            write_predictions(path, rows, [1, 2], 1)
            self.assertEqual(json.loads(path.read_text()), rows)


if __name__ == "__main__":
    unittest.main()


class ProjectionBoundsTests(unittest.TestCase):
    """Negative expected points are legitimate in FPL (issue #13). Only values
    outside the documented hard bounds may block publication."""

    def setUp(self):
        from prediction_safety import XPOINTS_HARD_MAX, XPOINTS_HARD_MIN
        self.hard_min = XPOINTS_HARD_MIN
        self.hard_max = XPOINTS_HARD_MAX

    def test_small_negative_projection_is_accepted_and_preserved(self):
        rows = [prediction(1, xpoints=-0.4), prediction(2, xpoints=4.0)]
        self.assertEqual(validation_problems(rows, [1, 2], 1), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            write_predictions(path, rows, [1, 2], 1)
            written = {r["player_id"]: r["xPoints"] for r in json.loads(path.read_text())}
        self.assertEqual(written[1], -0.4)

    def test_values_outside_hard_bounds_block_publication(self):
        too_low = [prediction(1, xpoints=self.hard_min - 1), prediction(2)]
        too_high = [prediction(1, xpoints=self.hard_max + 1), prediction(2)]
        for rows in (too_low, too_high):
            problems = validation_problems(rows, [1, 2], 1)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("out-of-bounds", problems[0])

    def test_anomalies_are_reported_but_do_not_block(self):
        from prediction_safety import XPOINTS_WARN_MAX, anomaly_warnings
        rows = [prediction(1, xpoints=-0.5), prediction(2, xpoints=XPOINTS_WARN_MAX + 1)]
        self.assertEqual(validation_problems(rows, [1, 2], 1), [])
        warnings = anomaly_warnings(rows)
        self.assertEqual(len(warnings), 2, warnings)
        self.assertIn("negative", warnings[0])
        self.assertIn("ceiling", warnings[1])
        with tempfile.TemporaryDirectory() as tmp:
            returned = write_predictions(Path(tmp) / "p.json", rows, [1, 2], 1)
        self.assertEqual(returned, warnings)


class DeadlineFreezeTests(unittest.TestCase):
    """The per-gameweek snapshot may only be rewritten before its deadline."""

    def test_archive_allowed_only_before_deadline(self):
        import datetime as dt
        from baseline import archive_allowed
        deadline = "2026-09-05T17:30:00Z"
        before = dt.datetime(2026, 9, 5, 17, 29, tzinfo=dt.timezone.utc)
        after = dt.datetime(2026, 9, 5, 17, 31, tzinfo=dt.timezone.utc)
        self.assertTrue(archive_allowed(before, deadline))
        self.assertFalse(archive_allowed(after, deadline))
        self.assertTrue(archive_allowed(after, None))

    def test_content_signature_ignores_run_timestamp(self):
        from baseline import content_signature
        a = [prediction(1) | {"generated_at": "2026-09-01T10:00:00Z"}]
        b = [prediction(1) | {"generated_at": "2026-09-01T11:00:00Z"}]
        c = [prediction(1, xpoints=4.1) | {"generated_at": "2026-09-01T11:00:00Z"}]
        self.assertEqual(content_signature(a), content_signature(b))
        self.assertNotEqual(content_signature(a), content_signature(c))
