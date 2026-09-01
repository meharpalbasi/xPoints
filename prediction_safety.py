"""Shared fixture and publication safety for baseline and model predictions."""

import json
import math
import os
from collections import defaultdict
from pathlib import Path


POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Documented bounds for one player's gameweek projection.
#
# FPL's own ep_next can be slightly negative (a player expected to concede or
# be carded with no attacking return), so negatives inside the hard bounds are
# LEGITIMATE and must be preserved. Rejecting every negative cost three
# publication runs on 29-30 Aug 2026 (issue #13). Values outside the hard
# bounds mean a broken upstream or a units bug and block publication; values
# past the warn bounds are published but surfaced as anomalies.
XPOINTS_HARD_MIN = -5.0
XPOINTS_HARD_MAX = 40.0  # a double-gameweek haul projection can exceed 20
XPOINTS_WARN_MIN = -2.0
XPOINTS_WARN_MAX = 20.0

REQUIRED_FIELDS = {
    "player_id",
    "player_code",
    "team",
    "element_type",
    "position",
    "xPoints",
    "fixture_count",
    "generated_at",
    "season",
    "gameweek",
    "source",
    "model_version",
}


class PredictionValidationError(ValueError):
    """Raised when a candidate must not replace the production feed."""


def season_from_events(events):
    if not events:
        raise PredictionValidationError("cannot derive season without events")
    first = min(events, key=lambda event: event["id"])
    year = int(first["deadline_time"][:4])
    return f"{year}/{str(year + 1)[-2:]}"


def build_player_fixture_rows(players, fixtures, target_gameweek):
    """Build per-player fixture features from the validated global feed.

    A missing fixture for one team is a confirmed BGW only after the global
    fixture request succeeded. A completely empty target GW is treated as
    unknown upstream state and blocks model publication.
    """
    if target_gameweek is None:
        raise PredictionValidationError("target gameweek is missing")

    target_fixtures = [
        fixture for fixture in fixtures
        if fixture.get("event") == target_gameweek
    ]
    if not target_fixtures:
        raise PredictionValidationError(
            f"global fixture feed has no fixtures for GW{target_gameweek}"
        )

    known_teams = {int(player["team"]) for player in players}
    stats = defaultdict(lambda: {"count": 0, "difficulty": [], "home": 0})

    for fixture in target_fixtures:
        home = int(fixture["team_h"])
        away = int(fixture["team_a"])
        if home not in known_teams or away not in known_teams:
            raise PredictionValidationError(
                f"fixture references unknown team: {home} vs {away}"
            )

        stats[home]["count"] += 1
        stats[home]["home"] += 1
        stats[home]["difficulty"].append(
            int(fixture.get("team_h_difficulty") or 3)
        )
        stats[away]["count"] += 1
        stats[away]["difficulty"].append(
            int(fixture.get("team_a_difficulty") or 3)
        )

    rows = []
    for player in players:
        team = int(player["team"])
        team_stats = stats.get(
            team,
            {"count": 0, "difficulty": [], "home": 0},
        )
        count = team_stats["count"]
        average_difficulty = (
            sum(team_stats["difficulty"]) / count if count else 0.0
        )
        home_proportion = team_stats["home"] / count if count else 0.0
        rows.append({
            "player_id": int(player["id"]),
            "fixture_count": count,
            "avg_difficulty": round(average_difficulty, 2),
            "fixture_difficulty": round(average_difficulty, 2),
            "home_proportion": round(home_proportion, 2),
            "home_dummy": round(home_proportion, 2),
        })

    return rows


def validation_problems(rows, expected_player_ids, target_gameweek):
    problems = []
    expected_ids = {int(player_id) for player_id in expected_player_ids}

    if not rows:
        return ["prediction output is empty"]

    row_ids = [row.get("player_id") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        problems.append("duplicate player IDs in prediction output")

    actual_ids = {int(player_id) for player_id in row_ids if player_id is not None}
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing:
        problems.append(f"missing {len(missing)} official player IDs: {missing[:5]}")
    if unexpected:
        problems.append(f"found {len(unexpected)} unknown player IDs: {unexpected[:5]}")

    sources = set()
    nonzero = 0
    for index, row in enumerate(rows):
        missing_fields = sorted(REQUIRED_FIELDS - row.keys())
        if missing_fields:
            problems.append(f"row {index} missing fields: {missing_fields}")
            continue

        element_type = row["element_type"]
        if element_type not in POSITIONS:
            problems.append(f"player {row['player_id']} has invalid element_type")
        elif row["position"] != POSITIONS[element_type]:
            problems.append(f"player {row['player_id']} has inconsistent position")

        try:
            xpoints = float(row["xPoints"])
        except (TypeError, ValueError):
            problems.append(f"player {row['player_id']} has non-numeric xPoints")
        else:
            if (
                not math.isfinite(xpoints)
                or xpoints < XPOINTS_HARD_MIN
                or xpoints > XPOINTS_HARD_MAX
            ):
                problems.append(
                    f"player {row['player_id']} has out-of-bounds xPoints {xpoints!r} "
                    f"(hard bounds {XPOINTS_HARD_MIN} to {XPOINTS_HARD_MAX})"
                )
            elif xpoints > 0:
                nonzero += 1

        fixture_count = row["fixture_count"]
        if not isinstance(fixture_count, (int, float)) or fixture_count < 0:
            problems.append(f"player {row['player_id']} has invalid fixture_count")
        if row["gameweek"] != target_gameweek:
            problems.append(f"player {row['player_id']} targets the wrong gameweek")
        if not row["season"] or not row["generated_at"] or not row["model_version"]:
            problems.append(f"player {row['player_id']} has incomplete provenance")
        sources.add(row["source"])

    minimum_nonzero = min(100, max(1, math.ceil(len(expected_ids) * 0.2)))
    if nonzero < minimum_nonzero:
        problems.append(
            f"only {nonzero} players have positive xPoints; "
            f"minimum is {minimum_nonzero}"
        )
    if len(sources) != 1 or None in sources or "unknown" in sources:
        problems.append(f"prediction source is inconsistent: {sorted(map(str, sources))}")

    return problems


def anomaly_warnings(rows):
    """Non-blocking anomalies worth a human glance. Publication proceeds."""
    negatives, below_warn, above_warn = [], [], []
    for row in rows:
        try:
            xpoints = float(row.get("xPoints"))
        except (TypeError, ValueError):
            continue
        if xpoints < 0:
            negatives.append(row.get("player_id"))
        if xpoints < XPOINTS_WARN_MIN:
            below_warn.append(row.get("player_id"))
        if xpoints > XPOINTS_WARN_MAX:
            above_warn.append(row.get("player_id"))

    warnings = []
    if negatives:
        warnings.append(
            f"{len(negatives)} players have negative xPoints (kept): {negatives[:5]}"
        )
    if below_warn:
        warnings.append(
            f"{len(below_warn)} players below warn floor {XPOINTS_WARN_MIN}: {below_warn[:5]}"
        )
    if above_warn:
        warnings.append(
            f"{len(above_warn)} players above warn ceiling {XPOINTS_WARN_MAX}: {above_warn[:5]}"
        )
    return warnings


def write_predictions(path, rows, expected_player_ids, target_gameweek):
    """Validate and atomically replace the production artifact.

    Returns the list of non-blocking anomaly warnings (possibly empty).
    """
    problems = validation_problems(rows, expected_player_ids, target_gameweek)
    if problems:
        raise PredictionValidationError("; ".join(problems))
    warnings = anomaly_warnings(rows)

    destination = Path(path)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(rows, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return warnings
