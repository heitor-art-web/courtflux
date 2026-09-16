from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.p2_real_video_probe import DEFAULT_NEUTRAL_MATCH_CONFIG, parse_args


class P2RealVideoProbeContractTest(unittest.TestCase):
    def test_probe_defaults_to_neutral_match_config(self) -> None:
        with patch.object(sys, "argv", ["p2_real_video_probe.py"]):
            args = parse_args()
        self.assertEqual(args.match_config, DEFAULT_NEUTRAL_MATCH_CONFIG)

    def test_neutral_match_config_has_no_team_identity(self) -> None:
        payload = json.loads(Path(DEFAULT_NEUTRAL_MATCH_CONFIG).read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata") or {})
        self.assertIsNone(metadata.get("home_team"))
        self.assertIsNone(metadata.get("away_team"))
        self.assertEqual(payload.get("teams"), {})

    def test_known_team_control_requires_explicit_override(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["p2_real_video_probe.py", "--match-config", "data/match_config.json"],
        ):
            args = parse_args()
        self.assertEqual(args.match_config, "data/match_config.json")


if __name__ == "__main__":
    unittest.main()
