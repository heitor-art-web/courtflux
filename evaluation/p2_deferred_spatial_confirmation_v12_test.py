from __future__ import annotations

import unittest

from services.p2.shot_event_tracker_v1_2 import ShotEventTrackerV12


class DeferredSpatialConfirmationV12Test(unittest.TestCase):
    def test_repeated_inside_mouth_confirms(self) -> None:
        reason = ShotEventTrackerV12._deferred_confirmation_reason(
            {
                "rim_observations": 3,
                "near_x_observations": 2,
                "inside_mouth_observations": 2,
                "incoherent_rim_steps": 0,
            }
        )
        self.assertEqual(
            reason,
            "deferred_rim_exit_repeated_inside_mouth_confirmed",
        )

    def test_single_inside_mouth_does_not_confirm(self) -> None:
        reason = ShotEventTrackerV12._deferred_confirmation_reason(
            {
                "rim_observations": 5,
                "near_x_observations": 1,
                "inside_mouth_observations": 1,
                "incoherent_rim_steps": 0,
            }
        )
        self.assertIsNone(reason)

    def test_sustained_near_x_confirms(self) -> None:
        reason = ShotEventTrackerV12._deferred_confirmation_reason(
            {
                "rim_observations": 9,
                "near_x_observations": 9,
                "inside_mouth_observations": 0,
                "incoherent_rim_steps": 0,
            }
        )
        self.assertEqual(
            reason,
            "deferred_rim_exit_persistent_near_x_confirmed",
        )

    def test_short_near_x_run_does_not_confirm(self) -> None:
        reason = ShotEventTrackerV12._deferred_confirmation_reason(
            {
                "rim_observations": 4,
                "near_x_observations": 4,
                "inside_mouth_observations": 0,
                "incoherent_rim_steps": 0,
            }
        )
        self.assertIsNone(reason)

    def test_partial_near_x_does_not_confirm(self) -> None:
        reason = ShotEventTrackerV12._deferred_confirmation_reason(
            {
                "rim_observations": 5,
                "near_x_observations": 4,
                "inside_mouth_observations": 0,
                "incoherent_rim_steps": 0,
            }
        )
        self.assertIsNone(reason)

    def test_incoherent_rim_blocks_both_confirmation_paths(self) -> None:
        for diagnostics in (
            {
                "rim_observations": 3,
                "near_x_observations": 2,
                "inside_mouth_observations": 2,
                "incoherent_rim_steps": 1,
            },
            {
                "rim_observations": 9,
                "near_x_observations": 9,
                "inside_mouth_observations": 0,
                "incoherent_rim_steps": 1,
            },
        ):
            with self.subTest(diagnostics=diagnostics):
                self.assertIsNone(
                    ShotEventTrackerV12._deferred_confirmation_reason(diagnostics)
                )


if __name__ == "__main__":
    unittest.main()
