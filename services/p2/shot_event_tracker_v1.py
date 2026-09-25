from services.p2.shot_event_tracker_v1_3 import ShotEventTrackerV13

# Canonical import remains stable for engine/callers while the V6 validation branch
# advances the implementation behind it. Frozen V1/V1.1/V1.2 remain untouched.
ShotEventTrackerV1 = ShotEventTrackerV13

__all__ = ["ShotEventTrackerV1"]
