from services.p2.shot_event_tracker_v1_2 import ShotEventTrackerV12

# Canonical import remains stable for engine/callers while the implementation
# revision advances behind it.
ShotEventTrackerV1 = ShotEventTrackerV12

__all__ = ["ShotEventTrackerV1"]
