# Outcome policy revision note

The canonical `services.p2.shot_event_tracker_v1.ShotEventTrackerV1` now layers a Gate-B outcome-policy revision over the frozen original implementation preserved in `shot_event_tracker_base_v1.py`.

The revision intentionally does not fit `post_gap_fall_ratio` thresholds. A pending bridge resolves MADE only after a later direct observation confirms continued relative descent with coherent rim geometry and horizontal proximity. Ambiguous rim exits after observation gaps are deferred briefly; without later confirmation they remain UNKNOWN.
