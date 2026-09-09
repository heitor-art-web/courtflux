# CourtFlux P0 Foundation Snapshot

Status: **P0 FOUNDATION COMPLETE**

Date: 2026-09-09

This snapshot freezes the technical foundation that CourtFlux carries into P1. It does **not** mean that basketball detection/tracking is solved, nor that the Ball Quality Maturity Gate has passed.

## Product boundary

P0 is complete because the vision foundation is safe, measurable, modular, and capable of supporting higher-level basketball analytics. P1 may therefore proceed while ball quality continues to improve in parallel.

The ball subsystem is classified as:

- **experimental / confidence-aware**;
- useful as supporting evidence for possession and events;
- allowed to abstain / emit unknown state;
- **not a blocking dependency for P1**.

The system-level Ball Quality Maturity Gate remains a separate product-quality milestone.

## Frozen ball benchmark contract

- model: `weights/model_v1.0.0.pt`
- detector confidence: `0.10`
- image size: `960`
- official GT match radius: `<=40 px`
- hard reacquisition: `BASKETBALL_BALL_REACQUIRE=0`
- detector backend: `BASKETBALL_BALL_DETECTOR_BACKEND=yolo`
- V1-V5 frozen generalization set remains unchanged

## Frozen V1-V5 ground truth

| Video | Window | GT run |
| --- | --- | --- |
| V1 | `data/hILf-ySowIo.mp4`, 10-40s | `20260907_121516_009508` |
| V2 | `data/ULUpcQXsRII.mp4`, 0-30s | `20260907_121948_601883` |
| V3 | `data/Vr4105n-cPM.mp4`, 0-30s | `20260907_122315_943004` |
| V4 | `data/WMWTbS8J2ec.mp4`, 0-30s | `20260907_122515_042534` |
| V5 | `data/C-JjAlX2Dyc.mp4`, 10-40s | `20260907_122807_977379` |

Aligned gameplay evidence: 4494 frames, of which 3189 are ball-visible and 1305 are not-visible.

## Frozen Smart-v1 detector evidence

Smart-v1 production runs:

- V1 `20260908_130530_855844`
- V2 `20260908_130718_769477`
- V3 `20260908_131032_611540`
- V4 `20260908_131204_724212`
- V5 `20260908_131609_006117`

Direct visible candidate recall:

- micro: **56.54%**
- macro: **58.74%**
- worst video: **50.69%**

Oracle short-gap coverage using GT-validated linear bridges reaches **72.81% micro at gap <=8**. This is an oracle upper bound, not production tracking quality.

## Historical production ball-state baseline

Frozen P0 system-gate baseline:

- usable visible coverage micro: **25.49%**
- usable visible coverage macro: **28.19%**
- worst-video usable coverage: **4.64%**
- output precision: **19.47%**
- not-visible false-state: **88.51%**
- unsupported visible gap P90: **40 frames**
- catastrophic error events: **129**
- direct candidate recall: **56.54%** supporting evidence

These values are retained as the historical production-state baseline. They are not the exit criteria for P0 Foundation Complete.

## Hypothesis tracking foundation

`OC-SORT + CIoU` is retained as the experimental multi-hypothesis tracker.

Frozen offline evidence:

- correct confirmed-hypothesis presence on visible frames: **41.58%**;
- current single-track direct correct coverage reference: **22.86%**;
- confirmed instance precision in the offline benchmark: **17.90%**;
- not-visible confirmed-track rate: **71.88%**.

Therefore OC-SORT is retained to preserve hypotheses, not to emit all confirmed tracks as balls.

Explicit lifecycle:

```text
TENTATIVE -> TRACKED -> LOST -> REMOVED
```

Safety invariant:

```text
LOST / TENTATIVE / REMOVED
-> never emit a concrete ball location
```

Hard reacquisition remains OFF by default.

## Temporal Ball Identity evidence

Temporal/detector features produced useful ranking signal but did not generalize strongly enough to become the sole production identity policy.

The role of the temporal arbiter at the P0 boundary is therefore:

- high-confidence supporting signal / seed;
- optional experimental identity layer;
- allowed to abstain;
- not a requirement for every frame.

## Court geometry / homography foundation

Implemented:

- reusable FIBA court geometry;
- versioned Roboflow basketball-court v19 33-landmark -> FIBA mapping;
- lightweight Serverless inference backend;
- RANSAC anchors / inliers / reprojection RMSE diagnostics;
- periodic keypoint refresh plus optical-flow hold;
- player feet -> `court_xy`;
- ball center -> explicitly planar/apparent court projection.

Real v19 validation example:

```text
model      basketball-court-detection-2/19
source     direct_keypoints
anchors    12
inliers     6
RMSE       1.5261 px
valid      True
identity H False
```

Ball court coordinates are soft context only because an airborne ball does not lie on the ground plane.

## Court/player context research

Held-out LOVO comparison at the 80% training-precision target:

| Profile | Held-out AUC | Output precision | Visible coverage | NV false |
| --- | ---: | ---: | ---: | ---: |
| temporal-only | 0.7656 | 73.09% | 37.72% | 15.63% |
| temporal + court/player context | 0.7232 | 72.25% | 39.10% | 16.09% |

Engineering delta from generic context:

- AUC: **-0.0424**;
- output precision: **-0.84 pp**;
- visible coverage: **+1.38 pp**;
- not-visible false-state: **+0.46 pp**.

Conclusion: generic court/player context is not promoted into the production Ball Identity policy at the P0 boundary. The underlying court/player features remain useful infrastructure for P1.

## Ball Quality Maturity Gate

The following targets remain active, measurable, and parallel to P1:

| Metric | Target |
| --- | ---: |
| usable visible coverage micro | `>=80%` |
| usable visible coverage macro | `>=78%` |
| worst-video usable coverage | `>=70%` |
| concrete output precision | `>=80%` |
| not-visible false-state | `<=5%` |
| unsupported visible gap P90 | `<8 frames` |
| catastrophic errors (`>160 px`) | `0` |

Approximately **90%** remains the later product North Star.

These targets do **not** block P1 entry.

## P1 entry contract

P1 may build on:

- player tracks;
- team assignments;
- court geometry / `court_xy`;
- temporal state;
- timestamps and diagnostics;
- ball evidence only when sufficiently reliable.

P1 must tolerate missing / unknown ball state and must not force possession or event output from low-confidence ball evidence.

Intended architecture:

```text
players + teams + court + temporal state
                 ↓
          possession / events
                 ↑
       ball when sufficiently reliable
```

## Next phase

P1 starts with:

```text
P1.1 — Player Tracking Quality
```

Primary focus:

- track fragmentation;
- ID switches;
- track duration;
- players per gameplay frame;
- team-label stability;
- court-position continuity;
- per-video robustness.

This snapshot is the reference point for future comparisons against the end of the P0 foundation phase.
