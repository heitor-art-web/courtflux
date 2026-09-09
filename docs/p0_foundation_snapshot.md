# CourtFlux P0 Foundation Snapshot

Status: **P0 FOUNDATION COMPLETE**

Date: 2026-09-09

This snapshot freezes the technical foundation that CourtFlux will carry into P1. It does **not** mean that basketball detection/tracking is solved, nor that the ball-quality maturity gate has passed.

## Product boundary

P0 is complete when the vision foundation is safe, measurable, modular, and capable of supporting higher-level basketball analytics. P1 may therefore proceed while ball quality continues to improve in parallel.

The ball subsystem is classified as:

- **experimental / confidence-aware**;
- useful as supporting evidence for possession and events;
- allowed to abstain / emit unknown state;
- **not a blocking dependency for P1**.

The system-level ball quality gate remains a separate product-maturity milestone.

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

## Current production ball-state baseline

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

Offline evidence on the frozen Smart-v1 candidate traces:

- confirmed true-hypothesis recall: **41.58%**
- candidate-to-confirmed-truth efficiency: **73.54%**
- confirmed instance precision: **17.90%**
- not-visible confirmed frames: **71.88%**

Interpretation: OC-SORT is valuable for preserving hypotheses, but confirmed tracks must **not** be emitted directly as the basketball.

## Safe lifecycle

The experimental ball-hypothesis layer exposes:

```text
TENTATIVE -> TRACKED -> LOST -> REMOVED
```

Safety invariants:

- `LOST` is internal recovery state and never emits a concrete ball location;
- `TENTATIVE` never emits a concrete ball location;
- `REMOVED` never emits a concrete ball location;
- hard reacquisition remains OFF by default;
- concrete hypothesis output is opt-in and identity-gated.

## Temporal Ball Identity evidence

Temporal/detector-only multivariate LOVO research showed that useful identity signal exists, but calibration is not robust enough across all cameras to serve as a standalone product output policy.

The later context-enabled LOVO benchmark, calibrated to the P0 engineering precision target, reported:

### Temporal-only, target 80% precision on training folds

- held-out AUC macro: **0.7656**
- output precision micro: **73.09%**
- visible coverage micro: **37.72%**
- not-visible false-state micro: **15.63%**

### Temporal + court/player context

- held-out AUC macro: **0.7232**
- output precision micro: **72.25%**
- visible coverage micro: **39.10%**
- not-visible false-state micro: **16.09%**

Context delta versus temporal-only:

- AUC: **-0.0424**
- precision: **-0.84 pp**
- visible coverage: **+1.38 pp**
- not-visible false-state: **+0.46 pp**

Decision: generic court/player context is retained as infrastructure and diagnostics but is **not promoted into the production ball identity policy** at the P0 boundary.

## Court geometry / homography foundation

CourtFlux now has:

- versioned Roboflow basketball-court v19 / 33-landmark -> FIBA mapping;
- Serverless Roboflow keypoint inference through the lightweight SDK;
- image-pixel -> court-metre homography;
- anchors, RANSAC inliers, reprojection RMSE and calibration diagnostics;
- periodic keypoint refresh with optical-flow hold between refreshes;
- player foot projection to `court_xy`;
- explicitly planar/apparent ball court projection.

A real V3 frame using `basketball-court-detection-2/19` produced:

- source: `direct_keypoints`
- backend: `serverless`
- anchors: **12**
- inliers: **6**
- reprojection RMSE: **1.5261 px**
- valid: **True**
- identity homography: **False**

The homography is therefore accepted as P0 infrastructure. Airborne ball projection remains soft/apparent evidence, never a hard physical constraint.

## Ball quality maturity gate

The following targets remain active, but are now a **parallel ball-quality milestone rather than a P1 entry gate**:

| Metric | Ball quality target |
| --- | ---: |
| correct usable visible coverage, micro | `>=80%` |
| correct usable visible coverage, macro | `>=78%` |
| worst-video usable visible coverage | `>=70%` |
| concrete output precision | `>=80%` |
| not-visible false-state | `<=5%` |
| unsupported visible gap P90 | `<8 frames` |
| catastrophic errors (`>160 px`) | `0` |

Approximately 90% remains a later product North Star.

## What P0 has established

P0 Foundation Complete means CourtFlux now has:

- frozen detector and GT benchmarks;
- repeatable system-level evaluation;
- safe uncertainty semantics;
- multi-hypothesis ball tracking infrastructure;
- conservative lifecycle rules;
- homography and court-coordinate infrastructure;
- evidence about what temporal and court/player context can and cannot currently do;
- a clear, measurable backlog for future ball-quality work.

It does **not** mean:

- ball tracking is solved;
- the 80% ball-quality milestone has passed;
- P1 events should force a ball location when confidence is low.

## P1 entry contract

P1 may use:

- player tracks;
- team assignments;
- court geometry and `court_xy`;
- timestamps and frame diagnostics;
- ball hypotheses / high-confidence ball evidence when available.

P1 must tolerate:

- missing or unknown ball state;
- abstention from the ball subsystem;
- ball confidence varying by camera and sequence.

The intended architecture is therefore:

```text
players + teams + court + temporal state
                 ↓
          possession / events
                 ↑
       ball when sufficiently reliable
```

Future ball-quality work continues in parallel and can improve P1 confidence without blocking P1 development.
