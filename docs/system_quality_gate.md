# CourtFlux P0 system quality gate

P0 now has two distinct concepts that must not be conflated:

1. **P0 Foundation Complete** — the vision/evaluation foundation is safe, measurable, modular, and sufficient to start P1.
2. **Ball Quality Maturity Gate** — the ball subsystem continues to be measured against explicit engineering targets while P1 progresses in parallel.

The product North Star remains approximately **90%+ correct usable ball-state coverage** and event precision/recall around 90% once possession/events are implemented. The current ball engineering target is lower and is **not a P1 entry blocker**.

## Frozen baseline

Do not change these while validating the ball maturity gate unless an experiment explicitly says otherwise:

```bash
export BASKETBALL_BALL_REACQUIRE=0
export BASKETBALL_BALL_DETECTOR_BACKEND=yolo
```

Detector benchmark contract:

- model: `weights/model_v1.0.0.pt`;
- confidence: `0.10`;
- image size: `960`;
- official point match: candidate/output center `<=40 px` from frozen GT;
- V1-V5 remain the frozen generalization set;
- hard reacquisition remains OFF by default.

The central evidence from the completed experiments is:

- Smart YOLO v1 direct visible candidate recall: **56.54% micro**;
- oracle-validated linear bridging to gap 8: **72.81% micro**;
- short temporal continuity helps materially, but cannot rescue the detector alone;
- hard state-aware reacquisition previously produced unsafe promotions, so the working rule is: **interpolation is cheap; reacquisition is dangerous**.

## P0 foundation workstreams

### P0.1 — detector candidate quality

Measure whether the correct ball exists in the candidate set before association.

Primary diagnostics:

- visible candidate frame recall (`<=40 px`);
- candidate-instance precision;
- candidates per labelled frame;
- not-visible candidate-frame rate;
- per-video recall/generalization.

The current direct detector target is supporting evidence, not a standalone blocker. A practical supporting target is approximately **65%+ visible candidate recall**.

### P0.2 — association / tracker

Measure whether the tracker selects the correct candidate when a correct candidate is available.

Primary diagnostics:

- direct correct output coverage;
- direct output precision;
- association-opportunity miss rate;
- rejected candidate counts and gate behaviour;
- catastrophic location errors;
- reacquisition promotions.

Safety properties covered by tests:

- an implausible high-confidence candidate cannot beat an in-gate candidate;
- `BASKETBALL_BALL_REACQUIRE=0` cannot promote an out-of-gate challenger;
- a long reset/reinitialization is not counted as a safe short interpolation.

#### OC-SORT hypothesis evidence

On the frozen Smart-v1 candidate traces, `OC-SORT + CIoU` preserved a correct confirmed hypothesis on **41.58%** of visible frames versus **22.86%** direct correct coverage from the current single-track association. It is therefore useful as a **hypothesis tracker**, not as a direct ball-output policy.

The integrated experimental layer is opt-in:

```bash
export BASKETBALL_BALL_HYPOTHESIS_TRACKER=1
```

It exposes an explicit lifecycle:

```text
TENTATIVE -> TRACKED -> LOST -> REMOVED
```

`LOST`, `TENTATIVE`, and `REMOVED` hypotheses are internal state only and must never emit a concrete ball location.

### P0.3 — trajectory / predicted hold

Prediction is a bounded bridge between direct observations; it must not become a recursive detector substitute.

Primary diagnostics:

- correct predicted visible frames;
- wrong predicted visible frames;
- predicted-output precision;
- coverage lift over correct direct association;
- hold-age distribution;
- unsupported-gap distribution.

Required invariant: **predicted holds never update the direct observation anchor**.

### P0.4 — robustness / ball state

The system must represent uncertainty instead of forcing a location.

Primary diagnostics:

- not-visible false-state rate;
- direct false locations during not-visible GT;
- predicted false locations during not-visible GT;
- catastrophic location-error events;
- hard reacquisition events;
- missing/unsupported visible gaps.

A not-visible GT frame is correct only when no concrete ball location is emitted.

### P0.5 — instrumentation / evaluation

`evaluation/p0_quality_gate.py` evaluates P0.1-P0.5 from the same production `frames.jsonl` and frozen point GT.

It reports per run plus micro/macro/worst-video aggregate metrics and emits an explicit `PASS` or `FAIL` for the **ball maturity gate**.

## Ball Quality Maturity Gate

### Engineering targets

| Metric | Gate |
| --- | ---: |
| correct usable visible ball-state coverage, micro | `>=80%` |
| correct usable visible ball-state coverage, macro | `>=78%` |
| worst-video correct usable visible coverage | `>=70%` |
| concrete output precision | `>=80%` |
| not-visible false-state rate | `<=5%` provisional ceiling |
| unsupported visible gap P90 | `<8 frames` |
| catastrophic error events (`>160 px`) | `0` |

The approximately 90% product target is a later North Star.

These checks remain active and measurable, but **they no longer block entry into P1**. They track the maturity of the ball subsystem in parallel with player/game-state/event development.

### Supporting checks

- direct visible candidate recall: `>=65%` supporting target;
- no camera/video collapse across V1-V5;
- hard reacquisition should remain zero in the frozen baseline;
- throughput is reported but is not currently a blocker.

## Frozen V1-V5 GT pairs

| Video | Production source window | GT run |
| --- | --- | --- |
| V1 | `data/hILf-ySowIo.mp4`, 10-40s | `20260907_121516_009508` |
| V2 | `data/ULUpcQXsRII.mp4`, 0-30s | `20260907_121948_601883` |
| V3 | `data/Vr4105n-cPM.mp4`, 0-30s | `20260907_122315_943004` |
| V4 | `data/WMWTbS8J2ec.mp4`, 0-30s | `20260907_122515_042534` |
| V5 | `data/C-JjAlX2Dyc.mp4`, 10-40s | `20260907_122807_977379` |

Frozen Smart YOLO v1 production runs:

```text
V1 20260908_130530_855844
V2 20260908_130718_769477
V3 20260908_131032_611540
V4 20260908_131204_724212
V5 20260908_131609_006117
```

## Frozen historical production baseline

The current historical production-state baseline is:

- usable visible coverage micro: **25.49%**;
- usable visible coverage macro: **28.19%**;
- worst-video usable coverage: **4.64%**;
- output precision: **19.47%**;
- not-visible false-state: **88.51%**;
- unsupported visible gap P90: **40 frames**;
- catastrophic error events: **129**;
- direct candidate recall: **56.54%** supporting evidence.

These values are retained so future improvements can be measured against the P0 exit snapshot. They are not criteria for `P0 FOUNDATION COMPLETE`.

## Court geometry / homography status

Court calibration is part of the established P0 foundation and is available to P1 as court context.

Implemented:

- reusable court geometry primitives;
- versioned Roboflow basketball-court v19 33-landmark -> FIBA mapping;
- Serverless inference through the lightweight SDK;
- anchors / RANSAC inliers / reprojection RMSE diagnostics;
- periodic keypoint refresh and optical-flow hold;
- player feet projected to `court_xy`;
- ball projection labelled planar/apparent.

Real v19 calibration was validated with:

```text
anchors       12
inliers        6
RMSE        1.5261 px
valid        True
```

Ball court coordinates remain soft context only because an airborne ball does not lie on the ground plane.

## Court/player context research

Held-out LOVO comparison at the 80% training-precision target:

| Profile | Held-out AUC | Output precision | Visible coverage | NV false |
| --- | ---: | ---: | ---: | ---: |
| temporal-only | 0.7656 | 73.09% | 37.72% | 15.63% |
| temporal + court/player context | 0.7232 | 72.25% | 39.10% | 16.09% |

Generic court/player context was therefore **not promoted** into the production Ball Identity policy at the P0 boundary.

## P0 exit boundary

CourtFlux declares **P0 FOUNDATION COMPLETE** because the following foundation now exists:

- frozen, reproducible evaluation contract;
- safe ball hypothesis lifecycle;
- explicit uncertainty / abstention semantics;
- bounded temporal prediction;
- court geometry / homography;
- player `court_xy` projection;
- repeatable diagnostics and failure attribution;
- known ball-quality limitations documented and measurable.

P1 may now proceed using player tracks, team assignments, court geometry, temporal state and ball evidence when reliable.

P1 must tolerate missing/unknown ball state and must not force events from low-confidence ball output.

## Validation order going forward

1. Preserve the frozen V1-V5 benchmark contract.
2. Run deterministic tests before ball-quality experiments.
3. Measure ball changes against the frozen historical baseline.
4. Keep hard reacquisition OFF unless a future experiment proves it safe.
5. Continue the Ball Quality Maturity Gate in parallel.
6. Do **not** use Ball Quality Gate failure to block P1 player/game-state work.

For the frozen P0 exit snapshot, see `docs/p0_foundation_snapshot.md`.
