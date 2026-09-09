# CourtFlux P0 system quality gate

P0 is a **system ball-state gate**, not a standalone detector-recall gate.

The product North Star remains approximately **90%+ correct usable ball-state coverage** and event precision/recall around 90% once possession/events are implemented. P0 uses a lower engineering gate so P1 is not blocked on an unrealistic direct-detector requirement.

## Frozen baseline

Do not change these while validating the P0 gate unless an experiment explicitly says otherwise:

```bash
export BASKETBALL_BALL_REACQUIRE=0
export BASKETBALL_BALL_DETECTOR_BACKEND=yolo
```

Detector benchmark contract:

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

## P0 workstreams

### P0.1 — detector candidate quality

Measure whether the correct ball exists in the candidate set before association.

Primary diagnostics:

- visible candidate frame recall (`<=40 px`);
- candidate-instance precision;
- candidates per labelled frame;
- not-visible candidate-frame rate;
- per-video recall/generalization.

The current direct detector target is **supporting evidence**, not a standalone blocker. A practical supporting target is approximately **65%+ visible candidate recall**, with 70% desirable if it can be achieved without a large noise increase.

### P0.2 — association / tracker

Measure whether the tracker selects the correct candidate when a correct candidate is available.

Primary diagnostics:

- direct correct output coverage;
- direct output precision;
- association-opportunity miss rate: a correct candidate existed but the emitted state was wrong/missing;
- rejected candidate counts and gate behaviour;
- catastrophic location errors;
- reacquisition promotions.

Safety properties covered by tests:

- an implausible high-confidence candidate cannot beat an in-gate candidate;
- `BASKETBALL_BALL_REACQUIRE=0` cannot promote an out-of-gate challenger;
- a long reset/reinitialization is not counted as a safe short interpolation.

Court homography may become a soft association feature only after its semantics and real-video calibration are independently validated. It is not allowed to change the frozen detector benchmark by default.

### P0.3 — trajectory / predicted hold

Prediction is a bounded bridge between direct observations; it must not become a recursive detector substitute.

Primary diagnostics:

- correct predicted visible frames;
- wrong predicted visible frames;
- predicted-output precision;
- coverage lift over correct direct association;
- hold-age distribution;
- unsupported-gap distribution.

Required invariant: **predicted holds never update the direct observation anchor**. This prevents prediction drift from recursively extending itself.

The oracle short-gap curve remains evidence about what temporal modelling could recover; it is not production tracker performance.

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

It reports per run plus micro/macro/worst-video aggregate metrics and emits an explicit `PASS` or `FAIL`.

## P0 engineering gate

### Blocking checks

| Metric | Gate |
| --- | ---: |
| correct usable visible ball-state coverage, micro | `>=80%` |
| correct usable visible ball-state coverage, macro | `>=78%` |
| worst-video correct usable visible coverage | `>=70%` |
| concrete output precision | `>=90%` |
| not-visible false-state rate | `<=5%` provisional ceiling |
| unsupported visible gap P90 | `<8 frames` |
| catastrophic error events (`>160 px`) | `0` |

The 5% not-visible ceiling is deliberately conservative and can be tightened after the first full V1-V5 system-gate run.

### Supporting checks

- direct visible candidate recall: `>=65%` is the current supporting target;
- no camera/video collapse across V1-V5;
- hard reacquisition should remain zero in the frozen baseline;
- throughput is reported but is not a blocker until an explicit offline throughput threshold is frozen.

The detector-supporting check does **not** turn the overall P0 gate into FAIL if the system passes the blocking checks.

## Frozen V1-V5 GT pairs

Current frozen GT:

| Video | Production source window | GT run |
| --- | --- | --- |
| V1 | `data/hILf-ySowIo.mp4`, 10-40s | `20260907_121516_009508` |
| V2 | `data/ULUpcQXsRII.mp4`, 0-30s | `20260907_121948_601883` |
| V3 | `data/Vr4105n-cPM.mp4`, 0-30s | `20260907_122315_943004` |
| V4 | `data/WMWTbS8J2ec.mp4`, 0-30s | `20260907_122515_042534` |
| V5 | `data/C-JjAlX2Dyc.mp4`, 10-40s | `20260907_122807_977379` |

The current Smart YOLO v1 production runs are:

```text
V1 20260908_130530_855844
V2 20260908_130718_769477
V3 20260908_131032_611540
V4 20260908_131204_724212
V5 20260908_131609_006117
```

## Run the end-to-end P0 gate

Use the frozen Smart v1 runs as the first baseline of the new metric contract:

```bash
python3 evaluation/p0_quality_gate.py \
  --pair 20260908_130530_855844=20260907_121516_009508 \
  --pair 20260908_130718_769477=20260907_121948_601883 \
  --pair 20260908_131032_611540=20260907_122315_943004 \
  --pair 20260908_131204_724212=20260907_122515_042534 \
  --pair 20260908_131609_006117=20260907_122807_977379 \
  --gt-radius-px 40 \
  --catastrophic-radius-px 160 \
  --output evaluation/p0_quality_gate_smart_v1
```

The result is written to:

```text
evaluation/p0_quality_gate_smart_v1/summary.json
```

Do not infer PASS from internal `effective_trajectory_frame_rate`; the gate requires GT-correct locations.

## Oracle short-gap evidence

The separate track-coverage evaluator intentionally measures an oracle upper bound:

```bash
python3 evaluation/ball_track_coverage_eval.py \
  --pair 20260908_130530_855844=20260907_121516_009508 \
  --pair 20260908_130718_769477=20260907_121948_601883 \
  --pair 20260908_131032_611540=20260907_122315_943004 \
  --pair 20260908_131204_724212=20260907_122515_042534 \
  --pair 20260908_131609_006117=20260907_122807_977379 \
  --gt-radius-px 40 \
  --gap 1 --gap 2 --gap 3 --gap 5 --gap 8 \
  --output evaluation/p0_track_coverage_smart_v1
```

Linear bridges count only when every interpolated point is GT-valid. These numbers must never be reported as production tracker quality.

## Court geometry / homography status

Court calibration is auxiliary P0.2 evidence, not a P0 blocker today.

### Geometry implementation

`services/vision/court_geometry.py` estimates image-pixel -> court-metre homography by running RANSAC in **image pixel space**. OpenCV applies the RANSAC threshold in the destination coordinate system, so CourtFlux estimates court->image first and then inverts the matrix. This keeps `ransac_reprojection_px` and `reprojection_rmse_px` genuinely pixel based.

Synthetic tests cover:

- exact projective recovery;
- rejection of a large pixel outlier;
- degenerate geometry rejection.

### Landmark semantics

The downloaded `basketball-court-detection-2` v19 COCO dataset contains 33 keypoints:

```text
01 02 04 05 07 08 09 10 11 12 13 14 15 16 17 19 21 23 25 26 27 28 29 30 31 32 33 34 35 37 38 40 41
```

The dataset does not include a verified `keypoint ID -> physical court coordinate` map. CourtFlux therefore **must not guess** a metric mapping for v19.

The legacy `FIBA_COURT_KEYPOINTS_14` remains a backward-compatible mapping only. It is not proof that Roboflow model version `/14` contains 14 landmarks, and it must not be applied to the v19 33-point schema.

### Hosted inference

Hosted/Serverless Roboflow inference returned 401 with both private and publishable authentication attempts in this environment. Dataset download works with a valid private key. P0 must therefore not depend on hosted court inference.

For dataset download use a separate private-key environment variable:

```bash
export ROBOFLOW_PRIVATE_API_KEY='...'
python3 evaluation/download_roboflow_benchmarks.py --only court
```

Never commit API keys.

## External detector evidence

The public Roboflow datasets remain useful for external/domain evidence even though hosted inference is not a P0 dependency.

Basketball detection dataset default:

```text
roboflow-jvuqo/basketball-player-detection-3-ycjdo v18
```

Court keypoint dataset default:

```text
roboflow-jvuqo/basketball-court-detection-2 v19
```

External image-level recall is not directly comparable to V1-V5 temporal tracking recall. Use it to distinguish general detector weakness from CourtFlux-domain weakness.

## Tests

Run the fast deterministic suite before any new V1-V5 analysis:

```bash
python3 evaluation/ball_candidate_backend_test.py
python3 evaluation/ball_candidate_pipeline_test.py
python3 evaluation/smart_candidate_filter_test.py
python3 evaluation/smart_ball_backend_integration_test.py
python3 evaluation/system_quality_gate_test.py
python3 evaluation/p0_quality_gate_test.py
```

`system_quality_gate_test.py` covers geometry, conservative association, bounded/non-recursive predicted hold, frozen reacquisition safety and oracle bridging.

`p0_quality_gate_test.py` covers the P0.1-P0.5 metric contract and verifies that the supporting detector target does not incorrectly block a system-level PASS.

## Validation order

1. Run the deterministic tests above.
2. Run `p0_quality_gate.py` on the existing frozen Smart v1 V1-V5 runs.
3. Inspect the exact failing checks; do not tune everything at once.
4. Improve the dominant blocker in P0.1/P0.2/P0.3/P0.4.
5. Re-run the same V1-V5 gate.
6. Only declare **P0 COMPLETE** when every blocking check passes on the frozen five-video set.
7. Then move to P1 possession/events.
