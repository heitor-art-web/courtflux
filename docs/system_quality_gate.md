# CourtFlux system quality gate

The P0 detector target is no longer treated as an isolated 90% prerequisite. CourtFlux now separates:

1. direct detector observation quality;
2. court calibration / homography quality;
3. usable temporal track coverage;
4. product-level event quality.

The product-level objective remains approximately 90%+ correct usable ball-state coverage with very low catastrophic switches. The direct detector target will be decided from measured system evidence rather than chosen in advance.

## 1. Homography / court calibration

Calibration is opt-in during detection-only P0 runs so frozen detector benchmarks are unchanged by default.

```bash
export BASKETBALL_COURT_CALIBRATION=1
export ROBOFLOW_API_KEY='...'
export BASKETBALL_COURT_MODEL_ID='basketball-court-detection-2/14'
```

Then run the existing analyzer command. Per-frame diagnostics record:

- `camera.valid`;
- `camera.anchor_count`;
- `camera.inlier_count`;
- `camera.reprojection_rmse_px`;
- calibration source/model/schema;
- `analytics.homography_active`.

Players receive `court_xy` from their bottom-center. The ball receives `projected_court_xy`, explicitly labelled `homography_planar_apparent`: an airborne basketball is not on the court plane, so this projection is contextual evidence rather than physical 3-D ground truth.

### 33-keypoint Roboflow models

The current Roboflow basketball court dataset uses 33 landmarks. CourtFlux does **not** guess their index-to-meter ordering. To use a 33-keypoint model for metric homography, set a versioned JSON map:

```bash
export BASKETBALL_COURT_MODEL_ID='basketball-court-detection-2/<MODEL_VERSION>'
export BASKETBALL_COURT_KEYPOINT_MAP='data/court_keypoint_map_33.json'
```

Accepted map form:

```json
{
  "points": [
    {"index": 0, "court_xy_m": [0.0, 0.0]},
    {"index": 1, "court_xy_m": [0.0, 15.0]}
  ]
}
```

Use only a map verified against the exact exported Roboflow COCO keypoint schema/version.

## 2. Dataset/model external benchmark

Install the optional Roboflow stack:

```bash
python3 -m pip install -r requirements-roboflow.txt
```

Download the frozen public datasets used for external evidence:

```bash
python3 evaluation/download_roboflow_benchmarks.py --only all
```

Defaults:

- basketball detection: `roboflow-jvuqo/basketball-player-detection-3-ycjdo`, dataset v18;
- basketball court keypoints: `roboflow-jvuqo/basketball-court-detection-2`, dataset v19.

Use the **test** split for external detector comparison. Locate its `_annotations.coco.json` and image directory, then run the same backend contract:

```bash
python3 evaluation/external_ball_dataset_benchmark.py \
  --images data/external/roboflow/basketball_player_detection_v18/test \
  --annotations data/external/roboflow/basketball_player_detection_v18/test/_annotations.coco.json \
  --backend yolo \
  --yolo-model weights/Model_v0.1.pt \
  --device mps \
  --conf 0.10 \
  --imgsz 960 \
  --output evaluation/external_ball_yolo
```

Roboflow public model:

```bash
export BASKETBALL_ROBOFLOW_BALL_MODEL='basketball-player-detection-3-ycjdo/18'
export BASKETBALL_ROBOFLOW_BALL_LABELS='ball'

python3 evaluation/external_ball_dataset_benchmark.py \
  --images data/external/roboflow/basketball_player_detection_v18/test \
  --annotations data/external/roboflow/basketball_player_detection_v18/test/_annotations.coco.json \
  --backend roboflow \
  --conf 0.10 \
  --output evaluation/external_ball_roboflow_v18
```

Hugging Face challengers use `--backend huggingface` and the existing `BASKETBALL_HF_*` variables.

External image-level recall is **not tracking recall** and is not directly comparable to V1-V5. It is domain/model evidence: it tells us whether failures are CourtFlux-domain-specific or general detector weakness.

### Roboflow model on frozen CourtFlux videos

The analyzer also accepts the public Roboflow model as the ball backend:

```bash
export BASKETBALL_BALL_REACQUIRE=0
export BASKETBALL_BALL_DETECTOR_BACKEND=roboflow
export BASKETBALL_ROBOFLOW_BALL_MODEL='basketball-player-detection-3-ycjdo/18'
export BASKETBALL_ROBOFLOW_BALL_LABELS='ball'
export BASKETBALL_SMART_BALL_PIPELINE=0
```

Run V3 first, then benchmark it with `evaluation/smart_ball_benchmark.py` against the frozen GT. Only expand to V1-V5 if V3 gives meaningful signal.

## 3. Track-coverage evaluation

`evaluation/ball_track_coverage_eval.py` measures two different quantities:

- direct correct candidate coverage;
- direct + short linear-gap coverage, where every interpolated frame is checked against GT.

The latter is intentionally an **oracle-validated upper bound**, not a production tracker result. It answers whether short temporal continuity can plausibly close enough of the direct-detection gap to justify a lower detector gate.

Five-video example using the current Smart YOLO v1 runs:

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

Interpret the curve as:

```text
direct
→ +1 frame safe bridge
→ +2
→ +3
→ +5
→ +8
```

If coverage rises rapidly with short gaps, the system should invest in trajectory/state estimation. If it remains far below 90%, the detector still needs materially better observations.

## 4. Decide the detector target from evidence

Do not lower the direct detector target simply because 90% is difficult. Change it only after the track-coverage curve is known.

The working system gate is:

- correct usable ball-state coverage: ~90%+;
- direct/output precision: ~90%+ where a concrete location is emitted;
- catastrophic switches: ~0;
- unsupported gap P90: <~8 frames;
- very low false state during not-visible periods;
- coherent confidence calibration;
- multi-video / multi-camera generalization;
- acceptable runtime.

Decision rule:

- if direct + conservative short-gap coverage cannot approach ~90%, keep pushing detector/model representation;
- if direct + <=3/5-frame GT-valid bridges approach ~90%, a lower direct detector minimum becomes defensible and P0.3 trajectory/state estimation becomes the priority;
- never count long-gap reacquisition as equivalent to interpolation; previous CourtFlux experiments showed hard reacquisition can be catastrophically wrong.

The numeric direct detector target is therefore intentionally **TBD pending this evaluation**, while ~90%+ remains the system-level usable-track target.

## Tests

```bash
python3 evaluation/ball_candidate_backend_test.py
python3 evaluation/smart_ball_backend_integration_test.py
python3 evaluation/ball_candidate_pipeline_test.py
python3 evaluation/smart_candidate_filter_test.py
python3 evaluation/system_quality_gate_test.py
```
