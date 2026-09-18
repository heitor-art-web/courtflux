# AirborneClassifierV1 development protocol

This directory is development-only. V1–V5 are forbidden for feature selection, weight training, confidence calibration or model selection.

## Split

`manifest.json` defines six 90-second candidate windows from three videos that are not V1–V5:

- D1–D4: training, across two source videos;
- D5–D6: calibration, from a third source video.

Train and calibration are video-disjoint. If a chosen window is mostly stoppage/non-gameplay after human inspection, replace it **before** training and keep the replacement outside V1–V5.

## 1. Generate development runs

Run with Jersey OFF, neutral match config and **no airborne model loaded**:

```bash
unset P2_AIRBORNE_MODEL

python3 evaluation/p2_real_video_probe.py \
  --video "<source from manifest>" \
  --start <start_time_sec> \
  --end <end_time_sec> \
  --model weights/model_v1.1.0.pt \
  --device mps \
  --match-config evaluation/match_configs/unknown.json \
  --no-jersey
```

After every run, place its run ID into the corresponding `run_id` field in `manifest.json`.

## 2. Human annotation

Create `ground_truth/d1.json` … `ground_truth/d6.json` using the existing `courtflux_p2_gt_v1` schema.

For AirborneClassifierV1, the required human field is `control_intervals`. Label only what is visually supportable:

- `possessed`
- `pass_flight`
- `shot_flight`
- `loose`
- `unknown`

Keep annotation blind to classifier predictions. Prefer `unknown` over guessing. Human windows do not need frame-perfect release boundaries.

The development labels are not Gate-A labels; they exist only to train/calibrate the airborne classifier.

## 3. Build the dataset

```bash
python3 evaluation/p2_airborne_build_dataset.py
```

The builder:

- rejects V1–V5 video IDs;
- rejects train/calibration video overlap;
- uses only engine-eligible frames with an active P2 flight and a human-labeled control state;
- maps PASS and SHOT to their classes and all other labeled airborne frames to `unresolved`;
- writes `evaluation/airborne_dev/airborne_dataset.jsonl`.

## 4. Train

```bash
python3 evaluation/p2_airborne_train.py \
  --minimum-probability 0.55
```

This writes:

- `weights/p2_airborne_classifier_v1.json`
- `evaluation/airborne_dev/training_report.json`

The model is class-balanced multinomial logistic regression over a frozen causal feature vector. The confidence threshold is development-only and must be frozen before V1–V5 validation.

## 5. Validation run

Load the frozen model only after all development choices are complete:

```bash
export P2_AIRBORNE_MODEL="weights/p2_airborne_classifier_v1.json"
```

Then rerun V1–V5 once and evaluate with the current P3-readiness Gate A:

```bash
python3 evaluation/p2_gate_a_scoreboard.py \
  --v1-run "$V1_RUN" \
  --v2-run "$V2_RUN" \
  --v3-run "$V3_RUN" \
  --v4-run "$V4_RUN" \
  --v5-run "$V5_RUN" \
  --output evaluation/gate_a_airborne_classifier_v1.json
```

Do not change model weights, features, confidence threshold or Gate-A thresholds in response to V1–V5 scores.
