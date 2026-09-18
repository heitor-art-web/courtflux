# AirborneClassifierV1 development protocol

V1–V5 are the development/training corpus for AirborneClassifierV1. Their reviewed human GT and existing no-model `p2_states.jsonl` runs are reused directly.

V6–V7 will be two new videos reserved exclusively for final validation.

## Development contract

The five development clips are video-disjoint. Model quality is estimated with leave-one-video-out grouped cross-validation:

1. train on four videos;
2. evaluate on the held-out fifth video;
3. repeat for all five videos;
4. inspect Stage 1, Stage 2 and end-to-end out-of-fold metrics;
5. once development choices are frozen, fit the final artefact on all V1–V5.

## Model contract

AirborneClassifierV1 is causal and separate from event generation:

```text
active P2 flight hypothesis
        ↓
Stage 1: AIRBORNE vs CONTINUITY
        ↓ AIRBORNE
Stage 2: PASS vs SHOT
        ↓
PASS_FLIGHT / SHOT_FLIGHT
```

For training:

- GT `pass_flight` and `shot_flight` are positive AIRBORNE examples for Stage 1;
- other human-labelled frames inside an active engine flight are CONTINUITY/not-airborne examples for Stage 1;
- Stage 2 trains only on GT `pass_flight` and `shot_flight` rows;
- Stage 2 may abstain when PASS-vs-SHOT confidence is below the frozen probability threshold.

Training weighting is interval-balanced, then video-balanced, then class-balanced independently for each binary stage.

## 1. Build the dataset from existing runs

```bash
python3 evaluation/p2_airborne_build_dataset.py
```

The builder uses only human-labelled, engine-eligible frames with an active P2 flight and writes:

```text
evaluation/airborne_dev/airborne_dataset.jsonl
```

Temporal features are FPS-normalized so clips at different frame rates use comparable physical-time features.

## 2. Train + grouped cross-validation

```bash
python3 evaluation/p2_airborne_train.py \
  --minimum-probability 0.55
```

This writes:

- `weights/p2_airborne_classifier_v1.json`
- `evaluation/airborne_dev/training_report.json`

The report contains, for every held-out video and in aggregate:

- `stage1_airborne_vs_continuity`;
- `stage2_pass_vs_shot`;
- `end_to_end`.

The final JSON artefact is then fitted on all V1–V5 using the same frozen feature/model contract.

## 3. Freeze development

Before V6–V7 are evaluated, freeze:

- feature contract;
- two-stage model family;
- learned weights;
- confidence threshold;
- live resolver integration;
- Gate A thresholds.

After this point, V6–V7 must not be used to change any of the above.

## 4. New validation videos

Choose two new basketball videos not present in V1–V5 and designate them V6 and V7. Create reviewed human GT blind to classifier predictions, run them with the frozen model, and validate with the P3-readiness Gate A.

V1–V5 are not part of the final Gate score.
