# AirborneClassifierV1 development protocol

V1–V5 are the development/training corpus for AirborneClassifierV1. Their reviewed human GT and existing no-model `p2_states.jsonl` runs are reused directly.

V6–V7 will be two new videos reserved exclusively for final validation.

## Development contract

The five development clips are video-disjoint:

- V1
- V2
- V3
- V4
- V5

Model quality inside development is estimated with leave-one-video-out grouped cross-validation:

1. train on four videos;
2. evaluate on the held-out fifth video;
3. repeat for all five videos;
4. inspect the aggregate out-of-fold metrics;
5. once development choices are frozen, fit the final artefact on all V1–V5.

This gives every V1–V5 sample an out-of-fold prediction during development while still allowing the final model to use all available labelled training data.

## 1. Build the dataset from existing runs

`manifest.json` points to the existing V1–V5 human GT and no-airborne-model state runs.

```bash
python3 evaluation/p2_airborne_build_dataset.py
```

The builder:

- uses only human-labelled frames;
- uses only engine-eligible frames with an active P2 flight;
- maps human `pass_flight` and `shot_flight` to their classifier classes;
- maps other labelled airborne frames to `unresolved`;
- reconstructs the classifier feature vector from fields already persisted in the existing V1–V5 `p2_states.jsonl` files;
- writes `evaluation/airborne_dev/airborne_dataset.jsonl`.

Features that were not persisted in those historical runs are deliberately neutralised. Constant columns receive zero model weight, so runtime-only information cannot silently influence the trained model.

## 2. Train + grouped cross-validation

```bash
python3 evaluation/p2_airborne_train.py \
  --minimum-probability 0.55
```

This writes:

- `weights/p2_airborne_classifier_v1.json`
- `evaluation/airborne_dev/training_report.json`

The report contains five leave-one-video-out folds plus the aggregate out-of-fold result. The final JSON model is then fitted on all V1–V5.

## 3. Freeze development

Before V6–V7 are evaluated, freeze:

- feature contract;
- model family;
- learned weights;
- confidence threshold;
- live resolver integration;
- Gate A thresholds.

After this point, V6–V7 must not be used to change any of the above.

## 4. New validation videos

Choose two new basketball videos not present in V1–V5 and designate them V6 and V7.

Create reviewed human GT using the same `courtflux_p2_gt_v1` schema and the same annotation principles:

- annotate blind to classifier predictions;
- prefer `unknown` over guessing;
- use human event/control windows rather than pretending exact release-frame precision;
- include enough possession, pass and shot episodes for the Gate to be meaningful.

Run them with the frozen model:

```bash
export P2_AIRBORNE_MODEL="weights/p2_airborne_classifier_v1.json"
```

Then validate V6–V7 with the P3-readiness Gate A. V1–V5 are not part of the final Gate score.
