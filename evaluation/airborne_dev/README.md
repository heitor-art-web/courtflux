# AirborneClassifierV1.1 development protocol

V1–V5 are the development/training corpus. Their reviewed human GT and existing no-model `p2_states.jsonl` runs are reused directly. V6–V7 remain reserved exclusively for final validation.

## Development contract

The five development clips are video-disjoint. Model quality is estimated with leave-one-video-out grouped cross-validation:

1. train on four videos;
2. replay the held-out fifth video as episode examples and causal prefixes;
3. repeat for all five videos;
4. inspect Stage 1, Stage 2, terminal episode and causal episode metrics;
5. once development choices are frozen, fit the final artefact on all V1–V5.

## Model contract

AirborneClassifierV1.1 is causal and separate from event generation:

```text
active P2 flight hypothesis
        ↓
causal episode accumulator
        ↓
Stage 1: AIRBORNE vs CONTINUITY
        ↓ AIRBORNE
Stage 2: PASS vs SHOT
        ↓
PASS_FLIGHT / SHOT_FLIGHT
```

### Training unit

One contiguous P2 flight hypothesis is one training example. Hundreds of frames from one physical hypothesis no longer become hundreds of statistically independent examples.

For each episode the builder stores:

- one terminal causal feature summary used for fitting;
- every cumulative prefix summary, used only to replay runtime behavior during cross-validation;
- overlap with human `possessed`, `pass_flight` and `shot_flight` intervals.

Stage-1 target:

- `airborne` when the flight episode overlaps any human PASS or SHOT interval;
- `continuity` when it does not overlap PASS/SHOT but overlaps human POSSESSED;
- episodes without useful core GT are excluded.

Stage-2 target:

- `pass_flight` when an airborne episode overlaps PASS but not SHOT;
- `shot_flight` when it overlaps SHOT but not PASS;
- an episode spanning both PASS and SHOT remains useful for Stage 1 but is excluded from Stage-2 fitting because its type target is ambiguous.

## Causal runtime equivalence

At runtime the episode accumulator sees only current and past frames. After each frame it recomputes the same episode feature vector used in training. The resolver latches the first confident PASS/SHOT decision for the physical flight.

Cross-validation therefore reports both:

- `terminal_episode_end_to_end`: classification using the completed episode summary;
- `causal_episode_end_to_end`: prefix replay using the first decision that runtime would have committed.

The causal metric is the more important development metric.

## 1. Build the episode dataset

```bash
rm -f evaluation/airborne_dev/airborne_dataset.jsonl
python3 evaluation/p2_airborne_build_dataset.py
```

This writes:

```text
evaluation/airborne_dev/airborne_dataset.jsonl
evaluation/airborne_dev/airborne_dataset_summary.json
```

Temporal frame features are FPS-normalized before episode aggregation.

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
- `terminal_episode_end_to_end`;
- `causal_episode_end_to_end`, including correct airborne decision latency.

Each episode is one statistical training unit. Weighting then equalizes source-video mass and balances the two classes independently per stage.

## 3. Freeze development

Before V6–V7 are evaluated, freeze:

- episode feature contract;
- model family;
- learned weights;
- confidence threshold;
- live resolver integration;
- Gate A thresholds.

After this point, V6–V7 must not be used to change any of the above.

## 4. New validation videos

Choose two new basketball videos not present in V1–V5 and designate them V6 and V7. Create reviewed human GT blind to classifier predictions, run them with the frozen model, and validate with the P3-readiness Gate A.

V1–V5 are not part of the final Gate score.
