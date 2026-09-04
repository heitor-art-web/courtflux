# CourtFlux P0.1 — Direct Ball Detection Ground Truth

This folder documents the ground-truth protocol used to evaluate the **direct ball detector only**.

The P0.1 evaluator deliberately does **not** use BoT-SORT, temporal association, `predicted_hold`, possession logic or downstream reconstruction. The goal is to measure what the detector itself sees.

Candidate-frame coverage is useful for diagnostics, but it is not precision or recall. A frame can contain a false ball candidate and still count as "covered" without ground truth.

## Fast workflow: one click per frame

For the first P0.1 baseline, use a sampled 30-second clip. At 25 FPS, `--every 3` produces roughly 250 frames.

### 1. Prepare frames

```bash
python evaluation/prepare_ball_ground_truth.py \
  --video data/OwiQgAMqyUs.mp4 \
  --start 0 \
  --end 30 \
  --every 3 \
  --output evaluation/ball_ground_truth/porto_ovarense_0000_0030
```

This creates:

```text
evaluation/ball_ground_truth/porto_ovarense_0000_0030/
  metadata.json
  annotations.jsonl
  frames/
```

### 2. Label with the one-click annotator

Do **not** edit `annotations.jsonl` manually.

```bash
python evaluation/ball_point_annotator.py \
  --annotations evaluation/ball_ground_truth/porto_ovarense_0000_0030/annotations.jsonl
```

Controls:

```text
left click   ball centre → save + next frame
N            ball genuinely not visible → save + next frame
B            go back one frame
Q / Esc      save and quit
```

The annotator saves after every decision and resumes at the first unlabelled frame when reopened.

The default truth format is only:

```json
{"visible": true, "point": [x, y]}
```

or:

```json
{"visible": false, "point": null}
```

A full bounding box is optional. It is not required for the fast P0.1 presence/recall baseline.

### 3. Run the direct detector evaluator

Use 1280 as the current accuracy baseline:

```bash
python evaluation/ball_detection_eval.py \
  --video data/OwiQgAMqyUs.mp4 \
  --annotations evaluation/ball_ground_truth/porto_ovarense_0000_0030/annotations.jsonl \
  --model weights/Model_v0.1.pt \
  --imgsz 1280 \
  --device mps \
  --capture-conf 0.01 \
  --eval-conf 0.10 \
  --max-det 80
```

The evaluator uses `YOLO.predict`, not tracking. It captures low-confidence ball candidates once and automatically evaluates a confidence sweep.

For point truth, a prediction is considered the correct ball when the clicked point falls inside the predicted ball box plus a small tolerance for tiny-object click error. Existing bounding-box truth remains supported and uses IoU.

Output:

```text
evaluation/ball_eval/<timestamp>/
  summary.json
  per_frame.json
  per_frame.csv
```

The key first-pass metrics are:

- precision
- recall
- F1
- visible-ball detection rate
- false positives per annotated frame
- confidence-threshold sweep

## P0.1 working gate

The current product milestone is approximately:

```text
direct visible-ball recall / coverage  >= 70%
precision                              protected; do not buy recall with junk detections
false positives                         low enough for temporal association to remain stable
localization                            good enough for trajectory and possession use
```

This must eventually hold across multiple games and camera conditions.

## Later: richer failure labels

Only after the fast baseline is useful should we spend time tagging failure categories such as:

```text
small / distant ball
motion blur
partial body occlusion
under-rim congestion
rim / backboard confusion
crowd / background hard negatives
ball next to player body
fast pass
shot arc
rebound traffic
broadcast compression
```

Those labels are useful for targeted dataset work, but they are **not required** to get the first P0.1 precision/recall baseline.
