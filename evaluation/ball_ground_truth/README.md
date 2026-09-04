# CourtFlux P0.1 — Direct Ball Detection Ground Truth

This folder documents the ground-truth protocol used to evaluate the **direct ball detector only**.

The P0.1 evaluator deliberately does **not** use BoT-SORT, temporal association, `predicted_hold`, possession logic or any other downstream reconstruction. The goal is to measure what the detector itself sees.

## Why this exists

Candidate-frame coverage is useful for diagnostics, but it is not precision or recall. A frame can contain a high-confidence false ball candidate and still count as "covered" without ground truth.

P0.1 should therefore be judged using manually verified frames and at least:

- detector precision
- detector recall
- F1
- visible-ball frame detection rate
- false positives per frame
- localization quality
- threshold sweep
- confidence calibration
- performance by ball size
- occlusion state
- court region
- camera view
- basketball event context

## 1. Prepare a clip

Use the same cached video used by the beta benchmarks. For the current Porto vs Ovarense reference clip:

```bash
python evaluation/prepare_ball_ground_truth.py \
  --video data/OwiQgAMqyUs.mp4 \
  --start 0 \
  --end 30 \
  --every 1 \
  --output evaluation/ball_ground_truth/porto_ovarense_0000_0030
```

For a faster first pass, `--every 2` or `--every 3` is acceptable. The full P0.1 gate should ultimately use every frame in multiple clips/games.

The command creates:

```text
evaluation/ball_ground_truth/porto_ovarense_0000_0030/
  metadata.json
  annotations.jsonl
  frames/
    frame_000000_t00000.000.jpg
    ...
```

## 2. Annotate the ball

Each line in `annotations.jsonl` represents one extracted frame.

Visible ball:

```json
{"frame_id": 123, "timestamp_sec": 4.92, "image": "frames/frame_000123_t00004.920.jpg", "visible": true, "bbox": [901, 341, 918, 358], "occlusion": "none", "court_region": "frontcourt", "camera": "broadcast_wide", "event_context": "dribble", "notes": ""}
```

Ball genuinely not visible:

```json
{"frame_id": 124, "timestamp_sec": 4.96, "image": "frames/frame_000124_t00004.960.jpg", "visible": false, "bbox": null, "occlusion": "full", "court_region": "frontcourt", "camera": "broadcast_wide", "event_context": "dribble", "notes": "behind player body"}
```

### Annotation rules

- `bbox` is `[x1, y1, x2, y2]` in **original source-frame pixels**.
- If any identifiable part of the real basketball is visible, use `visible=true` and annotate it.
- Use `visible=false` only when the real basketball cannot be visually located with reasonable confidence.
- Do not label logos, heads, rim details, scoreboard graphics or crowd objects as the ball.
- Keep the box tight around the visible ball rather than the expected full circular extent when strongly occluded.
- `occlusion` suggested values: `none`, `partial`, `full`.
- `court_region` suggested values: `backcourt`, `frontcourt`, `paint`, `rim_area`, `sideline`, `unknown`.
- `camera` suggested values: `broadcast_wide`, `broadcast_medium`, `closeup`, `replay`, `other`.
- `event_context` suggested values: `dribble`, `pass`, `shot`, `rebound`, `loose_ball`, `inbound`, `unknown`.

The categorical fields are optional, but filling them is valuable because P0.1 needs to identify *where* recall and precision fail, not only the global average.

## 3. Run direct detector evaluation

Capture low-confidence candidates once and evaluate the current operating threshold plus a full threshold sweep:

```bash
python evaluation/ball_detection_eval.py \
  --video data/OwiQgAMqyUs.mp4 \
  --annotations evaluation/ball_ground_truth/porto_ovarense_0000_0030/annotations.jsonl \
  --model weights/Model_v0.1.pt \
  --imgsz 960 \
  --device mps \
  --capture-conf 0.01 \
  --eval-conf 0.10 \
  --max-det 80
```

The evaluator uses `YOLO.predict`, not tracking, so these are P0.1 detector metrics.

Output:

```text
evaluation/ball_eval/<timestamp>/
  summary.json
  per_frame.json
  per_frame.csv
```

`summary.json` includes separate IoU@0.25 and IoU@0.50 results. IoU@0.50 remains the stricter localization metric; IoU@0.25 is also reported because a basketball can be only a handful of pixels wide in broadcast footage and small box shifts can heavily punish IoU.

## 4. P0.1 working gate

The current product milestone is approximately:

```text
direct visible-ball coverage / recall  >= 70%
precision                              protected; do not buy recall with junk detections
false positives                         low enough for temporal association to remain stable
localization                            good enough for trajectory and possession use
```

This gate must hold across multiple games and camera conditions before P0.1 is considered complete.

## 5. Use failures to drive data work

After evaluation, inspect false negatives and false positives by category. Training data should then target the measured failures, for example:

```text
small / distant ball
motion blur
partial body occlusion
under-rim congestion
rim / backboard confusion
crowd / background hard negatives
ball next to a player body
fast pass
shot arc
rebound traffic
broadcast compression
```

Do not retrain simply because aggregate frame coverage is below target. First identify the failure distribution and build the next dataset around it.
