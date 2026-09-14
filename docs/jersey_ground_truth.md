# Jersey OCR ground truth workflow

This workflow creates an auditable jersey-number ground truth from CourtFlux tracked detections. It is intentionally independent from Jersey model predictions so model output cannot leak into the labels.

For the active P1 exit gate, use the current v1.1 detector/tracker operating point documented in `docs/p1_jersey_exit_gate.md`.

## What is ground truth here?

Each reviewed sample links a concrete video frame/detection crop to the human-reviewed subject and, when applicable, the jersey number actually visible:

- `subject_type`: `player`, `referee`, `cameraman_staff`, `other_non_player`, or `unknown`;
- `jersey_gt`: for real players, `0`–`99` or `unreadable`; for non-players/unknown, `not_applicable`;
- `visibility`: for players, `clear`, `partial`, or `unreadable`; otherwise `not_applicable`;
- `team_gt`: optional manual team label;
- `identity_group`: optional stable physical-player identity across fragmented track IDs;
- reviewer, timestamp and notes.

The match roster is context only. **Never infer `jersey_gt` because a number is or is not present in the roster.** Likewise, do not consult Jersey recognizer output while annotating.

`unreadable` means a real basketball player whose number cannot be read from the sample. Referees, camera operators and other false-positive detections are not `unreadable`; label their `subject_type` instead. These detections remain in the ground truth as detector-negative evidence but are excluded from jersey OCR training/evaluation.

Generated crops/manifests live under `data/`, which is already ignored by git. The tooling itself is versioned under `tools/`.

## 1. Extract crops for the P1 exit gate

For the FC Porto vs Ovarense source video:

```bash
cd /Users/user/Documents/server/projects/courtflux
source .venv/bin/activate

PYTHONPATH=. python tools/jersey_ground_truth.py extract \
  --video data/OwiQgAMqyUs.mp4 \
  --match-config data/match_config.json \
  --output-dir data/jersey_ground_truth/porto_ovarense_p1_v110 \
  --model weights/model_v1.1.0.pt \
  --start 0 \
  --end 120 \
  --device mps \
  --tracker botsort.yaml \
  --imgsz 960 \
  --player-conf 0.40 \
  --inference-conf 0.03 \
  --sample-every-sec 1.0 \
  --max-per-track 8
```

The P1 exit dataset must be extracted with the current integrated detector `weights/model_v1.1.0.pt`, BoT-SORT and player confidence `0.40`. This supersedes the historical v1.0 extraction example.

The manifest stores `torso_sharpness`, player/torso dimensions and detector/team evidence. **Do not discard blurry samples during extraction.** Keep them and label real-player cases as `unreadable` when appropriate. That lets CourtFlux measure false acceptance on genuinely unreadable evidence instead of hiding difficult examples.

Outputs:

```text
data/jersey_ground_truth/porto_ovarense_p1_v110/
├── extraction.json
├── manifest.csv
├── annotations.csv
├── track_reviews.csv
└── crops/
    ├── player/
    └── torso/
```

`manifest.csv` is extraction evidence and should not be manually edited. `annotations.csv` contains the human labels.

## 2. Annotate visually

Launch the local Streamlit reviewer:

```bash
streamlit run tools/jersey_ground_truth_app.py -- \
  --manifest data/jersey_ground_truth/porto_ovarense_p1_v110/manifest.csv \
  --reviewer heitor
```

For each sample:

1. decide whether the detection is actually a basketball player;
2. if it is not a player, choose the appropriate `subject_type`; no jersey number is requested;
3. if it is a player, enter the number only when genuinely readable;
4. use `unreadable` only for a real player whose number cannot be determined from the images;
5. set player `visibility` to `clear`, `partial`, or `unreadable`;
6. optionally correct `team_gt`;
7. use `identity_group` when multiple fragmented track IDs are known to belong to the same physical player.

The P1 exit gate should start with **40–60 representative reviewed tracks**, blind to predictions. If both candidates pass and correct recall is within 10 percentage points, extend the reviewed set toward approximately 100 tracks before promotion.

The UI can propagate a label across all samples in a video+track, but that option is deliberately off by default. Only use it after visually checking the track for ID switches.

## 3. Validate annotations

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py validate \
  --manifest data/jersey_ground_truth/porto_ovarense_p1_v110/manifest.csv
```

Validation rejects malformed labels and enforces the subject contract:

- `player` → jersey `0`–`99` or `unreadable` + player visibility;
- non-player/unknown → `jersey_gt=not_applicable` and `visibility=not_applicable`;
- legacy `unreadable` without a reviewed `subject_type` is reported as an error;
- track-level conflicts where the same player track has multiple numeric jersey labels are reported for review.

A track conflict can indicate either an annotation mistake or a tracker ID switch and must be reviewed before export.

## 4. Export the evaluation ground truth

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py export \
  --manifest data/jersey_ground_truth/porto_ovarense_p1_v110/manifest.csv \
  --output data/jersey_ground_truth/porto_ovarense_p1_v110/ground_truth.csv
```

The export retains all reviewed evidence and adds:

- `ocr_eligible=true` only for real-player numeric/unreadable rows;
- `dataset_role=ocr` for jersey OCR samples;
- `dataset_role=detector_negative` for referees/camera staff/other non-player detections;
- `dataset_role=excluded_unknown` when the subject cannot be confidently classified;
- deterministic `train`/`val`/`test` splits.

OCR evaluation must use only `dataset_role=ocr` (equivalently `ocr_eligible=true`). Detector negatives remain available for later player/non-player gating work rather than being deleted.

Splitting is grouped by `identity_group` when supplied; otherwise it falls back to `video + track_id`. This grouping is important: random frame-level splitting would put nearly identical frames of the same player into training and validation and overstate OCR quality. Before using the split for a formal benchmark, fill `identity_group` across any fragmented tracks known to be the same physical player.

## P1 Jersey acceptance criteria

The promotion gate is defined in `docs/p1_jersey_exit_gate.md` and enforced by `evaluation/p1_jersey_exit_gate.py`:

```text
reviewed tracks                  >= 40
selective / decision precision   >= 90%
false accept on UNREADABLE       <= 5%
winner                           highest correct recall among passers
```

If neither candidate passes, promote neither. `UNKNOWN` remains valid.

## OCR quality-gate calibration

Do not hard-code a blur threshold while collecting the first ground truth. After annotation, analyse at least:

- `subject_type`;
- `visibility`;
- `torso_sharpness`;
- torso/player dimensions;
- detector confidence;
- team evidence;
- numeric vs `unreadable` player labels.

The goal is to find a cheap pre-OCR quality gate that rejects obviously hopeless crops while preserving readable numbers. A useful threshold must be measured against labelled data, not chosen from the sharpness value alone.

## Acceptance criteria before using the GT

Do not call the dataset ready merely because every row is labelled. Before evaluation:

- there must be no validation errors;
- every reviewed row must have a valid `subject_type`;
- labels must come from visual evidence, not roster inference;
- unreadable/partial real-player cases must be retained rather than silently discarded;
- non-player detector false positives must not enter the OCR evaluation set;
- obvious track ID switches must be separated or corrected;
- known fragments of the same physical player should share `identity_group`;
- report class, subject and visibility distributions before interpreting model accuracy.

A small clean, diverse and leakage-resistant ground truth is more valuable than thousands of near-duplicate frames.
