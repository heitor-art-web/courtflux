# Jersey OCR ground truth workflow

This workflow creates an auditable jersey-number ground truth from CourtFlux tracked detections. It is intentionally independent from both the legacy jersey OCR and the Roboflow challenger so model predictions cannot leak into the labels.

## What is ground truth here?

Each reviewed sample links a concrete video frame/detection crop to the human-reviewed subject and, when applicable, the jersey number actually visible:

- `subject_type`: `player`, `referee`, `cameraman_staff`, `other_non_player`, or `unknown`;
- `jersey_gt`: for real players, `0`–`99` or `unreadable`; for non-players/unknown, `not_applicable`;
- `visibility`: for players, `clear`, `partial`, or `unreadable`; otherwise `not_applicable`;
- `team_gt`: optional manual team label;
- `identity_group`: optional stable physical-player identity across fragmented track IDs;
- reviewer, timestamp and notes.

The match roster is context only. **Never infer `jersey_gt` because a number is or is not present in the roster.** Likewise, do not consult legacy/Roboflow OCR output while annotating.

`unreadable` means a real basketball player whose number cannot be read from the sample. Referees, camera operators and other false-positive detections are not `unreadable`; label their `subject_type` instead. These detections remain in the ground truth as detector-negative evidence but are excluded from jersey OCR training/evaluation.

Generated crops/manifests live under `data/`, which is already ignored by git. The tooling itself is versioned under `tools/`.

## 1. Extract crops

For the FC Porto vs Ovarense source video:

```bash
cd /Users/user/Documents/server/projects/courtflux-jersey
source /Users/user/Documents/server/projects/courtflux/.venv/bin/activate

PYTHONPATH=. python tools/jersey_ground_truth.py extract \
  --video /Users/user/Documents/server/projects/courtflux/data/OwiQgAMqyUs.mp4 \
  --match-config data/match_config.json \
  --output-dir data/jersey_ground_truth/porto_ovarense_v1 \
  --start 0 \
  --end 120 \
  --device mps \
  --tracker botsort.yaml \
  --imgsz 960 \
  --player-conf 0.10 \
  --inference-conf 0.03 \
  --sample-every-sec 1.0 \
  --max-per-track 8
```

The extractor uses the canonical `weights/model_v1.0.0.pt` detector, BoT-SORT and the same detector thresholds used by the P1 benchmark. It saves both the complete tracked crop and the same conservative torso region used by jersey recognition.

The manifest also stores `torso_sharpness`, player/torso dimensions and detector/team evidence. **Do not discard blurry samples during extraction.** Keep them and label real-player cases as `unreadable` when appropriate. That lets CourtFlux calibrate a cheap OCR quality gate from actual labelled data instead of choosing an arbitrary sharpness threshold.

Outputs:

```text
data/jersey_ground_truth/porto_ovarense_v1/
├── extraction.json
├── manifest.csv
├── annotations.csv
└── crops/
    ├── player/
    └── torso/
```

`manifest.csv` is extraction evidence and should not be manually edited. `annotations.csv` contains the human labels.

## 2. Annotate visually

Launch the local Streamlit reviewer:

```bash
streamlit run tools/jersey_ground_truth_app.py -- \
  --manifest data/jersey_ground_truth/porto_ovarense_v1/manifest.csv \
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

The UI can propagate a label across all samples in a video+track, but that option is deliberately off by default. Only use it after visually checking the track for ID switches.

The `Previous` button follows the actual viewing history, including a sample that disappeared from the default `unreviewed` queue after it was saved. This allows recent mistakes to be corrected without switching the whole queue to `reviewed`.

### Legacy annotations created before `subject_type`

Existing numeric annotations are safely interpreted as `subject_type=player`. Existing `unreadable` rows are deliberately **not** guessed because some may be referees/camera staff. The reviewer exposes a `needs subject review` queue containing only those ambiguous legacy rows. Review that short queue before final export.

## 3. Validate annotations

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py validate \
  --manifest data/jersey_ground_truth/porto_ovarense_v1/manifest.csv
```

Validation rejects malformed labels and enforces the subject contract:

- `player` → jersey `0`–`99` or `unreadable` + player visibility;
- non-player/unknown → `jersey_gt=not_applicable` and `visibility=not_applicable`;
- legacy `unreadable` without a reviewed `subject_type` is reported as an error;
- track-level conflicts where the same player track has multiple numeric jersey labels are reported for review.

A track conflict can indicate either an annotation mistake or a tracker ID switch and must be reviewed before export.

## 4. Export the training/evaluation ground truth

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py export \
  --manifest data/jersey_ground_truth/porto_ovarense_v1/manifest.csv \
  --output data/jersey_ground_truth/porto_ovarense_v1/ground_truth.csv
```

The export retains all reviewed evidence and adds:

- `ocr_eligible=true` only for real-player numeric/unreadable rows;
- `dataset_role=ocr` for jersey OCR samples;
- `dataset_role=detector_negative` for referees/camera staff/other non-player detections;
- `dataset_role=excluded_unknown` when the subject cannot be confidently classified;
- deterministic `train`/`val`/`test` splits.

OCR training/evaluation must use only `dataset_role=ocr` (equivalently `ocr_eligible=true`). Detector negatives remain available for later player/non-player gating work rather than being deleted.

Splitting is grouped by `identity_group` when supplied; otherwise it falls back to `video + track_id`. This grouping is important: random frame-level splitting would put nearly identical frames of the same player into training and validation and overstate OCR quality. Before using the split for a formal benchmark, fill `identity_group` across any fragmented tracks known to be the same physical player.

Default split ratios are 70/15/15 and the seed is fixed. A companion `ground_truth.summary.json` records split, subject, label and dataset-role counts.

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

Conceptually:

```text
detected track
    ↓
subject / team / track evidence
    ↓
cheap crop-quality gate
    ↓
keep best temporally separated crops
    ↓
jersey OCR
```

## Acceptance criteria before retraining

Do not call the dataset ready merely because every row is labelled. Before retraining:

- there must be no validation errors;
- every reviewed row must have a valid `subject_type`;
- labels must come from visual evidence, not roster inference;
- unreadable/partial real-player cases must be retained rather than silently discarded;
- non-player detector false positives must not enter the OCR training set;
- obvious track ID switches must be separated or corrected;
- known fragments of the same physical player should share `identity_group`;
- validation/test groups must not contain the same physical player/image sequence as training;
- report class, subject and visibility distributions before interpreting model accuracy.

The first dataset can be small. A clean, diverse and leakage-resistant ground truth is more valuable than thousands of near-duplicate frames.
