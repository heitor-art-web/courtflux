# Jersey OCR ground truth workflow

This workflow creates an auditable jersey-number ground truth from CourtFlux player tracks. It is intentionally independent from both the legacy jersey OCR and the Roboflow challenger so model predictions cannot leak into the labels.

## What is ground truth here?

Each reviewed sample links a concrete video frame/player crop to the number actually visible on the jersey:

- `jersey_gt`: `0`–`99`, or `unreadable`;
- `visibility`: `clear`, `partial`, or `unreadable`;
- `team_gt`: optional manual team label;
- `identity_group`: optional stable physical-player identity across fragmented track IDs;
- reviewer, timestamp and notes.

The match roster is context only. **Never infer `jersey_gt` because a number is or is not present in the roster.** Likewise, do not consult legacy/Roboflow OCR output while annotating.

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

The extractor uses the canonical `weights/model_v1.0.0.pt` player detector, BoT-SORT and the same detector thresholds used by the P1 benchmark. It saves both the complete tracked-player crop and the same conservative torso region used by jersey recognition.

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

1. inspect the torso crop and full-player crop;
2. enter the number only when it is genuinely readable;
3. use `unreadable` when it cannot be determined from the images;
4. set `visibility` to `clear`, `partial`, or `unreadable`;
5. optionally correct `team_gt`;
6. use `identity_group` when you know multiple fragmented track IDs belong to the same physical player.

The UI can propagate a label across all samples in a video+track, but that option is deliberately off by default. Only use it after visually checking the track for ID switches.

## 3. Validate annotations

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py validate \
  --manifest data/jersey_ground_truth/porto_ovarense_v1/manifest.csv
```

Validation rejects malformed labels and reports track-level conflicts where the same video+track has multiple numeric jersey labels. A conflict can indicate either an annotation mistake or a tracker ID switch and must be reviewed before export.

## 4. Export the training/evaluation ground truth

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py export \
  --manifest data/jersey_ground_truth/porto_ovarense_v1/manifest.csv \
  --output data/jersey_ground_truth/porto_ovarense_v1/ground_truth.csv
```

The exporter includes reviewed numeric and `unreadable` samples and creates deterministic `train`/`val`/`test` splits. Splitting is grouped by `identity_group` when supplied; otherwise it falls back to `video + track_id`.

This grouping is important: random frame-level splitting would put nearly identical frames of the same player into training and validation and overstate OCR quality. Before using the split for a formal benchmark, fill `identity_group` across any fragmented tracks that are known to be the same physical player.

Default split ratios are 70/15/15 and the seed is fixed. A companion `ground_truth.summary.json` records the split counts and policy.

## Acceptance criteria before retraining

Do not call the dataset ready merely because every row is labelled. Before retraining:

- there must be no validation errors;
- labels must come from visual evidence, not roster inference;
- unreadable/partial cases must be retained rather than silently discarded;
- obvious track ID switches must be separated or corrected;
- known fragments of the same physical player should share `identity_group`;
- validation/test groups must not contain the same physical player/image sequence as training;
- report class distribution before interpreting model accuracy.

The first dataset can be small. A clean, diverse and leakage-resistant ground truth is more valuable than thousands of near-duplicate frames.
