# Jersey Reconstruction R&D

This branch is an isolated research track for Jersey reconstruction. It starts from `main` and must not depend on the active P2 GT/calibration branch.

## Scope

The immediate question is not "which OCR model is best?". It is:

> When a player's jersey number is genuinely observable somewhere in a track, can CourtFlux retrieve the right frame/ROI and preserve enough visual evidence for a recognizer to read it?

The first benchmark therefore separates four stages that were previously collapsed into one final identification percentage:

1. **Visibility rate** — fraction of labelled tracks where a human can find at least one genuinely readable number.
2. **ROI localization recall** — overlap between the automatic number-region proposal and the human oracle ROI.
3. **Recognizer conditional accuracy** — recognition accuracy when the recognizer receives the human-selected oracle crop.
4. **Track-level identification** — final number after automatic retrieval/recognition/temporal aggregation.

A low final score must not be interpreted as an OCR failure until the oracle-crop result is known.

## Branch boundary

Do not modify the production P1/P2 control path in this research branch merely to improve the benchmark.

In particular:

- P2 GT/calibration continues independently on `validation/p2-gt-calibration`.
- Identity-v2 remains the single downstream temporal consensus/locking layer.
- Rosters remain validator-only and must not manufacture a visual number.
- `UNKNOWN` remains preferable to a false identity.
- Recognizer comparisons should use the same frozen oracle/automatic inputs whenever possible.

## Initial benchmark scaffold

`evaluation/jersey_recon_benchmark.py` scores a JSONL manifest containing one row per stable track.

Example:

```json
{"track_id":"17","gt_number":"21","visible":true,"oracle_roi":[0.30,0.24,0.68,0.58],"auto_roi":[0.32,0.25,0.69,0.59],"oracle_prediction":"21","auto_prediction":"21","final_track_prediction":"21"}
```

ROIs may be normalized or pixel coordinates, provided `oracle_roi` and `auto_roi` for a row use the same coordinate system.

Run the example:

```bash
python evaluation/jersey_recon_benchmark.py \
  evaluation/jersey_recon_manifest.example.jsonl
```

Run the lightweight test:

```bash
python evaluation/jersey_recon_benchmark_test.py
```

## First experiment

Create a small frozen diagnostic set, initially around 50–100 stable player tracks.

For each track:

1. inspect the track without looking at model predictions;
2. record the true jersey number if known from reviewed visual evidence;
3. record whether the number is ever genuinely readable;
4. select the best human/oracle frame and number ROI;
5. record the current automatic ROI, if available;
6. run the same recognizer on the oracle crop and automatic crop;
7. record the final Identity-v2 track prediction separately.

The key diagnostic is the gap between `oracle_recognizer.accuracy_on_visible` and `automatic_recognizer.accuracy_on_visible`.

A large gap means the primary bottleneck is evidence retrieval/localization rather than recognition. A low oracle score means recognizer/domain quality remains a primary problem.

## Next research steps after the oracle result

Only after the first oracle benchmark is measured should this branch decide whether to invest in:

- track-level frame ranking specific to jersey visibility;
- front/back/side orientation evidence;
- localized chest/back number-region proposals;
- legibility scoring;
- multi-frame recognizer observations;
- recognizer comparison or fine-tuning.

Do not tune all of these simultaneously: keep the frozen benchmark so improvements can be attributed to the correct stage.
