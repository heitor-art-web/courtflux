# P2 human ground truth

This directory is the frozen human-GT surface for the P2 exit gate.

## Scientific contract

- Annotate from video evidence, not from P2 event predictions.
- Keep P2 basketball thresholds frozen until the first reviewed GT evaluation is complete.
- Run all basketball GT/baseline probes with Jersey OFF.
- Prefer `unknown`/null to guessed actor, team or outcome labels.
- Do not copy event times, event types or control states from `p2_events.jsonl` or `p2_states.jsonl` into GT.
- The V1–V5 clips use `evaluation/match_configs/unknown.json` and therefore do not measure team attribution, pass-vs-turnover classification or team-aware rebound semantics.
- `PORTO_OVARENSE_60S` is the known-team control and uses `data/match_config.json` explicitly.

## Two-pass annotation workflow

### Pass A — blind basketball truth

Watch only the frozen clip and annotate:

1. control-state intervals: `unknown`, `possessed`, `loose`, `pass_flight`, `shot_flight`;
2. possession acquisition/change;
3. loose-ball start/recovery/end;
4. pass/turnover truth where team context is known;
5. shot attempts and outcome (`made`, `missed`, `unknown`);
6. human shooter/actor note when visually identifiable.

Do not inspect P2 predictions during Pass A.

### Pass B — attribution alignment

After basketball truth is frozen, use a tracking-only diagnostic/preview to map visually identified actors to stable `track_id` values. Do not display P2 event labels while doing this alignment. If the actor cannot be mapped reliably, leave `actor_track_id` or `secondary_track_id` null.

For V1–V5, team fields stay null. For Porto/Ovarense, team truth may be annotated as `FC Porto` or `Ovarense` when visually clear.

## GT file format

Use `courtflux_p2_gt_v1`, matching `evaluation/p2_ground_truth_template.json`.

Create each GT file only after the corresponding canonical Jersey-OFF probe exists. Copy source/window/FPS from the run summary rather than guessing FPS. The manifest lists the frozen source and window but intentionally does not invent source FPS.

Example event:

```json
{
  "event_type": "shot_attempt",
  "frame": 612,
  "outcome": "missed",
  "team_id": "FC Porto",
  "actor_track_id": 144,
  "secondary_track_id": null,
  "human_actor_note": "shooter visually confirmed; track mapped in Pass B"
}
```

Example control interval:

```json
{
  "state": "possessed",
  "frame_start": 580,
  "frame_end": 603,
  "human_note": "clear controlled dribble"
}
```

Extra human-only fields such as `human_note` are allowed and ignored by the evaluator.

## Frozen evaluation set

See `manifest.json` for V1–V5 plus `PORTO_OVARENSE_60S`.

The real-video probe now defaults to the neutral/unknown config. A frozen V1 command is therefore:

```bash
python evaluation/p2_real_video_probe.py \
  --video "https://www.youtube.com/watch?v=hILf-ySowIo" \
  --start 10 \
  --end 40 \
  --model weights/model_v1.1.0.pt \
  --device mps \
  --match-config evaluation/match_configs/unknown.json \
  --no-jersey
```

The known-team control must override that default explicitly:

```bash
python evaluation/p2_real_video_probe.py \
  --video "https://www.youtube.com/watch?v=OwiQgAMqyUs" \
  --start 0 \
  --end 60 \
  --model weights/model_v1.1.0.pt \
  --device mps \
  --match-config data/match_config.json \
  --no-jersey
```

Before accepting a V1–V5 run for GT, verify its summary reports `home_team = null`, `away_team = null` (or equivalent unknown team metadata) and no team-attributed P2 events. A known-team label in a V1–V5 summary invalidates that run for the frozen neutral battery.

## Evaluator

```bash
python evaluation/p2_ground_truth.py \
  --events evaluation/runs/<RUN_ID>/p2_events.jsonl \
  --states evaluation/runs/<RUN_ID>/p2_states.jsonl \
  --ground-truth evaluation/ground_truth/<CLIP>.json \
  --tolerance-frames 8 \
  --output evaluation/runs/<RUN_ID>/p2_gt_eval.json
```

The current evaluator reports per-event TP / FP / FN, precision / recall / F1 and mean absolute temporal frame error, plus control-state frame accuracy/confusion. Aggregate scoring can be added without changing the GT schema.

Threshold changes are allowed only after these metrics are reviewed across the frozen set. Any calibration must be re-evaluated against the full set, not only the clip that motivated the change.
