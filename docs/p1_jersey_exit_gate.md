# P1 Jersey Exit Gate

This is the final P1 Jersey correctness gate before CourtFlux can declare the P1 exit contract complete.

The gate is human-ground-truth based. No recognizer is promoted from no-GT coverage/stability evidence alone.

## Frozen P1 operating point

Use the current integrated P1 detector/tracker contract:

```text
detector            weights/model_v1.1.0.pt
tracker             botsort.yaml
player confidence   0.40
inference confidence 0.03
imgsz               960
Jersey              ON
```

Reference footage: FC Porto vs Ovarense, `data/OwiQgAMqyUs.mp4`.

## 1. Build the human GT

Start with 40–60 representative current tracks. Keep both readable-number and genuinely unreadable player evidence. Labels must be blind to model predictions and must never be inferred from roster membership.

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

Annotate:

```bash
streamlit run tools/jersey_ground_truth_app.py -- \
  --manifest data/jersey_ground_truth/porto_ovarense_p1_v110/manifest.csv \
  --reviewer heitor
```

For real players, use only the visually supported number or `unreadable`. Use `identity_group` where fragmented track IDs are known to be the same physical player.

Validate and export:

```bash
PYTHONPATH=. python tools/jersey_ground_truth.py validate \
  --manifest data/jersey_ground_truth/porto_ovarense_p1_v110/manifest.csv

PYTHONPATH=. python tools/jersey_ground_truth.py export \
  --manifest data/jersey_ground_truth/porto_ovarense_p1_v110/manifest.csv \
  --output data/jersey_ground_truth/porto_ovarense_p1_v110/ground_truth.csv
```

## 2. Evaluate candidate recognizers on the same GT

Both candidates must use exactly the same human-reviewed GT and P1 detector/tracker operating point.

Candidates currently under review:

```text
best_temporal.pt                 temporal EfficientNet-B0 + BiLSTM
uncertainty-aware ViT-B          frozen #37 uncertainty policy
```

A candidate summary used by the exit gate must expose the following under `evaluation.all` (or `gate_metrics`):

```text
numeric_tracks
unreadable_tracks
decision_precision
unreadable_false_accept_rate
correct_recall
```

No-GT operational acceptance/coverage is supporting evidence only and cannot satisfy this gate.

## 3. Promotion rules

A candidate passes only if all of the following are true:

```text
human-reviewed tracks                 >= 40
selective / decision precision        >= 90%
false accept on UNREADABLE            <= 5%
correct recall                        measured
```

Among candidates that pass, the candidate with the highest `correct_recall` wins.

If neither candidate passes, promote neither. `UNKNOWN` remains preferable to a forced jersey identity.

If multiple candidates pass and their correct recall is within 10 percentage points on the initial 40–60-track GT, extend the GT toward approximately 100 reviewed tracks before promotion.

Run the gate with:

```bash
PYTHONPATH=. python evaluation/p1_jersey_exit_gate.py \
  --candidate best_temporal=/path/to/best_temporal/summary.json \
  --candidate vit_b=/path/to/vit_b/summary.json \
  --output evaluation/p1_jersey_exit_gate_report.json
```

Possible decisions:

```text
PROMOTE        one candidate passes and wins by correct recall
EXTEND_GT      multiple candidates pass but the result is too close on the small GT
NO_PROMOTION   no candidate satisfies the correctness gate
```

## 4. P1 exit condition

After the Jersey gate resolves:

1. wire the promoted recognizer/policy into the integrated Jersey runtime, or retain abstention/baseline behavior if no candidate passes;
2. rerun the Porto/Ovarense integrated `0–60 s` P1 validation with Jersey ON and Smart Ball OFF;
3. confirm no regression in PlayerState, BallState, CourtState, team identity, visual QA or runtime performance;
4. record the final metrics and Jersey decision in the README;
5. close PR #43 and declare the P1 exit / v1.2.0 release gate complete.

The Jersey promotion decision is about correctness under uncertainty, not maximum raw coverage.
