# Gate D V6 diagnostic findings

**Branch:** `validation/v6-abcd-batch-20260925`

**Status:** Gate D remains **OPEN**. Gates A/B/C remain frozen for this cycle.

This note records the V6 identity diagnostics performed after the first formal Gate-D failure. It is a diagnostic record, not a new Gate-D contract and not authorization to tune Gate-D thresholds on V6.

## Boundary

Gate D is game-local anonymous identity: "is this the same player inside this game?"

Gate D must not use jersey-number semantics or P5 evidence as an identity key or association signal. In particular, Gate D must not depend on:

- `shirt_num` / decoded jersey number;
- Jersey OCR acceptance or rejection;
- roster or player name;
- digit posteriors;
- embeddings/features from models trained specifically to recognize jersey numbers;
- human jersey labels as resolver input.

Human labels may use all visible evidence for ground-truth review, but resolver inputs remain jersey-blind. Persistent semantic identity remains P5.

## V6 frozen human score target

The V6 human Gate-D set contains:

- 16 SAME pairs;
- 20 DIFFERENT pairs;
- 7 event actor observations.

Required Gate-D checks remain:

- SAME continuity recall >= 0.85;
- DIFFERENT false-merge rate <= 0.05;
- event -> game-player attribution coverage >= 0.80 when visually resolvable;
- per-clip continuity recall >= 0.70;
- simultaneous same-player conflicts = 0.

## Formal V6 baseline after the first resolver patch

- SAME continuity: **0/16 = 0%** — FAIL;
- DIFFERENT false merges: **1/20 = 5%** — at the formal limit;
- event -> game-player coverage: **7/7 = 100%** — PASS;
- simultaneous conflicts: **0** — PASS.

The main defect is identity continuity/fragmentation, not actor coverage.

## Diagnostic approaches tested on V6

These experiments were diagnostic only. None is approved for runtime integration.

| Approach | SAME / handoff result | DIFFERENT / false merge | Other signal | Decision |
| --- | ---: | ---: | ---: | --- |
| Generic OSNet/MSMT17 ReID | 1/16 = 6.25% | 0/20 | AUC 57.50% | reject |
| SportsReID OSNet/SoccerNet | 1/16 = 6.25% | 0/20 | AUC 58.44% | reject |
| Temporal torso/tracklet signature | 5/16 = 31.25% | 1/20 = 5% | AUC 74.53%; later audit found invalid co-visibility constraints on some SAME pairs | reject as implemented; not sufficient signal |
| Local SportsReID handoff detector | 1/3 = 33.33% | 1/1 SAME control false alarm | 60 change-points | reject |
| BoT-SORT / ByteTrack policy probe | 0/3 known handoffs split | 3/7 nearby false merges | 30 vs actual FPS made no difference | reject stock tracker-policy explanation |
| Detection-level geometry/motion/team/torso assignment | 0/3 handoffs for every ablation | 3/7 nearby false merges | all four ablations identical | reject frame-greedy association |
| Anonymous global micro-tracklet + beam association | **3/16 = 18.75%** | **2/20 = 10%** | 22/22 anchors mapped; 0 simultaneous conflicts | reject |

## Structural findings

1. Raw `track_id` is not a human identity. The same raw track can transfer between people.
2. The same human is frequently fragmented across several raw tracks.
3. Most V6 SAME examples require cross-track recovery, while at least some raw tracks also contain within-track identity switches.
4. Geometry/motion can help local continuity but do not provide enough information to recover the same player after long gaps or camera changes.
5. Team evidence cannot solve same-team crossings and can itself be wrong at critical frames.
6. Generic full-body ReID is not discriminative enough for these broadcast basketball crops.
7. Global anonymous assignment with the currently available non-semantic signals does not approach the frozen Gate-D target.

## Development-data consequence

Further Gate-D architecture/feature/weight work must not be selected on V6.

The existing development Gate-D human set in `evaluation/ground_truth/gate_d_human_observations_v1.json` currently has only:

- **2 SAME pairs**;
- **5 DIFFERENT pairs**.

That set is adequate as an early smoke/regression set but is too small to select a serious identity architecture against an 85% SAME / 5% false-merge contract.

Before another Gate-D architecture is tuned, expand prediction-blind human identity ground truth on **V1/V3/V4/V5**. The expanded set should include:

- same-player continuity across raw-track changes;
- same-team crossings/occlusions;
- within-raw-track human switches;
- short and long reappearances;
- hard DIFFERENT pairs from the same team;
- simultaneous/covisible negatives.

Candidate generation may use timing, geometry and raw-track structure to reduce review burden, but human identity labels must remain independent of `game_player_id` predictions. Jersey may be visible to the human reviewer, but must not become resolver input.

## Next engineering gate

Do not integrate another V6 identity resolver from these diagnostics.

Next sequence:

1. expand Gate-D development GT on V1/V3/V4/V5;
2. establish a development baseline on that expanded set;
3. develop anonymous identity only on the development corpus;
4. freeze architecture/features/weights/thresholds;
5. run final validation only after the development Gate-D contract passes;
6. keep P5 jersey/semantic identity separate.

V6 remains a documented failure/diagnostic clip for this branch, not a source of further tuning decisions.
