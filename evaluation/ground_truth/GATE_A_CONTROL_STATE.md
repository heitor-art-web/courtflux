# P2 Gate A — Control-State Correctness

## Purpose

Close the first P2 basketball-validation gate by proving that the canonical runtime can distinguish the temporal ball-control states required by downstream analytics:

- `unknown`
- `possessed`
- `loose`
- `pass_flight`
- `shot_flight`

Gate A is about control-state correctness only. Event existence, shot outcome, team-aware pass/turnover classification, rebounds and persistent identity are evaluated in later gates.

## Frozen evaluation set

Use the neutral V1–V5 clips from `evaluation/ground_truth/manifest.json` with:

- canonical P1/P2 runtime;
- Smart Ball OFF;
- Jersey OFF;
- `evaluation/match_configs/unknown.json`;
- human interval/window GT;
- no team-attribution scoring.

V1 is the development/diagnostic clip. V2–V5 are held as generalization clips.

## Scientific workflow

1. Run a canonical Jersey-OFF probe for each clip so source FPS and run metadata are measured rather than guessed.
2. Create V2–V5 human GT blind from video evidence only. Do not inspect `p2_states.jsonl`, `p2_events.jsonl` or evaluator output while authoring GT.
3. Freeze V2–V5 GT.
4. Before evaluating P2 predictions on V2–V5, freeze the numeric Gate-A acceptance thresholds using the human-GT class/interval distribution and the reliability needed by P3. The thresholds must not be chosen from observed P2 scores on V2–V5.
5. Use V1 for diagnosis and candidate runtime fixes. Any runtime change must have a demonstrated structural reason or a measured V1 failure; do not tune from intuition alone.
6. Once a candidate runtime is frozen on V1, evaluate exactly that runtime on V2–V5.
7. Produce per-clip and aggregate Gate-A metrics. If a generalization failure motivates another runtime change, re-evaluate the full V1–V5 set before accepting it.

## Primary Gate-A metrics

The primary human-GT interpretation is interval based:

- dominant predicted state per human interval;
- matching fraction per human interval;
- dominant-state interval accuracy;
- mean interval matching rate.

Frame-level accuracy and confusion remain diagnostic and must be reported, but coarse human intervals are not treated as millisecond-exact truth.

Also report, by truth state and clip:

- `unknown` rate;
- `possessed` coverage;
- `pass_flight` recall/confusions;
- `shot_flight` recall/confusions;
- `loose` recall/confusions;
- engine-ineligible frames inside labelled intervals.

## Frozen acceptance thresholds

These thresholds were frozen after blind V1–V5 human annotation and before inspecting P2 scores on V2–V5.
They are product-reliability gates, not calibration targets for an individual clip.

### Coverage sanity gate

- aggregate prediction coverage on human-labelled frames: **>= 0.98**;
- each individual clip: **>= 0.95**.

Coverage failure is treated as a runtime/evaluation-boundary defect rather than a semantic-control miss.

### Interval-continuity gate

Across all V1–V5 human control intervals:

- aggregate dominant-state interval accuracy: **>= 0.70**;
- aggregate mean interval matching rate: **>= 0.65**.

Generalization safeguard for every individual clip:

- dominant-state interval accuracy: **>= 0.50**;
- mean interval matching rate: **>= 0.45**.

### Core-state frame recall guardrails

Aggregate recall on human-labelled frames must be:

- `possessed`: **>= 0.60**;
- `pass_flight`: **>= 0.60**;
- `shot_flight`: **>= 0.60**.

For `loose`, require **>= 0.50** only when the frozen V1–V5 set contains at least two independently annotated loose-ball intervals. Otherwise report it as insufficient-GT rather than silently passing or failing it.

`unknown` has no positive-recall threshold because human `unknown` means evidence is insufficient, not a basketball state that P3 should optimize toward.

### Critical semantic-confusion guardrails

On aggregate labelled frames:

- truth `pass_flight` predicted as `shot_flight`: **<= 0.20**;
- truth `shot_flight` predicted as `pass_flight`: **<= 0.20**;
- predictions of `unknown` while truth is one of `possessed`, `pass_flight`, or `shot_flight`: **<= 0.30**.

These guardrails prevent a superficially acceptable overall score from hiding semantic failures that would corrupt P3 timelines and analytics.

## Live vs resolved contract

`control_state` / `live_control_state` is the online decision and remains the primary Gate-A result.

`resolved_control_state` may be reported as a secondary/provisional diagnostic channel. It must never hide or replace live metrics. `resolved_pass` experiments remain challengers unless separately promoted by full-set evidence.

## Non-goals

Gate A does not close:

- MADE/MISS correctness;
- team attribution;
- pass-vs-turnover classification with known teams;
- rebound correctness;
- persistent player identity;
- Jersey OCR.

## Gate closure

Gate A closes only when:

- V1–V5 all have reviewed frozen human control-state GT;
- one frozen canonical runtime has been evaluated on all five clips;
- the predeclared numeric acceptance thresholds above are met on the aggregate and required per-clip safeguards;
- remaining material confusions are documented and either accepted explicitly for P3 candidate/UNKNOWN handling or assigned to a later gate;
- focused P2 regressions remain green.

No threshold or runtime calibration is accepted from a single clip without full-set re-evaluation.
