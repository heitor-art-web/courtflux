# P1 Jersey / Identity Exit Gate

> [!IMPORTANT]
> ## HISTORICAL / SUPERSEDED
> This document is retained as the technical history of the pre-merge P1 Jersey/Identity gate. It is **not** an active release gate or current operational instruction set.
>
> PR #43 was merged on 2026-09-15. That merge closed P1 while explicitly deferring final Jersey/Identity correctness to the end of P2. The current sources of truth are `README.md` and `docs/p2_roadmap.md`.
>
> In particular, the historical instruction near the end of this file saying **“Do not merge PR #43” is superseded and no longer applies**. The validation sequence below is preserved only to document the reasoning and experiments that led to the P1 closure decision.

This is the remaining P1 exit gate for CourtFlux identity quality.

P1 does **not** require a formal human-ground-truth dataset to close this gate. The release decision is based on disciplined visual QA of the integrated Porto/Ovarense reference window, with conservative abstention and explicit diagnostics. Human GT tooling remains available for deeper investigation, model comparison, or later accuracy measurement, but it is not a blocking P1 dependency.

Do not report formal Jersey/team accuracy without ground truth. Runtime coverage, acceptance rate, lock rate, stability, and visual observations are diagnostics rather than accuracy claims.

## Frozen P1 operating point

Use the accepted integrated runtime contract:

```text
detector              weights/model_v1.1.0.pt
tracker               botsort.yaml
player confidence     0.40
inference confidence  0.03
ball confidence       0.075
imgsz                  960
single-forward        ON
Smart Ball            OFF
Jersey                ON
```

Reference footage: FC Porto vs Ovarense, integrated `0–60 s` validation.

Reference baseline before Identity v2:

```text
run                    20260914_140057_772692
analysis FPS           18.60
detector               44.81 ms/frame
total pipeline         49.43 ms/frame
ball effective traj.   89.51%
team coverage          99.5%
```

Team coverage is not team accuracy. Jersey identification/acceptance is not Jersey accuracy.

## Identity v2 scope

Identity v2 is the P1 challenger that must be evaluated before adding segmentation to the critical path.

### A. Best-crop temporal buffer

Do not use only the most recent eight torso crops. Maintain a bounded set of the strongest observations per track, with an age penalty so stale crops can eventually be displaced.

Crop quality should prefer readable evidence using inexpensive image properties such as sharpness, contrast, exposure and useful crop resolution.

### B. Match-adaptive team prototypes

Retain the active-kit HSV classifier as the conservative seed. Learn lightweight match-specific colour prototypes only from strong accepted observations, then use them as a bounded correction signal rather than an independent decoder.

The prototype path must not self-train from ambiguous evidence.

### C. Team evidence with decay/window and reversible correction

Team evidence must use a finite temporal window with decay. A previous team label may be corrected when sustained newer evidence wins by a stronger threshold.

A team reversal invalidates the old player-number identity. The safe output after a reversal is `UNKNOWN` until Jersey evidence supports a new identity.

### D. Raw Jersey temporal consensus

Run the existing single-frame Jersey checkpoint on the selected crops independently. Build temporal consensus from the **raw decoded numbers** across crops.

Do not average evidence and then search the roster for the most convenient number.

### E. Roster is validator-only

The roster may answer only:

```text
Is the raw temporal number compatible with this team's known roster?
```

The roster must never answer:

```text
Which roster number best explains these weak digit probabilities?
```

If raw evidence and roster disagree, abstain. `UNKNOWN` is preferable to a wrong identity.

### F. Reversible identity lock

Identity lock is a confidence state, not a permanent latch.

A locked track continues to receive lower-cadence team/Jersey revalidation. Sustained contradictory evidence can unlock the identity back to `UNKNOWN`; stronger consistent evidence may later relock it. Do not stop all team/Jersey evidence collection merely because an identity was once locked.

## SAM2 challenger policy

SAM2 is **not** part of the initial Identity v2 critical path.

Only after the Identity v2 Porto/Ovarense comparison should SAM2 be considered as an optional challenger for difficult observations, for example:

- identity still uncertain after temporal consensus;
- torso crop polluted by another player/background;
- partial occlusion where a mask could materially improve the number crop.

If tested, SAM2 should refine mask/crop evidence only. It must not become an identity decoder, and it must earn its runtime cost through visible improvement in hard cases.

## Validation sequence

1. Run the Identity v2 unit/regression checks.
2. Run the same Porto/Ovarense `0–60 s` integrated configuration used by the accepted baseline.
3. Keep Jersey ON and Smart Ball OFF.
4. Produce a comparable visual preview/video plus per-track diagnostics.
5. Review team + jersey number together, focusing on wrong confident identities, premature locks, corrections, and `UNKNOWN` behavior.
6. Compare runtime/performance against `20260914_140057_772692` so any identity gain has an explicit cost.
7. Decide whether Identity v2 is visually sufficient for P1 or whether SAM2 deserves a targeted challenger run.

## Visual QA rubric

The visual gate is intentionally conservative.

### Pass signals

- obvious Porto/Ovarense team assignments remain stable when visual evidence is clear;
- previously wrong team assignments can correct rather than remaining permanently latched;
- jersey numbers shown as identified are visually plausible across multiple observations;
- roster membership never rewrites a different raw number into an apparently valid identity;
- uncertain or contradictory tracks remain/return `UNKNOWN`;
- identity does not lock immediately from one weak crop;
- lock revisions are explainable in diagnostics;
- no visible regression in player/referee separation, CourtState, BallState, or ball continuity;
- runtime remains acceptable relative to the P1 baseline.

### Fail signals

- repeated confidently wrong team + number combinations;
- roster-driven substitutions unsupported by raw digit evidence;
- permanent wrong locks that cannot self-correct;
- material identity flicker caused by over-reacting to single frames;
- segmentation/model cost without visible identity benefit;
- regression in accepted P1 subsystems.

There is no numeric accuracy threshold in this no-GT visual gate. If the visual evidence is ambiguous, record the limitation rather than converting coverage into an accuracy claim.

## Optional human GT

The existing `docs/jersey_ground_truth.md` and evaluation scripts remain useful for an auditable later correctness study. They may also be used during P1 if visual QA reveals a disputed failure mode that benefits from formal measurement.

Their output must not be mixed with no-GT runtime coverage metrics. Any formal precision/recall/accuracy statement requires the corresponding reviewed GT.

## P1 exit condition

P1 Jersey/Identity may be declared complete only after all of the following are true:

1. Identity v2 has been exercised on the same Porto/Ovarense `0–60 s` reference window;
2. the integrated visual QA is explicitly accepted for team + jersey identity;
3. `UNKNOWN` remains the fallback for unsupported identity;
4. no accepted P1 subsystem shows a material regression;
5. performance impact is recorded against the final baseline;
6. the decision on SAM2 is recorded as either unnecessary for P1 or justified by a targeted challenger result;
7. PR #43 is reviewed for final integration.

**Do not merge PR #43 without explicit authorization.**
