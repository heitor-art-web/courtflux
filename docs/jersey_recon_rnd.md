# Jersey Reconstruction R&D

This branch is an isolated research track for Jersey reconstruction. It starts from `main` and must not depend on the active P2 GT/calibration branch.

## Status

**CLOSED / PAUSED pending CourtFlux-specific data.**

The v1-v5 research sequence is complete for the current no-GT phase. The experiments produced enough evidence to stop investing in the current SoccerNet/PARSeq retrieval stack without CourtFlux-specific labelled data.

Operational decision:

1. keep Jersey disabled for the frozen P2 validation baseline;
2. do not spend more engineering time on this Jersey architecture without CourtFlux-specific data;
3. do not proceed to pose-guided ROI / ViTPose from the current v5 result;
4. if Jersey research resumes, the next phase must be data-driven adaptation/fine-tuning rather than additional ROI/threshold heuristics;
5. Identity-v2 remains the only downstream cross-attempt identity consensus layer.

This is a research stop decision, not a claim that Jersey recognition is impossible. The current experiments did **not** use reviewed human ground truth, so the percentages below are diagnostic identification/yield/consistency metrics and must not be described as verified accuracy.

## Research question

The central question was not "which OCR model is best?". It was:

> When a player's jersey number is genuinely observable somewhere in a track, can CourtFlux retrieve the right frame/ROI and preserve enough visual evidence for a recognizer to read it?

The research decomposed the problem into:

1. **Visibility / legibility** — whether a usable observation exists.
2. **ROI localization** — whether the automatic crop isolates the jersey-number evidence.
3. **Recognizer behaviour** — whether the recognizer reads a tight usable crop without hallucinating numbers from weak evidence.
4. **Track-level identification** — whether repeated visual evidence is stable enough for temporal consensus.

## Branch boundary

Do not modify the production P1/P2 control path from this research branch merely to improve Jersey metrics.

In particular:

- P2 GT/calibration continues independently on `validation/p2-gt-calibration`.
- The frozen P2 validation baseline remains **Jersey OFF**.
- Identity-v2 remains the single downstream temporal consensus/locking layer.
- Rosters remain validator-only and must not manufacture a visual number.
- `UNKNOWN` remains preferable to a false identity.
- Basketball/P2 thresholds remain frozen until human GT provides evidence for calibration.

## Frozen experimental target

All v1-v5 comparison runs below used the same Porto vs Ovarense 0-60 s control:

```text
video             https://www.youtube.com/watch?v=OwiQgAMqyUs
model             weights/model_v1.1.0.pt
player conf       0.40
imgsz             960
device            mps
tracker           botsort.yaml
jersey backend    parseq_soccernet
match config      data/match_config.json
```

Known rosters were validator-only. They were never allowed to select or rewrite a visual prediction.

## v1-v5 results

| Version | Retrieval / gating strategy | Jersey identification | Identified tracks | Consensus numeric yield | Main observation |
| --- | --- | ---: | ---: | ---: | --- |
| v1 | single torso crop -> PARSeq | 13.2% | 23 | n/a | PARSeq emitted numeric outputs on ~95% of attempts; 232 `raw_not_in_roster` rejections showed severe weak-crop hallucination/eagerness. |
| v2 | multi-frame torso evidence -> consensus | 4.31% | 8 | 15.57% | `raw_not_in_roster` collapsed to 18, but 375 attempts failed low multi-frame consensus. Filtering noise worked, coverage collapsed. |
| v3 | multi-frame + geometric localized torso ROIs | 4.57% | 7 | 8.53% | 452/457 attempts contained conflicting ROI outputs; geometric boxes did not reliably localize the number. |
| v4 | OCR-blind contrast/component number-region proposals | 5.47% | 11 | 14.25% | Learned-looking visual structure was still not isolated: region numeric yield remained 87.77%, with 344 conflicting attempts and 341 low-consensus rejections. |
| v5 | official SoccerNet ResNet34 legibility gate -> v4 proposals -> PARSeq | 1.50% | 3 | 8.41% | Legibility passed only 29.47% of scored frames; 278/464 attempts had no legible frame. Coverage collapsed without enough consistency gain. |

### v1 — single-frame PARSeq baseline

Key diagnostics:

```text
attempts                         438
identified tracks                23
jersey identification          13.2%
raw numeric yield              95.2%
raw_not_in_roster                232
debug accepted                   153
```

The recognizer was too eager: generic crop quality and PARSeq confidence did not separate useful jersey evidence from unreadable texture. Long tracks frequently produced many different numbers.

Conclusion: the 13.2% identification metric could not be treated as recognizer accuracy; evidence acquisition/localization was already a major bottleneck.

### v2 — multi-frame evidence

Key diagnostics:

```text
attempts                         456
identified tracks                 8
jersey identification           4.31%
consensus numeric outputs          71
consensus numeric yield         15.57%
raw_not_in_roster                 18
low_multiframe_consensus         375
```

Per-crop numeric yield remained ~95%, but multi-frame disagreement exposed the instability. The method successfully rejected much of the hallucinated evidence, at the cost of coverage.

Conclusion: temporal evidence fusion was useful as a rejection mechanism, but the input crops were not reliably observing the number.

### v3 — localized geometric ROIs

Key diagnostics:

```text
attempts                         457
identified tracks                 7
jersey identification           4.57%
consensus numeric outputs          39
consensus numeric yield          8.53%
ROI outputs                      6632
ROI numeric yield               93.52%
ROI-output conflicts              452
low_multiframe_consensus          418
```

`center` was the best geometric ROI by consistency contribution, but nearly every attempt still produced contradictory readings across regions.

Conclusion: static upper/center/lower torso geometry was not a number localizer.

### v4 — visual number-region proposals

Key diagnostics:

```text
attempts                         456
identified tracks                11
jersey identification           5.47%
consensus numeric outputs          65
consensus numeric yield         14.25%
region proposal outputs          3271
region numeric yield            87.77%
attempts without proposals         39
region-output conflicts           344
low_multiframe_consensus          341
```

The proposal scorer improved consistency versus v3 and recovered consensus yield close to v2, so localization quality mattered. However, PARSeq still emitted a number for almost 88% of proposed regions and the final identification metric remained low.

Conclusion: better proposals help, but contrast/components/geometry are not sufficient to isolate jersey-number evidence robustly.

### v5 — official SoccerNet legibility gate

The v5 experiment added the original Koshkina/Elder SoccerNet ResNet34 legibility classifier before the otherwise unchanged v4 proposal/PARSeq stack.

Key diagnostics:

```text
attempts                         464
identified tracks                 3
jersey identification           1.50%
consensus numeric outputs          39
consensus numeric yield          8.41%
legibility frames scored         4917
legibility frames passed         1449
legibility frames rejected       3468
legibility pass rate            29.47%
attempts without legible frames   278
region numeric yield            90.02%
low_multiframe_consensus          143
legibility_rejected               278
```

The classifier behaved as a strong gate but removed too much usable coverage for this basketball broadcast domain. The surviving region crops still produced numeric outputs at ~90%, so the downstream recognizer remained eager even after the legibility filter.

Conclusion: the official SoccerNet legibility gate did not justify a follow-on ViTPose/pose-guided ROI investment in the current no-GT phase.

## Final technical conclusions

The v1-v5 sequence supports the following conclusions for the current CourtFlux domain:

- The main failure is not solved by swapping thresholds or adding more static ROIs.
- PARSeq confidence is not a reliable jersey-legibility confidence signal on these basketball crops.
- Multi-frame consensus is valuable for rejecting unstable evidence, but cannot manufacture observability.
- Geometric ROI proposals and OpenCV number-region proposals improve some local consistency but do not create enough stable evidence.
- The official SoccerNet legibility classifier shows meaningful domain shift on the Porto/Ovarense broadcast control: it rejects most observations and reduces final coverage substantially.
- Because there is no reviewed CourtFlux GT, none of the reported percentages are verified accuracy.
- The next credible research phase requires CourtFlux-specific labelled examples and adaptation/fine-tuning. More heuristic engineering on the current stack is explicitly out of scope.

## Resume condition

Jersey research should only be reopened when at least one of these is true:

- CourtFlux-specific labelled jersey/legibility data is available;
- a frozen CourtFlux evaluation set can be created with acceptable human effort;
- a materially different pretrained model is available with direct evidence on comparable basketball broadcast footage;
- product requirements change enough to justify a new data-collection/fine-tuning phase.

Until then:

> **Jersey remains experimental, P2 validation remains Jersey OFF, and no further engineering is planned on the current SoccerNet/PARSeq retrieval architecture.**

## Historical benchmark scaffold

`evaluation/jersey_recon_benchmark.py` and the example manifest remain available for a future labelled phase. They separate visibility, ROI localization, recognizer conditional accuracy and final track-level identification.

If the project later creates reviewed GT, the benchmark should be used to distinguish evidence-retrieval failures from recognizer failures before any new model tuning.
