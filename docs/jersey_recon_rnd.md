# Jersey Reconstruction R&D — v1-v5 closure

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

## Frozen experimental target

All v1-v5 comparison runs used the same Porto vs Ovarense 0-60 s control:

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
| v4 | OCR-blind contrast/component number-region proposals | 5.47% | 11 | 14.25% | Region numeric yield remained 87.77%, with 344 conflicting attempts and 341 low-consensus rejections. |
| v5 | official SoccerNet ResNet34 legibility gate -> v4 proposals -> PARSeq | 1.50% | 3 | 8.41% | Legibility passed only 29.47% of scored frames; 278/464 attempts had no legible frame. Coverage collapsed without enough consistency gain. |

## Main findings

- PARSeq confidence is not a reliable jersey-legibility confidence signal on these basketball crops.
- Multi-frame consensus is useful for rejecting unstable evidence, but cannot manufacture observability.
- Static geometric ROIs did not reliably localize jersey numbers.
- OCR-blind OpenCV region proposals improved some local consistency but did not create enough stable evidence.
- The official SoccerNet legibility classifier showed substantial domain shift on the Porto/Ovarense basketball broadcast control and reduced final coverage.
- The current no-GT phase does not justify a follow-on ViTPose / pose-guided ROI investment.
- None of the percentages above are verified accuracy because no reviewed CourtFlux GT was used.

## P2 boundary

- P2 validation remains **Jersey OFF**.
- P2 GT/calibration continues independently.
- Basketball/P2 thresholds remain frozen until reviewed GT provides evidence for calibration.
- Rosters remain validator-only and must not manufacture visual identity.
- `UNKNOWN` remains preferable to a false identity.

## Resume condition

Jersey research should only be reopened when at least one of these is true:

- CourtFlux-specific labelled jersey/legibility data is available;
- a frozen CourtFlux evaluation set can be created with acceptable human effort;
- a materially different pretrained model is available with direct evidence on comparable basketball broadcast footage;
- product requirements change enough to justify a new data-collection/fine-tuning phase.

When research resumes, the next credible phase is **CourtFlux-specific data + adaptation/fine-tuning**, not another round of ROI/threshold heuristics.

## Archived implementation

The experimental v1-v5 analyzers, probes and tests remain preserved on branch `research/jersey-recon-rnd`. They are intentionally not merged into production `main`; this document preserves the durable conclusions while keeping the production P1/P2 path clean.
