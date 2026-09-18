# P2 team colour space contract

## Purpose

Gate C must identify stable team slots (`team_a` / `team_b`) without treating a finite list of colour names as the team identity or as the complete visual decision space.

## Runtime contract

`TeamOnlyAnalyzer` uses continuous CIELAB colour prototypes.

A kit colour can be configured as:

- a legacy colour-name alias, for backward compatibility;
- a hex value such as `#F07820`;
- an RGB prototype such as `{ "name": "custom_orange", "rgb": [240, 120, 32] }`;
- an HSV prototype such as `{ "name": "custom_teal", "hsv": [185, 0.85, 0.80] }`.

Arbitrary numeric colours do not require a new Python branch, mask or class.

Legacy names are converted to numeric prototypes before inference. They are configuration aliases only; named masks are not used in the team decision.

## Multi-colour kits

A team may have multiple torso prototypes. Per-pixel affinity is computed continuously in CIELAB and the strongest relevant prototype contributes to the team score.

Colours shared or nearly shared by both teams are continuously downweighted according to prototype proximity. An identical shared colour therefore carries little discriminatory weight, while a unique colour carries substantially more.

## Identity-v2 compatibility

The historical `color_coverage` field remains available because Identity v2 consumes a fixed-length temporal vector. Its semantics changed from named binary-mask coverage to continuous affinity against a fixed anchor basis (`Black`, `Grey`, `White`, `Blue`, `Navy`, `Red`, `Green`, `Yellow`).

This anchor basis is an embedding basis, not a closed supported-colour list. Orange, purple, cyan, custom RGB colours and other arbitrary prototypes project continuously onto it.

`score_teams()` also emits a 4x4x4 normalized CIELAB histogram as `color_embedding` for diagnostics and future temporal-model migration.

## Human kit audit

The prediction-blind `human_kit_audit` in V1–V5 remains the source of reviewed equipment context.

- `jersey_body` is candidate classifier evidence.
- `shorts` and `lettering` are retained as human context and are not automatically injected into torso classification.
- colour names are human-readable metadata; they are not semantic team identity.
- a reviewed colour in GT does not by itself prove that the current numeric prototype is calibrated for the camera/video. Colour-capability validation is performed before Gate C is rerun and closed.

## Validation order

1. Unit-test arbitrary numeric colours and multi-colour kits.
2. Validate colour evidence on V1/V3/V4/V5 against the human kit audit.
3. Diagnose remaining temporal team-profile errors separately from instantaneous colour evidence.
4. Freeze colour representation and temporal policy.
5. Rerun Gate C with the frozen thresholds.
6. Keep V6/V7 unseen until the final P2 Exit Gate.
