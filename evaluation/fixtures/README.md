# Frozen P2 evaluation fixtures

These files are frozen regression inputs for CourtFlux P1/P2 validation. They are evaluation assets, not runtime dependencies.

## `g4UROqJoYBM.mp4`

Small real-video fixture carried forward from the legacy `p2` branch. Keep the bytes frozen so event/control-state regressions remain comparable across P2 revisions.

Use this fixture with Jersey OFF and Smart Ball OFF when validating the rebuilt P2 motor. Do not tune thresholds only against this clip; it belongs to the multi-video frozen validation set alongside Porto–Ovarense and the remaining local v1–v5 fixtures when available.

The fixture is intentionally committed to Git because it is small enough for deterministic CI/evaluation use.
