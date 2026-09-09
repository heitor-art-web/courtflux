from __future__ import annotations

from services.pipeline import Pipeline as BasePipeline
from services.vision.ball_context_features import BallContextFeatureEnricher


class Pipeline(BasePipeline):
    """Experimental pipeline that enriches OC-SORT hypotheses with court/player context.

    The base pipeline runs unchanged. Context is attached only after camera projection and
    detector processing have completed, so it cannot influence association or concrete
    ball output in this experiment.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ball_context_enricher = BallContextFeatureEnricher()

    def process(self, frame_state):
        frame_state = super().process(frame_state)
        if not bool(getattr(frame_state, "valid_gameplay", False)):
            return frame_state
        diagnostics = self.ball_context_enricher.enrich(frame_state)
        frame_state.diagnostics.setdefault("analytics", {})["ball_context"] = diagnostics
        return frame_state
