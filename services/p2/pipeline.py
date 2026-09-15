from __future__ import annotations

import os
from typing import Any

from services.pipeline import Pipeline


class P2Pipeline(Pipeline):
    """Compatibility wrapper that forces the canonical Pipeline's P2 stage on.

    The old reconstruction adapter used to inject a second P2 engine after P1 state.
    P2 now lives on the single canonical runtime path, so this wrapper only sets the
    opt-in flag before base initialization. It intentionally does not process P2 twice.
    """

    def __init__(self, *args: Any, p2_enabled: bool | None = None, **kwargs: Any) -> None:
        if p2_enabled is None:
            p2_enabled = True
        if p2_enabled:
            os.environ["BASKETBALL_P2_ENABLED"] = "1"
        else:
            os.environ["BASKETBALL_P2_ENABLED"] = "0"
        super().__init__(*args, **kwargs)
