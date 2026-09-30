from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_gate_d_tracker_policy_probe as base
from evaluation.p2_gate_d_tracker_policy_probe_v2 import _relation_windows


def main() -> None:
    geometry = base._load_json(base.DEFAULT_GEOMETRY)
    anchors = base._anchors_by_id(geometry)
    pairs = base._derive_probe_pairs(anchors)

    windows = _relation_windows(anchors, pairs, pad_sec=2.0)
    assert len(windows) == 2, windows

    # First continuous lifetime contains both the track-61 handoff and B9 SAME control.
    first_start, first_end = windows[0]
    assert first_start <= 737.0 + 1e-6, windows
    assert first_end >= 751.0 - 1e-6, windows

    # Second continuous lifetime contains both track-360 handoffs A23->A7->A23.
    second_start, second_end = windows[1]
    assert second_start <= 849.3 + 1e-6, windows
    assert second_end >= 861.783333 - 1e-6, windows

    # Every scored handoff/control must have both endpoints inside one and the same window.
    for key in ("handoffs", "stable_controls"):
        for pair in pairs[key]:
            left_ts = float(anchors[pair["left"]]["timestamp_sec"])
            right_ts = float(anchors[pair["right"]]["timestamp_sec"])
            covering = [
                (start, end)
                for start, end in windows
                if start <= left_ts <= end and start <= right_ts <= end
            ]
            assert len(covering) == 1, (key, pair, windows)

    print("p2_gate_d_tracker_policy_probe_v2_test: PASS")


if __name__ == "__main__":
    main()
