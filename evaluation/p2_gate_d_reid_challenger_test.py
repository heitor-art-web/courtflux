from __future__ import annotations

import numpy as np

from evaluation.p2_gate_d_reid_challenger import EndpointCrop, cosine_similarity, evaluate_pairs


def _endpoint(observation_id: str, index: int) -> EndpointCrop:
    return EndpointCrop(
        observation_id=observation_id,
        endpoint_index=index,
        frame_id=index,
        timestamp_sec=float(index),
        track_id=index + 1,
        episode_index=0,
        box=(0, 0, 2, 2),
        crop=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def main() -> None:
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    pairs = [
        (
            {"id": "same", "clip": "v3", "label": "SAME"},
            [_endpoint("same", 0), _endpoint("same", 1)],
        ),
        (
            {"id": "different", "clip": "v3", "label": "DIFFERENT"},
            [_endpoint("different", 0), _endpoint("different", 1)],
        ),
    ]

    def fake_extractor(crops: list[np.ndarray]) -> np.ndarray:
        assert len(crops) == 4
        return np.asarray(
            [
                [1.0, 0.0],
                [0.98, 0.2],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    result = evaluate_pairs(pairs, extractor=fake_extractor)
    rows = {row["id"]: row for row in result["pairs"]}
    assert rows["same"]["osnet_cosine_similarity"] > 0.97
    assert rows["different"]["osnet_cosine_similarity"] == 0.0
    assert result["same_similarity_min"] > result["different_similarity_max"]
    assert result["separation_margin"] > 0.97
    print("p2_gate_d_reid_challenger_test: PASS")


if __name__ == "__main__":
    main()
