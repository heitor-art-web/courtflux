from __future__ import annotations

import numpy as np

from evaluation.p2_gate_d_reid_tracklet_challenger import (
    aggregate_tracklet_features,
    tracklet_pair_metrics,
)


def main() -> None:
    left_same = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.1, 0.0],
            [0.97, -0.1, 0.0],
        ],
        dtype=np.float32,
    )
    right_same = np.asarray(
        [
            [0.99, 0.02, 0.0],
            [0.96, 0.08, 0.0],
            [1.0, -0.04, 0.0],
        ],
        dtype=np.float32,
    )
    right_different = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.1, 0.98, 0.0],
            [-0.08, 0.97, 0.0],
        ],
        dtype=np.float32,
    )

    centroid, coherence = aggregate_tracklet_features(left_same)
    assert centroid.shape == (3,)
    assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-6
    assert coherence > 0.99

    same = tracklet_pair_metrics(left_same, right_same)
    different = tracklet_pair_metrics(left_same, right_different)
    assert same["centroid_cosine_similarity"] > 0.99
    assert same["pairwise_top_quartile_mean"] > 0.99
    assert different["centroid_cosine_similarity"] < 0.10
    assert same["centroid_cosine_similarity"] > different["centroid_cosine_similarity"]
    print("p2_gate_d_reid_tracklet_challenger_test: PASS")


if __name__ == "__main__":
    main()
