from __future__ import annotations

from math import sqrt

import numpy as np

from services.vision.game_player_appearance_v1 import GamePlayerAppearanceV1


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def _player_crop(*, head, torso, lower, height=120, width=48) -> np.ndarray:
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    a = height // 4
    b = (height * 3) // 4
    crop[:a] = np.asarray(head, dtype=np.uint8)
    crop[a:b] = np.asarray(torso, dtype=np.uint8)
    crop[b:] = np.asarray(lower, dtype=np.uint8)
    return crop


def test_descriptor_is_deterministic_and_normalized() -> None:
    descriptor = GamePlayerAppearanceV1()
    crop = _player_crop(head=(40, 80, 150), torso=(30, 30, 200), lower=(20, 20, 20))
    first = descriptor.describe(crop)
    second = descriptor.describe(crop.copy())
    assert first is not None and second is not None
    assert first == second
    assert abs(sqrt(sum(value * value for value in first)) - 1.0) < 1e-6


def test_uniform_brightness_change_remains_similar() -> None:
    descriptor = GamePlayerAppearanceV1()
    crop = _player_crop(head=(45, 80, 140), torso=(25, 35, 190), lower=(20, 20, 30))
    brighter = np.clip(crop.astype(np.float32) * 1.35, 0, 255).astype(np.uint8)
    first = descriptor.describe(crop)
    second = descriptor.describe(brighter)
    assert first is not None and second is not None
    assert _cosine(first, second) > 0.96


def test_full_body_difference_reduces_similarity() -> None:
    descriptor = GamePlayerAppearanceV1()
    first = descriptor.describe(
        _player_crop(head=(35, 75, 150), torso=(25, 30, 200), lower=(20, 20, 20))
    )
    second = descriptor.describe(
        _player_crop(head=(170, 120, 70), torso=(190, 190, 35), lower=(190, 190, 190))
    )
    assert first is not None and second is not None
    assert _cosine(first, second) < 0.70


def test_resize_preserves_coarse_spatial_descriptor() -> None:
    descriptor = GamePlayerAppearanceV1()
    small = _player_crop(head=(45, 80, 140), torso=(25, 35, 190), lower=(20, 20, 30), height=120, width=48)
    large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    first = descriptor.describe(small)
    second = descriptor.describe(large)
    assert first is not None and second is not None
    assert _cosine(first, second) > 0.99


def test_invalid_or_tiny_crop_abstains() -> None:
    descriptor = GamePlayerAppearanceV1()
    assert descriptor.describe(None) is None
    assert descriptor.describe(np.zeros((5, 5, 3), dtype=np.uint8)) is None
    assert descriptor.describe(np.zeros((20, 20), dtype=np.uint8)) is None


def main() -> None:
    test_descriptor_is_deterministic_and_normalized()
    test_uniform_brightness_change_remains_similar()
    test_full_body_difference_reduces_similarity()
    test_resize_preserves_coarse_spatial_descriptor()
    test_invalid_or_tiny_crop_abstains()
    print("game_player_appearance_v1_test: PASS")


if __name__ == "__main__":
    main()
