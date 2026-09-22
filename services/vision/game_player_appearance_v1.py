from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GamePlayerAppearanceConfig:
    grid_rows: int = 6
    grid_cols: int = 3
    horizontal_margin_ratio: float = 0.08
    min_height_px: int = 12
    min_width_px: int = 6


class GamePlayerAppearanceV1:
    """Cheap deterministic full-player appearance descriptor for Gate-D development.

    This is deliberately not a person-ReID model. It keeps spatially coarse full-body
    colour/brightness/texture evidence so head, torso, shorts and footwear can all help
    distinguish teammates. Per-cell chromaticity and relative luminance make the vector
    less sensitive to uniform exposure changes than raw RGB/BGR means.
    """

    def __init__(self, config: GamePlayerAppearanceConfig | None = None) -> None:
        self.config = config or GamePlayerAppearanceConfig()

    @staticmethod
    def _cell_features(cell: np.ndarray, global_log_luma: float) -> list[float]:
        pixels = cell.reshape(-1, 3).astype(np.float32, copy=False)
        mean_channels = pixels.mean(axis=0)
        channel_total = float(mean_channels.sum())
        if channel_total <= 1e-6:
            chroma = np.zeros(3, dtype=np.float32)
        else:
            chroma = mean_channels / channel_total - (1.0 / 3.0)

        luma = pixels.mean(axis=1)
        mean_luma = float(luma.mean())
        relative_luma = float(np.clip((np.log1p(mean_luma) - global_log_luma) / 2.0, -1.0, 1.0))
        contrast = float(np.clip(float(luma.std()) / max(mean_luma, 12.0), 0.0, 1.0))

        gray = cell.astype(np.float32, copy=False).mean(axis=2)
        gradients: list[float] = []
        if gray.shape[0] > 1:
            gradients.append(float(np.abs(np.diff(gray, axis=0)).mean()))
        if gray.shape[1] > 1:
            gradients.append(float(np.abs(np.diff(gray, axis=1)).mean()))
        texture = float(np.clip((sum(gradients) / max(len(gradients), 1)) / max(mean_luma, 12.0), 0.0, 1.0))

        # Chroma is signed and therefore remains discriminative under cosine similarity.
        return [
            float(chroma[0] * 3.0),
            float(chroma[1] * 3.0),
            float(chroma[2] * 3.0),
            relative_luma,
            contrast,
            texture,
        ]

    def describe(self, crop: Any) -> tuple[float, ...] | None:
        try:
            array = np.asarray(crop)
        except Exception:
            return None
        if array.ndim != 3 or array.shape[2] < 3:
            return None
        height, width = int(array.shape[0]), int(array.shape[1])
        if height < self.config.min_height_px or width < self.config.min_width_px:
            return None

        array = array[:, :, :3].astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            return None
        array = np.clip(array, 0.0, 255.0)

        margin = int(round(width * self.config.horizontal_margin_ratio))
        if width - 2 * margin >= self.config.grid_cols:
            array = array[:, margin : width - margin]
        height, width = int(array.shape[0]), int(array.shape[1])
        if height < self.config.grid_rows or width < self.config.grid_cols:
            return None

        global_luma = float(array.mean(axis=2).mean())
        global_log_luma = float(np.log1p(global_luma))
        row_edges = np.linspace(0, height, self.config.grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, width, self.config.grid_cols + 1, dtype=int)

        features: list[float] = []
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                cell = array[
                    row_edges[row] : row_edges[row + 1],
                    col_edges[col] : col_edges[col + 1],
                ]
                if cell.size == 0:
                    return None
                features.extend(self._cell_features(cell, global_log_luma))

        norm = sqrt(sum(value * value for value in features))
        if norm <= 1e-12:
            return None
        return tuple(value / norm for value in features)
