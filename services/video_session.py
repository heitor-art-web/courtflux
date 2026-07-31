"""
Handles VideoCapture lifecycle.
"""

from dataclasses import dataclass

import cv2


@dataclass(slots=True)
class VideoSession:

    video_path: str
    start_time: float
    end_time: float
    cap: cv2.VideoCapture | None = None
    fps: float = 30.0
    width: int = 0
    height: int = 0

    def open(self):

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video.")

        self.cap.set(
            cv2.CAP_PROP_POS_MSEC,
            self.start_time * 1000,
        )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"⏩ Fast-forwarding to {self.start_time:.1f}s...")

    def close(self):

        if self.cap:
            self.cap.release()