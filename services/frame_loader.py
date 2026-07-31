from core.frame_state import FrameState
import cv2

class FrameLoader:

    def __init__(self, session):

        self.session = session
        self.frame_id = 0

    def next(self):

        ok, frame = self.session.cap.read()
        if not ok:
            return None

        timestamp = (
            self.session.cap.get(
                #self.session.cap.CAP_PROP_POS_MSEC
                cv2.CAP_PROP_POS_MSEC
            )
            / 1000
        )

        self.frame_id += 1
        return FrameState(
            frame_id=self.frame_id,
            timestamp=timestamp,
            frame=frame,
            fps=self.session.fps,
        )