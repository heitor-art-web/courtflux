from dataclasses import dataclass, field


@dataclass(slots=True)
class GameContext:
    """
    Global game state.

    Persists during the whole video.
    """

    fps: float
    frame_count: int = 0
    previous_offensive_team: str | None = None
    jersey_cache: dict[int, str] = field(default_factory=dict)
    court_loss_counter: int = 0
    was_out_of_bounds: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    
    def next_frame(self):
        self.frame_count += 1