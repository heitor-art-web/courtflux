import os
import sqlite3
import cv2


class DatabaseService:

    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.cursor = self.conn.cursor()

    def save(self, frame_state):
        """Saves player detections, frame metrics, team metrics, and events."""
        try:
            if not hasattr(frame_state, "players") or not frame_state.players:
                return

            frame_id = getattr(frame_state, "frame_id", 0)
            players_data = []

            for i, player in enumerate(frame_state.players):
                if isinstance(player, dict):
                    track_id = player.get("track_id")
                    team = player.get("team")
                    shirt_num = player.get("shirt_num")
                    has_ball = player.get("has_ball", False)
                    speed_mps = player.get("speed_mps")
                    dist = player.get("closest_defender_dist")

                    # Never silently store pixel coordinates in columns named *_meters.
                    x_m = player.get("x_m")
                    y_m = player.get("y_m")
                else:
                    track_id = getattr(player, "track_id", None)
                    team = getattr(player, "team", None)
                    shirt_num = getattr(player, "shirt_num", None)
                    has_ball = getattr(player, "has_ball", False)
                    speed_mps = getattr(player, "speed_mps", None)
                    dist = getattr(player, "closest_defender_dist", None)
                    x_m = getattr(player, "x_m", None)
                    y_m = getattr(player, "y_m", None)

                if track_id is None:
                    track_id = i + 1

                players_data.append(
                    (
                        frame_id,
                        track_id,
                        x_m,
                        y_m,
                        team,
                        shirt_num,
                        has_ball,
                        speed_mps,
                        dist,
                    )
                )

            if players_data:
                self.cursor.executemany(
                    """
                    INSERT INTO player_detections
                    (
                        frame_id, track_id, x_meters, y_meters,
                        team, shirt_number, has_ball, speed_mps, closest_defender_dist
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    players_data,
                )

            metrics = getattr(frame_state, "metrics", {}) or {}
            if metrics and isinstance(metrics, dict):
                self.cursor.execute(
                    """
                    INSERT OR REPLACE INTO frame_metrics
                    (
                        frame_id, offensive_team, corner_occupied, in_paint,
                        spacing_area_m2, is_pnr, transition_lag_sec
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        frame_id,
                        metrics.get("offensive_team"),
                        metrics.get("corner_occupied"),
                        metrics.get("in_paint"),
                        metrics.get("spacing_area_m2"),
                        metrics.get("is_pnr"),
                        metrics.get("transition_lag_sec"),
                    ),
                )

            team_metrics_list = getattr(frame_state, "team_metrics", [])
            if team_metrics_list and isinstance(team_metrics_list, list):
                team_data = []
                for tm in team_metrics_list:
                    team_data.append(
                        (
                            frame_id,
                            tm.get("team"),
                            tm.get("centroid_x_meters"),
                            tm.get("centroid_y_meters"),
                            tm.get("spacing_area_sqm"),
                            tm.get("avg_speed_mps"),
                            tm.get("has_possession", False),
                        )
                    )
                if team_data:
                    self.cursor.executemany(
                        """
                        INSERT INTO team_metrics
                        (
                            frame_id, team, centroid_x_meters, centroid_y_meters,
                            spacing_area_sqm, avg_speed_mps, has_possession
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        team_data,
                    )

            events = getattr(frame_state, "events", [])
            if events and isinstance(events, list):
                event_data = []
                for ev in events:
                    event_data.append(
                        (
                            frame_id,
                            getattr(ev, "type", str(ev)),
                            getattr(ev, "description", ""),
                        )
                    )
                if event_data:
                    self.cursor.executemany(
                        """
                        INSERT INTO game_events (frame_id, event_type, description)
                        VALUES (?, ?, ?)
                        """,
                        event_data,
                    )

            self.conn.commit()

        except sqlite3.Error as db_err:
            print(f"🚨 SQLITE ERROR at Frame {getattr(frame_state, 'frame_id', '?')}: {db_err}")
            self.conn.rollback()
        except Exception as e:
            print(f"🚨 PYTHON ERROR at Frame {getattr(frame_state, 'frame_id', '?')}: {e}")

    def save_events(self, frame_state, output_dir: str = "data/events", fps: float = 30.0):
        """Saves event snapshot images to disk and logs the event in the database."""
        if not hasattr(frame_state, "events") or not frame_state.events:
            return

        os.makedirs(output_dir, exist_ok=True)

        image_to_save = getattr(frame_state, "annotated_frame", None)
        if image_to_save is None:
            image_to_save = getattr(frame_state, "frame", None)
        if image_to_save is None:
            return

        frame_id = getattr(frame_state, "frame_id", 0)
        total_seconds = frame_id / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        time_stamp = f"{minutes:02d}m{seconds:02d}s"

        event_db_records = []

        for event in frame_state.events:
            try:
                event_str = str(event)
                safe_event_name = "".join(
                    c for c in event_str if c.isalnum() or c in ("_", "-")
                ).strip() or "event"

                filename = f"frame_{frame_id:06d}_{time_stamp}_{safe_event_name}.jpg"
                file_path = os.path.join(output_dir, filename)

                if cv2.imwrite(file_path, image_to_save):
                    print(f"📸 [SNAPSHOT SAVED] {time_stamp} - {safe_event_name} saved to {file_path}")

                event_db_records.append(
                    (
                        frame_id,
                        safe_event_name,
                        f"Timestamp {time_stamp}: {event_str}",
                    )
                )

            except Exception as e:
                print(f"[Warning] Failed to process event: {e}")

        if event_db_records:
            try:
                self.cursor.executemany(
                    """
                    INSERT INTO game_events (frame_id, event_type, description)
                    VALUES (?, ?, ?)
                    """,
                    event_db_records,
                )
                self.conn.commit()
            except sqlite3.Error as db_err:
                print(f"🚨 SQLITE ERROR saving events: {db_err}")
                self.conn.rollback()
