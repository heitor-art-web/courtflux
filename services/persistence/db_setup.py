import os
import sqlite3

os.makedirs("data", exist_ok=True)
os.makedirs("data/debug_frames", exist_ok=True)


def initialize_database():
    """Create a clean local beta database for one analysis run."""
    os.makedirs("data", exist_ok=True)
    db_path = os.path.abspath("data/tracking.db")
    print(f"--> Initializing database at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS player_detections")
    cursor.execute("DROP TABLE IF EXISTS team_metrics")
    cursor.execute("DROP TABLE IF EXISTS game_events")
    cursor.execute("DROP TABLE IF EXISTS passing_networks")
    cursor.execute("DROP TABLE IF EXISTS shot_events")
    cursor.execute("DROP TABLE IF EXISTS frame_metrics")

    cursor.execute(
        """
        CREATE TABLE player_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER,
            track_id INTEGER,
            x_meters REAL,
            y_meters REAL,
            team TEXT,
            shirt_number TEXT,
            has_ball BOOLEAN,
            speed_mps REAL,
            closest_defender_dist REAL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE team_metrics (
            frame_id INTEGER,
            team TEXT,
            centroid_x_meters REAL,
            centroid_y_meters REAL,
            spacing_area_sqm REAL,
            avg_speed_mps REAL,
            has_possession BOOLEAN
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE game_events (
            frame_id INTEGER,
            event_type TEXT,
            description TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE passing_networks (
            passer_id INTEGER,
            receiver_id INTEGER,
            team TEXT,
            pass_count INTEGER
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE shot_events (
            frame_id INTEGER,
            track_id INTEGER,
            team TEXT,
            x_meters REAL,
            y_meters REAL,
            defender_distance REAL,
            shot_outcome TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE frame_metrics (
            frame_id INTEGER PRIMARY KEY,
            offensive_team TEXT,
            corner_occupied BOOLEAN,
            in_paint BOOLEAN,
            spacing_area_m2 REAL,
            is_pnr BOOLEAN,
            transition_lag_sec REAL
        )
        """
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
