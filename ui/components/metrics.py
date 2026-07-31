"""
components/metrics.py

Basketball AI Analytics

Centralised KPI and tactical score calculations.
"""

from __future__ import annotations

import pandas as pd


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_possession(
    df: pd.DataFrame,
    my_team: str,
) -> float:
    """
    Estimate possession percentage using has_ball.
    """

    ball_frames = df[df["has_ball"] == 1]

    if ball_frames.empty:
        return 0.0

    my_ball = ball_frames[
        ball_frames["team"] == my_team
    ]

    return len(my_ball) / len(ball_frames) * 100


def calculate_distance(
    player_df: pd.DataFrame,
    fps: int = 30,
) -> float:
    """
    Approximate total distance covered.
    """

    if "speed_mps" not in player_df.columns:
        return 0.0

    return player_df["speed_mps"].sum() / fps


def calculate_average_spacing(
    player_df: pd.DataFrame,
) -> float:

    if player_df.empty:
        return 0.0

    return (
        player_df
        .groupby("frame_id")["x_meters"]
        .std()
        .mean()
    )


def calculate_average_speed(
    player_df: pd.DataFrame,
) -> float:

    if "speed_mps" not in player_df.columns:
        return 0.0

    return player_df["speed_mps"].mean()


def calculate_average_pressure(
    player_df: pd.DataFrame,
) -> float:

    if "closest_defender_dist" not in player_df.columns:
        return 0.0

    return player_df["closest_defender_dist"].mean()


# ============================================================
# TEAM SCORES
# ============================================================

def calculate_team_scores(
    spacing: float,
    pressure: float,
    speed: float,
) -> dict:
    """
    Convert physical KPIs into tactical scores (0-100).
    """

    offense = max(
        0,
        min(
            100,
            spacing * 15,
        ),
    )

    defense = max(
        0,
        min(
            100,
            100 - pressure * 25,
        ),
    )

    transition = max(
        0,
        min(
            100,
            speed * 18,
        ),
    )

    overall = (
        offense +
        defense +
        transition
    ) / 3

    return {
        "offense": round(offense, 1),
        "defense": round(defense, 1),
        "transition": round(transition, 1),
        "overall": round(overall, 1),
    }


# ============================================================
# MATCH GRADE
# ============================================================

def score_to_grade(
    score: float,
) -> str:

    if score >= 95:
        return "A+"

    if score >= 90:
        return "A"

    if score >= 85:
        return "A-"

    if score >= 80:
        return "B+"

    if score >= 75:
        return "B"

    if score >= 70:
        return "B-"

    if score >= 65:
        return "C+"

    if score >= 60:
        return "C"

    if score >= 55:
        return "C-"

    return "D"


# ============================================================
# DASHBOARD KPIs
# ============================================================

def build_dashboard_metrics(
    df: pd.DataFrame,
    my_team: str,
    fps: int = 30,
) -> dict:
    """
    Compute all dashboard metrics in one place.
    """

    team_df = df[
        df["team"] == my_team
    ].copy()

    spacing = calculate_average_spacing(team_df)

    speed = calculate_average_speed(team_df)

    pressure = calculate_average_pressure(team_df)

    possession = calculate_possession(
        df,
        my_team,
    )

    distance = calculate_distance(
        team_df,
        fps,
    )

    scores = calculate_team_scores(
        spacing,
        pressure,
        speed,
    )

    return {
        "spacing": spacing,
        "speed": speed,
        "pressure": pressure,
        "distance": distance,
        "possession": possession,
        "scores": scores,
        "grade": score_to_grade(
            scores["overall"]
        ),
    }