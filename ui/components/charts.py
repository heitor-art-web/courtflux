"""
components/charts.py
Basketball AI Analytics
Centralised Plotly chart creation.
The dashboard should never build Plotly figures directly.
"""
from __future__ import annotations
import pandas as pd
import plotly.express as px

# ============================================================
# TEAM POSITION HEATMAP
# ============================================================

def create_heatmap(
    player_df: pd.DataFrame,
    title: str,
):
    """
    Team positioning density.
    """
    if player_df.empty:
        return None

    fig = px.density_contour(
        player_df,
        x="x_meters",
        y="y_meters",
        range_x=[0, 28],
        range_y=[0, 15],
        color_discrete_sequence=["royalblue"],
        title=title,
    )

    fig.update_traces(
        contours_coloring="fill",
        opacity=0.45,
    )

    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=28,
        y1=15,
    )

    fig.add_shape(
        type="line",
        x0=14,
        y0=0,
        x1=14,
        y1=15,
    )

    fig.update_layout(
        height=550,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
    )

    return fig


# ============================================================
# PLAYER SPEED
# ============================================================

def create_speed_chart(
    player_summary: pd.DataFrame,
):
    """
    Average player speed.
    """

    if player_summary.empty:
        return None

    fig = px.bar(
        player_summary,
        x="Player",
        y="AvgSpeed",
        color="AvgSpeed",
        text="AvgSpeed",
        title="Average Player Speed",
    )

    fig.update_layout(
        height=420,
    )

    return fig


# ============================================================
# WORKLOAD
# ============================================================

def create_workload_chart(
    player_summary: pd.DataFrame,
):
    """
    Distance covered.
    """

    if player_summary.empty:
        return None

    fig = px.bar(
        player_summary,
        x="Player",
        y="Distance",
        color="Distance",
        text="Distance",
        title="Distance Covered",
    )

    fig.update_layout(
        height=420,
    )

    return fig


# ============================================================
# SCORE BREAKDOWN
# ============================================================

def create_score_breakdown(
    scores: dict,
):
    """
    Tactical score bars.
    """
    data = pd.DataFrame(

        {
            "Category":[

                "Offense",

                "Defense",

                "Transition",

            ],

            "Score":[

                scores["offense"],
                scores["defense"],
                scores["transition"],
            ]

        }

    )

    fig = px.bar(

        data,

        x="Category",
        y="Score",
        text="Score",
        color="Score",
        range_y=[0,100],

    )

    fig.update_layout(

        height=350,
    )

    return fig


# ============================================================
# POSSESSION
# ============================================================

def create_possession_chart(
    possession: float,
):
    """
    Simple possession pie chart.
    """

    fig = px.pie(

        values=[

            possession,

            100-possession,

        ],

        names=[

            "My Team",

            "Opponent",

        ],

        hole=.55,

    )

    fig.update_layout(
        height=350,
    )

    return fig


# ============================================================
# SHOT QUALITY
# ============================================================

def create_shot_quality_chart(
    metrics_df: pd.DataFrame,
):
    """
    Placeholder for Stage 2.
    """

    if (
        metrics_df.empty
        or "spacing_area_m2" not in metrics_df.columns
    ):
        return None

    fig = px.histogram(

        metrics_df,

        x="spacing_area_m2",

        nbins=20,

        title="Shot Quality Distribution",

    )

    fig.update_layout(
        height=350,
    )

    return fig


# ============================================================
# PLAYER SPEED DISTRIBUTION
# ============================================================

def create_speed_distribution(
    player_df: pd.DataFrame,
):
    """
    Histogram of player speeds.
    """

    if player_df.empty:
        return None

    fig = px.histogram(
        player_df,
        x="speed_mps",
        nbins=25,
        title="Speed Distribution",
    )

    fig.update_layout(
        height=350,
    )

    return fig