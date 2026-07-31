"""
dashboard.py

Basketball AI Analytics Dashboard
Stage 1

This dashboard is intentionally lightweight.
Business logic lives inside components/.
"""

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# COMPONENTS
# ============================================================

from components.metrics import (
    build_dashboard_metrics,
)

from components.coach_intelligence import (
    evaluate_team,
    report_summary,
)

from components.charts import (
    create_heatmap,
    create_speed_chart,
    create_score_breakdown,
    create_possession_chart,
)

# ============================================================
# CONFIG
# ============================================================

CACHE_SECONDS = 2

st.set_page_config(
    page_title="Basketball AI Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

DB_PATH = DATA_DIR / "tracking.db"
EVENTS_DIR = DATA_DIR / "events"

LOGO_PATH = ASSETS_DIR / "logo.png"

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container{
    padding-top:2rem;
    padding-left:2rem;
    padding-right:2rem;
    padding-bottom:2rem;
}

.metric-card{
    background:var(--secondary-background-color);
    border-radius:12px;
    padding:18px;
}

.section-title{
    font-size:30px;
    font-weight:700;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATABASE
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def load_database():

    conn = sqlite3.connect(DB_PATH)

    players = pd.read_sql(
        "SELECT * FROM player_detections",
        conn,
    )

    try:

        metrics = pd.read_sql(
            "SELECT * FROM frame_metrics",
            conn,
        )

    except Exception:

        metrics = pd.DataFrame()

    conn.close()

    return players, metrics


@st.cache_data(ttl=CACHE_SECONDS)
def load_event_images():

    if not EVENTS_DIR.exists():
        return []

    return sorted(
        EVENTS_DIR.glob("*.jpg"),
        reverse=True,
    )


# ============================================================
# LOAD DATA
# ============================================================

df, df_metrics = load_database()

if df.empty:

    st.warning(
        "No tracking data found.\nRun analyzer.py first."
    )

    st.stop()

required_columns = [

    "team",

    "frame_id",

    "track_id",

    "has_ball",

]

missing = [

    c

    for c in required_columns

    if c not in df.columns

]

if missing:

    st.error(
        f"Database missing columns: {missing}"
    )

    st.stop()

# ============================================================
# HEADER
# ============================================================

header = st.container()

with header:

    st.markdown(
        "<div style='height:15px'></div>",
        unsafe_allow_html=True,
    )

    logo_col, title_col, version_col = st.columns(
        [1.2, 6, 1]
    )

    with logo_col:

        if LOGO_PATH.exists():

            st.image(
                str(LOGO_PATH),
                width=90,
            )

        else:

            st.markdown("# 🏀")

    with title_col:

        st.markdown(
            """
<div class="section-title">
Basketball AI Analytics
</div>
""",
            unsafe_allow_html=True,
        )

        st.caption(
            "Automated Tactical Analysis Platform"
        )

    with version_col:

        st.metric(
            "Version",
            "Stage 1",
        )

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Game Session")

teams = sorted(

    t

    for t in df.team.dropna().unique()

    if "Unknown" not in t

)

if len(teams) >= 2:

    my_team = st.sidebar.selectbox(
        "My Team",
        teams,
    )

    opponent = next(
        t
        for t in teams
        if t != my_team
    )

else:

    my_team = "Light Team"
    opponent = "Dark Team"

phase = st.sidebar.radio(

    "Analysis Phase",

    [

        "Full Game",

        "Offense",

        "Defense",

    ],

)

st.sidebar.divider()

st.sidebar.caption(
    "Basketball AI Analytics"
)

# ============================================================
# TEAM DATA
# ============================================================

my_df = df[
    df.team == my_team
].copy()

opp_df = df[
    df.team == opponent
].copy()

# ============================================================
# DASHBOARD METRICS
# ============================================================

metrics = build_dashboard_metrics(
    df,
    my_team,
)

scores = metrics["scores"]

coach = evaluate_team(
    metrics,
)

# ============================================================
# MATCH SUMMARY
# ============================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Team",
    my_team,
)

k2.metric(
    "Opponent",
    opponent,
)

k3.metric(
    "Phase",
    phase,
)

k4.metric(
    "Frames",
    f"{df.frame_id.nunique():,}",
)

st.divider()

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

grade_col, summary_col = st.columns(
    [1,2]
)

with grade_col:

    st.subheader("Overall Rating")

    st.metric(
        "Match Grade",
        metrics["grade"],
        delta=f"{scores['overall']:.1f}/100",
    )

    st.progress(
        scores["overall"]/100
    )

with summary_col:

    st.subheader("Coach Summary")

    st.info(
        report_summary(
            coach
        )
    )

    if coach["strengths"]:

        st.success("### Strengths")

        for item in coach["strengths"]:

            st.write("✅", item)

    if coach["priorities"]:

        st.warning("### Priorities")

        for item in coach["priorities"]:

            st.write("•", item)

st.divider()

# ============================================================
# TABS
# ============================================================

overview_tab, offense_tab, defense_tab, players_tab, events_tab = st.tabs(

    [

        "📋 Overview",

        "🏀 Offense",

        "🛡 Defense",

        "👥 Players",

        "📸 Events",

    ]

)
# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    st.header("🏀 Match Overview")

    # --------------------------------------------------------
    # Dashboard KPIs
    # --------------------------------------------------------

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        "Possession",
        f"{metrics['possession']:.1f}%"
    )

    kpi2.metric(
        "Average Speed",
        f"{metrics['speed']:.2f} m/s"
    )

    kpi3.metric(
        "Average Pressure",
        f"{metrics['pressure']:.2f} m"
    )

    kpi4.metric(
        "Distance Covered",
        f"{metrics['distance']:.0f} m"
    )

    st.divider()

    # --------------------------------------------------------
    # Team Score Breakdown
    # --------------------------------------------------------

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Performance Breakdown")

        fig = create_score_breakdown(
            scores
        )

        if fig is not None:

            st.plotly_chart(
                fig,
                width="stretch",
                key="performance_breakdown"
            )

    with right:

        st.subheader("Possession")

        fig = create_possession_chart(
            metrics["possession"]
        )

        if fig is not None:

            st.plotly_chart(
                fig,
                width="stretch",
                key="possession_chart"
            )

    st.divider()

    # --------------------------------------------------------
    # Position Heatmap
    # --------------------------------------------------------

    st.subheader("Team Positioning")

    plot_df = my_df.copy()

    if phase == "Offense":

        offense_frames = my_df[
            my_df["has_ball"] == 1
        ]["frame_id"].unique()

        plot_df = plot_df[
            plot_df["frame_id"].isin(
                offense_frames
            )
        ]

    elif phase == "Defense":

        defense_frames = opp_df[
            opp_df["has_ball"] == 1
        ]["frame_id"].unique()

        plot_df = plot_df[
            plot_df["frame_id"].isin(
                defense_frames
            )
        ]

    fig = create_heatmap(
        plot_df,
        f"{my_team} Position Density"
    )

    if fig is not None:

        st.plotly_chart(
            fig,
            width="stretch",
            key="position Density"
        )

    else:

        st.info(
            "No positioning data available."
        )

    st.divider()

    # --------------------------------------------------------
    # Executive Notes
    # --------------------------------------------------------

    st.subheader("Coach Notes")

    left, right = st.columns(2)

    with left:

        st.success("### Strengths")

        if coach["strengths"]:

            for item in coach["strengths"]:

                st.write("✅", item)

        else:

            st.write("No significant strengths detected.")

    with right:

        st.error("### Areas to Improve")

        if coach["weaknesses"]:

            for item in coach["weaknesses"]:

                st.write("⚠️", item)

        else:

            st.write("No major tactical weaknesses detected.")

    st.divider()

    st.subheader("Training Priorities")

    if coach["priorities"]:

        for priority in coach["priorities"]:

            with st.expander(priority):

                st.write(priority)

    else:

        st.success(
            "No urgent coaching priorities detected."
        )
        
# ============================================================
# OFFENSE TAB
# ============================================================

with offense_tab:

    st.header("🏀 Offensive Analysis")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Spacing",
        f"{metrics['spacing']:.2f} m",
    )

    c2.metric(
        "Possession",
        f"{metrics['possession']:.1f} %",
    )

    c3.metric(
        "Offense Score",
        f"{scores['offense']:.1f}",
    )

    st.divider()

    if (
        not df_metrics.empty
        and "spacing_area_m2" in df_metrics.columns
    ):

        st.subheader("Spacing Area")

        st.line_chart(
            df_metrics["spacing_area_m2"],
            height=300,
        )

    else:

        st.info(
            "Frame metrics not available."
        )

    st.divider()

    st.subheader("Coach Insight")

    if scores["offense"] >= 80:

        st.success(
            """
Players generally maintained good floor spacing.

Continue emphasizing corner occupation and weak-side movement.
"""
        )

    elif scores["offense"] >= 65:

        st.warning(
            """
Spacing is acceptable but inconsistent.

Players occasionally collapsed toward the ball.
"""
        )

    else:

        st.error(
            """
Poor offensive spacing detected.

Focus on:

• Corner occupation

• Weak-side spacing

• Ball reversals
"""
        )

# ============================================================
# DEFENSE TAB
# ============================================================

with defense_tab:

    st.header("🛡 Defensive Analysis")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Pressure",
        f"{metrics['pressure']:.2f} m",
    )

    c2.metric(
        "Average Speed",
        f"{metrics['speed']:.2f} m/s",
    )

    c3.metric(
        "Defense Score",
        f"{scores['defense']:.1f}",
    )

    st.divider()

    if (
        not df_metrics.empty
        and "transition_lag_sec" in df_metrics.columns
    ):

        st.subheader("Transition Recovery")

        st.line_chart(
            df_metrics["transition_lag_sec"],
            height=300,
        )

    else:

        st.info(
            "Transition metrics unavailable."
        )

    st.divider()

    st.subheader("Coach Insight")

    if scores["defense"] >= 85:

        st.success(
            """
Excellent defensive pressure.

Ball handlers rarely had time to organize the offense.
"""
        )

    elif scores["defense"] >= 70:

        st.warning(
            """
Pressure is acceptable but can improve.

Focus on faster close-outs after passes.
"""
        )

    else:

        st.error(
            """
Opponents received too much space.

Practice:

• Close-outs

• Ball pressure

• Help-and-recover rotations
"""
        )

# ============================================================
# PLAYERS TAB
# ============================================================

with players_tab:

    st.header("👥 Player Performance")

    player_summary = (

        my_df

        .groupby(
            [
                "track_id",
                "shirt_number",
            ]
        )

        .agg(

            AvgSpeed=("speed_mps", "mean"),

            MaxSpeed=("speed_mps", "max"),

            Distance=("speed_mps", lambda x: x.sum()/30),

            BallTouches=("has_ball", "sum"),

        )

        .reset_index()

    )

    player_summary.rename(

        columns={

            "track_id": "Player",

            "shirt_number": "Jersey",

        },

        inplace=True,

    )

    st.dataframe(

        player_summary,

        width="stretch",

        hide_index=True,

    )

    st.divider()

    fig = create_speed_chart(
        player_summary
    )

    if fig is not None:

        st.plotly_chart(
            fig,
            width="stretch",
            key="speed_chart"
        )

# ============================================================
# EVENTS TAB
# ============================================================

with events_tab:

    st.header("📸 Key Moments")

    images = load_event_images()

    if not images:

        st.info(
            "No event snapshots available."
        )

    else:

        cols = st.columns(2)

        for i, image_path in enumerate(images[:8]):

            with cols[i % 2]:

                image = Image.open(image_path)

                st.image(
                    image,
                    width="stretch",
                )

                filename = image_path.stem

                st.caption(
                    filename.replace("_", " ")
                )

                name = filename.lower()

                if "turnover" in name:

                    st.error(
                        "Turnover detected."
                    )

                elif "shot" in name:

                    st.success(
                        "Shot attempt detected."
                    )

                elif "rebound" in name:

                    st.info(
                        "Rebound sequence."
                    )

                else:

                    st.info(
                        "Game event."
                    )
# ============================================================
# COACH INTELLIGENCE
# ============================================================

st.markdown("---")

st.header("🧠 Coach Intelligence")

left, right = st.columns([1, 2])

# ------------------------------------------------------------
# Executive Grade
# ------------------------------------------------------------

with left:

    st.metric(
        "Overall Match Grade",
        metrics["grade"],
    )

    st.progress(
        scores["overall"] / 100
    )

    st.caption(
        f"Overall Tactical Score: {scores['overall']:.1f}/100"
    )

# ------------------------------------------------------------
# AI Summary
# ------------------------------------------------------------

with right:

    st.subheader("Executive Summary")

    st.info(
        coach["headline"]
    )

    if coach["summary"]:

        st.write("### Match Summary")

        for line in coach["summary"]:

            st.write(f"• {line}")

st.divider()

# ------------------------------------------------------------
# Tactical Scores
# ------------------------------------------------------------

st.subheader("Score Breakdown")

fig = create_score_breakdown(
    scores,
)

if fig is not None:

    st.plotly_chart(
        fig,
        width="stretch",
        key="score_breakdown"
    )

st.divider()

# ------------------------------------------------------------
# Strengths & Weaknesses
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.success("### Strengths")

    if coach["strengths"]:

        for item in coach["strengths"]:

            st.write(f"✅ {item}")

    else:

        st.write(
            "No major strengths detected."
        )

with col2:

    st.error("### Coaching Priorities")

    if coach["priorities"]:

        for item in coach["priorities"]:

            st.write(f"⚠️ {item}")

    else:

        st.success(
            "No tactical priorities identified."
        )

st.divider()

# ------------------------------------------------------------
# Raw Metrics
# ------------------------------------------------------------

with st.expander("Advanced Metrics"):

    metric_table = pd.DataFrame({

        "Metric": [

            "Average Spacing",

            "Average Pressure",

            "Average Speed",

            "Possession",

            "Distance",

        ],

        "Value": [

            round(metrics["spacing"], 2),

            round(metrics["pressure"], 2),

            round(metrics["speed"], 2),

            round(metrics["possession"], 1),

            round(metrics["distance"], 1),

        ]

    })

    st.dataframe(
        metric_table,
        width="stretch",
        hide_index=True,
    )

# ============================================================
# TACTICAL COURT (STAGE 2)
# ============================================================

st.markdown("---")

st.header("🏀 Tactical Court")

st.info(
    """
🚧 Coming in Stage 2

This module will include:

• Interactive basketball court

• Player tracking

• Ball trajectory

• Offensive spacing

• Defensive shell

• Shot chart

• Pass network

• Pick & Roll recognition

• Transition lanes

• AI tactical overlays
"""
)

# ------------------------------------------------------------
# Future Integration
# ------------------------------------------------------------
#
# from components.tactical_court import render_tactical_court
#
# render_tactical_court(
#     tracking_df=my_df,
#     metrics_df=df_metrics,
#     team=my_team,
#     phase=phase,
# )
#
# ------------------------------------------------------------

# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_left, footer_center, footer_right = st.columns(
    [2, 2, 1]
)

with footer_left:

    st.caption(
        "🏀 Basketball AI Analytics"
    )

with footer_center:

    st.caption(
        "Stage 1 • Component Architecture"
    )

with footer_right:

    st.caption(
        "v1.0"
    )