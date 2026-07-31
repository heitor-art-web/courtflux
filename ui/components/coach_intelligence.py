"""
components/coach_intelligence.py

Basketball AI Analytics

Converts KPIs into coaching insights.

This module never calculates metrics.
It only interprets them.
"""

from __future__ import annotations


# ============================================================
# MATCH EVALUATION
# ============================================================

def evaluate_team(metrics: dict) -> dict:
    """
    Generate tactical observations from dashboard metrics.

    Parameters
    ----------
    metrics : dict
        Output from metrics.build_dashboard_metrics()

    Returns
    -------
    dict
    """

    scores = metrics["scores"]

    spacing = metrics["spacing"]
    pressure = metrics["pressure"]
    speed = metrics["speed"]
    possession = metrics["possession"]

    strengths = []
    weaknesses = []
    priorities = []
    summary = []

    # ========================================================
    # OFFENSIVE SPACING
    # ========================================================

    if scores["offense"] >= 85:

        strengths.append(
            "Excellent offensive spacing."
        )

        summary.append(
            "Players maintained good floor balance."
        )

    elif scores["offense"] >= 70:

        summary.append(
            "Spacing was acceptable for most possessions."
        )

    else:

        weaknesses.append(
            "Players frequently crowded the ball."
        )

        priorities.append(
            "Increase weak-side spacing and corner occupation."
        )

    # ========================================================
    # DEFENSIVE PRESSURE
    # ========================================================

    if scores["defense"] >= 85:

        strengths.append(
            "Strong on-ball defensive pressure."
        )

        summary.append(
            "Ball handlers rarely received comfortable space."
        )

    elif scores["defense"] >= 70:

        summary.append(
            "Defensive pressure was generally acceptable."
        )

    else:

        weaknesses.append(
            "Opponents received too much space."
        )

        priorities.append(
            "Close out earlier and reduce perimeter cushion."
        )

    # ========================================================
    # TRANSITION
    # ========================================================

    if scores["transition"] >= 85:

        strengths.append(
            "High collective playing pace."
        )

    elif scores["transition"] < 60:

        weaknesses.append(
            "Movement intensity was below expected."
        )

        priorities.append(
            "Increase off-ball movement and transition speed."
        )

    # ========================================================
    # POSSESSION
    # ========================================================

    if possession >= 60:

        strengths.append(
            "Controlled the majority of possessions."
        )

    elif possession < 40:

        weaknesses.append(
            "Limited offensive possessions."
        )

        priorities.append(
            "Improve rebounding and reduce turnovers."
        )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    if scores["overall"] >= 90:

        headline = (
            "Excellent collective tactical performance."
        )

    elif scores["overall"] >= 80:

        headline = (
            "Strong performance with minor tactical improvements available."
        )

    elif scores["overall"] >= 70:

        headline = (
            "Competitive performance with several coaching opportunities."
        )

    else:

        headline = (
            "Multiple tactical deficiencies were detected."
        )

    return {

        "headline": headline,

        "summary": summary,

        "strengths": strengths,

        "weaknesses": weaknesses,

        "priorities": priorities,

        "scores": scores,

        "metrics": {

            "spacing": spacing,

            "pressure": pressure,

            "speed": speed,

            "possession": possession,

        }

    }


# ============================================================
# SHORT EXECUTIVE REPORT
# ============================================================

def report_summary(report: dict) -> str:
    """
    One sentence shown on the dashboard.
    """

    return report["headline"]


# ============================================================
# COACHING REPORT
# ============================================================

def build_coach_report(report: dict) -> str:
    """
    Multi-line report for future PDF export.
    """

    lines = []

    lines.append(report["headline"])
    lines.append("")

    if report["strengths"]:

        lines.append("Strengths:")

        for item in report["strengths"]:

            lines.append(f"• {item}")

        lines.append("")

    if report["priorities"]:

        lines.append("Coaching Priorities:")

        for item in report["priorities"]:

            lines.append(f"• {item}")

    return "\n".join(lines)