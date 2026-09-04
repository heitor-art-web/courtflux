from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import settings


RUNS_DIR = ROOT / "evaluation" / "runs"
MATCH_CONFIG_PATH = ROOT / "data" / "match_config.json"
DEFAULT_VIDEO = "https://www.youtube.com/watch?v=OwiQgAMqyUs"

configured_model = ROOT / settings.model
fallback_model = ROOT / "weights" / "Model_v0.1.pt"
if configured_model.exists():
    DEFAULT_MODEL = settings.model
elif fallback_model.exists():
    DEFAULT_MODEL = "weights/Model_v0.1.pt"
else:
    DEFAULT_MODEL = settings.model

st.set_page_config(
    page_title="Basketball AI Analytics — Beta Lab",
    page_icon="🏀",
    layout="wide",
)


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_match_config() -> dict:
    try:
        return json.loads(MATCH_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_match_config(config: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["The match config must be a JSON object."]

    teams = config.get("teams")
    if not isinstance(teams, dict) or len(teams) < 2:
        errors.append("'teams' must contain at least two teams.")
        return errors

    metadata = config.get("metadata", {}) or {}
    for key in ("home_team", "away_team"):
        team_name = metadata.get(key)
        if team_name and team_name not in teams:
            errors.append(f"metadata.{key}='{team_name}' is not present in teams.")

    for team_name, team_data in teams.items():
        if not isinstance(team_data, dict):
            errors.append(f"Team '{team_name}' must be a JSON object.")
            continue

        kits = team_data.get("kits")
        if not isinstance(kits, dict) or not kits:
            errors.append(f"Team '{team_name}' needs at least one kit in 'kits'.")

        roster = team_data.get("roster")
        if not isinstance(roster, dict) or not roster:
            errors.append(f"Team '{team_name}' needs a non-empty 'roster'.")

    return errors


def render_match_summary(config: dict) -> None:
    teams = config.get("teams", {}) or {}
    metadata = config.get("metadata", {}) or {}

    home = metadata.get("home_team") or (list(teams.keys())[0] if teams else "Home")
    away = metadata.get("away_team") or (list(teams.keys())[1] if len(teams) > 1 else "Away")

    st.subheader("Match configuration")
    st.caption(
        "This file is used by the player distinction pipeline for team colours and roster-constrained jersey recognition."
    )

    cols = st.columns(2)
    for col, team_name in zip(cols, [home, away]):
        team_data = teams.get(team_name, {}) or {}
        roster = team_data.get("roster", {}) or {}
        kits = team_data.get("kits", {}) or {}

        with col:
            st.markdown(f"**{team_name}**")
            st.caption(f"{len(roster)} roster entries · kits: {', '.join(kits.keys()) or 'none'}")

            rows = []
            for number, player in roster.items():
                player = player if isinstance(player, dict) else {}
                rows.append(
                    {
                        "#": number,
                        "Player": player.get("name", ""),
                        "Position": player.get("position", ""),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=220)


def latest_summary_path() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    summaries = sorted(RUNS_DIR.glob("*/summary.json"), reverse=True)
    return summaries[0] if summaries else None


def flatten_summary(summary: dict) -> dict:
    ball = summary.get("ball", {})
    players = summary.get("players", {})
    match = summary.get("match_config", {}) or {}
    return {
        "run_id": summary.get("run_id"),
        "match": f"{match.get('home_team') or '?'} vs {match.get('away_team') or '?'}",
        "model": summary.get("model"),
        "conf": summary.get("detector_conf"),
        "imgsz": summary.get("imgsz"),
        "frames": summary.get("gameplay_frames"),
        "raw players/frame": players.get("raw_per_gameplay_frame"),
        "final players/frame": players.get("final_per_gameplay_frame"),
        "team classified %": players.get("team_classification_rate"),
        "jersey identified %": players.get("jersey_identification_rate"),
        "raw ball candidate %": ball.get("raw_candidate_frame_rate"),
        "selected ball %": ball.get("selected_detection_frame_rate"),
        "missing ball %": ball.get("missing_frame_rate"),
        "ball mean conf": ball.get("selected_conf_mean"),
        "elapsed s": summary.get("elapsed_sec"),
        "status": summary.get("extra", {}).get("status"),
    }


def render_summary(summary_path: Path) -> None:
    summary = load_summary(summary_path)
    if not summary:
        return

    st.subheader("Latest results")
    st.caption(f"Run: {summary.get('run_id')} · {summary_path.parent}")

    if summary.get("extra", {}).get("status") == "failed":
        st.error(summary.get("extra", {}).get("error", "Run failed"))

    ball = summary.get("ball", {})
    players = summary.get("players", {})

    st.markdown("**Players**")
    a, b, c, d = st.columns(4)
    a.metric("Raw players / frame", players.get("raw_per_gameplay_frame", 0))
    b.metric("Final players / frame", players.get("final_per_gameplay_frame", 0))
    c.metric("Team classified", f"{float(players.get('team_classification_rate') or 0):.1f}%")
    d.metric("Jersey identified", f"{float(players.get('jersey_identification_rate') or 0):.1f}%")

    team_counts = players.get("team_counts", {}) or {}
    if team_counts:
        st.caption(
            "Classified player detections: "
            + " · ".join(f"{team}: {count}" for team, count in team_counts.items())
        )

    st.markdown("**Ball**")
    a, b, c, d = st.columns(4)
    a.metric("Raw candidate frames", f"{float(ball.get('raw_candidate_frame_rate') or 0):.1f}%")
    b.metric("Selected detection", f"{float(ball.get('selected_detection_frame_rate') or 0):.1f}%")
    c.metric("Missing", f"{float(ball.get('missing_frame_rate') or 0):.1f}%")
    d.metric("Mean confidence", ball.get("selected_conf_mean") or 0)

    st.caption(
        f"Gameplay frames: {summary.get('gameplay_frames', 0)} · "
        f"Predicted/held ball: {float(ball.get('predicted_hold_frame_rate') or 0):.1f}% · "
        f"Ball rejected by court filter: {ball.get('rejected_outside_court', 0)}"
    )

    run_dir = summary_path.parent
    download_cols = st.columns(4)
    with download_cols[0]:
        st.download_button(
            "summary.json",
            data=summary_path.read_bytes(),
            file_name=f"{summary.get('run_id', 'run')}_summary.json",
            mime="application/json",
            width="stretch",
        )

    frames_path = run_dir / "frames.jsonl"
    if frames_path.exists():
        with download_cols[1]:
            st.download_button(
                "frames.jsonl",
                data=frames_path.read_bytes(),
                file_name=f"{summary.get('run_id', 'run')}_frames.jsonl",
                mime="application/x-ndjson",
                width="stretch",
            )

    console_path = run_dir / "console.log"
    if console_path.exists():
        with download_cols[2]:
            st.download_button(
                "console.log",
                data=console_path.read_bytes(),
                file_name=f"{summary.get('run_id', 'run')}_console.log",
                mime="text/plain",
                width="stretch",
            )

    match_snapshot = run_dir / "match_config.json"
    if match_snapshot.exists():
        with download_cols[3]:
            st.download_button(
                "match_config.json",
                data=match_snapshot.read_bytes(),
                file_name=f"{summary.get('run_id', 'run')}_match_config.json",
                mime="application/json",
                width="stretch",
            )


def render_comparison() -> None:
    if not RUNS_DIR.exists():
        return

    paths = sorted(RUNS_DIR.glob("*/summary.json"), reverse=True)
    rows = [flatten_summary(load_summary(path)) for path in paths[:20]]
    rows = [row for row in rows if row.get("run_id")]
    if not rows:
        return

    st.subheader("Run comparison")
    st.caption("Keep the same video interval and match config, then change one detector parameter at a time.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


st.title("🏀 Basketball AI Analytics — Beta Lab")
st.write("YouTube → detection/tracking → measurable results → run comparison.")

# ----------------------------------------------------------------------
# Match configuration is part of the experiment, so show it before runs.
# ----------------------------------------------------------------------
if "match_config_editor" not in st.session_state:
    if MATCH_CONFIG_PATH.exists():
        st.session_state["match_config_editor"] = MATCH_CONFIG_PATH.read_text(encoding="utf-8")
    else:
        st.session_state["match_config_editor"] = "{}"

try:
    current_match_config = json.loads(st.session_state["match_config_editor"])
    config_parse_error = None
except Exception as exc:
    current_match_config = {}
    config_parse_error = str(exc)

config_errors = validate_match_config(current_match_config) if config_parse_error is None else [config_parse_error]
config_ready = not config_errors

if config_ready:
    render_match_summary(current_match_config)
else:
    st.error("match_config.json is not ready: " + " | ".join(config_errors))

with st.expander("Edit match_config.json", expanded=False):
    st.text_area(
        "JSON",
        key="match_config_editor",
        height=420,
        help="This file controls team colour matching and the roster used to validate jersey numbers.",
    )

    save_config = st.button("Save match_config.json")
    if save_config:
        try:
            candidate = json.loads(st.session_state["match_config_editor"])
            errors = validate_match_config(candidate)
            if errors:
                st.error("Cannot save: " + " | ".join(errors))
            else:
                formatted = json.dumps(candidate, indent=2, ensure_ascii=False) + "\n"
                MATCH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                MATCH_CONFIG_PATH.write_text(formatted, encoding="utf-8")
                st.session_state["match_config_editor"] = formatted
                st.success("match_config.json saved. The next run will use this configuration.")
                st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON: {exc}")

st.divider()

with st.sidebar:
    st.header("Beta run")
    video_source = st.text_input("YouTube URL / local video", value=DEFAULT_VIDEO)
    start_time = st.text_input("Start", value="00:00")
    end_time = st.text_input("End", value="01:00")
    model_path = st.text_input("Model", value=DEFAULT_MODEL)

    detector_conf = st.slider(
        "Detector confidence",
        min_value=0.01,
        max_value=0.60,
        value=0.40,
        step=0.01,
        help="0.40 reproduces the previous hard-coded detector baseline.",
    )
    imgsz = st.selectbox("Inference image size", [640, 768, 960, 1280], index=0)

    device_options = ["mps", "cpu", "cuda"]
    default_device = settings.device if settings.device in device_options else "mps"
    device = st.selectbox("Device", device_options, index=device_options.index(default_device))
    tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)

    if not config_ready:
        st.warning("Fix match_config.json before running the analysis.")

    run_clicked = st.button(
        "🚀 Run beta analysis",
        type="primary",
        width="stretch",
        disabled=not config_ready,
    )

if run_clicked:
    analyzer = ROOT / "services" / "analytics" / "analyzer.py"
    command = [
        sys.executable,
        str(analyzer),
        "--video",
        video_source,
        "--start",
        start_time,
        "--end",
        end_time,
        "--model",
        model_path,
        "--conf",
        str(detector_conf),
        "--imgsz",
        str(imgsz),
        "--device",
        device,
        "--tracker",
        tracker,
        "--reset-db",
    ]

    status = st.status("Running analysis...", expanded=True)
    log_box = st.empty()

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    captured: list[str] = []
    run_dir: Path | None = None
    summary_path: Path | None = None

    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if not text:
            continue
        captured.append(text)
        log_box.code("\n".join(captured[-80:]), language="text")

        if text.startswith("[RUN_DIR]"):
            run_dir = Path(text.replace("[RUN_DIR]", "", 1).strip())
        elif text.startswith("[SUMMARY]"):
            summary_path = Path(text.replace("[SUMMARY]", "", 1).strip())

    process.wait()

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "console.log").write_text("\n".join(captured) + "\n", encoding="utf-8")

    if summary_path is not None and summary_path.exists():
        st.session_state["last_summary_path"] = str(summary_path)

    if process.returncode == 0:
        status.update(label="Analysis completed", state="complete", expanded=False)
        st.success("Run completed. Results and diagnostic logs were saved for comparison.")
    else:
        status.update(label=f"Analysis failed (exit {process.returncode})", state="error", expanded=True)
        st.error("The run failed. Its summary and console log are still useful diagnostic evidence.")

st.divider()

selected_summary = None
if st.session_state.get("last_summary_path"):
    candidate = Path(st.session_state["last_summary_path"])
    if candidate.exists():
        selected_summary = candidate
if selected_summary is None:
    selected_summary = latest_summary_path()

if selected_summary is not None:
    render_summary(selected_summary)
else:
    st.info("No beta diagnostic runs found yet. Start with a short baseline run.")

st.divider()
render_comparison()
