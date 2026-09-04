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
DEFAULT_MODEL = settings.model if configured_model.exists() else (
    "weights/Model_v0.1.pt" if fallback_model.exists() else settings.model
)

st.set_page_config(page_title="Basketball AI Analytics — Beta Lab", page_icon="🏀", layout="wide")


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_match_config(config: dict) -> list[str]:
    errors = []
    if not isinstance(config, dict):
        return ["The match config must be a JSON object."]
    teams = config.get("teams")
    if not isinstance(teams, dict) or len(teams) < 2:
        return ["'teams' must contain at least two teams."]
    metadata = config.get("metadata", {}) or {}
    for key in ("home_team", "away_team"):
        name = metadata.get(key)
        if name and name not in teams:
            errors.append(f"metadata.{key}='{name}' is not present in teams.")
    for team_name, team_data in teams.items():
        if not isinstance(team_data, dict):
            errors.append(f"Team '{team_name}' must be a JSON object.")
            continue
        if not isinstance(team_data.get("kits"), dict) or not team_data.get("kits"):
            errors.append(f"Team '{team_name}' needs kits.")
        if not isinstance(team_data.get("roster"), dict) or not team_data.get("roster"):
            errors.append(f"Team '{team_name}' needs a roster.")
    return errors


def render_match_summary(config: dict) -> None:
    teams = config.get("teams", {}) or {}
    metadata = config.get("metadata", {}) or {}
    home = metadata.get("home_team") or list(teams.keys())[0]
    away = metadata.get("away_team") or list(teams.keys())[1]
    st.subheader("Match configuration")
    st.caption("Used for team colour distinction and roster-constrained jersey recognition.")
    for col, team_name in zip(st.columns(2), [home, away]):
        data = teams.get(team_name, {}) or {}
        with col:
            st.markdown(f"**{team_name}**")
            st.caption(f"{len(data.get('roster', {}) or {})} roster entries · kits: {', '.join((data.get('kits', {}) or {}).keys())}")
            rows = [
                {"#": number, "Player": (player or {}).get("name", ""), "Position": (player or {}).get("position", "")}
                for number, player in (data.get("roster", {}) or {}).items()
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=220)


def latest_summary_path() -> Path | None:
    paths = sorted(RUNS_DIR.glob("*/summary.json"), reverse=True) if RUNS_DIR.exists() else []
    return paths[0] if paths else None


def flatten_summary(summary: dict) -> dict:
    players = summary.get("players", {}) or {}
    ball = summary.get("ball", {}) or {}
    perf = summary.get("performance", {}) or {}
    return {
        "run_id": summary.get("run_id"),
        "mode": "detect" if summary.get("detection_only") else "full",
        "jersey": summary.get("jersey_enabled"),
        "conf": summary.get("detector_conf"),
        "imgsz": summary.get("imgsz"),
        "players/frame": players.get("raw_per_gameplay_frame"),
        "team %": players.get("team_classification_rate"),
        "jersey %": players.get("jersey_identification_rate"),
        "ball %": ball.get("selected_detection_frame_rate"),
        "missing ball %": ball.get("missing_frame_rate"),
        "analysis FPS": perf.get("analysis_fps"),
        "elapsed s": summary.get("elapsed_sec"),
        "status": (summary.get("extra", {}) or {}).get("status"),
    }


def render_summary(path: Path) -> None:
    summary = load_summary(path)
    if not summary:
        return
    players = summary.get("players", {}) or {}
    ball = summary.get("ball", {}) or {}
    performance = summary.get("performance", {}) or {}

    st.subheader("Latest results")
    st.caption(f"Run: {summary.get('run_id')} · mode: {(summary.get('extra', {}) or {}).get('mode', 'unknown')}")
    if (summary.get("extra", {}) or {}).get("status") == "failed":
        st.error((summary.get("extra", {}) or {}).get("error", "Run failed"))

    a, b, c, d = st.columns(4)
    a.metric("Players / frame", players.get("raw_per_gameplay_frame", 0))
    b.metric("Team classified", f"{float(players.get('team_classification_rate') or 0):.1f}%")
    c.metric("Jersey identified", f"{float(players.get('jersey_identification_rate') or 0):.1f}%")
    d.metric("Ball detected", f"{float(ball.get('selected_detection_frame_rate') or 0):.1f}%")

    a, b, c, d = st.columns(4)
    a.metric("Ball missing", f"{float(ball.get('missing_frame_rate') or 0):.1f}%")
    b.metric("Analysis FPS", performance.get("analysis_fps", 0))
    c.metric("Real-time factor", performance.get("realtime_factor", 0))
    d.metric("Elapsed", f"{float(summary.get('elapsed_sec') or 0):.1f}s")

    timing = summary.get("detector_timing", {}) or {}
    if timing:
        st.markdown("**Detector timing**")
        timing_rows = [
            {"Stage": name, "Calls": values.get("calls"), "Total s": values.get("total_sec"), "Avg ms": values.get("avg_ms")}
            for name, values in timing.items()
        ]
        st.dataframe(pd.DataFrame(timing_rows), width="stretch", hide_index=True)

    stages = performance.get("stages", {}) or {}
    if stages:
        st.markdown("**Pipeline timing**")
        stage_rows = [
            {
                "Stage": name,
                "Calls": values.get("calls"),
                "Total s": values.get("total_sec"),
                "Avg ms": values.get("avg_ms"),
                "Wall %": values.get("wall_pct"),
            }
            for name, values in stages.items()
        ]
        st.dataframe(pd.DataFrame(stage_rows), width="stretch", hide_index=True)

    run_dir = path.parent
    cols = st.columns(4)
    files = [
        (path, "summary.json", "application/json"),
        (run_dir / "frames.jsonl", "frames.jsonl", "application/x-ndjson"),
        (run_dir / "console.log", "console.log", "text/plain"),
        (run_dir / "match_config.json", "match_config.json", "application/json"),
    ]
    for col, (file_path, label, mime) in zip(cols, files):
        if file_path.exists():
            with col:
                st.download_button(label, data=file_path.read_bytes(), file_name=f"{summary.get('run_id', 'run')}_{label}", mime=mime, width="stretch")


st.title("🏀 Basketball AI Analytics — Performance Beta")
st.write("Results-first benchmark for detection, team distinction, ball recall and optional jersey recognition.")

if "match_config_editor" not in st.session_state:
    st.session_state["match_config_editor"] = MATCH_CONFIG_PATH.read_text(encoding="utf-8") if MATCH_CONFIG_PATH.exists() else "{}"
try:
    current_match_config = json.loads(st.session_state["match_config_editor"])
    config_errors = validate_match_config(current_match_config)
except Exception as exc:
    current_match_config = {}
    config_errors = [str(exc)]
config_ready = not config_errors

if config_ready:
    render_match_summary(current_match_config)
else:
    st.error("match_config.json is not ready: " + " | ".join(config_errors))

with st.expander("Edit match_config.json", expanded=False):
    st.text_area("JSON", key="match_config_editor", height=420)
    if st.button("Save match_config.json"):
        try:
            candidate = json.loads(st.session_state["match_config_editor"])
            errors = validate_match_config(candidate)
            if errors:
                st.error("Cannot save: " + " | ".join(errors))
            else:
                MATCH_CONFIG_PATH.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                st.success("match_config.json saved.")
        except Exception as exc:
            st.error(f"Invalid JSON: {exc}")

with st.sidebar:
    st.header("Performance run")
    video_source = st.text_input("YouTube URL / local video", value=DEFAULT_VIDEO)
    start_time = st.text_input("Start", value="00:00")
    end_time = st.text_input("End", value="01:00")
    model_path = st.text_input("Model", value=DEFAULT_MODEL)
    detector_conf = st.slider("Detector confidence", 0.01, 0.60, 0.10, 0.01)
    imgsz = st.selectbox("Inference image size", [640, 768, 960, 1280], index=0)
    device_options = ["mps", "cpu", "cuda"]
    default_device = settings.device if settings.device in device_options else "mps"
    device = st.selectbox("Device", device_options, index=device_options.index(default_device))
    tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)

    st.divider()
    detection_only = st.toggle("Detection-only mode", value=True, help="Skips camera/homography and tactical analytics.")
    enable_jersey = st.toggle("Jersey recognition", value=False, help="Off is the fastest clean detection benchmark.")
    jersey_interval = st.number_input("Jersey attempt interval (frames)", min_value=1, max_value=300, value=12, step=1, disabled=not enable_jersey)
    persist_results = st.toggle("Persist SQLite results", value=True, help="Writes are batched; turn off to measure pure processing speed.")

    if not config_ready:
        st.warning("Fix match_config.json before running.")
    run_clicked = st.button("🚀 Run performance beta", type="primary", width="stretch", disabled=not config_ready)

if run_clicked:
    command = [
        sys.executable,
        str(ROOT / "services" / "analytics" / "analyzer.py"),
        "--video", video_source,
        "--start", start_time,
        "--end", end_time,
        "--model", model_path,
        "--conf", str(detector_conf),
        "--imgsz", str(imgsz),
        "--device", device,
        "--tracker", tracker,
        "--jersey-interval", str(int(jersey_interval)),
        "--db-batch-size", "100",
        "--diagnostics-batch-size", "100",
        "--reset-db",
    ]
    if detection_only:
        command.append("--detection-only")
    if not enable_jersey:
        command.append("--no-jersey")
    if not persist_results:
        command.append("--no-persist")

    status = st.status("Running analysis...", expanded=True)
    log_box = st.empty()
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1)
    captured = []
    run_dir = None
    summary_path = None
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
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "console.log").write_text("\n".join(captured) + "\n", encoding="utf-8")
    if summary_path and summary_path.exists():
        st.session_state["last_summary_path"] = str(summary_path)
    if process.returncode == 0:
        status.update(label="Analysis completed", state="complete", expanded=False)
        st.success("Performance run completed.")
    else:
        status.update(label=f"Analysis failed (exit {process.returncode})", state="error", expanded=True)

st.divider()
selected = Path(st.session_state["last_summary_path"]) if st.session_state.get("last_summary_path") else latest_summary_path()
if selected and selected.exists():
    render_summary(selected)
else:
    st.info("No diagnostic runs found yet.")

st.divider()
if RUNS_DIR.exists():
    rows = [flatten_summary(load_summary(path)) for path in sorted(RUNS_DIR.glob("*/summary.json"), reverse=True)[:20]]
    rows = [row for row in rows if row.get("run_id")]
    if rows:
        st.subheader("Run comparison")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
