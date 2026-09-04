from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import settings


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "evaluation" / "runs"
DEFAULT_VIDEO = "https://www.youtube.com/watch?v=OwiQgAMqyUs"

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


def latest_summary_path() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    summaries = sorted(RUNS_DIR.glob("*/summary.json"), reverse=True)
    return summaries[0] if summaries else None


def flatten_summary(summary: dict) -> dict:
    ball = summary.get("ball", {})
    players = summary.get("players", {})
    return {
        "run_id": summary.get("run_id"),
        "model": summary.get("model"),
        "conf": summary.get("detector_conf"),
        "imgsz": summary.get("imgsz"),
        "device": summary.get("device"),
        "frames": summary.get("gameplay_frames"),
        "raw players/frame": players.get("raw_per_gameplay_frame"),
        "parsed players/frame": players.get("parsed_per_gameplay_frame"),
        "raw ball candidate %": ball.get("raw_candidate_frame_rate"),
        "selected ball %": ball.get("selected_detection_frame_rate"),
        "predicted ball %": ball.get("predicted_hold_frame_rate"),
        "missing ball %": ball.get("missing_frame_rate"),
        "ball rejected by court": ball.get("rejected_outside_court"),
        "ball mean conf": ball.get("selected_conf_mean"),
        "homography %": summary.get("homography_active_rate"),
        "elapsed s": summary.get("elapsed_sec"),
    }


def render_summary(summary_path: Path) -> None:
    summary = load_summary(summary_path)
    if not summary:
        return

    st.subheader("Latest diagnostic summary")
    st.caption(f"Run: {summary.get('run_id')} · {summary_path.parent}")

    ball = summary.get("ball", {})
    players = summary.get("players", {})

    a, b, c, d, e = st.columns(5)
    a.metric("Gameplay frames", summary.get("gameplay_frames", 0))
    b.metric("Raw players/frame", players.get("raw_per_gameplay_frame", 0))
    c.metric("Raw ball candidate", f"{ball.get('raw_candidate_frame_rate', 0):.1f}%")
    d.metric("Selected ball", f"{ball.get('selected_detection_frame_rate', 0):.1f}%")
    e.metric("Ball rejected by court", ball.get("rejected_outside_court", 0))

    a, b, c, d = st.columns(4)
    a.metric("Predicted/held ball", f"{ball.get('predicted_hold_frame_rate', 0):.1f}%")
    b.metric("Missing ball", f"{ball.get('missing_frame_rate', 0):.1f}%")
    c.metric("Mean ball confidence", ball.get("selected_conf_mean") or 0)
    d.metric("Homography active", f"{summary.get('homography_active_rate', 0):.1f}%")

    run_dir = summary_path.parent
    summary_bytes = summary_path.read_bytes()
    st.download_button(
        "Download summary.json",
        data=summary_bytes,
        file_name=f"{summary.get('run_id', 'run')}_summary.json",
        mime="application/json",
    )

    frames_path = run_dir / "frames.jsonl"
    if frames_path.exists():
        st.download_button(
            "Download frame diagnostics JSONL",
            data=frames_path.read_bytes(),
            file_name=f"{summary.get('run_id', 'run')}_frames.jsonl",
            mime="application/x-ndjson",
        )

    console_path = run_dir / "console.log"
    if console_path.exists():
        st.download_button(
            "Download console.log",
            data=console_path.read_bytes(),
            file_name=f"{summary.get('run_id', 'run')}_console.log",
            mime="text/plain",
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
    st.caption("Use the same video interval and change one detector parameter at a time.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


st.title("🏀 Basketball AI Analytics — Beta Lab")
st.write(
    "Local test bench for one repeatable workflow: YouTube → pipeline → frame-level diagnostics → comparison."
)

with st.sidebar:
    st.header("Beta run")
    video_source = st.text_input("YouTube URL / local video", value=DEFAULT_VIDEO)
    start_time = st.text_input("Start", value="00:00")
    end_time = st.text_input("End", value="01:00")
    model_path = st.text_input("Model", value=settings.model)

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

    run_clicked = st.button("🚀 Run beta analysis", type="primary", width="stretch")

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

    st.code(" ".join(command), language="bash")
    status = st.status("Running analysis...", expanded=True)
    log_box = st.empty()
    preview_box = st.empty()

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

        elif text.startswith("[PREVIEW]"):
            preview_path = Path(text.replace("[PREVIEW]", "", 1).strip())
            if preview_path.exists():
                preview_box.image(str(preview_path), caption="Latest analysed frame", width="stretch")

    process.wait()

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "console.log").write_text("\n".join(captured) + "\n", encoding="utf-8")

    if process.returncode == 0:
        status.update(label="Analysis completed", state="complete", expanded=False)
        st.success("Run completed. Diagnostic files were saved for comparison.")
        if summary_path is not None and summary_path.exists():
            st.session_state["last_summary_path"] = str(summary_path)
    else:
        status.update(label=f"Analysis failed (exit {process.returncode})", state="error", expanded=True)
        st.error("The run failed. Keep the console.log: the failure itself is useful beta evidence.")

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
    st.info("No beta diagnostic runs found yet. Start with a short 60-second baseline run.")

st.divider()
render_comparison()
