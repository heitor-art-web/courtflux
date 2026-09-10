from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ANNOTATION_FIELDS = [
    "sample_id", "jersey_gt", "visibility", "team_gt", "identity_group",
    "reviewer", "notes", "annotated_at",
]
VISIBILITY = ["clear", "partial", "unreadable"]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--reviewer", default="")
    args, _ = parser.parse_known_args()
    return args


def _read_annotations(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=ANNOTATION_FIELDS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ANNOTATION_FIELDS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[ANNOTATION_FIELDS]


def _atomic_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, quoting=csv.QUOTE_MINIMAL)
    temp.replace(path)


def _annotation_for(frame: pd.DataFrame, sample_id: str) -> dict[str, str]:
    rows = frame[frame["sample_id"] == sample_id]
    if rows.empty:
        return {field: "" for field in ANNOTATION_FIELDS}
    return {field: str(rows.iloc[-1].get(field, "")) for field in ANNOTATION_FIELDS}


def _valid_label(value: str) -> bool:
    value = value.strip().lower()
    return value == "unreadable" or (value.isdigit() and 0 <= int(value) <= 99)


def _save_rows(
    annotations: pd.DataFrame,
    sample_ids: list[str],
    jersey_gt: str,
    visibility: str,
    team_gt: str,
    identity_group: str,
    reviewer: str,
    notes: str,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    annotations = annotations[~annotations["sample_id"].isin(sample_ids)].copy()
    additions = pd.DataFrame([
        {
            "sample_id": sample_id,
            "jersey_gt": jersey_gt,
            "visibility": visibility,
            "team_gt": team_gt,
            "identity_group": identity_group,
            "reviewer": reviewer,
            "notes": notes,
            "annotated_at": now,
        }
        for sample_id in sample_ids
    ])
    return pd.concat([annotations, additions], ignore_index=True)[ANNOTATION_FIELDS]


def main() -> None:
    st.set_page_config(page_title="CourtFlux Jersey OCR Ground Truth", layout="wide")
    args = _args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations
        else manifest_path.parent / "annotations.csv"
    )
    if not manifest_path.exists():
        st.error(f"Manifest not found: {manifest_path}")
        st.stop()

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    annotations = _read_annotations(annotations_path)
    reviewed_ids = set(annotations["sample_id"].tolist())

    st.title("CourtFlux · Jersey OCR Ground Truth")
    st.caption(
        "Annotate what is actually visible in the crop. Do not infer a number from the roster "
        "or from previous OCR predictions."
    )

    with st.sidebar:
        st.header("Queue")
        status = st.selectbox("Status", ["unreviewed", "all", "reviewed"], index=0)
        teams = ["all"] + sorted(
            value for value in manifest.get("team_suggested", pd.Series(dtype=str)).unique()
            if value
        )
        team_filter = st.selectbox("Suggested team", teams)
        track_filter = st.text_input("Track ID", "").strip()
        reviewer_default = args.reviewer or st.session_state.get("reviewer", "")
        reviewer = st.text_input("Reviewer", reviewer_default)
        st.session_state["reviewer"] = reviewer

    queue = manifest.copy()
    if status == "unreviewed":
        queue = queue[~queue["sample_id"].isin(reviewed_ids)]
    elif status == "reviewed":
        queue = queue[queue["sample_id"].isin(reviewed_ids)]
    if team_filter != "all":
        queue = queue[queue["team_suggested"] == team_filter]
    if track_filter:
        queue = queue[queue["track_id"] == track_filter]
    queue = queue.reset_index(drop=True)

    total = len(manifest)
    reviewed = len(reviewed_ids.intersection(set(manifest["sample_id"])))
    st.progress(reviewed / total if total else 0.0)
    st.write(f"Reviewed: **{reviewed}/{total}** · Current queue: **{len(queue)}**")
    if queue.empty:
        st.success("No samples match the current filters.")
        st.stop()

    index_key = f"queue_index:{status}:{team_filter}:{track_filter}"
    index = int(st.session_state.get(index_key, 0))
    index = max(0, min(index, len(queue) - 1))
    st.session_state[index_key] = index
    row = queue.iloc[index]
    sample_id = str(row["sample_id"])
    current = _annotation_for(annotations, sample_id)

    st.subheader(f"Sample {index + 1}/{len(queue)} · {sample_id}")
    st.write(
        f"Track **{row.get('track_id', '')}** · t={row.get('timestamp_sec', '')}s · "
        f"suggested team **{row.get('team_suggested', '') or 'unknown'}** · "
        f"sharpness {row.get('torso_sharpness', '')}"
    )

    torso_path = manifest_path.parent / str(row["torso_crop"])
    player_path = manifest_path.parent / str(row["player_crop"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Torso crop**")
        if torso_path.exists():
            st.image(str(torso_path), use_container_width=True)
        else:
            st.error(f"Missing crop: {torso_path}")
    with right:
        st.markdown("**Full player crop**")
        if player_path.exists():
            st.image(str(player_path), use_container_width=True)
        else:
            st.error(f"Missing crop: {player_path}")

    teams_for_gt = [""] + sorted(
        value for value in manifest.get("team_suggested", pd.Series(dtype=str)).unique()
        if value
    )
    current_team = current.get("team_gt", "")
    if current_team and current_team not in teams_for_gt:
        teams_for_gt.append(current_team)

    form_key = sample_id
    jersey_gt = st.text_input(
        "Jersey GT (0–99, or unreadable)",
        value=current.get("jersey_gt", ""),
        key=f"jersey:{form_key}",
    ).strip().lower()
    visibility_default = current.get("visibility", "")
    visibility_index = VISIBILITY.index(visibility_default) if visibility_default in VISIBILITY else 0
    visibility = st.selectbox(
        "Visibility", VISIBILITY, index=visibility_index, key=f"visibility:{form_key}"
    )
    team_gt = st.selectbox(
        "Team GT (optional)",
        teams_for_gt,
        index=teams_for_gt.index(current_team) if current_team in teams_for_gt else 0,
        key=f"team:{form_key}",
    )
    identity_group = st.text_input(
        "Identity group (optional, same physical player across fragmented tracks)",
        value=current.get("identity_group", ""),
        key=f"identity:{form_key}",
    ).strip()
    notes = st.text_area(
        "Notes", value=current.get("notes", ""), key=f"notes:{form_key}"
    ).strip()
    propagate = st.checkbox(
        "Apply this annotation to every sample in this video+track",
        value=False,
        key=f"propagate:{form_key}",
        help="Use only after visually checking that the track does not contain an ID switch.",
    )

    def target_ids() -> list[str]:
        if not propagate:
            return [sample_id]
        same_track = manifest[
            (manifest["video"] == row["video"]) & (manifest["track_id"] == row["track_id"])
        ]
        return same_track["sample_id"].astype(str).tolist()

    def advance_after_save() -> None:
        # In the default unreviewed queue the saved sample disappears on rerun, so keep
        # the same index. Incrementing here would skip the next unreviewed sample.
        if status == "unreviewed":
            st.session_state[index_key] = index
        else:
            st.session_state[index_key] = min(len(queue) - 1, index + 1)

    previous_col, save_col, unreadable_col, next_col = st.columns(4)
    with previous_col:
        if st.button("← Previous", use_container_width=True):
            st.session_state[index_key] = max(0, index - 1)
            st.rerun()
    with next_col:
        if st.button("Next →", use_container_width=True):
            st.session_state[index_key] = min(len(queue) - 1, index + 1)
            st.rerun()

    with save_col:
        if st.button("Save & next", type="primary", use_container_width=True):
            if not reviewer.strip():
                st.error("Reviewer is required.")
            elif not _valid_label(jersey_gt):
                st.error("Jersey GT must be a number from 0 to 99, or 'unreadable'.")
            else:
                updated = _save_rows(
                    annotations,
                    target_ids(),
                    jersey_gt,
                    visibility,
                    team_gt,
                    identity_group,
                    reviewer.strip(),
                    notes,
                )
                _atomic_write(annotations_path, updated)
                advance_after_save()
                st.rerun()

    with unreadable_col:
        if st.button("Unreadable & next", use_container_width=True):
            if not reviewer.strip():
                st.error("Reviewer is required.")
            else:
                updated = _save_rows(
                    annotations,
                    target_ids(),
                    "unreadable",
                    "unreadable",
                    team_gt,
                    identity_group,
                    reviewer.strip(),
                    notes,
                )
                _atomic_write(annotations_path, updated)
                advance_after_save()
                st.rerun()

    st.divider()
    st.caption(f"Annotations file: {annotations_path}")


if __name__ == "__main__":
    main()
