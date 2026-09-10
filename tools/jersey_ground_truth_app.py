from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ANNOTATION_FIELDS = [
    "sample_id",
    "subject_type",
    "jersey_gt",
    "visibility",
    "team_gt",
    "identity_group",
    "reviewer",
    "notes",
    "annotated_at",
]
TRACK_REVIEW_FIELDS = [
    "video", "track_id", "labels", "resolution", "reviewer", "notes", "reviewed_at",
]
SUBJECT_TYPES = ["player", "referee", "cameraman_staff", "other_non_player", "unknown"]
VISIBILITY = ["clear", "partial", "unreadable"]
SUBJECT_LABELS = {
    "": "— needs review —",
    "player": "Player",
    "referee": "Referee",
    "cameraman_staff": "Cameraman / staff",
    "other_non_player": "Other non-player",
    "unknown": "Unknown subject",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--track-reviews")
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
    numeric = frame["jersey_gt"].str.fullmatch(r"\d{1,2}", na=False)
    frame.loc[numeric & (frame["subject_type"] == ""), "subject_type"] = "player"
    return frame[ANNOTATION_FIELDS]


def _read_track_reviews(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=TRACK_REVIEW_FIELDS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in TRACK_REVIEW_FIELDS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[TRACK_REVIEW_FIELDS]


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


def _valid_player_label(value: str) -> bool:
    value = value.strip().lower()
    return value == "unreadable" or (value.isdigit() and 0 <= int(value) <= 99)


def _canonical_number(value: str) -> str:
    value = str(value or "").strip()
    if value.isdigit() and 0 <= int(value) <= 99:
        return str(int(value))
    return ""


def _labels_token(values: list[str]) -> str:
    return "|".join(sorted({_canonical_number(value) for value in values if _canonical_number(value)}, key=int))


def _review_for(
    track_reviews: pd.DataFrame,
    video: str,
    track_id: str,
) -> dict[str, str] | None:
    rows = track_reviews[
        (track_reviews["video"] == video) & (track_reviews["track_id"] == track_id)
    ]
    if rows.empty:
        return None
    return {field: str(rows.iloc[-1].get(field, "")) for field in TRACK_REVIEW_FIELDS}


def _save_track_review(
    track_reviews: pd.DataFrame,
    video: str,
    track_id: str,
    labels: list[str],
    reviewer: str,
    notes: str,
) -> pd.DataFrame:
    track_reviews = track_reviews[
        ~((track_reviews["video"] == video) & (track_reviews["track_id"] == track_id))
    ].copy()
    row = {
        "video": video,
        "track_id": track_id,
        "labels": _labels_token(labels),
        "resolution": "confirmed_id_switch",
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    return pd.concat([track_reviews, pd.DataFrame([row])], ignore_index=True)[TRACK_REVIEW_FIELDS]


def _track_conflicts(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    track_reviews: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    numeric = annotations[
        (annotations["subject_type"] == "player")
        & annotations["jersey_gt"].str.fullmatch(r"\d{1,2}", na=False)
    ].copy()
    if numeric.empty:
        return {}

    numeric["canonical_number"] = numeric["jersey_gt"].map(_canonical_number)
    locations = manifest[["sample_id", "video", "track_id", "timestamp_sec"]].copy()
    joined = numeric.merge(locations, on="sample_id", how="inner")

    conflicts: dict[tuple[str, str], dict[str, object]] = {}
    for (video, track_id), rows in joined.groupby(["video", "track_id"], sort=False):
        labels = sorted({value for value in rows["canonical_number"].tolist() if value}, key=int)
        if len(labels) <= 1:
            continue
        video = str(video)
        track_id = str(track_id)
        track_rows = manifest[(manifest["video"] == video) & (manifest["track_id"] == track_id)]
        review = _review_for(track_reviews, video, track_id)
        confirmed = bool(
            review
            and review.get("resolution") == "confirmed_id_switch"
            and review.get("labels") == _labels_token(labels)
        )
        conflicts[(video, track_id)] = {
            "labels": labels,
            "samples": int(len(track_rows)),
            "numeric_samples": int(len(rows)),
            "confirmed": confirmed,
            "review": review,
        }
    return conflicts


def _save_rows(
    annotations: pd.DataFrame,
    sample_ids: list[str],
    subject_type: str,
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
            "subject_type": subject_type,
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
        if args.annotations else manifest_path.parent / "annotations.csv"
    )
    track_reviews_path = (
        Path(args.track_reviews).expanduser().resolve()
        if args.track_reviews else manifest_path.parent / "track_reviews.csv"
    )
    if not manifest_path.exists():
        st.error(f"Manifest not found: {manifest_path}")
        st.stop()

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    annotations = _read_annotations(annotations_path)
    track_reviews = _read_track_reviews(track_reviews_path)
    reviewed_ids = set(annotations["sample_id"].tolist())
    needs_subject_ids = set(
        annotations.loc[
            (annotations["jersey_gt"].str.lower() == "unreadable")
            & (annotations["subject_type"] == ""),
            "sample_id",
        ].tolist()
    )
    conflict_tracks = _track_conflicts(manifest, annotations, track_reviews)
    unresolved_conflicts = {
        key: info for key, info in conflict_tracks.items() if not bool(info["confirmed"])
    }

    st.title("CourtFlux · Jersey OCR Ground Truth")
    st.caption(
        "Label what is actually visible. First decide whether the detection is a real player. "
        "Do not infer a jersey number from the roster or from OCR predictions."
    )

    selected_conflict: tuple[str, str] | None = None
    with st.sidebar:
        st.header("Queue")
        status = st.selectbox(
            "Status",
            ["unreviewed", "needs subject review", "track conflicts", "all", "reviewed"],
            index=0,
        )

        if status == "track conflicts" and conflict_tracks:
            conflict_keys = sorted(
                conflict_tracks,
                key=lambda item: (bool(conflict_tracks[item]["confirmed"]), int(item[1])),
            )
            conflict_labels = {}
            for key in conflict_keys:
                info = conflict_tracks[key]
                numbers = "/".join(info["labels"])
                state = "✅ confirmed" if info["confirmed"] else "⬜ needs confirmation"
                conflict_labels[key] = (
                    f"Track {key[1]} → {numbers} · {state} "
                    f"({info['samples']} crops)"
                )
            selected_conflict = st.selectbox(
                "Conflict track",
                conflict_keys,
                format_func=lambda key: conflict_labels[key],
                help="A confirmation is stored separately and tied to the exact current labels.",
            )
            st.caption(
                f"{len(unresolved_conflicts)} need confirmation · "
                f"{len(conflict_tracks) - len(unresolved_conflicts)} confirmed"
            )

        teams = ["all"] + sorted(
            value for value in manifest.get("team_suggested", pd.Series(dtype=str)).unique() if value
        )
        team_filter = st.selectbox("Suggested team", teams)
        track_filter = ""
        if status != "track conflicts":
            track_filter = st.text_input("Track ID", "").strip()
        reviewer_default = args.reviewer or st.session_state.get("reviewer", "")
        reviewer = st.text_input("Reviewer", reviewer_default)
        st.session_state["reviewer"] = reviewer

        if needs_subject_ids:
            st.info(f"Legacy unreadable rows needing subject review: {len(needs_subject_ids)}")
        if conflict_tracks:
            with st.expander("All conflict tracks", expanded=status == "track conflicts"):
                for key in sorted(conflict_tracks, key=lambda item: int(item[1])):
                    info = conflict_tracks[key]
                    icon = "✅" if info["confirmed"] else "⬜"
                    st.write(
                        f"{icon} Track **{key[1]}** → **{' / '.join(info['labels'])}** "
                        f"· {info['samples']} crops"
                    )

    queue = manifest.copy()
    if status == "unreviewed":
        queue = queue[~queue["sample_id"].isin(reviewed_ids)]
    elif status == "needs subject review":
        queue = queue[queue["sample_id"].isin(needs_subject_ids)]
    elif status == "track conflicts":
        if not conflict_tracks:
            st.success("No conflicting tracks remain.")
            st.stop()
        assert selected_conflict is not None
        video, track_id = selected_conflict
        queue = queue[(queue["video"] == video) & (queue["track_id"] == track_id)]
        queue = queue.sort_values("timestamp_sec", key=lambda col: pd.to_numeric(col, errors="coerce"))
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
    st.write(
        f"Reviewed: **{reviewed}/{total}** · Current queue: **{len(queue)}** · "
        f"Needs subject review: **{len(needs_subject_ids)}** · "
        f"Unconfirmed conflict tracks: **{len(unresolved_conflicts)}**"
    )

    if status == "track conflicts" and selected_conflict is not None:
        info = conflict_tracks[selected_conflict]
        if info["confirmed"]:
            review = info.get("review") or {}
            st.success(
                f"Track {selected_conflict[1]} is confirmed as a tracker ID switch for labels "
                f"{' / '.join(info['labels'])} by {review.get('reviewer', 'reviewer')}."
            )
        else:
            st.warning(
                f"Reviewing track {selected_conflict[1]} with conflicting labels "
                f"{' / '.join(info['labels'])}. If the labels are both correct and this is a real "
                "tracker ID switch, confirm it below after checking the crops."
            )

    if queue.empty:
        st.success("No samples match the current filters.")
        st.stop()

    history_key = f"history:{manifest_path}"
    forced_key = f"forced_sample:{manifest_path}"
    conflict_token = selected_conflict[1] if selected_conflict else "none"
    index_key = f"queue_index:{status}:{team_filter}:{track_filter}:{conflict_token}"
    history = list(st.session_state.get(history_key, []))
    forced_sample_id = str(st.session_state.get(forced_key, "") or "")

    if forced_sample_id:
        forced_rows = manifest[manifest["sample_id"] == forced_sample_id]
        if forced_rows.empty:
            st.session_state[forced_key] = ""
            forced_sample_id = ""
        else:
            row = forced_rows.iloc[0]
            index = 0
    if not forced_sample_id:
        index = int(st.session_state.get(index_key, 0))
        index = max(0, min(index, len(queue) - 1))
        st.session_state[index_key] = index
        row = queue.iloc[index]

    sample_id = str(row["sample_id"])
    if not history or history[-1] != sample_id:
        history.append(sample_id)
        st.session_state[history_key] = history[-500:]

    current = _annotation_for(annotations, sample_id)
    has_annotation = sample_id in reviewed_ids
    location_text = "history" if forced_sample_id else f"{index + 1}/{len(queue)}"
    st.subheader(f"Sample {location_text} · {sample_id}")
    st.write(
        f"Track **{row.get('track_id', '')}** · t={row.get('timestamp_sec', '')}s · "
        f"current GT **{current.get('jersey_gt', '') or 'unlabelled'}** · "
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
        value for value in manifest.get("team_suggested", pd.Series(dtype=str)).unique() if value
    )
    current_team = current.get("team_gt", "")
    if current_team and current_team not in teams_for_gt:
        teams_for_gt.append(current_team)

    form_key = sample_id
    current_subject = current.get("subject_type", "")
    if not has_annotation and not current_subject:
        current_subject = "player"
    subject_options = [""] + SUBJECT_TYPES
    subject_type = st.selectbox(
        "Subject type",
        subject_options,
        index=subject_options.index(current_subject) if current_subject in subject_options else 0,
        format_func=lambda value: SUBJECT_LABELS.get(value, value),
        key=f"subject:{form_key}",
        help="Only real basketball players are eligible for jersey OCR.",
    )

    if subject_type == "player":
        jersey_gt = st.text_input(
            "Jersey GT (0–99, or unreadable)",
            value=current.get("jersey_gt", "") if current.get("jersey_gt") != "not_applicable" else "",
            key=f"jersey:{form_key}",
        ).strip().lower()
        visibility_default = current.get("visibility", "")
        visibility_index = VISIBILITY.index(visibility_default) if visibility_default in VISIBILITY else 0
        visibility = st.selectbox(
            "Visibility", VISIBILITY, index=visibility_index, key=f"visibility:{form_key}"
        )
    else:
        jersey_gt = "not_applicable"
        visibility = "not_applicable"
        if subject_type:
            st.info(
                "Non-player/unknown subject: retained as detector evidence but excluded "
                "from jersey OCR training/evaluation."
            )
        else:
            st.warning("Choose a subject type before saving this annotation.")

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
        help="For a confirmed ID switch, identity_group can additionally identify the physical player.",
    ).strip()
    notes = st.text_area("Notes", value=current.get("notes", ""), key=f"notes:{form_key}").strip()

    propagate = st.checkbox(
        "Apply this annotation to every sample in this video+track",
        value=False,
        key=f"propagate:{form_key}",
        help="Do NOT use this for a conflict track unless you are certain there is no ID switch.",
        disabled=status == "track conflicts",
    )

    def target_ids() -> list[str]:
        if not propagate:
            return [sample_id]
        same_track = manifest[
            (manifest["video"] == row["video"]) & (manifest["track_id"] == row["track_id"])
        ]
        return same_track["sample_id"].astype(str).tolist()

    def advance_after_save() -> None:
        st.session_state[forced_key] = ""
        if status in {"unreviewed", "needs subject review"}:
            st.session_state[index_key] = index
        else:
            st.session_state[index_key] = min(max(len(queue) - 1, 0), index + 1)

    previous_col, save_col, unreadable_col, next_col = st.columns(4)
    with previous_col:
        if st.button("← Previous", use_container_width=True, disabled=len(history) < 2):
            history = list(st.session_state.get(history_key, []))
            if len(history) >= 2:
                history.pop()
                previous_sample = history[-1]
                st.session_state[history_key] = history
                st.session_state[forced_key] = previous_sample
                st.rerun()
    with next_col:
        if st.button("Next →", use_container_width=True):
            if forced_sample_id:
                st.session_state[forced_key] = ""
            else:
                st.session_state[index_key] = min(len(queue) - 1, index + 1)
            st.rerun()

    save_label = "Save & next" if subject_type == "player" else "Save subject & next"
    with save_col:
        if st.button(save_label, type="primary", use_container_width=True):
            if not reviewer.strip():
                st.error("Reviewer is required.")
            elif not subject_type:
                st.error("Subject type is required.")
            elif subject_type == "player" and not _valid_player_label(jersey_gt):
                st.error("For a player, Jersey GT must be a number from 0 to 99, or 'unreadable'.")
            else:
                updated = _save_rows(
                    annotations,
                    target_ids(),
                    subject_type,
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
        if st.button(
            "Player unreadable & next",
            use_container_width=True,
            disabled=subject_type != "player",
        ):
            if not reviewer.strip():
                st.error("Reviewer is required.")
            else:
                updated = _save_rows(
                    annotations,
                    target_ids(),
                    "player",
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

    if status == "track conflicts" and selected_conflict is not None:
        st.divider()
        info = conflict_tracks[selected_conflict]
        st.subheader("Track conflict review")
        if info["confirmed"]:
            review = info.get("review") or {}
            st.success(
                f"Confirmed ID switch · labels {review.get('labels', '')} · "
                f"reviewer {review.get('reviewer', '')} · {review.get('reviewed_at', '')}"
            )
        else:
            review_notes = st.text_input(
                "Conflict review notes (optional)",
                value="Manual review confirmed both jersey labels are correct",
                key=f"conflict_notes:{selected_conflict[1]}",
            )
            if st.button("Confirm reviewed tracker ID switch", type="primary"):
                if not reviewer.strip():
                    st.error("Reviewer is required before confirming an ID switch.")
                else:
                    updated_reviews = _save_track_review(
                        track_reviews,
                        selected_conflict[0],
                        selected_conflict[1],
                        list(info["labels"]),
                        reviewer.strip(),
                        review_notes.strip(),
                    )
                    _atomic_write(track_reviews_path, updated_reviews)
                    st.rerun()

        st.subheader("Current track labels")
        track_sample_ids = set(queue["sample_id"].astype(str))
        track_annotations = annotations[annotations["sample_id"].isin(track_sample_ids)].copy()
        display = queue[["sample_id", "timestamp_sec"]].merge(
            track_annotations[["sample_id", "jersey_gt", "identity_group"]],
            on="sample_id",
            how="left",
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(f"Annotations file: {annotations_path}")
    st.caption(f"Track reviews file: {track_reviews_path}")


if __name__ == "__main__":
    main()
