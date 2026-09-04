from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional until dependencies are installed
    Client = Any  # type: ignore[misc,assignment]
    create_client = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flatten_numeric_metrics(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric_metrics(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and prefix:
        rows.append((prefix, float(value)))
    return rows


class SupabaseRunStore:
    """Best-effort platform persistence; inference must not depend on network writes."""

    def __init__(self) -> None:
        url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        self.client: Client | None = None
        self.enabled = bool(url and key and create_client is not None)
        self._retention_stop = threading.Event()

        if self.enabled:
            try:
                self.client = create_client(url, key)
                print("☁️ CourtFlux Supabase persistence: on")
            except Exception as exc:
                self.enabled = False
                print(f"⚠️ CourtFlux Supabase persistence disabled: {exc}")
        else:
            print("☁️ CourtFlux Supabase persistence: off (configure SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY)")

        if self.enabled and self.client:
            threading.Thread(
                target=self._retention_loop,
                name="courtflux-video-retention",
                daemon=True,
            ).start()

    def _warn(self, action: str, exc: Exception) -> None:
        print(f"⚠️ Supabase {action} failed: {type(exc).__name__}: {exc}")

    def _retention_loop(self) -> None:
        interval = max(3600, int(os.getenv("VIDEO_RETENTION_SWEEP_SECONDS", "21600")))
        # Give the worker time to finish startup before the first storage sweep.
        if self._retention_stop.wait(10):
            return
        while not self._retention_stop.is_set():
            try:
                self.purge_expired_analysis_videos()
            except Exception as exc:  # pragma: no cover - retention must never stop inference
                self._warn("video retention sweep", exc)
            self._retention_stop.wait(interval)

    def update_job(self, job_id: str | None, values: dict[str, Any]) -> None:
        if not self.client or not job_id:
            return
        try:
            self.client.table("analysis_jobs").update(values).eq("id", job_id).execute()
        except Exception as exc:
            self._warn("job update", exc)

    def update_run(self, run_id: str | None, values: dict[str, Any]) -> None:
        if not self.client or not run_id:
            return
        try:
            self.client.table("analysis_runs").update(values).eq("id", run_id).execute()
        except Exception as exc:
            self._warn("run update", exc)

    def purge_expired_analysis_videos(self, limit: int = 100) -> int:
        """Delete expired source videos while preserving analysis rows and derived metrics."""
        if not self.client:
            return 0

        purged = 0
        try:
            result = (
                self.client.table("analysis_video_uploads")
                .select("id,bucket,object_path")
                .is_("deleted_at", "null")
                .lte("expires_at", utc_now())
                .order("expires_at")
                .limit(max(1, min(limit, 500)))
                .execute()
            )
            rows = result.data or []
        except Exception as exc:
            self._warn("video retention query", exc)
            return 0

        for row in rows:
            row_id = row.get("id")
            bucket = row.get("bucket")
            object_path = row.get("object_path")
            if not row_id or not bucket or not object_path:
                continue

            attempted_at = utc_now()
            try:
                self.client.storage.from_(str(bucket)).remove([str(object_path)])
                self.client.table("analysis_video_uploads").update(
                    {
                        "purge_attempted_at": attempted_at,
                        "deleted_at": utc_now(),
                        "deletion_error": None,
                    }
                ).eq("id", row_id).execute()
                purged += 1
            except Exception as exc:
                try:
                    self.client.table("analysis_video_uploads").update(
                        {
                            "purge_attempted_at": attempted_at,
                            "deletion_error": f"{type(exc).__name__}: {exc}"[:1000],
                        }
                    ).eq("id", row_id).execute()
                except Exception as update_exc:
                    self._warn("video retention failure update", update_exc)
                self._warn("video retention delete", exc)

        if purged:
            print(f"🧹 CourtFlux retention: removed {purged} expired source video(s)")
        return purged

    def mark_running(self, job_id: str | None, run_id: str | None) -> None:
        started_at = utc_now()
        self.update_job(job_id, {"status": "running", "started_at": started_at})
        self.update_run(run_id, {"status": "running", "started_at": started_at})

    def mark_failed(
        self,
        job_id: str | None,
        run_id: str | None,
        error: str,
        log_tail: str | None = None,
    ) -> None:
        completed_at = utc_now()
        job_values: dict[str, Any] = {
            "status": "failed",
            "completed_at": completed_at,
            "error": error,
        }
        if log_tail:
            job_values["log_tail"] = log_tail[-12000:]
        self.update_job(job_id, job_values)
        self.update_run(run_id, {"status": "failed", "completed_at": completed_at})

    def mark_completed(
        self,
        job_id: str | None,
        run_id: str | None,
        summary: dict[str, Any] | None,
        request_data: dict[str, Any],
    ) -> None:
        completed_at = utc_now()
        summary = summary or {}
        self.update_job(job_id, {"status": "completed", "completed_at": completed_at, "error": None})
        self.update_run(
            run_id,
            {
                "status": "completed",
                "completed_at": completed_at,
                "summary": summary,
            },
        )

        if not self.client or not run_id:
            return

        try:
            metric_rows = [
                {
                    "run_id": run_id,
                    "scope": "run",
                    "metric_key": key,
                    "metric_value": value,
                }
                for key, value in _flatten_numeric_metrics(summary)
            ]
            if metric_rows:
                self.client.table("analysis_metrics").upsert(
                    metric_rows,
                    on_conflict="run_id,scope,metric_key",
                ).execute()
        except Exception as exc:
            self._warn("metric upsert", exc)

        workspace_id = request_data.get("workspace_id")
        user_id = request_data.get("user_id")
        start = request_data.get("start_time_sec")
        end = request_data.get("end_time_sec")
        try:
            if workspace_id and start is not None and end is not None:
                duration_minutes = max(0.0, float(end) - float(start)) / 60.0
                self.client.table("usage_events").insert(
                    {
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "analysis_run_id": run_id,
                        "metric_code": "analysis_minutes",
                        "quantity": duration_minutes,
                        "metadata": {
                            "device": request_data.get("device"),
                            "imgsz": request_data.get("imgsz"),
                            "subscription_plan": request_data.get("subscription_plan"),
                        },
                    }
                ).execute()
        except Exception as exc:
            self._warn("usage insert", exc)
