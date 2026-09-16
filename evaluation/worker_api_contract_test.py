from __future__ import annotations

import os
import sys
import types
import unittest
import uuid

# The worker contract tests exercise API/auth/path logic only. Keep CI lightweight by
# stubbing Torch instead of installing the full ML runtime just to import worker_api.
torch_stub = types.ModuleType("torch")
torch_stub.backends = types.SimpleNamespace(
    mps=types.SimpleNamespace(is_available=lambda: False)
)
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_stub)

from fastapi import HTTPException

from services import worker_api


class WorkerApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._host = os.environ.get("WORKER_HOST")
        self._token = os.environ.get("WORKER_API_TOKEN")
        os.environ.pop("WORKER_API_TOKEN", None)
        os.environ["WORKER_HOST"] = "127.0.0.1"

    def tearDown(self) -> None:
        if self._host is None:
            os.environ.pop("WORKER_HOST", None)
        else:
            os.environ["WORKER_HOST"] = self._host
        if self._token is None:
            os.environ.pop("WORKER_API_TOKEN", None)
        else:
            os.environ["WORKER_API_TOKEN"] = self._token

    def test_worker_exposes_core_routes(self) -> None:
        paths = {route.path for route in worker_api.app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/jobs", paths)

    def test_loopback_worker_can_run_without_shared_token(self) -> None:
        worker_api.require_worker_token(None)

    def test_non_loopback_worker_requires_shared_token(self) -> None:
        os.environ["WORKER_HOST"] = "0.0.0.0"
        with self.assertRaises(HTTPException) as raised:
            worker_api.require_worker_token(None)
        self.assertEqual(raised.exception.status_code, 503)

    def test_shared_token_rejects_missing_and_invalid_bearer(self) -> None:
        os.environ["WORKER_API_TOKEN"] = "ci-secret"

        with self.assertRaises(HTTPException) as missing:
            worker_api.require_worker_token(None)
        self.assertEqual(missing.exception.status_code, 401)

        with self.assertRaises(HTTPException) as invalid:
            worker_api.require_worker_token("Bearer wrong")
        self.assertEqual(invalid.exception.status_code, 403)

        worker_api.require_worker_token("Bearer ci-secret")

    def test_platform_ids_are_normalized_as_uuids(self) -> None:
        value = str(uuid.uuid4()).upper()
        self.assertEqual(worker_api.normalize_uuid(value, "platform_run_id"), value.lower())
        self.assertIsNone(worker_api.normalize_uuid(None, "platform_run_id"))

        with self.assertRaises(HTTPException) as invalid:
            worker_api.normalize_uuid("not-a-uuid", "platform_run_id")
        self.assertEqual(invalid.exception.status_code, 422)

    def test_job_ids_reject_path_traversal(self) -> None:
        valid = str(uuid.uuid4())
        self.assertEqual(worker_api.job_dir(valid).name, valid)

        for invalid in ("../escape", "job/child", "not_a_uuid"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(HTTPException) as raised:
                    worker_api.job_dir(invalid)
                self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
