from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "courtflux" / "worker-api-token"


def load_worker_environment() -> None:
    explicit_token = os.getenv("WORKER_API_TOKEN", "").strip()

    # Keep worker/backend configuration in the repository-root .env, but do not
    # let a legacy token there override an explicitly supplied process token.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    if not explicit_token:
        configured_path = os.getenv("WORKER_API_TOKEN_FILE", "").strip()
        token_path = Path(configured_path).expanduser() if configured_path else DEFAULT_TOKEN_FILE
        try:
            file_token = token_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            file_token = ""
        except OSError as exc:
            raise RuntimeError(f"Could not read worker token file: {token_path}") from exc

        if file_token:
            os.environ["WORKER_API_TOKEN"] = file_token

    if not os.getenv("WORKER_API_TOKEN", "").strip():
        raise RuntimeError(
            "WORKER_API_TOKEN is not configured. Set it in the process environment "
            "or create ~/.config/courtflux/worker-api-token."
        )

    os.environ.setdefault("WORKER_HOST", "127.0.0.1")
    os.environ.setdefault("WORKER_PORT", "8787")
    os.environ.setdefault("WORKER_MAX_JOBS", "1")


def main() -> None:
    load_worker_environment()
    os.chdir(PROJECT_ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "services.worker_api"])


if __name__ == "__main__":
    main()
