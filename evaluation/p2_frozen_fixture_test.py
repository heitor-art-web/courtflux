from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "g4UROqJoYBM.mp4"
EXPECTED_SIZE = 633_294
EXPECTED_GIT_BLOB_SHA1 = "966f136f41478dceba45c4b7068c2a433e17c86d"


class P2FrozenFixtureTest(unittest.TestCase):
    def test_fixture_is_exact_legacy_p2_blob(self) -> None:
        data = FIXTURE.read_bytes()
        self.assertEqual(len(data), EXPECTED_SIZE)
        header = f"blob {len(data)}\0".encode("ascii")
        digest = hashlib.sha1(header + data, usedforsecurity=False).hexdigest()
        self.assertEqual(digest, EXPECTED_GIT_BLOB_SHA1)


if __name__ == "__main__":
    unittest.main()
