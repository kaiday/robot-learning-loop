"""Where runs go. Local disk is always the source of truth; the Cloudflare Worker is a mirror.

Set RLL_WORKER_URL (and RLL_WORKER_TOKEN) to upload every run to the Worker, which puts
files in R2 and indexes iterations in D1. Without them, runs stay local.
"""
from __future__ import annotations

import json
import mimetypes
import os
import urllib.request
from pathlib import Path


class LocalStorage:
    name = "local"

    def put_file(self, key: str, path: Path) -> None:
        pass

    def upload_dir(self, local_dir: Path, prefix: str) -> int:
        return 0

    def record_iteration(self, run_id: str, record: dict) -> None:
        pass


class WorkerStorage:
    name = "cloudflare"

    def __init__(self, base_url: str, token: str = "") -> None:
        self.base = base_url.rstrip("/")
        self.token = token

    def _req(self, method: str, path: str, body: bytes, content_type: str) -> dict:
        req = urllib.request.Request(f"{self.base}{path}", data=body, method=method)
        req.add_header("Content-Type", content_type)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read() or b"{}")

    def put_file(self, key: str, path: Path) -> None:
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._req("PUT", f"/blobs/{key}", Path(path).read_bytes(), ctype)

    def upload_dir(self, local_dir: Path, prefix: str) -> int:
        n = 0
        for f in sorted(Path(local_dir).rglob("*")):
            if f.is_file():
                self.put_file(f"{prefix}/{f.relative_to(local_dir).as_posix()}", f)
                n += 1
        return n

    def record_iteration(self, run_id: str, record: dict) -> None:
        self._req("POST", f"/runs/{run_id}/iterations", json.dumps(record).encode(), "application/json")


def get_storage() -> LocalStorage | WorkerStorage:
    url = os.environ.get("RLL_WORKER_URL")
    return WorkerStorage(url, os.environ.get("RLL_WORKER_TOKEN", "")) if url else LocalStorage()
