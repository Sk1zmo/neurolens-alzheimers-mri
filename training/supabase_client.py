"""Minimal Supabase client used by the retraining pipeline.

Deliberately dependency-free beyond `requests`: it talks to PostgREST and the
Storage HTTP API directly rather than pulling in the supabase-py stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import requests

try:
    from dotenv import load_dotenv
    for candidate in (
        Path(__file__).resolve().parent.parent / "web" / ".env.local",
        Path(__file__).resolve().parent.parent / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate)
except ImportError:  # dotenv is optional
    pass


class SupabaseError(RuntimeError):
    pass


class Supabase:
    def __init__(self, url: str | None = None, service_key: str | None = None):
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = service_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not self.url or not self.key:
            raise SupabaseError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
                "(put them in alzheimer-app/web/.env.local or the environment)."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
        })

    # ------------------------------------------------------------------ REST
    def select(self, table: str, params: dict[str, str] | None = None,
               page_size: int = 1000) -> Iterator[dict[str, Any]]:
        """Paginated SELECT via PostgREST Range headers."""
        offset = 0
        while True:
            r = self.session.get(
                f"{self.url}/rest/v1/{table}",
                params={"select": "*", **(params or {})},
                headers={"Range": f"{offset}-{offset + page_size - 1}"},
                timeout=60,
            )
            if r.status_code not in (200, 206):
                raise SupabaseError(f"select {table}: {r.status_code} {r.text}")
            rows = r.json()
            if not rows:
                return
            yield from rows
            if len(rows) < page_size:
                return
            offset += page_size

    def update(self, table: str, match: dict[str, str], payload: dict[str, Any]) -> None:
        r = self.session.patch(
            f"{self.url}/rest/v1/{table}",
            params=match,
            json=payload,
            headers={"Content-Type": "application/json", "Prefer": "return=minimal"},
            timeout=60,
        )
        if r.status_code not in (200, 204):
            raise SupabaseError(f"update {table}: {r.status_code} {r.text}")

    def insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        r = self.session.post(
            f"{self.url}/rest/v1/{table}",
            json=rows,
            headers={"Content-Type": "application/json", "Prefer": "return=minimal"},
            timeout=60,
        )
        if r.status_code not in (200, 201, 204):
            raise SupabaseError(f"insert {table}: {r.status_code} {r.text}")

    # --------------------------------------------------------------- Storage
    def download(self, bucket: str, path: str) -> bytes:
        r = self.session.get(
            f"{self.url}/storage/v1/object/{bucket}/{path}", timeout=120)
        if r.status_code != 200:
            raise SupabaseError(f"download {bucket}/{path}: {r.status_code} {r.text}")
        return r.content
