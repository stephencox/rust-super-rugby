"""HTTP caching with conditional GET support."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)


@dataclass
class CacheMeta:
    """Metadata sidecar for cached HTTP responses."""
    last_modified: Optional[str] = None
    etag: Optional[str] = None

    def save(self, path: Path):
        data = {}
        if self.last_modified:
            data["last_modified"] = self.last_modified
        if self.etag:
            data["etag"] = self.etag
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "CacheMeta":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                last_modified=data.get("last_modified"),
                etag=data.get("etag"),
            )
        except (json.JSONDecodeError, OSError):
            return cls()


class HttpCache:
    """HTTP cache with conditional GET (If-Modified-Since / If-None-Match).

    Cache path: sanitize URL (strip protocol, replace / and ? with _)
    → data/cache/{sanitized}.html
    Metadata sidecar: .meta.json with last_modified and etag.

    Produces the same filenames as the Rust cache (shared data/cache/ directory).
    """

    def __init__(self, cache_dir: Path, offline: bool = False):
        self.cache_dir = cache_dir
        self.offline = offline
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "rugby-predictor/0.1"

    def _cache_path(self, url: str) -> Path:
        """Sanitize URL to a cache filename matching the Rust implementation."""
        filename = (
            url.replace("https://", "")
            .replace("http://", "")
            .replace("/", "_")
            .replace("?", "_")
            + ".html"
        )
        return self.cache_dir / filename

    def _meta_path(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(".meta.json")

    def fetch(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch URL content, using cache with conditional GET.

        Returns HTML content or None if unavailable.
        """
        cache_path = self._cache_path(url)
        meta_path = self._meta_path(cache_path)
        has_cache = cache_path.exists()

        if self.offline:
            if has_cache:
                log.debug("Offline mode: using cached %s", cache_path.name)
                return cache_path.read_text()
            log.warning("Offline mode: no cache for %s", url)
            return None

        # Build conditional request headers
        headers = {}
        meta = CacheMeta.load(meta_path) if has_cache else CacheMeta()
        if has_cache and meta.last_modified:
            headers["If-Modified-Since"] = meta.last_modified
        if has_cache and meta.etag:
            headers["If-None-Match"] = meta.etag

        try:
            resp = self.session.get(url, headers=headers, timeout=timeout)

            if resp.status_code == 304:
                log.debug("304 Not Modified: using cached %s", cache_path.name)
                return cache_path.read_text()

            resp.raise_for_status()
            html = resp.text

            # Save to cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html)

            # Save metadata
            new_meta = CacheMeta(
                last_modified=resp.headers.get("Last-Modified"),
                etag=resp.headers.get("ETag"),
            )
            new_meta.save(meta_path)

            log.debug("Fetched and cached %s", cache_path.name)
            return html

        except requests.RequestException as e:
            if has_cache:
                log.warning("Network error for %s, using stale cache: %s", url, e)
                return cache_path.read_text()
            log.error("Failed to fetch %s: %s", url, e)
            return None

    def get_cached(self, url: str) -> Optional[str]:
        """Get cached content without making a network request."""
        cache_path = self._cache_path(url)
        if cache_path.exists():
            return cache_path.read_text()
        return None

    def is_cached(self, url: str) -> bool:
        """Check if a URL is cached."""
        return self._cache_path(url).exists()
