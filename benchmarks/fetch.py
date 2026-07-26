"""Fetch a benchmark CSV once, verify it, and cache it.

Downloaded benchmark data never enters the repository — the cache directory is
git-ignored. The content hash is verified on download AND on every cache read,
so a truncated download or a half-written file fails loudly instead of silently
poisoning every later run.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

_urlopen = urllib.request.urlopen       # module-level seam so tests can inject


class BenchmarkFetchError(Exception):
    """A benchmark file could not be fetched, or failed verification."""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_csv(url: str, *, sha256: str, cache_dir: Path, refresh: bool = False) -> str:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / url.rsplit("/", 1)[-1]

    if cached.exists() and not refresh:
        text = cached.read_text(encoding="utf-8")
        if _digest(text) != sha256:
            raise BenchmarkFetchError(
                f"Cached file {cached} (from {url}) failed sha256 verification "
                f"(expected {sha256}). Re-run with --refresh-benchmark.")
        return text

    try:
        with _urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        raise BenchmarkFetchError(f"Failed to download {url}: {e}") from e

    actual = _digest(text)
    if actual != sha256:
        raise BenchmarkFetchError(
            f"Download of {url} failed sha256 verification: expected {sha256}, got {actual}.")

    tmp = cached.with_suffix(cached.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, cached)             # atomic: a crash never leaves a half file
    return text
