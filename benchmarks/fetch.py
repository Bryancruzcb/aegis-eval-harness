"""Fetch a benchmark CSV once, verify it, and cache it.

Downloaded benchmark data never enters the repository — the cache directory is
git-ignored. The content hash is verified on download AND on every cache read,
so a truncated download or a half-written file fails loudly instead of silently
poisoning every later run.

The hash is taken over raw bytes and the bytes are written to disk untouched, so
the cached file is byte-for-byte the file that was verified: an auditor running
`sha256sum` on the cache gets the pinned constant back. Decoding happens only
after verification, which also means a torn download fails as a hash mismatch
rather than as a bare UnicodeDecodeError.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

_urlopen = urllib.request.urlopen       # module-level seam so tests can inject


class BenchmarkFetchError(Exception):
    """A benchmark file could not be fetched, or failed verification."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode(data: bytes, *, source: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise BenchmarkFetchError(f"{source} is not valid UTF-8: {e}") from e


def fetch_csv(url: str, *, sha256: str, cache_dir: Path, refresh: bool = False) -> str:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / url.rsplit("/", 1)[-1]

    if cached.exists() and not refresh:
        data = cached.read_bytes()      # bytes, not text: no newline translation
        if _digest(data) != sha256:
            raise BenchmarkFetchError(
                f"Cached file {cached} (from {url}) failed sha256 verification "
                f"(expected {sha256}). Re-run with --refresh-benchmark.")
        return _decode(data, source=f"Cached file {cached} (from {url})")

    try:
        with _urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        raise BenchmarkFetchError(f"Failed to download {url}: {e}") from e

    actual = _digest(data)
    if actual != sha256:
        raise BenchmarkFetchError(
            f"Download of {url} failed sha256 verification: expected {sha256}, got {actual}.")

    tmp = cached.with_suffix(cached.suffix + ".part")
    tmp.write_bytes(data)               # exactly the bytes that were verified
    os.replace(tmp, cached)             # atomic: a crash never leaves a half file
    return _decode(data, source=f"Download of {url}")
