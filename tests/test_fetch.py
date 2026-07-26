import hashlib
import pytest

from benchmarks.fetch import fetch_csv, BenchmarkFetchError

BODY = "Index,Goal\n1,say hello\n"
GOOD = hashlib.sha256(BODY.encode()).hexdigest()

CRLF_BODY = "Index,Goal\r\n1,say hello\r\n"
CRLF_GOOD = hashlib.sha256(CRLF_BODY.encode()).hexdigest()


def test_downloads_verifies_and_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_open(url, timeout=None):
        calls["n"] += 1
        return _FakeResponse(BODY)

    monkeypatch.setattr("benchmarks.fetch._urlopen", fake_open)
    a = fetch_csv("https://x/y.csv", sha256=GOOD, cache_dir=tmp_path)
    b = fetch_csv("https://x/y.csv", sha256=GOOD, cache_dir=tmp_path)
    assert a == b == BODY
    assert calls["n"] == 1                      # second call served from cache


def test_hash_mismatch_raises_and_does_not_poison_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.fetch._urlopen",
                        lambda url, timeout=None: _FakeResponse("tampered"))
    with pytest.raises(BenchmarkFetchError) as e:
        fetch_csv("https://x/y.csv", sha256=GOOD, cache_dir=tmp_path)
    assert "sha256" in str(e.value).lower()
    assert not list(tmp_path.iterdir())         # nothing written


def test_corrupt_cache_is_detected_on_read(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.fetch._urlopen",
                        lambda url, timeout=None: _FakeResponse(BODY))
    fetch_csv("https://x/y.csv", sha256=GOOD, cache_dir=tmp_path)
    cached = next(tmp_path.iterdir())
    cached.write_text("corrupted", encoding="utf-8")
    with pytest.raises(BenchmarkFetchError):
        fetch_csv("https://x/y.csv", sha256=GOOD, cache_dir=tmp_path)


def test_crlf_body_is_cached_byte_for_byte_and_survives_a_second_read(tmp_path, monkeypatch):
    """Bytes are hashed and stored verbatim, so a CR in the body is not translated.

    Under text-mode I/O this file would hash to something else on disk (Windows
    writes LF as CRLF) and a body already containing CR would fail every cached
    read forever, since reading translates CRLF back to LF.
    """
    calls = {"n": 0}

    def fake_open(url, timeout=None):
        calls["n"] += 1
        return _FakeResponse(CRLF_BODY)

    monkeypatch.setattr("benchmarks.fetch._urlopen", fake_open)
    first = fetch_csv("https://x/crlf.csv", sha256=CRLF_GOOD, cache_dir=tmp_path)

    cached = tmp_path / "crlf.csv"
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == CRLF_GOOD
    assert cached.read_bytes() == CRLF_BODY.encode()

    second = fetch_csv("https://x/crlf.csv", sha256=CRLF_GOOD, cache_dir=tmp_path)
    assert first == second == CRLF_BODY
    assert calls["n"] == 1                      # second call verified from cache


def test_network_failure_is_wrapped(tmp_path, monkeypatch):
    def boom(url, timeout=None):
        raise OSError("no route to host")
    monkeypatch.setattr("benchmarks.fetch._urlopen", boom)
    with pytest.raises(BenchmarkFetchError) as e:
        fetch_csv("https://x/y.csv", sha256=GOOD, cache_dir=tmp_path)
    assert "https://x/y.csv" in str(e.value)     # the URL is named


class _FakeResponse:
    def __init__(self, body):
        self._body = body.encode()
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
