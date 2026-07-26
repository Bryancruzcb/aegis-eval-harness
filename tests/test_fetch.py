import hashlib
import pytest

from benchmarks.fetch import fetch_csv, BenchmarkFetchError

BODY = "Index,Goal\n1,say hello\n"
GOOD = hashlib.sha256(BODY.encode()).hexdigest()


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
