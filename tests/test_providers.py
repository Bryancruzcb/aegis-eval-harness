"""Tests for retry-eligibility classification of provider errors."""
from providers import is_retryable_error


class FakeHTTPError(Exception):
    """Stand-in for an SDK error carrying an HTTP status code."""
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def test_rate_limit_is_retryable():
    assert is_retryable_error(FakeHTTPError(429)) is True


def test_server_error_is_retryable():
    assert is_retryable_error(FakeHTTPError(503)) is True
    assert is_retryable_error(FakeHTTPError(500)) is True


def test_client_error_is_not_retryable():
    # A 400/401/404 is a bug or bad key — retrying just burns quota.
    assert is_retryable_error(FakeHTTPError(400)) is False
    assert is_retryable_error(FakeHTTPError(404)) is False


def test_value_error_is_not_retryable():
    assert is_retryable_error(ValueError("missing key")) is False


def test_timeout_message_is_retryable():
    assert is_retryable_error(Exception("Deadline exceeded: request timed out")) is True
