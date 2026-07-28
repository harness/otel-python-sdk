"""Unit tests for BaseInstrumentorWrapper sensitive-header redaction.

Covers the choke point (`add_headers_to_span`) shared by the HTTP
(requests/httpx/aiohttp) and RPC (gRPC) handlers, so credential-bearing
headers never reach span attributes verbatim.
"""

from harness_sdk.instrumentation import BaseInstrumentorWrapper


class _FakeSpan:
    """Minimal Span stand-in that records set_attribute calls."""

    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_sensitive_headers_are_redacted(agent):  # noqa: ARG001 (agent inits Config)
    wrapper = BaseInstrumentorWrapper()
    span = _FakeSpan()

    headers = {
        "authorization": "Bearer sk-secret-openai-key",
        "x-api-key": "sk-ant-secret-anthropic-key",
        "cookie": "session=abc123",
        "x-harness-service-token": "pat.acct.secret",
        "content-type": "application/json",
        "accept": "*/*",
    }

    wrapper.add_headers_to_span("http.request.header.", span, headers)

    # Sensitive headers must be masked.
    assert span.attributes["http.request.header.authorization"] == BaseInstrumentorWrapper.REDACTED_VALUE
    assert span.attributes["http.request.header.x-api-key"] == BaseInstrumentorWrapper.REDACTED_VALUE
    assert span.attributes["http.request.header.cookie"] == BaseInstrumentorWrapper.REDACTED_VALUE
    assert span.attributes["http.request.header.x-harness-service-token"] == BaseInstrumentorWrapper.REDACTED_VALUE

    # Benign headers must pass through untouched.
    assert span.attributes["http.request.header.content-type"] == "application/json"
    assert span.attributes["http.request.header.accept"] == "*/*"

    # No raw secret value should survive anywhere on the span.
    assert "sk-secret-openai-key" not in span.attributes.values()
    assert "sk-ant-secret-anthropic-key" not in span.attributes.values()


def test_redaction_is_case_insensitive(agent):  # noqa: ARG001
    wrapper = BaseInstrumentorWrapper()
    span = _FakeSpan()

    wrapper.add_headers_to_span(
        "http.request.header.", span, {"Authorization": "Bearer sk-secret"}
    )

    assert span.attributes["http.request.header.Authorization"] == BaseInstrumentorWrapper.REDACTED_VALUE


def test_redact_helper_returns_value_for_benign_header(agent):  # noqa: ARG001
    wrapper = BaseInstrumentorWrapper()
    assert wrapper._redact_if_sensitive("accept", "*/*") == "*/*"
    assert wrapper._redact_if_sensitive("authorization", "Bearer x") == BaseInstrumentorWrapper.REDACTED_VALUE
