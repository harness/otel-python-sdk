"""Defense-in-depth span processor that scrubs GenAI payload attributes.

Instrumentation wrappers in this SDK gate payload-bearing attributes on
``gen_ai.payload_capture_enabled`` (see ``instrumentation/genai_env.py``,
``instrumentation/litellm/``, ``instrumentation/mcp/``), but third-party OTel
contrib instrumentations we do not wrap may not consult Harness config at all.
This processor is the last line of defense before export: when capture is
disabled it strips any payload-bearing GenAI/Traceloop attribute from every
span, regardless of which instrumentation set it.
"""
from opentelemetry.sdk.trace import SpanProcessor

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

_SCRUBBED_ATTRIBUTES = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instruction",
    "traceloop.entity.input",
    "traceloop.entity.output",
})
_SCRUBBED_PREFIXES = ("gen_ai.prompt", "gen_ai.completion")


def _is_payload_attribute(key: str) -> bool:
    if key in _SCRUBBED_ATTRIBUTES:
        return True
    return any(key.startswith(prefix) for prefix in _SCRUBBED_PREFIXES)


class GenAiPayloadScrubSpanProcessor(SpanProcessor):
    """Strips GenAI payload attributes from spans when capture is disabled.

    Cheap no-op passthrough when capture is enabled. Mutates the ended span's
    underlying attribute store directly: ``ReadableSpan.attributes`` is a
    read-only view (``MappingProxyType``) over the same dict the concrete SDK
    ``Span`` still owns at ``on_end`` time, so deleting keys from it here is
    reflected in whatever export path runs downstream.
    """

    def __init__(self, processor):
        self._processor = processor

    def on_start(self, span, parent_context=None):
        self._processor.on_start(span, parent_context)

    def on_end(self, span):
        if Config().config.gen_ai.payload_capture_enabled.value:
            self._processor.on_end(span)
            return
        self._scrub(span)
        self._processor.on_end(span)

    @staticmethod
    def _scrub(span) -> None:
        attributes = getattr(span, "_attributes", None)
        if not attributes:
            return
        scrubbed_keys = [key for key in attributes if _is_payload_attribute(key)]
        for key in scrubbed_keys:
            del attributes[key]
        if scrubbed_keys:
            logger.debug(
                "GenAI: scrubbed payload attributes from span %s: %s",
                span.name,
                scrubbed_keys,
            )

    def force_flush(self, timeout_millis=30000):
        return self._processor.force_flush(timeout_millis)

    def shutdown(self):
        return self._processor.shutdown()
