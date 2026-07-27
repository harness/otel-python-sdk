import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider

from harness_sdk.gen_ai_payload_scrub_span_processor import GenAiPayloadScrubSpanProcessor


class TestGenAiPayloadScrubSpanProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_processor = MagicMock()
        self.processor = GenAiPayloadScrubSpanProcessor(processor=self.mock_processor)
        self.tracer_provider = TracerProvider()
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def create_test_span(self, attributes=None):
        """Helper method to create a test span with the given attributes."""
        with self.tracer.start_as_current_span("test-span") as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span

    def _mock_config(self, payload_capture_enabled: bool):
        gen = MagicMock()
        gen.payload_capture_enabled.value = payload_capture_enabled
        cfg = MagicMock()
        cfg.gen_ai = gen
        root = MagicMock()
        root.config = cfg
        return root

    def test_on_start_delegates_to_processor(self):
        span = self.create_test_span()
        parent_context = MagicMock()

        self.processor.on_start(span, parent_context)

        self.mock_processor.on_start.assert_called_once_with(span, parent_context)

    def test_capture_enabled_is_noop_passthrough(self):
        span = self.create_test_span({
            "gen_ai.input.messages": "[secret prompt]",
            "keep.me": "value",
        })

        with patch(
            "harness_sdk.gen_ai_payload_scrub_span_processor.Config",
            return_value=self._mock_config(payload_capture_enabled=True),
        ):
            self.processor.on_end(span)

        self.mock_processor.on_end.assert_called_once_with(span)
        assert span.attributes.get("gen_ai.input.messages") == "[secret prompt]"

    def test_capture_disabled_strips_known_payload_attributes(self):
        span = self.create_test_span({
            "gen_ai.input.messages": "[secret prompt]",
            "gen_ai.output.messages": "[secret response]",
            "gen_ai.system_instruction": "system prompt",
            "gen_ai.prompt.0.content": "hi",
            "gen_ai.completion.0.content": "hello",
            "traceloop.entity.input": "input payload",
            "traceloop.entity.output": "output payload",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 3,
        })

        with patch(
            "harness_sdk.gen_ai_payload_scrub_span_processor.Config",
            return_value=self._mock_config(payload_capture_enabled=False),
        ):
            self.processor.on_end(span)

        attrs = span.attributes
        for key in (
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.system_instruction",
            "gen_ai.prompt.0.content",
            "gen_ai.completion.0.content",
            "traceloop.entity.input",
            "traceloop.entity.output",
        ):
            assert key not in attrs, f"{key} should have been scrubbed"

        # Non-payload metadata attributes must survive the scrub.
        assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
        assert attrs.get("gen_ai.usage.input_tokens") == 3

        self.mock_processor.on_end.assert_called_once_with(span)

    def test_capture_disabled_on_span_without_payload_attributes_is_harmless(self):
        span = self.create_test_span({"gen_ai.request.model": "gpt-4o-mini"})

        with patch(
            "harness_sdk.gen_ai_payload_scrub_span_processor.Config",
            return_value=self._mock_config(payload_capture_enabled=False),
        ):
            self.processor.on_end(span)

        assert span.attributes.get("gen_ai.request.model") == "gpt-4o-mini"
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_force_flush_delegates_to_processor(self):
        self.processor.force_flush(5000)
        self.mock_processor.force_flush.assert_called_once_with(5000)

    def test_shutdown_delegates_to_processor(self):
        self.processor.shutdown()
        self.mock_processor.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
