"""
Anthropic SDK instrumentation using ``opentelemetry-util-genai`` TelemetryHandler
for all span/metric/event telemetry, plus Traceable pre-call policy evaluation.

Coverage:
  - Messages.create / AsyncMessages.create  (non-streaming and stream=True)
  - Messages.stream  / AsyncMessages.stream  (MessageStream API)

The OTEL contrib ``AnthropicInstrumentor.instrument()`` only handles non-streaming
``Messages.create``; we call its inner ``messages_create(handler)`` wrapper directly
and add policy evaluation and streaming support ourselves.

Optional: ``pip install agent-trace-sdk[anthropic]``
"""

from __future__ import annotations

from typing import Any, Callable

import wrapt
from opentelemetry.instrumentation.anthropic.messages_extractors import (
    extract_params,
    get_input_messages,
    get_llm_request_attributes,
    get_system_instruction,
    set_invocation_response_attributes,
)
from opentelemetry.instrumentation.anthropic.wrappers import (
    AsyncMessagesStreamWrapper as _AsyncMessagesStreamWrapper,
    MessagesStreamManagerWrapper as _MessagesStreamManagerWrapper,
    AsyncMessagesStreamManagerWrapper as _AsyncMessagesStreamManagerWrapper,
    MessagesStreamWrapper as _MessagesStreamWrapper,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, LLMInvocation
from opentelemetry.util.genai.utils import should_capture_content_on_spans_in_experimental_mode

from agent_trace.config.config import Config
from agent_trace.custom_logger import get_custom_logger
from agent_trace.plugins.control import get_control_registry
from agent_trace.gen_ai.exceptions import ControlEvaluationBlocked
from agent_trace.instrumentation import BaseInstrumentorWrapper

logger = get_custom_logger(__name__)

_ANTHROPIC_MESSAGES_MODULE = "anthropic.resources.messages"
_ANTHROPIC = "anthropic"


def _get_handler() -> TelemetryHandler:
    return TelemetryHandler()


def _evaluate_invocation(invocation: LLMInvocation) -> None:
    """Run Traceable policy evaluation against the live span; raise if blocked."""
    if not Config().config.gen_ai.payload_evaluation_enabled.value:
        logger.debug("Anthropic: evaluate_body disabled, skipping policy evaluation")
        return
    inf = getattr(invocation, "_inference_invocation", None)
    if inf is None:
        logger.debug("Anthropic: no _inference_invocation on invocation, skipping evaluation")
        return
    span = getattr(inf, "span", None)
    if span is None or not span.is_recording():
        logger.debug("Anthropic: no active span, skipping evaluation")
        return
    try:
        logger.debug("Anthropic: running policy evaluation on span %s", span.get_span_context().span_id)
        result = get_control_registry().evaluate_agent_span(span, body="")
        if result.block:
            logger.debug("Traceable policy blocked Anthropic request")
            raise ControlEvaluationBlocked(result)
        logger.debug("Anthropic: policy evaluation passed")
    except ControlEvaluationBlocked:
        raise
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("Anthropic span evaluation error: %s", err)


def _build_invocation(
    handler: TelemetryHandler,
    params: Any,
    instance: Any,
    capture_content: bool,
) -> LLMInvocation:
    attributes = get_llm_request_attributes(params, instance)
    model_attr = attributes.get("gen_ai.request.model")
    request_model = model_attr if isinstance(model_attr, str) else params.model
    invocation = LLMInvocation(
        request_model=request_model,
        provider=_ANTHROPIC,
        input_messages=get_input_messages(params.messages) if capture_content else [],
        system_instruction=get_system_instruction(params.system) if capture_content else [],
        attributes=attributes,
    )
    handler.start_llm(invocation)
    return invocation


def _make_create_sync(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()
    logger.debug("Anthropic Messages.create (sync): capture_content=%s", capture_content)

    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Anthropic Messages.create (sync): gen_ai disabled, passthrough")
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream") is True
        logger.debug("Anthropic Messages.create (sync): streaming=%s", is_streaming)
        params = extract_params(**kwargs)
        invocation = _build_invocation(handler, params, instance, capture_content)
        if is_streaming:
            invocation.attributes["gen_ai.request.streaming"] = True
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = wrapped(*args, **kwargs)
            if is_streaming:
                logger.debug("Anthropic Messages.create (sync): returning stream wrapper")
                return _MessagesStreamWrapper(result, handler, invocation, capture_content)
            set_invocation_response_attributes(invocation, result, capture_content)
            handler.stop_llm(invocation)
            logger.debug("Anthropic Messages.create (sync): complete")
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Anthropic Messages.create (sync): exception=%s", exc)
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_create_async(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()
    logger.debug("Anthropic Messages.create (async): capture_content=%s", capture_content)

    async def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Anthropic Messages.create (async): gen_ai disabled, passthrough")
            return await wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream") is True
        logger.debug("Anthropic Messages.create (async): streaming=%s", is_streaming)
        params = extract_params(**kwargs)
        invocation = _build_invocation(handler, params, instance, capture_content)
        if is_streaming:
            invocation.attributes["gen_ai.request.streaming"] = True
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = await wrapped(*args, **kwargs)
            if is_streaming:
                logger.debug("Anthropic Messages.create (async): returning stream wrapper")
                return _AsyncMessagesStreamWrapper(result, handler, invocation, capture_content)
            set_invocation_response_attributes(invocation, result, capture_content)
            handler.stop_llm(invocation)
            logger.debug("Anthropic Messages.create (async): complete")
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Anthropic Messages.create (async): exception=%s", exc)
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


class _TraceableMessagesStreamWrapper(_MessagesStreamWrapper):  # type: ignore[misc]
    """Extends the vendored wrapper to capture responses when the user iterates
    via text_stream or other SDK shortcuts that bypass __next__."""

    def _stop(self) -> None:
        if self._finalized:
            return
        if self._message is None:
            try:
                self._message = self.stream.current_message_snapshot
            except Exception:  # pylint: disable=broad-except
                pass
        super()._stop()


class _TraceableAsyncMessagesStreamWrapper(_AsyncMessagesStreamWrapper):  # type: ignore[misc]
    """Async counterpart of _TraceableMessagesStreamWrapper."""

    def _stop(self) -> None:
        if self._finalized:
            return
        if self._message is None:
            try:
                self._message = self.stream.current_message_snapshot
            except Exception:  # pylint: disable=broad-except
                pass
        super()._stop()


class MessagesStreamManagerWrapper(_MessagesStreamManagerWrapper):  # type: ignore[misc]  # pylint: disable=too-few-public-methods
    """Sync stream manager that injects our stream wrapper subclass."""

    def __enter__(self):  # type: ignore[override]
        stream = self._manager.__enter__()
        self._stream_wrapper = _TraceableMessagesStreamWrapper(
            stream, self._handler, self._invocation, self._capture_content
        )
        return self._stream_wrapper


class AsyncMessagesStreamManagerWrapper(_AsyncMessagesStreamManagerWrapper):  # type: ignore[misc]  # pylint: disable=too-few-public-methods
    """Async stream manager that injects our stream wrapper subclass."""

    async def __aenter__(self):  # type: ignore[override]
        msg_stream = await self._manager.__aenter__()
        self._stream_wrapper = _TraceableAsyncMessagesStreamWrapper(
            msg_stream, self._handler, self._invocation, self._capture_content
        )
        return self._stream_wrapper



def _make_stream_sync(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()
    logger.debug("Anthropic Messages.stream (sync): capture_content=%s", capture_content)

    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Anthropic Messages.stream (sync): gen_ai disabled, passthrough")
            return wrapped(*args, **kwargs)

        merged_kwargs = dict(kwargs)
        merged_kwargs["stream"] = True
        params = extract_params(**merged_kwargs)
        invocation = _build_invocation(handler, params, instance, capture_content)
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            manager = wrapped(*args, **kwargs)
            logger.debug("Anthropic Messages.stream (sync): returning stream manager wrapper")
            return MessagesStreamManagerWrapper(manager, handler, invocation, capture_content)
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Anthropic Messages.stream (sync): exception=%s", exc)
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_stream_async(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()
    logger.debug("Anthropic Messages.stream (async): capture_content=%s", capture_content)

    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Anthropic Messages.stream (async): gen_ai disabled, passthrough")
            return wrapped(*args, **kwargs)

        merged_kwargs = dict(kwargs)
        merged_kwargs["stream"] = True
        params = extract_params(**merged_kwargs)
        invocation = _build_invocation(handler, params, instance, capture_content)
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            manager = wrapped(*args, **kwargs)
            logger.debug("Anthropic Messages.stream (async): returning stream manager wrapper")
            return AsyncMessagesStreamManagerWrapper(manager, handler, invocation, capture_content)
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Anthropic Messages.stream (async): exception=%s", exc)
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


class AnthropicInstrumentorWrapper(BaseInstrumentorWrapper):
    """Instrument Anthropic Messages API with OTEL contrib telemetry and Traceable policy evaluation."""

    _applied: bool = False

    def __init__(self) -> None:
        BaseInstrumentorWrapper.__init__(self)

    def instrument(self, **_kwargs: Any) -> None:
        if self._applied:
            logger.debug("Anthropic instrumentation already applied.")
            return
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Gen AI instrumentation disabled; skip Anthropic wraps.")
            return
        try:
            handler = _get_handler()
            wrapt.wrap_function_wrapper(
                _ANTHROPIC_MESSAGES_MODULE, "Messages.create", _make_create_sync(handler)
            )
            wrapt.wrap_function_wrapper(
                _ANTHROPIC_MESSAGES_MODULE, "AsyncMessages.create", _make_create_async(handler)
            )
            wrapt.wrap_function_wrapper(
                _ANTHROPIC_MESSAGES_MODULE, "Messages.stream", _make_stream_sync(handler)
            )
            wrapt.wrap_function_wrapper(
                _ANTHROPIC_MESSAGES_MODULE, "AsyncMessages.stream", _make_stream_async(handler)
            )
            AnthropicInstrumentorWrapper._applied = True
            logger.debug("Traceable Anthropic instrumentation applied.")
        except ImportError as err:
            logger.error("Anthropic SDK not available: %s", err)
            raise

    def uninstrument(self, **_kwargs: Any) -> None:
        if not self._applied:
            return
        from importlib import import_module  # pylint: disable=import-outside-toplevel
        try:
            mod = import_module(_ANTHROPIC_MESSAGES_MODULE)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Anthropic SDK not available: %s", err)
            raise

        errors: list[Exception] = []
        for cls_name, method in [
            ("Messages", "create"),
            ("AsyncMessages", "create"),
            ("Messages", "stream"),
            ("AsyncMessages", "stream"),
        ]:
            try:
                unwrap(getattr(mod, cls_name), method)
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to uninstrument Anthropic %s.%s: %s", cls_name, method, err)
                errors.append(err)

        AnthropicInstrumentorWrapper._applied = False
        logger.debug("Anthropic instrumentation removed.")
        if errors:
            raise errors[0]


__all__ = ["AnthropicInstrumentorWrapper"]
