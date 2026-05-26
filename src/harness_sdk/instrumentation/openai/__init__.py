"""
OpenAI SDK instrumentation using ``opentelemetry-util-genai`` TelemetryHandler
for all span/metric/event telemetry, plus Traceable pre-call policy evaluation.

Coverage:
  - ChatCompletions.create / AsyncCompletions.create (non-streaming and stream=True)
  - Embeddings.create / AsyncEmbeddings.create

The OTEL contrib ``openai_v2`` package provides ``TelemetryHandler``-based
wrappers for chat completions (including streaming) and the ``EmbeddingInvocation``
type for embeddings.

Optional: ``pip install harness-sdk[openai]``
"""
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any, Callable

import wrapt
from opentelemetry.instrumentation.openai_v2.patch import (
    ChatStreamWrapper,
    _set_response_properties,
)
from opentelemetry.instrumentation.openai_v2.utils import (
    create_chat_invocation,
    get_server_address_and_port,
    is_streaming,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ContentCapturingMode, Error, LLMInvocation
from opentelemetry.util.genai.utils import get_content_capturing_mode, is_experimental_mode

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.plugins.control import get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation import BaseInstrumentorWrapper

logger = get_custom_logger(__name__)

_OPENAI_COMPLETIONS_MODULE = "openai.resources.chat.completions"
_OPENAI_EMBEDDINGS_MODULE = "openai.resources.embeddings"


def _get_handler() -> TelemetryHandler:
    return TelemetryHandler()


def _evaluate_invocation(invocation: Any) -> None:
    """Run Traceable policy evaluation against the live span; raise if blocked.

    Works for both LLMInvocation (span at ._inference_invocation.span) and
    EmbeddingInvocation (span at .span directly on the GenAIInvocation base).
    """
    if not Config().config.gen_ai.payload_evaluation_enabled.value:
        logger.debug("OpenAI: evaluate_body disabled, skipping policy evaluation")
        return
    span = getattr(invocation, "span", None)
    if span is None:
        inf = getattr(invocation, "_inference_invocation", None)
        if inf is not None:
            span = getattr(inf, "span", None)
    if span is None or not span.is_recording():
        logger.debug("OpenAI: no active span, skipping evaluation")
        return
    try:
        logger.debug("OpenAI: running policy evaluation on span %s", span.get_span_context().span_id)
        result = get_control_registry().evaluate_agent_span(span, body="")
        if result.block:
            logger.debug("Traceable policy blocked OpenAI request")
            raise ControlEvaluationBlocked(result)
        logger.debug("OpenAI: policy evaluation passed")
    except ControlEvaluationBlocked:
        raise
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("OpenAI span evaluation error: %s", err)


def _make_chat_create_sync(handler: TelemetryHandler, capture_content: bool) -> Callable[..., Any]:
    logger.debug("OpenAI Completions.create (sync): capture_content=%s", capture_content)

    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            logger.debug("OpenAI Completions.create (sync): gen_ai disabled, passthrough")
            return wrapped(*args, **kwargs)

        invocation = handler.start_llm(
            create_chat_invocation(kwargs, instance, capture_content=capture_content)
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = wrapped(*args, **kwargs)
            parsed_result = result.parse() if hasattr(result, "parse") else result
            if is_streaming(kwargs):
                logger.debug("OpenAI Completions.create (sync): returning stream wrapper")
                invocation.attributes["gen_ai.request.streaming"] = True
                return ChatStreamWrapper(parsed_result, handler, invocation, capture_content)
            _set_response_properties(invocation, parsed_result, capture_content)
            handler.stop_llm(invocation)
            logger.debug("OpenAI Completions.create (sync): complete")
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("OpenAI Completions.create (sync): exception=%s", exc)
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_chat_create_async(handler: TelemetryHandler, capture_content: bool) -> Callable[..., Any]:
    logger.debug("OpenAI AsyncCompletions.create (async): capture_content=%s", capture_content)

    async def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            logger.debug("OpenAI AsyncCompletions.create (async): gen_ai disabled, passthrough")
            return await wrapped(*args, **kwargs)

        invocation = handler.start_llm(
            create_chat_invocation(kwargs, instance, capture_content=capture_content)
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = await wrapped(*args, **kwargs)
            parsed_result = result.parse() if hasattr(result, "parse") else result
            if is_streaming(kwargs):
                logger.debug("OpenAI AsyncCompletions.create (async): returning stream wrapper")
                invocation.attributes["gen_ai.request.streaming"] = True
                return ChatStreamWrapper(parsed_result, handler, invocation, capture_content)
            _set_response_properties(invocation, parsed_result, capture_content)
            handler.stop_llm(invocation)
            logger.debug("OpenAI AsyncCompletions.create (async): complete")
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("OpenAI AsyncCompletions.create (async): exception=%s", exc)
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_embeddings_create_sync(handler: TelemetryHandler) -> Callable[..., Any]:
    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            return wrapped(*args, **kwargs)

        address, port = get_server_address_and_port(instance)
        invocation = handler.start_embedding(
            "openai",
            request_model=kwargs.get("model"),
            server_address=address,
            server_port=port,
        )
        enc = kwargs.get("encoding_format")
        if enc is not None:
            invocation.encoding_formats = [enc]
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            invocation.fail(Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = wrapped(*args, **kwargs)
            invocation.response_model_name = getattr(result, "model", None)
            usage = getattr(result, "usage", None)
            if usage is not None:
                invocation.input_tokens = getattr(usage, "prompt_tokens", None)
            data = getattr(result, "data", None)
            if data:
                emb = getattr(data[0], "embedding", None)
                if isinstance(emb, list):
                    invocation.dimension_count = len(emb)
            invocation.stop()
            logger.debug("OpenAI Embeddings.create (sync): complete")
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("OpenAI Embeddings.create (sync): exception=%s", exc)
            invocation.fail(Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_embeddings_create_async(handler: TelemetryHandler) -> Callable[..., Any]:
    async def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            return await wrapped(*args, **kwargs)

        address, port = get_server_address_and_port(instance)
        invocation = handler.start_embedding(
            "openai",
            request_model=kwargs.get("model"),
            server_address=address,
            server_port=port,
        )
        enc = kwargs.get("encoding_format")
        if enc is not None:
            invocation.encoding_formats = [enc]
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            invocation.fail(Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = await wrapped(*args, **kwargs)
            invocation.response_model_name = getattr(result, "model", None)
            usage = getattr(result, "usage", None)
            if usage is not None:
                invocation.input_tokens = getattr(usage, "prompt_tokens", None)
            data = getattr(result, "data", None)
            if data:
                emb = getattr(data[0], "embedding", None)
                if isinstance(emb, list):
                    invocation.dimension_count = len(emb)
            invocation.stop()
            logger.debug("OpenAI AsyncEmbeddings.create (async): complete")
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("OpenAI AsyncEmbeddings.create (async): exception=%s", exc)
            invocation.fail(Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


class OpenAIInstrumentorWrapper(BaseInstrumentorWrapper):
    """Instrument OpenAI chat completions and embeddings (sync + async) with OTEL contrib telemetry and Traceable policy evaluation."""  # pylint: disable=line-too-long

    _applied: bool = False

    def __init__(self) -> None:
        BaseInstrumentorWrapper.__init__(self)

    def instrument(self, **_kwargs: Any) -> None:
        if self._applied:
            logger.debug("OpenAI instrumentation already applied.")
            return
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Gen AI instrumentation disabled; skip OpenAI wraps.")
            return
        try:
            handler = _get_handler()
            latest_experimental = is_experimental_mode()
            content_mode = (
                get_content_capturing_mode()
                if latest_experimental
                else ContentCapturingMode.NO_CONTENT
            )
            capture_content = content_mode != ContentCapturingMode.NO_CONTENT
            wrapt.wrap_function_wrapper(
                _OPENAI_COMPLETIONS_MODULE,
                "Completions.create",
                _make_chat_create_sync(handler, capture_content),
            )
            wrapt.wrap_function_wrapper(
                _OPENAI_COMPLETIONS_MODULE,
                "AsyncCompletions.create",
                _make_chat_create_async(handler, capture_content),
            )
            wrapt.wrap_function_wrapper(
                _OPENAI_EMBEDDINGS_MODULE,
                "Embeddings.create",
                _make_embeddings_create_sync(handler),
            )
            wrapt.wrap_function_wrapper(
                _OPENAI_EMBEDDINGS_MODULE,
                "AsyncEmbeddings.create",
                _make_embeddings_create_async(handler),
            )
            OpenAIInstrumentorWrapper._applied = True
            logger.debug("Traceable OpenAI instrumentation applied.")
        except ImportError as err:
            logger.error("OpenAI SDK not available: %s", err)
            logger.error("Install with: pip install 'harness-sdk[openai]'")
            raise

    def uninstrument(self, **_kwargs: Any) -> None:
        if not self._applied:
            return
        from importlib import import_module  # pylint: disable=import-outside-toplevel

        errors: list[Exception] = []
        for module_path, cls_name, method in [
            (_OPENAI_COMPLETIONS_MODULE, "Completions", "create"),
            (_OPENAI_COMPLETIONS_MODULE, "AsyncCompletions", "create"),
            (_OPENAI_EMBEDDINGS_MODULE, "Embeddings", "create"),
            (_OPENAI_EMBEDDINGS_MODULE, "AsyncEmbeddings", "create"),
        ]:
            try:
                mod = import_module(module_path)
                unwrap(getattr(mod, cls_name), method)
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to uninstrument OpenAI %s.%s: %s", cls_name, method, err)
                errors.append(err)

        OpenAIInstrumentorWrapper._applied = False
        logger.debug("OpenAI instrumentation removed.")
        if errors:
            raise errors[0]


__all__ = ["OpenAIInstrumentorWrapper"]
