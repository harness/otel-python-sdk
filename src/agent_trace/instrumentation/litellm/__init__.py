"""
LiteLLM instrumentation using the LiteLLM OpenTelemetry integration for span
telemetry, plus Traceable pre-call policy evaluation via libtraceable.

Coverage:
  - litellm.completion / litellm.acompletion
  - litellm.embedding / litellm.aembedding

Registers a ``TraceableLiteLLMOpenTelemetry`` callback (subclass of LiteLLM's
``OpenTelemetry``) and wraps the public entry points so evaluation runs on an
active span before the provider call. LiteLLM's OTEL callback enriches that
span on success when it is the active parent context.

Optional: ``pip install agent-trace-sdk[litellm]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import wrapt
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Status, StatusCode

from agent_trace.config.config import Config
from agent_trace.custom_logger import get_custom_logger
from agent_trace.plugins.control import get_control_registry
from agent_trace.gen_ai.exceptions import ControlEvaluationBlocked
from agent_trace.instrumentation import BaseInstrumentorWrapper

logger = get_custom_logger(__name__)

_LITELLM_MAIN = "litellm.main"
_LITELLM_REQUEST_SPAN_NAME = "litellm_request"

_WRAPPED_FUNCTIONS = (
    ("completion", False),
    ("acompletion", True),
    ("embedding", False),
    ("aembedding", True),
)


def _evaluate_span(span: Any) -> None:
    """Run Traceable policy evaluation against the live span; raise if blocked."""
    if not Config().config.gen_ai.payload_evaluation_enabled.value:
        logger.debug("LiteLLM: evaluate_body disabled, skipping policy evaluation")
        return
    if span is None or not span.is_recording():
        logger.debug("LiteLLM: no active span, skipping evaluation")
        return
    try:
        logger.debug(
            "LiteLLM: running policy evaluation on span %s",
            span.get_span_context().span_id,
        )
        result = get_control_registry().evaluate_agent_span(span, body="")
        if result.block:
            logger.debug("Traceable policy blocked LiteLLM request")
            raise ControlEvaluationBlocked(result)
        logger.debug("LiteLLM: policy evaluation passed")
    except ControlEvaluationBlocked:
        raise
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM span evaluation error: %s", err)


def _extract_model_and_input(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Optional[str], Any]:
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    payload = kwargs.get("messages")
    if payload is None:
        payload = kwargs.get("input")
    if payload is None and len(args) > 1:
        payload = args[1]
    return model, payload


def _resolve_provider(model: Optional[str], kwargs: dict[str, Any]) -> str:
    provider = kwargs.get("custom_llm_provider")
    if provider:
        return str(provider)
    litellm_params = kwargs.get("litellm_params") or {}
    provider = litellm_params.get("custom_llm_provider")
    if provider:
        return str(provider)
    if not model:
        return "Unknown"
    try:
        from litellm.litellm_core_utils.get_llm_provider_logic import (  # pylint: disable=import-outside-toplevel
            get_llm_provider,
        )

        _model, inferred, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=kwargs.get("custom_llm_provider"),
        )
        if inferred:
            return str(inferred)
        if _model:
            return str(_model)
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM: could not infer provider for model %s: %s", model, err)
    return "Unknown"


def _operation_name(func_name: str) -> str:
    if func_name in ("embedding", "aembedding"):
        return "embeddings"
    return "chat"


@dataclass(frozen=True)
class _PreCallSpanContext:
    model: Optional[str]
    payload: Any
    kwargs: dict[str, Any]
    call_type: str


def _set_pre_call_request_attributes(
    otel_logger: Any,
    span: Any,
    pre_call: _PreCallSpanContext,
) -> None:
    """Populate request-side gen_ai attributes for libtraceable evaluation."""
    provider = _resolve_provider(pre_call.model, pre_call.kwargs)
    optional_params = pre_call.kwargs.get("optional_params") or pre_call.kwargs

    otel_logger.safe_set_attribute(span, "gen_ai.request.model", pre_call.model or "")
    otel_logger.safe_set_attribute(
        span, "gen_ai.operation.name", _operation_name(pre_call.call_type)
    )
    otel_logger.safe_set_attribute(span, "gen_ai.system", provider)
    otel_logger.safe_set_attribute(span, "gen_ai.framework", "litellm")
    otel_logger.safe_set_attribute(
        span,
        "gen_ai.request.streaming",
        str(optional_params.get("stream", False)),
    )

    if optional_params.get("max_tokens") is not None:
        otel_logger.safe_set_attribute(
            span, "gen_ai.request.max_tokens", optional_params.get("max_tokens")
        )
    if optional_params.get("temperature") is not None:
        otel_logger.safe_set_attribute(
            span, "gen_ai.request.temperature", optional_params.get("temperature")
        )
    if optional_params.get("top_p") is not None:
        otel_logger.safe_set_attribute(
            span, "gen_ai.request.top_p", optional_params.get("top_p")
        )

    if pre_call.payload is None:
        return
    try:
        import litellm  # pylint: disable=import-outside-toplevel
        from litellm.litellm_core_utils.safe_json_dumps import (  # pylint: disable=import-outside-toplevel
            safe_dumps,
        )

        if litellm.turn_off_message_logging or not otel_logger.message_logging:
            return
        if pre_call.call_type in ("embedding", "aembedding"):
            otel_logger.safe_set_attribute(
                span, "gen_ai.input.messages", safe_dumps(pre_call.payload)
            )
            return
        transformed = otel_logger._transform_messages_to_otel_semantic_conventions(  # pylint: disable=protected-access
            pre_call.payload
        )
        otel_logger.safe_set_attribute(span, "gen_ai.input.messages", safe_dumps(transformed))
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM: failed to set input attributes on span: %s", err)


def _build_traceable_otel_class() -> type:
    from litellm.integrations.opentelemetry import (  # pylint: disable=import-outside-toplevel
        OpenTelemetry,
        OpenTelemetryConfig,
    )

    class TraceableLiteLLMOpenTelemetry(OpenTelemetry):  # pylint: disable=abstract-method
        """LiteLLM ``OpenTelemetry`` callback using Traceable's global tracer provider."""

        def __init__(self) -> None:
            super().__init__(config=OpenTelemetryConfig.from_env())

    return TraceableLiteLLMOpenTelemetry


_TraceableLiteLLMOpenTelemetry: Optional[type] = None
_otel_logger: Any = None


def _get_otel_logger() -> Any:
    global _otel_logger, _TraceableLiteLLMOpenTelemetry  # pylint: disable=global-statement
    if _otel_logger is None:
        if _TraceableLiteLLMOpenTelemetry is None:
            _TraceableLiteLLMOpenTelemetry = _build_traceable_otel_class()
        _otel_logger = _TraceableLiteLLMOpenTelemetry()
    return _otel_logger


def _register_otel_callback(otel_logger: Any) -> None:
    import litellm  # pylint: disable=import-outside-toplevel
    from litellm.integrations.opentelemetry import OpenTelemetry  # pylint: disable=import-outside-toplevel

    updated: list[Any] = []
    replaced = False
    for callback in litellm.callbacks:
        if callback == "otel":
            replaced = True
            continue
        if type(callback) is OpenTelemetry:  # pylint: disable=unidiomatic-typecheck
            replaced = True
            continue
        updated.append(callback)
    traceable_cls = type(otel_logger)
    if not any(isinstance(cb, traceable_cls) for cb in updated):
        updated.append(otel_logger)
    elif not replaced:
        for index, callback in enumerate(updated):
            if isinstance(callback, traceable_cls):
                updated[index] = otel_logger
                break
    litellm.callbacks = updated
    logger.debug("LiteLLM OpenTelemetry callback registered for Traceable instrumentation.")


def _unregister_otel_callback() -> None:
    import litellm  # pylint: disable=import-outside-toplevel

    if _TraceableLiteLLMOpenTelemetry is None:
        return
    litellm.callbacks = [
        callback
        for callback in litellm.callbacks
        if not isinstance(callback, _TraceableLiteLLMOpenTelemetry)
    ]


def _activate_span(span: Any) -> object:
    return otel_context.attach(trace.set_span_in_context(span))


def _deactivate_span(token: object, span: Any) -> None:
    otel_context.detach(token)
    span.end()


def _fail_pre_call_span(span: Any, exc: BaseException, *, blocked: bool = False) -> None:
    if blocked:
        span.set_status(Status(StatusCode.ERROR, "blocked"))
    else:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.end()


def _start_evaluated_span(
    otel_logger: Any,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    model, payload = _extract_model_and_input(args, kwargs)
    pre_call = _PreCallSpanContext(
        model=model, payload=payload, kwargs=kwargs, call_type=func_name
    )
    span = otel_logger.tracer.start_span(_LITELLM_REQUEST_SPAN_NAME)
    try:
        _set_pre_call_request_attributes(otel_logger, span, pre_call)
        _evaluate_span(span)
    except ControlEvaluationBlocked as exc:
        _fail_pre_call_span(span, exc, blocked=True)
        raise
    except Exception as exc:  # pylint: disable=broad-except
        _fail_pre_call_span(span, exc)
        raise
    return span


def _make_wrapper(func_name: str, is_async: bool) -> Callable[..., Any]:
    otel_logger = _get_otel_logger()

    def _sync_wrapper(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            return wrapped(*args, **kwargs)

        span = _start_evaluated_span(otel_logger, func_name, args, kwargs)
        token = _activate_span(span)
        try:
            return wrapped(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            _deactivate_span(token, span)

    async def _async_wrapper(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            return await wrapped(*args, **kwargs)

        span = _start_evaluated_span(otel_logger, func_name, args, kwargs)
        token = _activate_span(span)
        try:
            return await wrapped(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            _deactivate_span(token, span)

    return _async_wrapper if is_async else _sync_wrapper


class LiteLLMInstrumentorWrapper(BaseInstrumentorWrapper):
    """Instrument LiteLLM with its OpenTelemetry SDK and Traceable policy evaluation."""

    _applied: bool = False

    def __init__(self) -> None:
        BaseInstrumentorWrapper.__init__(self)

    def instrument(self, **_kwargs: Any) -> None:
        if self._applied:
            logger.debug("LiteLLM instrumentation already applied.")
            return
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Gen AI instrumentation disabled; skip LiteLLM wraps.")
            return
        try:
            import litellm  # pylint: disable=import-outside-toplevel

            otel_logger = _get_otel_logger()
            _register_otel_callback(otel_logger)
            main_mod = __import__(_LITELLM_MAIN, fromlist=["*"])
            for func_name, is_async in _WRAPPED_FUNCTIONS:
                wrapt.wrap_function_wrapper(
                    _LITELLM_MAIN,
                    func_name,
                    _make_wrapper(func_name, is_async),
                )
                if hasattr(litellm, func_name):
                    setattr(litellm, func_name, getattr(main_mod, func_name))
            LiteLLMInstrumentorWrapper._applied = True
            logger.debug("Traceable LiteLLM instrumentation applied.")
        except ImportError as err:
            logger.error("LiteLLM not available: %s", err)
            logger.error("Install with: pip install 'agent-trace-sdk[litellm]'")
            raise

    def uninstrument(self, **_kwargs: Any) -> None:
        if not self._applied:
            return
        from importlib import import_module  # pylint: disable=import-outside-toplevel

        import litellm  # pylint: disable=import-outside-toplevel

        errors: list[Exception] = []
        mod = import_module(_LITELLM_MAIN)
        for func_name, _ in _WRAPPED_FUNCTIONS:
            try:
                unwrap(mod, func_name)
                if hasattr(litellm, func_name):
                    setattr(litellm, func_name, getattr(mod, func_name))
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to uninstrument LiteLLM %s: %s", func_name, err)
                errors.append(err)

        _unregister_otel_callback()
        global _otel_logger  # pylint: disable=global-statement
        _otel_logger = None
        LiteLLMInstrumentorWrapper._applied = False
        logger.debug("LiteLLM instrumentation removed.")
        if errors:
            raise errors[0]


__all__ = ["LiteLLMInstrumentorWrapper"]
