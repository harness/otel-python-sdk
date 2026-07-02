'''Hypertrace wrapper around OTel botocore Instrumentor''' # pylint: disable=R0801
from typing import Any, Callable, Optional

from opentelemetry.instrumentation.botocore import BotocoreInstrumentor # pylint:disable=E0611,E0401

from harness_sdk.instrumentation import BaseInstrumentorWrapper


from harness_sdk.custom_logger import get_custom_logger
logger = get_custom_logger(__name__)


_BEDROCK_RUNTIME_SERVICE_NAMES = frozenset({"bedrock-runtime", "Bedrock Runtime"})
_BEDROCK_CONVERSE_OPERATIONS = frozenset({"Converse", "ConverseStream"})
_BEDROCK_MODEL_ID_HEADER = "x-amzn-bedrock-model-id"


def _is_bedrock_converse(service_name: str, operation_name: str) -> bool:
    return (
        service_name in _BEDROCK_RUNTIME_SERVICE_NAMES
        and operation_name in _BEDROCK_CONVERSE_OPERATIONS
    )


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _set_if_present(span: Any, key: str, value: Any) -> None:
    if value is None:
        return
    span.set_attribute(key, value)


def _bedrock_request_model(api_params: dict[str, Any]) -> Optional[str]:
    model_id = api_params.get("modelId")
    return str(model_id) if model_id else None


def _is_inference_profile_identifier(model_id: Optional[str]) -> bool:
    return bool(
        model_id
        and (
            ":inference-profile/" in model_id
            or ":application-inference-profile/" in model_id
            or model_id.startswith("inference-profile/")
            or model_id.startswith("application-inference-profile/")
        )
    )


def _bedrock_inference_config(api_params: dict[str, Any]) -> dict[str, Any]:
    config = api_params.get("inferenceConfig") or {}
    if not isinstance(config, dict):
        return {}
    return config


def _set_bedrock_request_attributes(
    span: Any,
    operation_name: str,
    api_params: dict[str, Any],
) -> None:
    if span is None or not span.is_recording():
        return

    inference_config = _bedrock_inference_config(api_params)
    request_model = _bedrock_request_model(api_params)

    _set_if_present(span, "gen_ai.request.model", request_model)
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", "aws.bedrock")
    span.set_attribute("gen_ai.framework", "boto3")
    if _is_inference_profile_identifier(request_model):
        _set_if_present(span, "aws.bedrock.inference_profile_arn", request_model)
    span.set_attribute(
        "gen_ai.request.streaming",
        operation_name == "ConverseStream",
    )
    _set_if_present(
        span,
        "gen_ai.request.max_tokens",
        inference_config.get("maxTokens"),
    )
    _set_if_present(
        span,
        "gen_ai.request.temperature",
        inference_config.get("temperature"),
    )
    _set_if_present(span, "gen_ai.request.top_p", inference_config.get("topP"))


def _bedrock_execution_model(response: Any) -> Optional[str]:
    response_metadata = _get_value(response, "ResponseMetadata") or {}
    headers = _get_value(response_metadata, "HTTPHeaders") or {}
    model_id = _get_value(headers, _BEDROCK_MODEL_ID_HEADER)
    return str(model_id) if model_id else None


def _set_bedrock_response_attributes(
    span: Any,
    response: Any,
    request_model: Optional[str],
) -> None:
    if span is None or not span.is_recording():
        return

    execution_model = _bedrock_execution_model(response)
    response_model = execution_model or request_model
    _set_if_present(span, "gen_ai.response.model", response_model)
    _set_if_present(
        span,
        "aws.bedrock.execution_model_id",
        execution_model,
    )
    _set_if_present(
        span,
        "gen_ai.response.finish_reasons",
        _get_value(response, "stopReason"),
    )

    usage = _get_value(response, "usage") or {}
    _set_if_present(
        span,
        "gen_ai.usage.input_tokens",
        _get_value(usage, "inputTokens"),
    )
    _set_if_present(
        span,
        "gen_ai.usage.output_tokens",
        _get_value(usage, "outputTokens"),
    )
    _set_if_present(
        span,
        "gen_ai.usage.total_tokens",
        _get_value(usage, "totalTokens"),
    )


def _call_user_hook(hook: Optional[Callable[..., Any]], *args: Any) -> None:
    if not callable(hook):
        return
    hook(*args)


class BotocoreInstrumentationWrapper(BotocoreInstrumentor, BaseInstrumentorWrapper):
    '''Hypertrace wrapper around OTel Botocore Instrumentor class'''

    def _instrument(self, **kwargs):
        user_request_hook = kwargs.get("request_hook")
        user_response_hook = kwargs.get("response_hook")

        def request_hook(span, service_name, operation_name, api_params):
            _call_user_hook(user_request_hook, span, service_name, operation_name, api_params)
            if _is_bedrock_converse(service_name, operation_name):
                _set_bedrock_request_attributes(span, operation_name, api_params)

        def response_hook(span, service_name, operation_name, result):
            _call_user_hook(user_response_hook, span, service_name, operation_name, result)
            if _is_bedrock_converse(service_name, operation_name):
                request_model = span.attributes.get("gen_ai.request.model")
                _set_bedrock_response_attributes(span, result, request_model)

        kwargs["request_hook"] = request_hook
        kwargs["response_hook"] = response_hook
        super()._instrument(**kwargs)
