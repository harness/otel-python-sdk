import boto3
import pytest
from botocore.stub import Stubber

from harness_sdk.instrumentation.botocore import BotocoreInstrumentationWrapper


_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
_INFERENCE_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:"
    "application-inference-profile/profile123"
)


@pytest.fixture
def botocore_instrumentor():
    wrapper = BotocoreInstrumentationWrapper()
    yield wrapper
    if getattr(wrapper, "_is_instrumented_by_opentelemetry", False):
        wrapper.uninstrument()


def _bedrock_span(spans):
    for span in spans:
        attrs = span.attributes or {}
        if attrs.get("gen_ai.provider.name") == "aws.bedrock":
            return span
    raise AssertionError("No Bedrock GenAI span found")


def _converse_params(model_id=_MODEL_ID):
    return {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {
            "maxTokens": 32,
            "temperature": 0.1,
            "topP": 0.9,
        },
    }


def test_bedrock_converse_span_has_gen_ai_attributes(agent, exporter, botocore_instrumentor):  # pylint: disable=unused-argument
    botocore_instrumentor.instrument()
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hello"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8},
        "metrics": {"latencyMs": 42},
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "HTTPHeaders": {"x-amzn-bedrock-model-id": _MODEL_ID},
        },
    }

    with Stubber(client) as stubber:
        stubber.add_response("converse", response, _converse_params(_INFERENCE_PROFILE_ARN))
        client.converse(**_converse_params(_INFERENCE_PROFILE_ARN))

    attrs = _bedrock_span(exporter.get_finished_spans()).attributes
    exporter.clear()
    assert attrs.get("gen_ai.request.model") == _INFERENCE_PROFILE_ARN
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "aws.bedrock"
    assert attrs.get("gen_ai.framework") == "boto3"
    assert attrs.get("aws.bedrock.inference_profile_arn") == _INFERENCE_PROFILE_ARN
    assert attrs.get("gen_ai.request.streaming") is False
    assert attrs.get("gen_ai.request.max_tokens") == 32
    assert attrs.get("gen_ai.request.temperature") == 0.1
    assert attrs.get("gen_ai.request.top_p") == 0.9
    assert attrs.get("gen_ai.response.model") == _MODEL_ID
    assert attrs.get("aws.bedrock.execution_model_id") == _MODEL_ID
    assert attrs.get("gen_ai.response.finish_reasons") == "end_turn"
    assert attrs.get("gen_ai.usage.input_tokens") == 3
    assert attrs.get("gen_ai.usage.output_tokens") == 5
    assert attrs.get("gen_ai.usage.total_tokens") == 8


def test_bedrock_converse_falls_back_to_request_model_when_response_header_missing(
    agent,
    exporter,
    botocore_instrumentor,
):  # pylint: disable=unused-argument
    botocore_instrumentor.instrument()
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hello"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8},
        "metrics": {"latencyMs": 42},
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}},
    }

    with Stubber(client) as stubber:
        stubber.add_response("converse", response, _converse_params(_INFERENCE_PROFILE_ARN))
        client.converse(**_converse_params(_INFERENCE_PROFILE_ARN))

    attrs = _bedrock_span(exporter.get_finished_spans()).attributes
    exporter.clear()
    assert attrs.get("gen_ai.request.model") == _INFERENCE_PROFILE_ARN
    assert attrs.get("gen_ai.response.model") == _INFERENCE_PROFILE_ARN
    assert attrs.get("aws.bedrock.execution_model_id") is None
    assert attrs.get("aws.bedrock.inference_profile_arn") == _INFERENCE_PROFILE_ARN
    assert "aws.bedrock.inference_profile_model_arns" not in attrs
    assert "aws.bedrock.resolved_model_id" not in attrs


def test_bedrock_converse_stream_request_has_gen_ai_attributes(agent, exporter, botocore_instrumentor):  # pylint: disable=unused-argument
    calls = []

    def request_hook(span, service_name, operation_name, api_params):
        calls.append((service_name, operation_name, api_params["modelId"]))

    botocore_instrumentor.instrument(request_hook=request_hook)

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    params = {
        "modelId": _MODEL_ID,
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }

    with Stubber(client) as stubber:
        stubber.add_client_error(
            "converse_stream",
            service_error_code="ThrottlingException",
            service_message="rate limited",
            http_status_code=429,
            expected_params=params,
        )
        try:
            client.converse_stream(**params)
        except client.exceptions.ThrottlingException:
            pass

    attrs = _bedrock_span(exporter.get_finished_spans()).attributes
    exporter.clear()
    assert calls == [("bedrock-runtime", "ConverseStream", _MODEL_ID)]
    assert attrs.get("gen_ai.request.model") == _MODEL_ID
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "aws.bedrock"
    assert attrs.get("gen_ai.framework") == "boto3"
    assert attrs.get("gen_ai.request.streaming") is True
