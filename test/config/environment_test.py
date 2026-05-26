'''Environment config test for harness-sdk (HA_ prefix).'''

import os

from harness_sdk.config import config_pb2
from harness_sdk.config.config import Config


def test_env_config() -> None:
    Config._instance = None
    os.environ["HA_SERVICE_NAME"] = "pythonagent_002"
    os.environ["HA_REPORTING_ENDPOINT"] = "http://localhost:9411/api/v2/spans2"
    os.environ["HA_REPORTING_TRACE_REPORTER_TYPE"] = "OTLP"
    os.environ["HA_REPORTING_SECURE"] = "true"
    os.environ["HA_DATA_CAPTURE_HTTP_HEADERS_REQUEST"] = "False"
    os.environ["HA_DATA_CAPTURE_HTTP_HEADERS_RESPONSE"] = "False"
    os.environ["HA_DATA_CAPTURE_HTTP_BODY_REQUEST"] = "False"
    os.environ["HA_DATA_CAPTURE_HTTP_BODY_RESPONSE"] = "False"
    os.environ["HA_DATA_CAPTURE_RPC_METADATA_REQUEST"] = "False"
    os.environ["HA_DATA_CAPTURE_RPC_METADATA_RESPONSE"] = "False"
    os.environ["HA_DATA_CAPTURE_RPC_BODY_REQUEST"] = "False"
    os.environ["HA_DATA_CAPTURE_RPC_BODY_RESPONSE"] = "False"
    os.environ["HA_DATA_CAPTURE_BODY_MAX_SIZE_BYTES"] = "123456"
    os.environ["HA_PROPAGATION_FORMATS"] = "B3,TRACECONTEXT"
    os.environ["HA_ENABLED"] = "true"
    os.environ["HA_ENABLE_CONSOLE_SPAN_EXPORTER"] = "True"
    os.environ["HA_RESOURCE_ATTRIBUTES"] = "1=123,b=456,d=89123"
    os.environ["HA_DATA_CAPTURE_ALLOWED_CONTENT_TYPES"] = "json,foo,bar"
    config = Config().config
    assert config.service_name.value == "pythonagent_002"
    assert config.reporting.endpoint.value == "http://localhost:9411/api/v2/spans2"
    assert config.reporting.trace_reporter_type == config_pb2.TraceReporterType.OTLP
    assert config.data_capture.http_headers.request.value is False
    assert config.data_capture.http_body.request.value is False
    assert config.data_capture.body_max_size_bytes.value == 123456
    assert config_pb2.PropagationFormat.B3 in config.propagation_formats
    assert config_pb2.PropagationFormat.TRACECONTEXT in config.propagation_formats
    assert config.enabled.value is True
    content_type_values = [item.value for item in config.data_capture.allowed_content_types]
    assert "json" in content_type_values
    resource_attrs = config.resource_attributes
    assert resource_attrs['1'] == '123'
    unset_env_variables()
    Config._instance = None


def test_gen_ai_env_overrides() -> None:
    Config._instance = None
    os.environ["HA_GEN_AI_ENABLED"] = "true"
    os.environ["HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED"] = "false"
    os.environ["HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED"] = "false"
    cfg = Config()
    assert cfg.config.gen_ai.enabled.value is True
    assert cfg.config.gen_ai.payload_capture_enabled.value is False
    assert cfg.config.gen_ai.payload_evaluation_enabled.value is False
    unset_env_variables()
    Config._instance = None


def unset_env_variables():
    keys_to_delete = [key for key in os.environ if key.startswith("HA_")]
    for key in keys_to_delete:
        del os.environ[key]
