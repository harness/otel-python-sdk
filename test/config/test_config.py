# pylint: disable=no-member
import os

from harness_sdk.config import config_pb2
from harness_sdk.config.config import Config


def prepare():
    if hasattr(Config, '_instance'):
        del Config._instance


def reset():
    keys_to_delete = [key for key in os.environ if key.startswith("HA_")]
    for key in keys_to_delete:
        del os.environ[key]
    if hasattr(Config, '_instance'):
        Config._instance = None


def with_config_file(filepath='./test_config_file.yaml'):
    os.environ['HA_CONFIG_FILE'] = os.path.join(os.path.dirname(__file__), filepath)


def test_load_sdk_gen_ai_from_file():
    prepare()
    with_config_file()
    traceable_config = Config().config

    assert traceable_config.gen_ai.enabled.value is False
    assert traceable_config.gen_ai.payload_capture_enabled.value is False
    assert traceable_config.gen_ai.payload_evaluation_enabled.value is True
    reset()


def test_sdk_default_config():
    prepare()
    traceable_config = Config().config

    reporting = traceable_config.reporting
    assert traceable_config.service_name.value == "otel-sdk"
    assert reporting.endpoint.value == "http://localhost:5442"
    assert reporting.trace_reporter_type == config_pb2.TraceReporterType.OTLP_HTTP
    assert reporting.compression_type == config_pb2.CompressionType.COMPRESSION_TYPE_UNSPECIFIED
    assert Config().reporting_encoding == 'proto'

    gen_ai = traceable_config.gen_ai
    assert gen_ai.enabled.value is True
    assert gen_ai.payload_capture_enabled.value is True
    assert gen_ai.payload_evaluation_enabled.value is True

    assert len(traceable_config.span_attributes) == 0
    reset()


def test_with_minimal_config():
    prepare()
    with_config_file('./minimal_config.yaml')
    config = Config().config
    assert config.gen_ai.enabled.value is True
    reset()


def test_sdk_environ():
    prepare()
    os.environ["HA_GEN_AI_ENABLED"] = 'true'
    os.environ["HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED"] = 'true'
    os.environ["HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED"] = 'true'
    os.environ["HA_SPAN_ATTRIBUTES"] = 'env=prod,team=platform'
    os.environ["HA_DEPLOYMENT_NAME"] = 'my-deployment'
    traceable_config = Config().config

    assert traceable_config.agent_identity.deployment_name.value == 'my-deployment'

    gen_ai = traceable_config.gen_ai
    assert gen_ai.enabled.value is True
    assert gen_ai.payload_capture_enabled.value is True
    assert gen_ai.payload_evaluation_enabled.value is True

    assert traceable_config.span_attributes['env'] == 'prod'
    assert traceable_config.span_attributes['team'] == 'platform'
    reset()
