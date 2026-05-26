"""Apply HA_* environment variables to harness-sdk configuration."""
import os

from google.protobuf import wrappers_pb2

from harness_sdk.config import config_pb2
from harness_sdk.otlp_reporting import compression_type_to_enum


def _env(name):
    """Prefer HA_ prefix; accept legacy AT_/TA_ for SDK settings during migration."""
    return os.environ.get(f"HA_{name}") or os.environ.get(f"AT_{name}") or os.environ.get(f"TA_{name}")


def overwrite_with_environment(config, reporting_encoding=None):  # pylint: disable=R0912,R0914,R0915
    # Log configuration
    log_mode = _env("LOG_MODE")
    if log_mode:
        mode = getattr(config_pb2.LogMode, log_mode, None)
        if mode is not None:
            config.logging.log_mode = mode

    log_level = _env("LOG_LEVEL")
    if log_level:
        level = getattr(config_pb2.LogLevel, log_level, None)
        if level is not None:
            config.logging.log_level = level

    service_name = _env("SERVICE_NAME")
    if service_name:
        config.service_name.value = service_name

    reporting_endpoint = _env("REPORTING_ENDPOINT")
    if reporting_endpoint:
        config.reporting.endpoint.value = reporting_endpoint

    reporter_type = _env("REPORTING_TRACE_REPORTER_TYPE")
    if reporter_type:
        enum_value = getattr(config_pb2.TraceReporterType, reporter_type, None)
        if enum_value is not None:
            config.reporting.trace_reporter_type = enum_value

    reporting_secure = _env("REPORTING_SECURE")
    if reporting_secure:
        config.reporting.secure.value = reporting_secure.lower() == 'true'

    reporting_token = _env("REPORTING_TOKEN")
    if reporting_token:
        config.reporting.token.value = reporting_token

    reporting_compression = _env("REPORTING_COMPRESSION")
    if reporting_compression:
        config.reporting.compression_type = compression_type_to_enum(reporting_compression)

    reporting_encoding_env = _env("REPORTING_ENCODING")
    if reporting_encoding_env:
        reporting_encoding = reporting_encoding_env

    headers_request = _env("DATA_CAPTURE_HTTP_HEADERS_REQUEST")
    if headers_request:
        config.data_capture.http_headers.request.value = headers_request.lower() == 'true'

    headers_response = _env("DATA_CAPTURE_HTTP_HEADERS_RESPONSE")
    if headers_response:
        config.data_capture.http_headers.response.value = headers_response.lower() == 'true'

    body_request = _env("DATA_CAPTURE_HTTP_BODY_REQUEST")
    if body_request:
        config.data_capture.http_body.request.value = body_request.lower() == 'true'

    body_response = _env("DATA_CAPTURE_HTTP_BODY_RESPONSE")
    if body_response:
        config.data_capture.http_body.response.value = body_response.lower() == 'true'

    rpc_metadata_request = _env("DATA_CAPTURE_RPC_METADATA_REQUEST")
    if rpc_metadata_request:
        config.data_capture.rpc_metadata.request.value = rpc_metadata_request.lower() == 'true'

    rpc_metadata_response = _env("DATA_CAPTURE_RPC_METADATA_RESPONSE")
    if rpc_metadata_response:
        config.data_capture.rpc_metadata.response.value = rpc_metadata_response.lower() == 'true'

    rpc_body_request = _env("DATA_CAPTURE_RPC_BODY_REQUEST")
    if rpc_body_request:
        config.data_capture.rpc_body.request.value = rpc_body_request.lower() == 'true'

    rpc_body_response = _env("DATA_CAPTURE_RPC_BODY_RESPONSE")
    if rpc_body_response:
        config.data_capture.rpc_body.response.value = rpc_body_response.lower() == 'true'

    body_max_size_bytes = _env("DATA_CAPTURE_BODY_MAX_SIZE_BYTES")
    if body_max_size_bytes:
        config.data_capture.body_max_size_bytes.value = int(body_max_size_bytes)

    allowed_content_types = _env("DATA_CAPTURE_ALLOWED_CONTENT_TYPES")
    if allowed_content_types:
        configured_content_types = allowed_content_types.split(",")
        del config.data_capture.allowed_content_types[:]
        wrapped_content_types = [
            wrappers_pb2.StringValue(value=content_type.strip())
            for content_type in configured_content_types
        ]
        config.data_capture.allowed_content_types.extend(wrapped_content_types)

    propagation_formats = _env("PROPAGATION_FORMATS")
    if propagation_formats:
        tmp_propagation_formats = set()
        configured_propagation_formats = propagation_formats.split(",")
        if "TRACECONTEXT" in configured_propagation_formats:
            tmp_propagation_formats.add(config_pb2.PropagationFormat.TRACECONTEXT)
        if "B3" in configured_propagation_formats:
            tmp_propagation_formats.add(config_pb2.PropagationFormat.B3)
        if not tmp_propagation_formats:
            tmp_propagation_formats.add(config_pb2.PropagationFormat.TRACECONTEXT)
        config.propagation_formats[:] = list(tmp_propagation_formats)

    gen_ai_enabled = _env("GEN_AI_ENABLED")
    if gen_ai_enabled:
        config.gen_ai.enabled.value = gen_ai_enabled.lower() == 'true'

    gen_ai_capture = _env("GEN_AI_PAYLOAD_CAPTURE_ENABLED")
    if gen_ai_capture:
        config.gen_ai.payload_capture_enabled.value = gen_ai_capture.lower() == 'true'

    gen_ai_eval = _env("GEN_AI_PAYLOAD_EVALUATION_ENABLED")
    if gen_ai_eval:
        config.gen_ai.payload_evaluation_enabled.value = gen_ai_eval.lower() == 'true'

    deployment_name = _env("DEPLOYMENT_NAME")
    if deployment_name:
        config.agent_identity.deployment_name.value = deployment_name

    enabled = _env("ENABLED")
    if enabled:
        config.enabled.value = enabled.lower() == 'true'

    span_attributes = _env("SPAN_ATTRIBUTES")
    if span_attributes:
        for group in span_attributes.split(","):
            key, value = group.split("=")
            config.span_attributes[key] = value

    resource_attributes = _env("RESOURCE_ATTRIBUTES")
    if resource_attributes:
        for group in resource_attributes.split(","):
            key, value = group.split("=")
            config.resource_attributes[key] = value

    return config, reporting_encoding
