"""Tests for OTLP exporter initialization from reporting config."""

from unittest import mock

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
from opentelemetry.exporter.otlp.proto.http import Compression as HttpCompression
from harness_sdk.config import config_pb2


def test_otlp_http_exporter_appends_traces_path(agent):
    agent._config.config.reporting.endpoint.value = 'http://collector:4318'
    agent._config.config.reporting.trace_reporter_type = config_pb2.TraceReporterType.OTLP_HTTP

    exporter = agent._init._init_exporter(config_pb2.TraceReporterType.OTLP_HTTP)
    try:
        assert exporter._endpoint == 'http://collector:4318/v1/traces'
    finally:
        exporter.shutdown()


def test_otlp_http_exporter_uses_gzip_compression(agent):
    agent._config.config.reporting.endpoint.value = 'http://collector:4318/v1/traces'
    agent._config.config.reporting.compression_type = (
        config_pb2.CompressionType.COMPRESSION_TYPE_GZIP
    )

    exporter = agent._init._init_exporter(config_pb2.TraceReporterType.OTLP_HTTP)
    try:
        assert isinstance(exporter, OTLPHttpSpanExporter)
        assert exporter._compression == HttpCompression.Gzip
    finally:
        exporter.shutdown()


def test_otlp_http_exporter_warns_on_unsupported_encoding(agent):
    agent._config.reporting_encoding = 'json'
    agent._config.config.reporting.endpoint.value = 'http://collector:4318/v1/traces'

    with mock.patch('harness_sdk.agent_init.logger') as logger:
        exporter = agent._init._init_exporter(config_pb2.TraceReporterType.OTLP_HTTP)
        try:
            logger.warning.assert_called_once()
        finally:
            exporter.shutdown()
