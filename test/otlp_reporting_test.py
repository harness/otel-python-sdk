"""Tests for OTLP reporting configuration helpers."""

import pytest

from harness_sdk.config import config_pb2
from harness_sdk.otlp_reporting import (
    compression_type_to_otlp_grpc,
    compression_type_to_otlp_http,
    compression_value_to_enum_name,
    normalize_otlp_http_traces_endpoint,
    normalize_reporting_dict,
)


def test_normalize_otlp_http_traces_endpoint_appends_path():
    assert normalize_otlp_http_traces_endpoint(
        'http://localhost:5442'
    ) == 'http://localhost:5442/v1/traces'
    assert normalize_otlp_http_traces_endpoint(
        'http://localhost:5442/'
    ) == 'http://localhost:5442/v1/traces'


def test_normalize_otlp_http_traces_endpoint_keeps_existing_path():
    endpoint = 'http://localhost:5442/v1/traces'
    assert normalize_otlp_http_traces_endpoint(endpoint) == endpoint
    assert normalize_otlp_http_traces_endpoint(
        'http://localhost:5442/v1/traces/'
    ) == 'http://localhost:5442/v1/traces/'


def test_normalize_reporting_dict_maps_compression_and_encoding():
    reporting = {
        'endpoint': 'http://localhost:5442',
        'encoding': 'proto',
        'compression': 'gzip',
    }
    encoding = normalize_reporting_dict(reporting)
    assert encoding == 'proto'
    assert reporting['compression_type'] == 'COMPRESSION_TYPE_GZIP'
    assert 'compression' not in reporting
    assert 'encoding' not in reporting


def test_compression_value_to_enum_name():
    assert compression_value_to_enum_name('none') == 'COMPRESSION_TYPE_UNSPECIFIED'
    assert compression_value_to_enum_name('gzip') == 'COMPRESSION_TYPE_GZIP'


def test_compression_type_to_otlp_exporters():
    from opentelemetry.exporter.otlp.proto.http import Compression as HttpCompression
    from grpc import Compression as GrpcCompression

    assert compression_type_to_otlp_http(
        config_pb2.CompressionType.COMPRESSION_TYPE_UNSPECIFIED
    ) == HttpCompression.NoCompression
    assert compression_type_to_otlp_http(
        config_pb2.CompressionType.COMPRESSION_TYPE_GZIP
    ) == HttpCompression.Gzip
    assert compression_type_to_otlp_grpc(
        config_pb2.CompressionType.COMPRESSION_TYPE_GZIP
    ) == GrpcCompression.Gzip


def test_invalid_compression_raises():
    with pytest.raises(ValueError):
        compression_value_to_enum_name('invalid')
