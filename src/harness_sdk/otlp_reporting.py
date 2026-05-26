"""Helpers for OTLP trace exporter configuration."""
from urllib.parse import urlparse, urlunparse

from grpc import Compression as GrpcCompression
from opentelemetry.exporter.otlp.proto.http import Compression as HttpCompression

from harness_sdk.config import config_pb2

TRACES_PATH = 'v1/traces'


def normalize_reporting_dict(reporting: dict) -> str | None:
    """Map reporting YAML keys to protobuf fields; return encoding if set."""
    encoding = reporting.pop('encoding', None)

    if 'compression' in reporting:
        reporting['compression_type'] = compression_value_to_enum_name(
            reporting.pop('compression')
        )
    elif 'compression_type' not in reporting:
        reporting['compression_type'] = 'COMPRESSION_TYPE_UNSPECIFIED'

    return encoding


def compression_value_to_enum_name(value) -> str:
    if value is None:
        return 'COMPRESSION_TYPE_UNSPECIFIED'
    normalized = str(value).strip().lower()
    if normalized in ('', 'none', 'unspecified', 'compression_type_unspecified'):
        return 'COMPRESSION_TYPE_UNSPECIFIED'
    if normalized in ('gzip', 'compression_type_gzip'):
        return 'COMPRESSION_TYPE_GZIP'
    if normalized.startswith('compression_type_'):
        return str(value).strip().upper()
    raise ValueError(f'Unsupported reporting compression value: {value}')


def compression_type_to_enum(value: str) -> int:
    enum_name = compression_value_to_enum_name(value)
    return getattr(config_pb2.CompressionType, enum_name)


def normalize_otlp_http_traces_endpoint(endpoint: str) -> str:
    """Append v1/traces to HTTP OTLP endpoints that do not already include it."""
    if not endpoint:
        return endpoint

    parsed = urlparse(endpoint)
    path = parsed.path or ''
    if TRACES_PATH in path.lower():
        return endpoint

    path = path.rstrip('/')
    new_path = f'{path}/{TRACES_PATH}' if path else f'/{TRACES_PATH}'
    return urlunparse(parsed._replace(path=new_path))


def compression_type_to_otlp_http(compression_type: int):
    if compression_type == config_pb2.CompressionType.COMPRESSION_TYPE_GZIP:
        return HttpCompression.Gzip
    return HttpCompression.NoCompression


def compression_type_to_otlp_grpc(compression_type: int):
    if compression_type == config_pb2.CompressionType.COMPRESSION_TYPE_GZIP:
        return GrpcCompression.Gzip
    return GrpcCompression.NoCompression


def is_supported_encoding(encoding: str | None) -> bool:
    if encoding is None:
        return True
    normalized = encoding.strip().lower()
    return normalized in ('', 'none', 'proto')
