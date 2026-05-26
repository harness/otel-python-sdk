"""Default configuration for harness-sdk (generic instrumentation and telemetry)."""

DEFAULT = {
    'enabled': True,
    'logging': {
        'log_mode': 'LOG_MODE_STDOUT',
        'log_level': 'LOG_LEVEL_INFO',
    },
    'propagation_formats': ['TRACECONTEXT'],
    'service_name': 'otel-sdk',
    'reporting': {
        'endpoint': 'http://localhost:5442',
        'secure': False,
        'trace_reporter_type': 'OTLP_HTTP',
        'token': '',
        'encoding': 'proto',
        'compression': 'none',
    },
    'data_capture': {
        'http_headers': {
            'request': True,
            'response': True,
        },
        'http_body': {
            'request': True,
            'response': True,
        },
        'rpc_metadata': {
            'request': True,
            'response': True,
        },
        'rpc_body': {
            'request': True,
            'response': True,
        },
        'body_max_size_bytes': 131072,
        'allowed_content_types': ['json', 'xml', 'grpc', 'x-www-form-urlencoded', 'graphql']
    },
    'resource_attributes': {},
    'span_attributes': {},
    'agent_identity': {
        'deployment_name': '',
    },
    'gen_ai': {
        'enabled': True,
        'payload_capture_enabled': True,
        'payload_evaluation_enabled': True,
    },
    'plugins': {
        'control': [],
        'observability': [
            'builtin_pipeline',
            'builtin_span_attributes',
        ],
    },
}

SDK_CONFIG_KEYS = frozenset(DEFAULT.keys())
