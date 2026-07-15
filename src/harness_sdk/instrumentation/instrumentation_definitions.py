'''this module acts as a driver for instrumentation definitions + application'''
import importlib
from importlib import metadata as importlib_metadata

from harness_sdk.custom_logger import get_custom_logger

FLASK_KEY = 'flask'
DJANGO_KEY = 'django'
FAST_API_KEY = 'fastapi'
GRPC_SERVER_KEY = 'grpc:server'
GRPC_CLIENT_KEY = 'grpc:client'
POSTGRESQL_KEY = 'postgresql'
MYSQL_KEY = 'mysql'
REQUESTS_KEY = 'requests'
HTTPX_KEY = 'httpx'
AIOHTTP_CLIENT_KEY = 'aiohttp:client'
BOTOCORE = 'botocore'
ANTHROPIC_KEY = 'anthropic'
OPENAI_KEY = 'openai'
LITELLM_KEY = 'litellm'
GOOGLE_GENAI_KEY = 'google_genai'
MCP_KEY = 'mcp'

SUPPORTED_LIBRARIES = [
    FLASK_KEY, DJANGO_KEY, FAST_API_KEY,
    GRPC_SERVER_KEY, GRPC_CLIENT_KEY,
    POSTGRESQL_KEY, MYSQL_KEY,
    REQUESTS_KEY, HTTPX_KEY, AIOHTTP_CLIENT_KEY,
    BOTOCORE,
    ANTHROPIC_KEY,
    OPENAI_KEY,
    LITELLM_KEY,
    GOOGLE_GENAI_KEY,
    MCP_KEY,
]

# map of library_key => instrumentation wrapper instance
_INSTRUMENTATION_STATE = {}
_GENERIC_INSTRUMENTATION_STATE = {}

logger = get_custom_logger(__name__)

def _safe_uninstrument(key, value):
    if not hasattr(value, "uninstrument"):
        return
    try:
        value.uninstrument()
    except Exception as _err:  # pylint:disable=W0703
        logger.debug("Failed to uninstrument %s: %s", key, _err, exc_info=True)


def _uninstrument_all():
    for key, value in list(_INSTRUMENTATION_STATE.items()):
        logger.debug("Uninstrumenting %s", key)
        _safe_uninstrument(key, value)

    for key, value in list(_GENERIC_INSTRUMENTATION_STATE.items()):
        logger.debug("Uninstrumenting generic contrib library %s", key)
        _safe_uninstrument(key, value)

    _INSTRUMENTATION_STATE.clear()
    _GENERIC_INSTRUMENTATION_STATE.clear()

def is_already_instrumented(library_key):
    """check if an instrumentation wrapper is already registered"""
    return library_key in _INSTRUMENTATION_STATE


def _mark_as_instrumented(library_key, wrapper_instance):
    """mark an instrumentation wrapper as registered"""
    _INSTRUMENTATION_STATE[library_key] = wrapper_instance


_WRAPPER_MODULE_AND_CLASS = {
    DJANGO_KEY: ("harness_sdk.instrumentation.django", "DjangoInstrumentationWrapper"),
    FLASK_KEY: ("harness_sdk.instrumentation.flask", "FlaskInstrumentorWrapper"),
    FAST_API_KEY: ("harness_sdk.instrumentation.fast_api", "FastAPIInstrumentorWrapper"),
    GRPC_SERVER_KEY: ("harness_sdk.instrumentation.grpc", "GrpcInstrumentorServerWrapper"),
    GRPC_CLIENT_KEY: ("harness_sdk.instrumentation.grpc", "GrpcInstrumentorClientWrapper"),
    POSTGRESQL_KEY: ("harness_sdk.instrumentation.postgresql", "PostgreSQLInstrumentorWrapper"),
    MYSQL_KEY: ("harness_sdk.instrumentation.mysql", "MySQLInstrumentorWrapper"),
    REQUESTS_KEY: ("harness_sdk.instrumentation.requests", "RequestsInstrumentorWrapper"),
    HTTPX_KEY: ("harness_sdk.instrumentation.httpx", "HTTPXClientInstrumentorWrapper"),
    AIOHTTP_CLIENT_KEY: ("harness_sdk.instrumentation.aiohttp", "AioHttpClientInstrumentorWrapper"),
    BOTOCORE: ("harness_sdk.instrumentation.botocore", "BotocoreInstrumentationWrapper"),
    ANTHROPIC_KEY: ("harness_sdk.instrumentation.anthropic", "AnthropicInstrumentorWrapper"),
    OPENAI_KEY: ("harness_sdk.instrumentation.openai", "OpenAIInstrumentorWrapper"),
    LITELLM_KEY: ("harness_sdk.instrumentation.litellm", "LiteLLMInstrumentorWrapper"),
    GOOGLE_GENAI_KEY: ("harness_sdk.instrumentation.google_genai", "GoogleGenAIInstrumentorWrapper"),
    MCP_KEY: ("harness_sdk.instrumentation.mcp", "McpInstrumentorWrapper"),
}

_WRAPPER_LIBRARY_ALIASES = {
    "psycopg2",
    "grpc",
}

# Contrib instrumentors without a custom wrapper that must not be auto-enabled.
# aws-lambda: marks itself instrumented off-Lambda but never sets _wrapped_module_name,
# so uninstrument() raises AttributeError (upstream OTel behavior).
# aiobotocore: optional async variant; botocore wrapper covers sync usage.
_GENERIC_CONTRIB_DENYLIST = {
    "awslambda",
    "aiobotocore",
}


def _normalize_library_name(library_name):
    return ''.join(ch for ch in str(library_name).lower() if ch.isalnum())


def _get_normalized_skip_libraries(skip_libraries):
    return {_normalize_library_name(library_name) for library_name in skip_libraries}


def _get_wrapper_normalized_names():
    wrapper_names = {
        _normalize_library_name(library_name)
        for library_name in _WRAPPER_MODULE_AND_CLASS
    }
    wrapper_names.update(_WRAPPER_LIBRARY_ALIASES)
    return wrapper_names


def _get_contrib_instrumentation_entry_points():
    try:
        entry_points = importlib_metadata.entry_points()
        if hasattr(entry_points, "select"):
            return list(entry_points.select(group="opentelemetry_instrumentor"))
        return list(entry_points.get("opentelemetry_instrumentor", []))
    except Exception as _err:  # pylint:disable=W0703
        logger.debug("Unable to load opentelemetry_instrumentor entry points: %s", str(_err), exc_info=True)
        return []


def _instrument_generic_contrib_library(entry_point):
    try:
        instrumentor_class = entry_point.load()
        instrumentor_instance = instrumentor_class()
        instrumentor_instance.instrument()
        if not getattr(instrumentor_instance, "_is_instrumented_by_opentelemetry", False):
            return
        _GENERIC_INSTRUMENTATION_STATE[entry_point.name] = instrumentor_instance
        logger.debug("Successfully instrumented generic contrib library %s", entry_point.name)
    except Exception as _err:  # pylint:disable=W0703
        logger.debug("Failed to instrument contrib library %s: %s", entry_point.name, str(_err), exc_info=True)


def instrument_supported_contrib_without_wrapper(skip_libraries=None):
    """Instrument installed OTel contrib instrumentors lacking a custom wrapper."""
    if skip_libraries is None:
        skip_libraries = []

    normalized_skip_libraries = _get_normalized_skip_libraries(skip_libraries)
    normalized_wrapper_library_names = _get_wrapper_normalized_names()
    normalized_denylist = set(_GENERIC_CONTRIB_DENYLIST)

    for entry_point in _get_contrib_instrumentation_entry_points():
        normalized_entry_name = _normalize_library_name(entry_point.name)
        if normalized_entry_name in normalized_wrapper_library_names:
            continue
        if normalized_entry_name in normalized_denylist:
            logger.debug("Skipping denylisted generic contrib instrumentation for %s", entry_point.name)
            continue
        if normalized_entry_name in normalized_skip_libraries:
            logger.debug("Skipping generic contrib instrumentation for %s", entry_point.name)
            continue
        if entry_point.name in _GENERIC_INSTRUMENTATION_STATE:
            logger.debug("Generic contrib instrumentation already enabled for %s", entry_point.name)
            continue
        _instrument_generic_contrib_library(entry_point)


def get_instrumentation_wrapper(library_key):
    """load an initialize an instrumentation wrapper"""
    if is_already_instrumented(library_key):
        logger.debug("Already instrumented %s", library_key)
        return None
    spec = _WRAPPER_MODULE_AND_CLASS.get(library_key)
    if spec is None:
        logger.debug("No instrumentation wrapper available for %s", library_key)
        return None
    try:
        module_path, class_name = spec
        module = importlib.import_module(module_path)
        wrapper_cls = getattr(module, class_name)
        wrapper_instance = wrapper_cls()
        _mark_as_instrumented(library_key, wrapper_instance)
        return wrapper_instance
    except Exception as _err: # pylint:disable=W0703
        logger.debug("Error while attempting to load instrumentation wrapper for %s: %s", library_key, str(_err),
                     exc_info=True)
        return None
