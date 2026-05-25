from unittest.mock import MagicMock, patch

import pytest

from agent_trace.instrumentation.instrumentation_definitions import (
    _GENERIC_INSTRUMENTATION_STATE,
    _INSTRUMENTATION_STATE,
    _WRAPPER_MODULE_AND_CLASS,
    _get_contrib_instrumentation_entry_points,
    _get_normalized_skip_libraries,
    _get_wrapper_normalized_names,
    _instrument_generic_contrib_library,
    _normalize_library_name,
    _uninstrument_all,
    instrument_supported_contrib_without_wrapper,
)


@pytest.fixture(autouse=True)
def clear_state():
    _INSTRUMENTATION_STATE.clear()
    _GENERIC_INSTRUMENTATION_STATE.clear()
    yield
    _INSTRUMENTATION_STATE.clear()
    _GENERIC_INSTRUMENTATION_STATE.clear()


# --- _normalize_library_name ---

def test_normalize_strips_non_alnum():
    assert _normalize_library_name("aws-lambda") == "awslambda"

def test_normalize_lowercases():
    assert _normalize_library_name("Flask") == "flask"

def test_normalize_strips_colons():
    assert _normalize_library_name("grpc:server") == "grpcserver"

def test_normalize_handles_underscores():
    assert _normalize_library_name("my_library") == "mylibrary"


# --- _get_normalized_skip_libraries ---

def test_get_normalized_skip_libraries():
    result = _get_normalized_skip_libraries(["Flask", "aws-lambda", "grpc:client"])
    assert result == {"flask", "awslambda", "grpcclient"}


# --- _get_wrapper_normalized_names ---

def test_get_wrapper_normalized_names_includes_all_wrapper_keys():
    result = _get_wrapper_normalized_names()
    for key in _WRAPPER_MODULE_AND_CLASS:
        assert _normalize_library_name(key) in result

def test_get_wrapper_normalized_names_includes_aliases():
    result = _get_wrapper_normalized_names()
    assert "awslambda" in result
    assert "psycopg2" in result
    assert "grpc" in result


# --- _get_contrib_instrumentation_entry_points ---

def test_get_contrib_entry_points_uses_select_when_available():
    mock_ep = MagicMock()
    mock_ep.name = "some-library"
    mock_entry_points = MagicMock()
    mock_entry_points.select.return_value = [mock_ep]

    with patch("agent_trace.instrumentation.instrumentation_definitions.importlib_metadata.entry_points",
               return_value=mock_entry_points):
        result = _get_contrib_instrumentation_entry_points()

    mock_entry_points.select.assert_called_once_with(group="opentelemetry_instrumentor")
    assert result == [mock_ep]

def test_get_contrib_entry_points_falls_back_to_get():
    # Simulate Python <=3.11 where entry_points() returns a plain dict, not a SelectableGroups object
    mock_ep = MagicMock()
    mock_ep.name = "some-library"
    legacy_entry_points = {"opentelemetry_instrumentor": [mock_ep]}
    assert not hasattr(legacy_entry_points, "select")

    with patch("agent_trace.instrumentation.instrumentation_definitions.importlib_metadata.entry_points",
               return_value=legacy_entry_points):
        result = _get_contrib_instrumentation_entry_points()

    assert result == [mock_ep]

def test_get_contrib_entry_points_returns_empty_on_exception():
    with patch("agent_trace.instrumentation.instrumentation_definitions.importlib_metadata.entry_points",
               side_effect=Exception("boom")):
        result = _get_contrib_instrumentation_entry_points()

    assert result == []


# --- _instrument_generic_contrib_library ---

def test_instrument_generic_contrib_library_instruments_and_registers():
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    mock_ep = MagicMock()
    mock_ep.name = "some-library"
    mock_ep.load.return_value = mock_class

    _instrument_generic_contrib_library(mock_ep)

    mock_instance.instrument.assert_called_once()
    assert _GENERIC_INSTRUMENTATION_STATE["some-library"] is mock_instance

def test_instrument_generic_contrib_library_swallows_exceptions():
    mock_ep = MagicMock()
    mock_ep.name = "bad-library"
    mock_ep.load.side_effect = Exception("import error")

    _instrument_generic_contrib_library(mock_ep)  # should not raise

    assert "bad-library" not in _GENERIC_INSTRUMENTATION_STATE


# --- instrument_supported_contrib_without_wrapper ---

def _make_entry_point(name):
    ep = MagicMock()
    ep.name = name
    return ep

def test_instrument_skips_wrapper_covered_libraries():
    flask_ep = _make_entry_point("flask")

    with patch("agent_trace.instrumentation.instrumentation_definitions._get_contrib_instrumentation_entry_points",
               return_value=[flask_ep]):
        instrument_supported_contrib_without_wrapper()

    assert "flask" not in _GENERIC_INSTRUMENTATION_STATE

def test_instrument_skips_alias_covered_libraries():
    psycopg2_ep = _make_entry_point("psycopg2")

    with patch("agent_trace.instrumentation.instrumentation_definitions._get_contrib_instrumentation_entry_points",
               return_value=[psycopg2_ep]):
        instrument_supported_contrib_without_wrapper()

    assert "psycopg2" not in _GENERIC_INSTRUMENTATION_STATE

def test_instrument_skips_skip_libraries():
    ep = _make_entry_point("redis")

    with patch("agent_trace.instrumentation.instrumentation_definitions._get_contrib_instrumentation_entry_points",
               return_value=[ep]):
        instrument_supported_contrib_without_wrapper(skip_libraries=["redis"])

    assert "redis" not in _GENERIC_INSTRUMENTATION_STATE

def test_instrument_skips_already_instrumented():
    original = MagicMock()
    _GENERIC_INSTRUMENTATION_STATE["redis"] = original
    ep = _make_entry_point("redis")

    with patch("agent_trace.instrumentation.instrumentation_definitions._get_contrib_instrumentation_entry_points",
               return_value=[ep]):
        instrument_supported_contrib_without_wrapper()

    # entry point should never have been loaded — existing instance unchanged
    ep.load.assert_not_called()
    assert _GENERIC_INSTRUMENTATION_STATE["redis"] is original

def test_instrument_instruments_unknown_library():
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    ep = _make_entry_point("redis")
    ep.load.return_value = mock_class

    with patch("agent_trace.instrumentation.instrumentation_definitions._get_contrib_instrumentation_entry_points",
               return_value=[ep]):
        instrument_supported_contrib_without_wrapper()

    mock_instance.instrument.assert_called_once()
    assert _GENERIC_INSTRUMENTATION_STATE["redis"] is mock_instance

def test_instrument_skip_libraries_normalizes_names():
    ep = _make_entry_point("my-redis")

    with patch("agent_trace.instrumentation.instrumentation_definitions._get_contrib_instrumentation_entry_points",
               return_value=[ep]), \
         patch("agent_trace.instrumentation.instrumentation_definitions._instrument_generic_contrib_library") as mock_instr:
        instrument_supported_contrib_without_wrapper(skip_libraries=["My_Redis"])

    mock_instr.assert_not_called()


# --- _uninstrument_all ---

def test_uninstrument_all_calls_uninstrument_on_each_and_clears():
    flask_wrapper = MagicMock()
    django_wrapper = MagicMock()
    redis_generic = MagicMock()
    _INSTRUMENTATION_STATE["flask"] = flask_wrapper
    _INSTRUMENTATION_STATE["django"] = django_wrapper
    _GENERIC_INSTRUMENTATION_STATE["redis"] = redis_generic

    _uninstrument_all()

    flask_wrapper.uninstrument.assert_called_once()
    django_wrapper.uninstrument.assert_called_once()
    redis_generic.uninstrument.assert_called_once()
    assert len(_INSTRUMENTATION_STATE) == 0
    assert len(_GENERIC_INSTRUMENTATION_STATE) == 0

def test_uninstrument_all_tolerates_missing_uninstrument():
    # Objects without uninstrument should not raise — e.g. a partially initialised wrapper
    _INSTRUMENTATION_STATE["flask"] = object()
    _GENERIC_INSTRUMENTATION_STATE["redis"] = object()

    _uninstrument_all()  # should not raise

    assert len(_INSTRUMENTATION_STATE) == 0
    assert len(_GENERIC_INSTRUMENTATION_STATE) == 0
