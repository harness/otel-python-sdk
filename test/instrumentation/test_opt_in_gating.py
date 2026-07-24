"""Opt-in instrumentation gating: default-off, one API toggle, per-provider AI toggles."""
import os

import pytest

from harness_sdk.agent import Agent
from harness_sdk.instrumentation.genai_env import maybe_set_genai_payload_capture_env_vars
from harness_sdk.instrumentation.instrumentation_definitions import (
    _uninstrument_all,
    is_api_instrumentation_enabled,
    any_ai_provider_enabled,
    is_library_enabled,
    API_LIBRARIES,
    AI_LIBRARY_ENV_FLAGS,
    OPENAI_KEY,
    REQUESTS_KEY,
)


# --------------------------------------------------------------------------- #
# Pure categorization helpers
# --------------------------------------------------------------------------- #
def test_default_off_nothing_enabled():
    assert is_api_instrumentation_enabled() is False
    assert any_ai_provider_enabled() is False
    for key in list(API_LIBRARIES) + list(AI_LIBRARY_ENV_FLAGS):
        assert is_library_enabled(key) is False


def test_api_flag_enables_all_non_ai_only():
    os.environ["HARNESS_ENABLE_API"] = "true"
    assert is_api_instrumentation_enabled() is True
    for key in API_LIBRARIES:
        assert is_library_enabled(key) is True
    for key in AI_LIBRARY_ENV_FLAGS:
        assert is_library_enabled(key) is False
    assert any_ai_provider_enabled() is False


@pytest.mark.parametrize("provider_key,flag", list(AI_LIBRARY_ENV_FLAGS.items()))
def test_ai_provider_enabled_independently(provider_key, flag):
    os.environ[flag] = "true"
    assert is_library_enabled(provider_key) is True
    assert any_ai_provider_enabled() is True
    for other_key, other_flag in AI_LIBRARY_ENV_FLAGS.items():
        if other_key != provider_key:
            assert is_library_enabled(other_key) is False
    assert is_api_instrumentation_enabled() is False
    for key in API_LIBRARIES:
        assert is_library_enabled(key) is False


def test_flag_requires_exact_true_value():
    os.environ["HARNESS_ENABLE_API"] = "1"
    assert is_api_instrumentation_enabled() is False
    os.environ["HARNESS_ENABLE_API"] = "yes"
    assert is_api_instrumentation_enabled() is False
    os.environ["HARNESS_ENABLE_API"] = "false"
    assert is_api_instrumentation_enabled() is False
    os.environ["HARNESS_ENABLE_API"] = "TRUE"
    assert is_api_instrumentation_enabled() is True
    os.environ["HARNESS_ENABLE_API"] = "  true  "
    assert is_api_instrumentation_enabled() is True


def test_ai_flags_have_no_legacy_aliases():
    os.environ["HA_ENABLE_AI_OPENAI"] = "true"
    os.environ["AT_ENABLE_AI_OPENAI"] = "true"
    os.environ["TA_ENABLE_AI_OPENAI"] = "true"
    try:
        assert is_library_enabled(OPENAI_KEY) is False
    finally:
        for legacy in ("AT_ENABLE_AI_OPENAI", "TA_ENABLE_AI_OPENAI"):
            os.environ.pop(legacy, None)


def test_api_flag_has_no_legacy_aliases():
    os.environ["HA_ENABLE_API"] = "true"
    os.environ["AT_ENABLE_API"] = "true"
    try:
        assert is_api_instrumentation_enabled() is False
    finally:
        os.environ.pop("AT_ENABLE_API", None)


# --------------------------------------------------------------------------- #
# Agent.instrument() gating integration
# --------------------------------------------------------------------------- #
def _build_agent():
    _uninstrument_all()
    maybe_set_genai_payload_capture_env_vars()
    ag = Agent()
    ag._init.init_trace_provider()
    return ag


def _record_instrumented(monkeypatch):
    instrumented = []
    monkeypatch.setattr(
        Agent,
        "_instrument",
        lambda self, library_key, app=None, auto_instrument=False: instrumented.append(library_key),
    )
    return instrumented


def test_instrument_default_off_instruments_nothing(monkeypatch):
    ag = _build_agent()
    instrumented = _record_instrumented(monkeypatch)
    contrib_calls = []
    monkeypatch.setattr(
        "harness_sdk.agent.instrument_supported_contrib_without_wrapper",
        lambda skip_libraries=None: contrib_calls.append(skip_libraries),
    )
    ag.instrument()
    assert instrumented == []
    assert contrib_calls == []


def test_instrument_single_ai_provider_only(monkeypatch):
    os.environ["HARNESS_ENABLE_AI_OPENAI"] = "true"
    ag = _build_agent()
    instrumented = _record_instrumented(monkeypatch)
    contrib_calls = []
    monkeypatch.setattr(
        "harness_sdk.agent.instrument_supported_contrib_without_wrapper",
        lambda skip_libraries=None: contrib_calls.append(skip_libraries),
    )
    ag.instrument()
    assert instrumented == [OPENAI_KEY]
    # API-only generic contrib must stay off when only an AI provider is enabled.
    assert contrib_calls == []


def test_instrument_api_flag_enables_api_and_generic_contrib(monkeypatch):
    os.environ["HARNESS_ENABLE_API"] = "true"
    ag = _build_agent()
    instrumented = _record_instrumented(monkeypatch)
    contrib_calls = []
    monkeypatch.setattr(
        "harness_sdk.agent.instrument_supported_contrib_without_wrapper",
        lambda skip_libraries=None: contrib_calls.append(skip_libraries),
    )
    ag.instrument()
    assert set(instrumented) == set(API_LIBRARIES)
    for key in AI_LIBRARY_ENV_FLAGS:
        assert key not in instrumented
    assert len(contrib_calls) == 1


def test_skip_libraries_overrides_enabled_category(monkeypatch):
    os.environ["HARNESS_ENABLE_API"] = "true"
    ag = _build_agent()
    instrumented = _record_instrumented(monkeypatch)
    monkeypatch.setattr(
        "harness_sdk.agent.instrument_supported_contrib_without_wrapper",
        lambda skip_libraries=None: None,
    )
    ag.instrument(skip_libraries=[REQUESTS_KEY])
    assert REQUESTS_KEY not in instrumented
    assert set(instrumented) == set(API_LIBRARIES) - {REQUESTS_KEY}
