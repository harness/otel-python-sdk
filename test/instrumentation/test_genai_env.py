"""Tests for GenAI OTel env var forcing (harness_sdk.instrumentation.genai_env).

Focus: config-disabled payload capture must force OTel content-capture off,
overwriting any pre-existing OTEL_* env vars and the OTel semconv stability
cache, while config-enabled must only supply defaults and never clobber env
vars the user already set.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from harness_sdk.instrumentation import genai_env as genai_env_mod
from harness_sdk.instrumentation.genai_env import (
    _OTEL_GENAI_CAPTURE_VAR,
    _OTEL_SEMCONV_STABILITY_VAR,
    maybe_set_genai_payload_capture_env_vars,
)
from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
    _OpenTelemetryStabilitySignalType,
    _StabilityMode,
)


@pytest.fixture(autouse=True)
def _isolate_genai_env_state():
    """Reset the module-level `_applied` flag and the OTel semconv cache.

    Both are process-global mutable state that would otherwise leak between
    tests (and between test modules, since other instrumentation suites also
    call `maybe_set_genai_payload_capture_env_vars`).
    """
    genai_env_mod._applied = False
    saved_capture_var = os.environ.pop(_OTEL_GENAI_CAPTURE_VAR, None)
    saved_semconv_var = os.environ.pop(_OTEL_SEMCONV_STABILITY_VAR, None)
    saved_mapping = dict(
        _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING  # pylint: disable=protected-access
    )
    saved_initialized = _OpenTelemetrySemanticConventionStability._initialized  # pylint: disable=protected-access

    yield

    genai_env_mod._applied = False
    for key, value in (
        (_OTEL_GENAI_CAPTURE_VAR, saved_capture_var),
        (_OTEL_SEMCONV_STABILITY_VAR, saved_semconv_var),
    ):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING = saved_mapping  # pylint: disable=protected-access
    _OpenTelemetrySemanticConventionStability._initialized = saved_initialized  # pylint: disable=protected-access


def _mock_config(payload_capture_enabled: bool):
    gen = MagicMock()
    gen.payload_capture_enabled.value = payload_capture_enabled
    cfg = MagicMock()
    cfg.gen_ai = gen
    root = MagicMock()
    root.config = cfg
    return root


def _gen_ai_mapping():
    return _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING.get(  # pylint: disable=protected-access
        _OpenTelemetryStabilitySignalType.GEN_AI
    )


def test_disable_forces_no_content_overwriting_preset_env_vars():
    os.environ[_OTEL_GENAI_CAPTURE_VAR] = "SPAN_ONLY"
    os.environ[_OTEL_SEMCONV_STABILITY_VAR] = "gen_ai_latest_experimental"

    with patch(
        "harness_sdk.instrumentation.genai_env.Config",
        return_value=_mock_config(payload_capture_enabled=False),
    ):
        maybe_set_genai_payload_capture_env_vars()

    assert os.environ.get(_OTEL_GENAI_CAPTURE_VAR) == "NO_CONTENT"


def test_disable_patches_semconv_cache_to_default():
    _OpenTelemetrySemanticConventionStability._initialized = True  # pylint: disable=protected-access
    _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING[  # pylint: disable=protected-access
        _OpenTelemetryStabilitySignalType.GEN_AI
    ] = _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL

    with patch(
        "harness_sdk.instrumentation.genai_env.Config",
        return_value=_mock_config(payload_capture_enabled=False),
    ):
        maybe_set_genai_payload_capture_env_vars()

    assert _gen_ai_mapping() == _StabilityMode.DEFAULT
    assert _OpenTelemetrySemanticConventionStability._initialized is True  # pylint: disable=protected-access


def test_disable_with_no_preexisting_env_vars_still_forces_no_content():
    with patch(
        "harness_sdk.instrumentation.genai_env.Config",
        return_value=_mock_config(payload_capture_enabled=False),
    ):
        maybe_set_genai_payload_capture_env_vars()

    assert os.environ.get(_OTEL_GENAI_CAPTURE_VAR) == "NO_CONTENT"
    assert _gen_ai_mapping() == _StabilityMode.DEFAULT


def test_enable_respects_preexisting_env_vars():
    os.environ[_OTEL_GENAI_CAPTURE_VAR] = "EVENT_ONLY"
    os.environ[_OTEL_SEMCONV_STABILITY_VAR] = "some_user_value"

    with patch(
        "harness_sdk.instrumentation.genai_env.Config",
        return_value=_mock_config(payload_capture_enabled=True),
    ):
        maybe_set_genai_payload_capture_env_vars()

    assert os.environ.get(_OTEL_GENAI_CAPTURE_VAR) == "EVENT_ONLY"
    assert os.environ.get(_OTEL_SEMCONV_STABILITY_VAR) == "some_user_value"


def test_enable_sets_defaults_when_absent():
    with patch(
        "harness_sdk.instrumentation.genai_env.Config",
        return_value=_mock_config(payload_capture_enabled=True),
    ):
        maybe_set_genai_payload_capture_env_vars()

    assert os.environ.get(_OTEL_GENAI_CAPTURE_VAR) == "SPAN_ONLY"
    assert os.environ.get(_OTEL_SEMCONV_STABILITY_VAR) == "gen_ai_latest_experimental"
    assert _gen_ai_mapping() == _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL


def test_applied_flag_short_circuits_subsequent_calls():
    mock_config_cls = MagicMock(return_value=_mock_config(payload_capture_enabled=False))
    with patch("harness_sdk.instrumentation.genai_env.Config", mock_config_cls):
        maybe_set_genai_payload_capture_env_vars()
        maybe_set_genai_payload_capture_env_vars()

    assert mock_config_cls.call_count == 1
