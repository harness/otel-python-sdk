"""Apply GenAI OTel env var defaults from Traceable config before instrumentation."""

from __future__ import annotations

import os

from agent_trace.config.config import Config
from agent_trace.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

_OTEL_GENAI_CAPTURE_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_OTEL_SEMCONV_STABILITY_VAR = "OTEL_SEMCONV_STABILITY_OPT_IN"
_GENAI_EXPERIMENTAL_VALUE = "gen_ai_latest_experimental"

_applied: bool = False


def maybe_set_genai_payload_capture_env_vars() -> None:
    """Set OTEL payload-capture env vars from Traceable config if not already set by the user.

    Must be called before any GenAI instrumentation wrapper evaluates
    should_capture_content_on_spans_in_experimental_mode(), because the OTel
    semconv stability class caches its mode on first access.  We also patch the
    cache directly to handle the case where OTel initialised before this runs.
    """
    global _applied  # pylint: disable=global-statement
    if _applied:
        return

    capture_var_set = _OTEL_GENAI_CAPTURE_VAR in os.environ
    semconv_var_set = _OTEL_SEMCONV_STABILITY_VAR in os.environ
    if capture_var_set or semconv_var_set:
        logger.debug(
            "GenAI: OTEL payload capture env vars already set; leaving them unchanged."
        )
        _applied = True
        return

    if not Config().config.gen_ai.payload_capture_enabled.value:
        _applied = True
        return

    os.environ[_OTEL_SEMCONV_STABILITY_VAR] = _GENAI_EXPERIMENTAL_VALUE
    os.environ[_OTEL_GENAI_CAPTURE_VAR] = "SPAN_ONLY"
    logger.debug(
        "GenAI: payload_capture_enabled=True; set %s=%s and %s=%s",
        _OTEL_SEMCONV_STABILITY_VAR,
        _GENAI_EXPERIMENTAL_VALUE,
        _OTEL_GENAI_CAPTURE_VAR,
        "SPAN_ONLY",
    )

    # The semconv stability class caches its mode on first access behind a _initialized flag.
    # Patch the cache directly so the env var takes effect even if OTel initialized early.
    try:
        from opentelemetry.instrumentation._semconv import (  # pylint: disable=import-outside-toplevel
            _OpenTelemetrySemanticConventionStability,
            _OpenTelemetryStabilitySignalType,
            _StabilityMode,
        )
        with _OpenTelemetrySemanticConventionStability._lock:  # pylint: disable=protected-access
            _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING[  # pylint: disable=protected-access
                _OpenTelemetryStabilitySignalType.GEN_AI
            ] = _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL
            _OpenTelemetrySemanticConventionStability._initialized = True  # pylint: disable=protected-access
        logger.debug("GenAI: patched OTel semconv stability cache for GEN_AI experimental mode.")
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("GenAI: could not patch OTel semconv stability cache: %s", err)

    _applied = True
