"""Apply GenAI OTel env var defaults from Traceable config before instrumentation."""

from __future__ import annotations

import os

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

_OTEL_GENAI_CAPTURE_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_OTEL_SEMCONV_STABILITY_VAR = "OTEL_SEMCONV_STABILITY_OPT_IN"
_GENAI_EXPERIMENTAL_VALUE = "gen_ai_latest_experimental"
_NO_CONTENT_VALUE = "NO_CONTENT"

_applied: bool = False


def maybe_set_genai_payload_capture_env_vars() -> None:
    """Sync OTEL payload-capture env vars with the resolved Traceable config.

    Must be called before any GenAI instrumentation wrapper evaluates
    should_capture_content_on_spans_in_experimental_mode(), because the OTel
    semconv stability class caches its mode on first access.  We also patch the
    cache directly to handle the case where OTel initialised before this runs.

    Precedence differs by direction, because payload capture is a privacy
    control and "false" must always mean false:
      - Disabled (``payload_capture_enabled`` resolves to False, whether via
        explicit config or the untouched default): force capture off,
        overwriting any pre-existing OTEL_* env vars and the semconv stability
        cache. A deployment that pre-sets these vars (e.g. via a shared base
        image or another OTel auto-instrumentation layer) must not be able to
        resurrect capture.
      - Enabled: only supply defaults. If the user already set either OTEL_*
        var, leave both alone.
    """
    global _applied  # pylint: disable=global-statement
    if _applied:
        return

    if not Config().config.gen_ai.payload_capture_enabled.value:
        _force_disable_payload_capture()
        _applied = True
        return

    capture_var_set = _OTEL_GENAI_CAPTURE_VAR in os.environ
    semconv_var_set = _OTEL_SEMCONV_STABILITY_VAR in os.environ
    if capture_var_set or semconv_var_set:
        logger.debug(
            "GenAI: OTEL payload capture env vars already set; leaving them unchanged."
        )
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


def _force_disable_payload_capture() -> None:
    """Force GenAI content capture off, overwriting any pre-existing OTel env vars.

    Sets the capture-mode env var to NO_CONTENT (the value every OTel GenAI
    instrumentation treats as "do not capture", regardless of which semconv
    stability mode it ends up in) and patches the cached semconv stability
    mode for the GEN_AI signal to DEFAULT (non-experimental), so instrumentation
    that only checks "is experimental mode" also sees capture as off. Both are
    forced unconditionally: this is the disable direction of a privacy control,
    so config wins over whatever the deployment environment set.
    """
    os.environ[_OTEL_GENAI_CAPTURE_VAR] = _NO_CONTENT_VALUE
    logger.debug(
        "GenAI: payload_capture_enabled=False; forcing %s=%s regardless of pre-existing env vars.",
        _OTEL_GENAI_CAPTURE_VAR,
        _NO_CONTENT_VALUE,
    )

    try:
        from opentelemetry.instrumentation._semconv import (  # pylint: disable=import-outside-toplevel
            _OpenTelemetrySemanticConventionStability,
            _OpenTelemetryStabilitySignalType,
            _StabilityMode,
        )
        with _OpenTelemetrySemanticConventionStability._lock:  # pylint: disable=protected-access
            _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING[  # pylint: disable=protected-access
                _OpenTelemetryStabilitySignalType.GEN_AI
            ] = _StabilityMode.DEFAULT
            _OpenTelemetrySemanticConventionStability._initialized = True  # pylint: disable=protected-access
        logger.debug("GenAI: patched OTel semconv stability cache to DEFAULT (non-experimental) for GEN_AI.")
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("GenAI: could not patch OTel semconv stability cache: %s", err)
