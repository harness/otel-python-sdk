"""Tests for :class:`agent_trace.instrumentation.mcp.McpInstrumentorWrapper`."""
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.instrumentation.mcp import McpInstrumentor

from agent_trace.instrumentation.mcp import McpInstrumentorWrapper


@pytest.fixture
def gen_ai_enabled():
    gen = MagicMock()
    gen.enabled.value = True
    gen.payload_capture_enabled.value = True
    gen.payload_evaluation_enabled.value = True
    root = MagicMock()
    root.config.gen_ai = gen
    with patch("agent_trace.instrumentation.mcp.Config") as mock_cfg:
        mock_cfg.return_value = root
        yield gen


def test_mcp_wrapper_skips_when_gen_ai_disabled():
    gen = MagicMock()
    gen.enabled.value = False
    root = MagicMock()
    root.config.gen_ai = gen
    with patch("agent_trace.instrumentation.mcp.Config") as mock_cfg:
        mock_cfg.return_value = root
        with patch("agent_trace.instrumentation.mcp.apply_gen_ai_env_for_mcp") as mock_apply:
            with patch.object(McpInstrumentor, "_instrument") as parent_instr:
                w = McpInstrumentorWrapper()
                w._instrument()
    mock_apply.assert_not_called()
    parent_instr.assert_not_called()


def test_mcp_wrapper_restores_get_tracer_after_instrument(gen_ai_enabled):
    import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst

    orig_get_tracer = mcp_inst.get_tracer
    with patch.object(McpInstrumentor, "_instrument", autospec=True):
        w = McpInstrumentorWrapper()
        w._instrument(tracer_provider=None)
    assert mcp_inst.get_tracer is orig_get_tracer


def test_mcp_wrapper_calls_apply_env_and_parent_when_gen_ai_on(gen_ai_enabled):
    import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst

    orig_get_tracer = mcp_inst.get_tracer
    with patch("agent_trace.instrumentation.mcp.apply_gen_ai_env_for_mcp") as mock_apply:
        with patch.object(McpInstrumentor, "_instrument", autospec=True) as parent_instr:
            w = McpInstrumentorWrapper()
            w._instrument(tracer_provider="tp")
    mock_apply.assert_called_once()
    parent_instr.assert_called_once()
    assert mcp_inst.get_tracer is orig_get_tracer
    assert parent_instr.call_args.kwargs.get("tracer_provider") == "tp"


def test_mcp_wrapper_restores_get_tracer_if_parent_raises(gen_ai_enabled):
    import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst

    orig_get_tracer = mcp_inst.get_tracer

    def boom(**_kwargs):
        raise RuntimeError("instrument failed")

    with patch.object(McpInstrumentor, "_instrument", side_effect=boom):
        w = McpInstrumentorWrapper()
        with pytest.raises(RuntimeError, match="instrument failed"):
            w._instrument()
    assert mcp_inst.get_tracer is orig_get_tracer
