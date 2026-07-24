'''Traceable wrapper around OpenTelemetry MCP instrumentation (Model Context Protocol).'''
from opentelemetry.instrumentation.mcp import McpInstrumentor

from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.instrumentation import BaseInstrumentorWrapper
from harness_sdk.instrumentation.mcp.gen_ai_mirror import (
    apply_gen_ai_env_for_mcp,
    mcp_instrumentation_get_tracer_patched,
)

logger = get_custom_logger(__name__)


class McpInstrumentorWrapper(McpInstrumentor, BaseInstrumentorWrapper):
    """
    Instruments the MCP Python SDK with GenAI-aware configuration.

    MCP instrumentation is opt-in via ``HARNESS_ENABLE_AI_MCP`` (enforced by the
    Agent before this wrapper is loaded). Payload capture mirrors
    ``TRACELOOP_TRACE_CONTENT`` for FastMCP based on
    ``HARNESS_GEN_AI_PAYLOAD_CAPTURE_ENABLED``; tool spans additionally record
    ``gen_ai.operation.name``, ``gen_ai.system``, ``gen_ai.tool.name``, ``mcp.method.name``,
    and opt-in ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``.
    """

    def __init__(self):
        McpInstrumentor.__init__(self)
        BaseInstrumentorWrapper.__init__(self)

    def _instrument(self, **kwargs) -> None:
        apply_gen_ai_env_for_mcp()

        with mcp_instrumentation_get_tracer_patched():
            super()._instrument(**kwargs)
