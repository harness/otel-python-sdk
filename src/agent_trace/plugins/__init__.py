"""Plugin discovery and registration for control and observability extensions."""

from agent_trace.plugins.control import ControlResult, ControlRegistry, get_control_registry
from agent_trace.plugins.loader import load_control_plugins, load_observability_plugins

__all__ = [
    "ControlResult",
    "ControlRegistry",
    "get_control_registry",
    "load_control_plugins",
    "load_observability_plugins",
]
