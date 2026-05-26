"""Load control and observability plugins in config order via setuptools entry points."""
from __future__ import annotations

import traceback
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List

from opentelemetry import trace

from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.plugins.control import get_control_registry

logger = get_custom_logger(__name__)

CONTROL_ENTRY_GROUP = "harness_sdk_control_plugin"
OBSERVABILITY_ENTRY_GROUP = "harness_sdk_observability_plugin"


def _entry_points(group: str) -> list:
    try:
        eps = importlib_metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=group))
        return list(eps.get(group, []))
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("Unable to load entry points for %s: %s", group, err)
        return []


def _entry_points_by_name(group: str) -> Dict[str, Any]:
    return {entry_point.name: entry_point for entry_point in _entry_points(group)}


def _ordered_plugin_names(config: Any, plugin_type: str) -> List[str]:
    """Return plugin names in config/env order (only explicitly configured plugins)."""
    names = getattr(config, f"enabled_{plugin_type}_plugins", None)
    if not names:
        return []
    return list(names)


def _load_plugin(entry_point, config: Any):
    factory = entry_point.load()
    plugin = factory(config) if callable(factory) else factory
    plugin.on_init(config)
    return plugin


def load_control_plugins(config: Any) -> None:
    """Load and register control plugins in the order listed in config."""
    registry = get_control_registry()
    registry.clear()
    entry_points_by_name = _entry_points_by_name(CONTROL_ENTRY_GROUP)

    for name in _ordered_plugin_names(config, "control"):
        entry_point = entry_points_by_name.get(name)
        if entry_point is None:
            logger.warning(
                "Control plugin '%s' is configured but not installed "
                "(pip install the package that provides harness_sdk_control_plugin)",
                name,
            )
            continue
        try:
            plugin = _load_plugin(entry_point, config)
            registry.register(plugin)
            logger.info("Registered control plugin '%s'", name)
        except Exception as err:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load control plugin '%s': %s\n%s",
                name,
                err,
                traceback.format_exc(),
            )


def load_observability_plugins(config: Any) -> None:
    """Register span processors from observability plugins in config order."""
    entry_points_by_name = _entry_points_by_name(OBSERVABILITY_ENTRY_GROUP)
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        logger.warning("Tracer provider does not support add_span_processor")
        return

    for name in _ordered_plugin_names(config, "observability"):
        entry_point = entry_points_by_name.get(name)
        if entry_point is None:
            logger.warning(
                "Observability plugin '%s' is configured but not installed "
                "(pip install the package that provides harness_sdk_observability_plugin)",
                name,
            )
            continue
        try:
            plugin = _load_plugin(entry_point, config)
            for processor in plugin.create_span_processors(config):
                provider.add_span_processor(processor)
            logger.info("Registered observability plugin '%s'", name)
        except Exception as err:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load observability plugin '%s': %s\n%s",
                name,
                err,
                traceback.format_exc(),
            )
