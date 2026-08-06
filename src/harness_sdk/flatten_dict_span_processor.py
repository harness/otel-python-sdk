"""Span processor that expands dict span attributes into dot-notation keys.

``set_span_attribute("agent", {"action": "generate"})`` parks the dict in
``flatten_dict_registry`` instead of handing it to OTel, which would reject it.
This processor drains the registry at ``on_end`` and writes
``agent.action=generate`` so the backend gets individually queryable
attributes instead of an opaque JSON blob.

It must be the outermost processor: downstream scrubbing and exclusion logic
matches on attribute keys, so the flattened keys have to exist before those
run. Mutating the ended span works the same way ``GenAiPayloadScrubSpanProcessor``
relies on: ``ReadableSpan.attributes`` is a read-only view over the attribute
store the concrete SDK ``Span`` still owns at ``on_end`` time.
"""
import json
from typing import Mapping

from opentelemetry.sdk.trace import SpanProcessor

from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.flatten_dict_registry import (
    get_registry,
    get_flatten_max_depth,
    get_flatten_max_leaves,
    is_raw_json_enabled,
)

logger = get_custom_logger(__name__)

_SCALAR_TYPES = (bool, int, float, str)


def _is_scalar(value):
    return isinstance(value, _SCALAR_TYPES)


def _scalar_kind(value):
    # bool is a subclass of int, but OTel treats them as distinct array types.
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def _to_json(value):
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _sequence_leaf(value):
    """Homogeneous scalar sequences stay arrays; anything else becomes JSON."""
    items = tuple(value)
    if not items:
        return items
    kinds = {_scalar_kind(item) for item in items if _is_scalar(item)}
    if len(kinds) == 1 and all(_is_scalar(item) for item in items):
        return items
    return _to_json(value)


def _leaf_value(value):
    """Convert a non-mapping value to something OTel accepts, or None to skip."""
    if value is None:
        return None
    if _is_scalar(value):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return _sequence_leaf(value)
    return str(value)


def _collect(prefix, mapping, depth, flattened, max_depth, max_leaves):
    """Walk ``mapping`` into ``flattened``; returns False once the cap is hit."""
    for key, value in mapping.items():
        if len(flattened) >= max_leaves:
            return False
        flat_key = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            if depth < max_depth:
                if not _collect(flat_key, value, depth + 1, flattened, max_depth, max_leaves):
                    return False
            else:
                flattened[flat_key] = _to_json(value)
            continue
        leaf = _leaf_value(value)
        if leaf is not None:
            flattened[flat_key] = leaf
    return True


class FlattenDictSpanProcessor(SpanProcessor):
    """Flattens registered dict attributes onto the span before export."""

    def __init__(self, processor):
        self._processor = processor

    def on_start(self, span, parent_context=None):
        self._processor.on_start(span, parent_context)

    def on_end(self, span):
        pending = get_registry().pop(span)
        if pending:
            try:
                self._flatten(span, pending)
            except Exception as err:  # pylint: disable=W0703
                logger.debug(
                    "Flatten: failed to flatten dict attributes on span %s: %s",
                    getattr(span, "name", None),
                    err,
                )
        self._processor.on_end(span)

    @staticmethod
    def _flatten(span, pending):
        attributes = getattr(span, "_attributes", None)
        if attributes is None:
            return
        raw_json = is_raw_json_enabled()
        max_depth = get_flatten_max_depth()
        max_leaves = get_flatten_max_leaves()
        for root_key, value in pending.items():
            flattened = {}
            if not _collect(root_key, value, 1, flattened, max_depth, max_leaves):
                logger.debug(
                    "Flatten: dict attribute %r on span %s exceeded %s leaf "
                    "attributes; remaining entries dropped",
                    root_key,
                    getattr(span, "name", None),
                    max_leaves,
                )
            for flat_key, leaf in flattened.items():
                # An explicitly set attribute always wins over a flattened one.
                if flat_key in attributes:
                    continue
                attributes[flat_key] = leaf
            if raw_json and root_key not in attributes:
                attributes[root_key] = _to_json(value)

    def force_flush(self, timeout_millis=30000):
        return self._processor.force_flush(timeout_millis)

    def shutdown(self):
        return self._processor.shutdown()
