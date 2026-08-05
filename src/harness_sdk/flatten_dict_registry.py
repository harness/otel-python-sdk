"""Thread-safe hand-off of dict span attributes from the hot path to span end.

``set_span_attribute("agent", {...})`` cannot go through
``span.set_attribute``: OTel rejects mapping values outright. Serializing on
the caller's thread would put JSON encoding on the application hot path, so
dict values are parked here untouched and flattened later by
``FlattenDictSpanProcessor.on_end``.

Entries are keyed by span context rather than object identity because
``Span.end()`` hands ``on_end`` a fresh ``ReadableSpan`` snapshot, not the
recording ``Span`` the enrichment helper saw.
"""
import threading
from collections import OrderedDict

from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.env import get_env_value, is_env_flag_enabled

logger = get_custom_logger(__name__)

FLATTEN_ENABLED_ENV = "SPAN_ATTRIBUTE_FLATTEN_ENABLED"
FLATTEN_RAW_JSON_ENV = "SPAN_ATTRIBUTE_FLATTEN_RAW_JSON"

# Bound on spans holding pending dicts. A span that is never ended would
# otherwise leak its entry forever; oldest entries are evicted instead.
_MAX_TRACKED_SPANS = 2048


def is_flatten_enabled():
    """Dict flattening is on by default; only explicit ``false`` disables it."""
    value = get_env_value(FLATTEN_ENABLED_ENV)
    if value is None:
        return True
    return value.strip().lower() != "false"


def is_raw_json_enabled():
    """Opt in to additionally keeping the original key as a JSON string."""
    return is_env_flag_enabled(FLATTEN_RAW_JSON_ENV)


def _span_key(span):
    get_context = getattr(span, "get_span_context", None)
    if get_context is None:
        return None
    context = get_context()
    if context is None or not context.trace_id:
        return None
    return (context.trace_id, context.span_id)


class FlattenDictRegistry:
    """Maps span identity to the dict attributes awaiting flattening."""

    def __init__(self, max_tracked_spans=_MAX_TRACKED_SPANS):
        self._lock = threading.Lock()
        self._pending = OrderedDict()
        self._max_tracked_spans = max_tracked_spans

    def register(self, span, key, value):
        """Park ``value`` under ``key`` for ``span``; last write wins."""
        span_key = _span_key(span)
        if span_key is None:
            return
        with self._lock:
            attributes = self._pending.get(span_key)
            if attributes is None:
                attributes = OrderedDict()
                self._pending[span_key] = attributes
            attributes[key] = value
            self._pending.move_to_end(span_key)
            while len(self._pending) > self._max_tracked_spans:
                evicted, _ = self._pending.popitem(last=False)
                logger.debug(
                    "Flatten: evicted pending dict attributes for span %s "
                    "(registry limit %s reached)",
                    evicted,
                    self._max_tracked_spans,
                )

    def pop(self, span):
        """Remove and return the pending dict attributes for ``span``."""
        span_key = _span_key(span)
        if span_key is None:
            return {}
        with self._lock:
            return self._pending.pop(span_key, {})

    def clear(self):
        with self._lock:
            self._pending.clear()


_REGISTRY = FlattenDictRegistry()


def get_registry():
    return _REGISTRY
