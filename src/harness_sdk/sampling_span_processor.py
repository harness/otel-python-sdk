"""Backwards-compatibility shim — use db_control_span_processor.DbControlSpanProcessor."""
from harness_sdk.db_control_span_processor import DbControlSpanProcessor as SamplingSpanProcessor

__all__ = ["SamplingSpanProcessor"]
