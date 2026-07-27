# Current-Span Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable Harness SDK helpers that write dynamic attributes to the current recording OpenTelemetry span.

**Architecture:** A small public module resolves `trace.get_current_span()`, no-ops when it is not recording, and uses the public `Span.set_attribute()` API. Package-root exports provide a stable customer surface. No processor, baggage, propagation, or configuration changes.

**Tech Stack:** Python 3.10+, OpenTelemetry API 1.41.1, pytest.

## Global Constraints

- Affect only the active recording span at call time.
- Accept arbitrary names with OpenTelemetry-compatible attribute values.
- Preserve native OpenTelemetry value types.
- Return `None`.
- No active recording span must be a silent no-op.
- Do not enrich descendants or propagate attributes.
- Do not change processors, plugins, configuration, propagators, or dependencies.
- Do not open a pull request.

---

### Task 1: Public Current-Span Helpers

**Files:**
- Create: `test/span_enrichment_test.py`
- Create: `src/harness_sdk/span_enrichment.py`
- Modify: `src/harness_sdk/__init__.py`

**Interfaces:**
- Produces: `set_span_attribute(key: str, value: AttributeValue) -> None`
- Produces: `set_span_attributes(attributes: Mapping[str, AttributeValue]) -> None`
- Produces: package-root imports for both helpers.

- [ ] **Step 1: Write failing public API tests**

```python
import asyncio

from opentelemetry.sdk.trace import TracerProvider

from harness_sdk import set_span_attribute, set_span_attributes


def _tracer():
    return TracerProvider().get_tracer(__name__)


def test_set_span_attribute_updates_current_recording_span():
    with _tracer().start_as_current_span("parent") as span:
        set_span_attribute("request.client.name", "acme")
        assert span.attributes["request.client.name"] == "acme"


def test_set_span_attributes_preserves_supported_types():
    attributes = {
        "string": "value",
        "boolean": True,
        "integer": 7,
        "float": 1.5,
        "sequence": ("a", "b"),
    }
    with _tracer().start_as_current_span("parent") as span:
        set_span_attributes(attributes)
        assert dict(span.attributes) == attributes


def test_last_write_wins():
    with _tracer().start_as_current_span("parent") as span:
        set_span_attribute("custom.key", "first")
        set_span_attribute("custom.key", "second")
        assert span.attributes["custom.key"] == "second"


def test_no_active_recording_span_is_noop():
    assert set_span_attribute("custom.key", "value") is None
    assert set_span_attributes({"custom.key": "value"}) is None


def test_empty_mapping_is_noop():
    with _tracer().start_as_current_span("parent") as span:
        assert set_span_attributes({}) is None
        assert not span.attributes


def test_attributes_do_not_inherit_to_child_span():
    tracer = _tracer()
    with tracer.start_as_current_span("parent") as parent:
        set_span_attribute("custom.key", "parent")
        with tracer.start_as_current_span("child") as child:
            assert "custom.key" not in child.attributes
        assert parent.attributes["custom.key"] == "parent"


def test_async_code_updates_active_span():
    async def enrich():
        set_span_attribute("agent.action.type", "search")

    with _tracer().start_as_current_span("parent") as span:
        asyncio.run(enrich())
        assert span.attributes["agent.action.type"] == "search"
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
PYTHONPATH=src /Users/shreyas/code/wd/aicm-agent-sdk/otel-python-sdk/.venv/bin/python \
  -m pytest test/span_enrichment_test.py -q
```

Expected: collection fails because package-root helpers do not exist.

- [ ] **Step 3: Implement minimal helpers**

```python
"""Public helpers for enriching the current OpenTelemetry span."""

from typing import Mapping

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue


def set_span_attribute(key: str, value: AttributeValue) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, value)


def set_span_attributes(attributes: Mapping[str, AttributeValue]) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)
```

Update `src/harness_sdk/__init__.py`:

```python
from harness_sdk.agent import Agent
from harness_sdk.span_enrichment import set_span_attribute, set_span_attributes

__all__ = ["Agent", "set_span_attribute", "set_span_attributes"]
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=src /Users/shreyas/code/wd/aicm-agent-sdk/otel-python-sdk/.venv/bin/python \
  -m pytest test/span_enrichment_test.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Run existing span-attribute tests**

Run:

```bash
PYTHONPATH=src /Users/shreyas/code/wd/aicm-agent-sdk/otel-python-sdk/.venv/bin/python \
  -m pytest test/span_attributes_processor_test.py -q
```

Expected: existing tests pass unchanged.

### Task 2: Customer Documentation and Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: package-root `set_span_attribute` and `set_span_attributes`.
- Produces: customer usage, scope, timing, and value constraints.

- [ ] **Step 1: Add README usage**

Document:

```python
from harness_sdk import set_span_attributes

set_span_attributes({
    "request.client.name": client_name,
    "agent.action.type": action_type,
})
```

State that the helper:

- must run while the target span is active;
- changes only that recording span;
- silently no-ops without an active recording span;
- does not affect child spans or downstream services;
- accepts OpenTelemetry-compatible values;
- uses last-write-wins;
- cannot influence head sampling.

- [ ] **Step 2: Run focused test set**

Run:

```bash
PYTHONPATH=src /Users/shreyas/code/wd/aicm-agent-sdk/otel-python-sdk/.venv/bin/python \
  -m pytest \
  test/span_enrichment_test.py \
  test/span_attributes_processor_test.py \
  test/test_plugin_loader.py \
  test/plugins/builtin/test_pipeline.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run static checks**

Run:

```bash
git diff --check
python -m compileall -q src/harness_sdk test/span_enrichment_test.py
```

Expected: both commands exit successfully.

- [ ] **Step 4: Inspect final scope**

Run:

```bash
git status --short
git diff --stat
```

Expected changes only:

- `README.md`
- `src/harness_sdk/__init__.py`
- `src/harness_sdk/span_enrichment.py`
- `test/span_enrichment_test.py`
- approved design and implementation-plan documents.
