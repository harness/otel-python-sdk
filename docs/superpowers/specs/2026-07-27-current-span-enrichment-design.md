# Current-Span Enrichment Design

## Goal

Give customers a stable Harness SDK API for adding dynamic custom attributes to the currently active OpenTelemetry span, including spans created by automatic or internal instrumentation.

Example:

```python
from harness_sdk import set_span_attributes

set_span_attributes({
    "request.client.name": client_name,
    "agent.action.type": action_type,
})
```

## Scope

The API affects only the active recording span at call time.

It does not:

- enrich child or future spans;
- store values in OpenTelemetry Context or Baggage;
- propagate values to downstream services;
- install a `SpanProcessor`;
- influence head sampling;
- expose an internal Harness span object.

These exclusions are intentional. Scoped descendant enrichment and distributed enrichment have different lifecycle, privacy, trust, and propagation requirements and should be separate future designs.

## Public API

Export two functions from the `harness_sdk` package root:

```python
def set_span_attribute(key: str, value: AttributeValue) -> None: ...

def set_span_attributes(
    attributes: Mapping[str, AttributeValue],
) -> None: ...
```

`AttributeValue` uses OpenTelemetry's supported attribute values: strings, booleans, integers, floats, and homogeneous sequences of those scalar types.

Both functions:

- accept arbitrary customer-defined attribute names;
- preserve native OpenTelemetry value types without string coercion;
- return `None`;
- silently do nothing when no recording span is active;
- rely on OpenTelemetry for attribute limits and value validation.

The plural helper applies every mapping entry to the same span snapshot. The singular helper provides the common one-attribute form.

## Architecture

Add a focused `harness_sdk.span_enrichment` module. It obtains the current span through the public OpenTelemetry API:

```python
span = trace.get_current_span()
if not span.is_recording():
    return
```

It then calls the span's public `set_attribute()` method. No private span fields or SDK-owned span registry are needed.

The package root re-exports both helpers so customers do not depend on the module layout.

## Data Flow

1. Harness or third-party instrumentation starts a span and makes it current.
2. Customer code computes a runtime value.
3. Customer calls a Harness enrichment helper.
4. The helper resolves the current span from OpenTelemetry context.
5. If the span is recording, the helper writes the requested attribute.
6. Existing processors and exporters receive the enriched span when it ends.

OpenTelemetry context resolution keeps the operation correct for normal synchronous and `asyncio` execution. A new raw thread does not automatically inherit the caller's current span.

## Precedence and Timing

OpenTelemetry uses last-write-wins behavior for duplicate span attribute keys.

- A customer call overwrites a value already present under the same key.
- Instrumentation writing the same key later can overwrite the customer value.
- Customers should prefer their own namespace and avoid semantic-convention or Harness-owned keys unless replacement is deliberate.
- Calls after a span has ended have no effect.
- Attributes added after span creation cannot affect the sampler because head sampling has already run.

## Failure Behavior

Absence of an active recording span is expected and is a silent no-op. This keeps telemetry calls from affecting application control flow.

The helpers do not coerce arbitrary Python objects into strings. Unsupported values follow standard OpenTelemetry validation behavior. The SDK does not add hidden truncation, serialization, propagation, or logging side effects.

## Java Reference

The Transposit implementation solves a different problem. It stores an allow-listed value in Baggage and copies it onto every newly started span through `SpanProcessor.onStart()`. Its useful general lessons are:

- use public OpenTelemetry APIs;
- mutate spans before export;
- keep enrichment work synchronous and cheap;
- define lifecycle and propagation semantics explicitly.

Its Baggage and processor mechanics do not fit this design because this API targets only the already-active span. Copying Baggage would broaden scope to descendants, introduce inbound trust and privacy risks, and potentially propagate values over the network.

## Files

- Create `src/harness_sdk/span_enrichment.py` for the two helpers.
- Modify `src/harness_sdk/__init__.py` to re-export them.
- Create `test/span_enrichment_test.py` for focused unit tests.
- Modify `README.md` with usage and lifecycle constraints.

No configuration schema, environment variables, processors, plugins, propagators, or dependencies change.

## Tests

Unit tests will verify:

- one custom attribute is written to the active recording span;
- multiple attributes are written in one call;
- strings, booleans, integers, floats, and supported sequences preserve their types;
- repeated writes use last-write-wins behavior;
- calls without an active recording span are no-ops;
- empty mappings are no-ops;
- attributes do not automatically appear on a child span;
- normal async context resolves and updates the active span;
- package-root imports expose both public helpers.

Tests will use a standalone `TracerProvider` and scoped current spans without replacing OpenTelemetry's process-global provider.

## Documentation

README documentation will state:

- call the helper while the desired span is active;
- only that span is changed;
- no descendant or network propagation occurs;
- no active span means no-op;
- values must be valid OpenTelemetry attribute values;
- later writes to the same key win;
- enrichment cannot affect head sampling.

## Acceptance Criteria

- Customers can enrich an internally created active span without accessing Harness internals.
- Both singular and bulk helpers are importable from `harness_sdk`.
- Existing static span attributes and processor/export pipelines remain unchanged.
- Enrichment remains local to the current recording span.
- No-active-span calls do not raise.
- Focused unit tests pass.
