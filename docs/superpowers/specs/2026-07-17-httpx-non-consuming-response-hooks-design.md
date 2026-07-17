# HTTPX Non-Consuming Response Hooks

## Problem

The HTTPX response hooks synchronously drain the transport response stream to
capture its body, then replace HTTPX's private `_httpcore_stream` state with a
custom replay iterator. This buffers streaming responses, depends on private
HTTPX internals, and can leave a partially consumed stream when draining fails.

## Design

HTTPX response hooks will capture response headers and status without reading
the response stream. They will pass `None` as the response body to the generic
response handler.

The response-draining helpers and replay stream will be removed. Request-body
capture remains unchanged.

## Compatibility

HTTPX spans will no longer contain `http.response.body`. Existing response
header and status attributes remain. This intentional telemetry reduction
preserves application correctness and true streaming behavior.

## Tests

- A synchronous response hook must not iterate or mutate its response stream.
- An asynchronous response hook must not iterate or mutate its response stream.
- Existing HTTPX integration tests must continue validating request capture,
  response headers, and status without expecting response-body capture.
- The complete unit suite must pass.

## Non-Goals

- Buffering, truncating, or sampling streaming response bodies.
- Extending span lifetime until application response consumption.
- Changing request-body capture or other instrumentations.
