# HTTPX Non-Consuming Response Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make synchronous and asynchronous HTTPX response hooks preserve headers and status telemetry without reading or replacing application response streams.

**Architecture:** Keep request handling unchanged. Both response paths extract only headers and call `generic_response_handler(headers, None, span)`; OpenTelemetry continues owning status capture. Remove response-body decoding, draining, and replay code once no production caller remains.

**Tech Stack:** Python 3.10+, pytest 7.4, pytest-asyncio, OpenTelemetry HTTPX instrumentation 0.62b1

## Global Constraints

- Apply the change to all HTTPX responses, not only compressed or streaming responses.
- Preserve HTTPX request-body capture.
- Preserve HTTPX response headers and status.
- Remove `http.response.body` from HTTPX spans.
- Never iterate or mutate synchronous or asynchronous response streams.
- Make no changes to other instrumentations.
- Keep the existing design commit `61dd1b4` unchanged.

---

### Task 1: Add response-stream regression tests

**Files:**
- Create: `test/instrumentation/httpx_client/test_httpx_hooks.py`

**Interfaces:**
- Consumes: `HTTPXClientInstrumentorWrapper.response_hook(span, request_info, response_info)` and `async_response_hook(span, request_info, response_info)`.
- Produces: focused regression coverage requiring response headers plus a `None` body while preserving the original `_httpcore_stream` object without iteration.

- [ ] **Step 1: Write the synchronous failing test**

```python
from types import SimpleNamespace

import pytest

from harness_sdk.instrumentation.httpx import HTTPXClientInstrumentorWrapper


class _SyncStream:
    def __init__(self):
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield b"response"


class _AsyncStream:
    def __init__(self):
        self.iterations = 0

    async def __aiter__(self):
        self.iterations += 1
        yield b"response"


def _wrapper_with_response_capture():
    wrapper = HTTPXClientInstrumentorWrapper()
    calls = []
    wrapper.generic_response_handler = (
        lambda headers, body, span: calls.append((headers, body, span))
    )
    return wrapper, calls


def test_response_hook_does_not_consume_or_replace_stream():
    wrapper, calls = _wrapper_with_response_capture()
    span = object()
    transport_stream = _SyncStream()
    response_stream = SimpleNamespace(_httpcore_stream=transport_stream)
    response_info = SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/plain"},
        stream=response_stream,
        extensions={},
    )

    wrapper.response_hook(span, None, response_info)

    assert calls == [({"content-type": "text/plain"}, None, span)]
    assert transport_stream.iterations == 0
    assert response_stream._httpcore_stream is transport_stream
```

- [ ] **Step 2: Write the asynchronous failing test**

```python
@pytest.mark.asyncio
async def test_async_response_hook_does_not_consume_or_replace_stream():
    wrapper, calls = _wrapper_with_response_capture()
    span = object()
    transport_stream = _AsyncStream()
    response_stream = SimpleNamespace(_httpcore_stream=transport_stream)
    response_info = SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/plain"},
        stream=response_stream,
        extensions={},
    )

    await wrapper.async_response_hook(span, None, response_info)

    assert calls == [({"content-type": "text/plain"}, None, span)]
    assert transport_stream.iterations == 0
    assert response_stream._httpcore_stream is transport_stream
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
python3 -m pytest -q \
  test/instrumentation/httpx_client/test_httpx_hooks.py
```

Expected: both tests fail because current hooks iterate `_httpcore_stream`, replace it, and pass captured bytes instead of `None`.

- [ ] **Step 4: Commit the regression tests**

```bash
git add test/instrumentation/httpx_client/test_httpx_hooks.py
git commit -m "test: cover non-consuming HTTPX response hooks"
```

### Task 2: Stop response capture and remove dead helpers

**Files:**
- Modify: `src/harness_sdk/instrumentation/httpx/__init__.py:8-73`
- Modify: `src/harness_sdk/instrumentation/httpx/utils.py:1-160`
- Delete: `test/instrumentation/httpx_client/test_httpx_utils.py`

**Interfaces:**
- Consumes: `headers_from_httpx(response_info.headers)` and `generic_response_handler(response_headers, response_body, span)`.
- Produces: `_process_response(span, response_info) -> None`, used by both sync and async response hooks, always forwarding `None` as `response_body`.

- [ ] **Step 1: Unify response processing without stream access**

Replace response-specific imports and methods in `src/harness_sdk/instrumentation/httpx/__init__.py` with:

```python
from harness_sdk.instrumentation.httpx.utils import (
    headers_from_httpx,
    read_request_body,
    url_from_request_info,
)

    def _process_response(self, span, response_info):
        headers = headers_from_httpx(response_info.headers)
        self.generic_response_handler(headers, None, span)

    async def async_response_hook(self, span, request_info, response_info):  # pylint: disable=unused-argument
        '''Capture async client response headers.'''
        self._process_response(span, response_info)
```

Remove `_process_response_async`. Update the synchronous response-hook docstring to say it captures response headers.

- [ ] **Step 2: Remove obsolete response helpers**

Delete from `src/harness_sdk/instrumentation/httpx/utils.py`:

```python
import gzip
import zlib

_content_encoding_chain
_apply_content_encoding_decodings
decode_response_body_for_capture
read_response_body
read_response_body_async
_ReplayAsyncStream
```

Keep `headers_from_httpx`, `url_from_request_info`, `_body_from_byte_stream`, and `read_request_body` unchanged. Delete `test/instrumentation/httpx_client/test_httpx_utils.py` because every test in it covers the removed response-only decoder.

- [ ] **Step 3: Run focused tests to verify GREEN**

Run:

```bash
python3 -m pytest -q \
  test/instrumentation/httpx_client/test_httpx_hooks.py
```

Expected: `2 passed`; neither stream reports iteration or replacement.

- [ ] **Step 4: Confirm response helpers have no references**

Run:

```bash
rg "decode_response_body_for_capture|read_response_body|read_response_body_async|_ReplayAsyncStream" src test
```

Expected: no matches.

- [ ] **Step 5: Commit implementation**

```bash
git add src/harness_sdk/instrumentation/httpx/__init__.py \
  src/harness_sdk/instrumentation/httpx/utils.py \
  test/instrumentation/httpx_client/test_httpx_utils.py
git commit -m "fix: avoid consuming HTTPX response streams"
```

### Task 3: Update telemetry expectations and verify

**Files:**
- Modify: `test/instrumentation/httpx_client/httpx_integration_test.py:19-127`

**Interfaces:**
- Consumes: finished HTTPX client span attributes.
- Produces: integration assertions preserving request body, response headers, and status while requiring `http.response.body` to be absent.

- [ ] **Step 1: Replace response-body assertions**

In all three HTTPX integration tests, replace:

```python
assert client_span['attributes']['http.response.body'] == '{ "a": "a", "xyz": "xyz" }'
```

or the equivalent `httpx_span` assertion with:

```python
assert 'http.response.body' not in client_span['attributes']
```

Use `httpx_span` in the async test. Retain all request-body, response-header, and status assertions. Add the existing `tester3` response-header assertion to the POST test so both POST response paths explicitly retain header coverage.

- [ ] **Step 2: Run the complete HTTPX test directory**

Run:

```bash
python3 -m pytest -q test/instrumentation/httpx_client
```

Expected: all HTTPX tests pass.

- [ ] **Step 3: Commit telemetry expectations**

```bash
git add test/instrumentation/httpx_client/httpx_integration_test.py
git commit -m "test: update HTTPX response telemetry expectations"
```

- [ ] **Step 4: Run repository verification**

Run:

```bash
python3 -m compileall -q src/harness_sdk
pylint src/harness_sdk --disable=C,R --ignore-patterns=config_pb2.py
./scripts/run-unit-tests.sh
```

Expected: each command exits 0. The unit script may report only the repository's configured DB integration-test skips when `RUN_SDK_INTEGRATION_TESTS` is unset.

- [ ] **Step 5: Review scope and final diff**

Run:

```bash
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- \
  src/harness_sdk/instrumentation/httpx \
  test/instrumentation/httpx_client \
  docs/superpowers
git status --short
```

Expected: no whitespace errors; only the approved design, plan, HTTPX source, and HTTPX tests differ; working tree is clean.

- [ ] **Step 6: Request code review and reverify findings**

Give a reviewer `origin/main` as the base, current `HEAD` as the head, this plan, and the approved design spec. Fix only valid high-confidence correctness or scope findings, commit those fixes, then rerun the focused HTTPX directory plus any affected lint command.

- [ ] **Step 7: Push and create the pull request**

```bash
git push -u origin fix/httpx-non-consuming-response-hooks
gh pr create --base main \
  --title "fix: preserve HTTPX response streams during instrumentation" \
  --body-file /tmp/httpx-stream-safe-pr.md
```

The PR body must state the root cause, the intentional removal of `http.response.body` from HTTPX spans, retained headers/status/request capture, exact verification results, and that this removes unsafe stream consumption/private mutation implicated by instrumentation without claiming it alone proves the production gzip issue.
