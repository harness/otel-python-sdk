# Send traces to Harness UDP ingest

A short integration: install the SDK, opt into the instrumentation you want,
call `Agent().instrument()` once at startup, and point a few `HARNESS_*`
environment variables at Harness UDP ingest. Instrumentation is opt-in — you
choose HTTP/API instrumentation and each AI provider explicitly — and the SDK
exports OTLP/HTTP spans to ingest.

> Environment variables use the `HARNESS_` prefix. The legacy `HA_`, `AT_`, and
> `TA_` prefixes still work for backwards compatibility (precedence:
> `HARNESS_` > `HA_` > `AT_` > `TA_`).

## 1. Install

The SDK is published on PyPI as `harness-sdk`. Pin the published release in
applications and CI builds:

```bash
pip install "harness-sdk==1.0.1"
```

Add the extra for the LLM client you use:

```bash
pip install "harness-sdk[litellm]==1.0.1"     # for LiteLLM
pip install "harness-sdk[anthropic]==1.0.1"   # for the Anthropic client
```

> Anthropic in-process spans additionally require the OTel GenAI helpers, which
> are not bundled:
> `pip install opentelemetry-instrumentation-anthropic opentelemetry-util-genai`

## 2. Configure (environment)

| Variable | Value |
|---|---|
| `HARNESS_SERVICE_NAME` | your service name |
| `HARNESS_REPORTING_ENDPOINT` | `https://app.harness.io/udp-ingest/otel/v1/traces?accountIdentifier=<ACCOUNT_ID>&routingId=<ACCOUNT_ID>` |
| `HARNESS_REPORTING_TRACE_REPORTER_TYPE` | `OTLP_HTTP` |
| `HARNESS_REPORTING_TOKEN` | a Harness **service account token** |

```bash
export HARNESS_SERVICE_NAME="my-service"
export HARNESS_REPORTING_ENDPOINT="https://app.harness.io/udp-ingest/otel/v1/traces?accountIdentifier=<ACCOUNT_ID>&routingId=<ACCOUNT_ID>"
export HARNESS_REPORTING_TRACE_REPORTER_TYPE=OTLP_HTTP
export HARNESS_REPORTING_TOKEN="<HARNESS_SERVICE_TOKEN>"
```

The token is sent to ingest as the `x-harness-service-token` header.

## 3. Opt into instrumentation

Instrumentation is opt-in: nothing is instrumented until you enable it. Set one
flag for all HTTP/API instrumentation, and one flag per AI provider you use. A
flag is on only when its value is `true`.

| Variable | Enables |
|---|---|
| `HARNESS_ENABLE_API` | All non-AI instrumentation (HTTP servers/clients, gRPC, DB, botocore, generic OTel contrib) |
| `HARNESS_ENABLE_AI_OPENAI` | OpenAI |
| `HARNESS_ENABLE_AI_ANTHROPIC` | Anthropic |
| `HARNESS_ENABLE_AI_LITELLM` | LiteLLM |
| `HARNESS_ENABLE_AI_GOOGLE_GENAI` | Google GenAI (Gemini / Vertex AI) |
| `HARNESS_ENABLE_AI_MCP` | Model Context Protocol |

```bash
# Example: HTTP/API instrumentation plus LiteLLM
export HARNESS_ENABLE_API=true
export HARNESS_ENABLE_AI_LITELLM=true
```

## 4. Instrument (two lines)

Call this once, as early as possible in your process startup — before you make
LLM calls.

```python
from harness_sdk.agent import Agent

Agent().instrument()
```

That's the entire code change. Everything below is just your normal app code.

---

## Example: LiteLLM

```python
# Requires: export HARNESS_ENABLE_AI_LITELLM=true
from harness_sdk.agent import Agent
Agent().instrument()  # do this first, at startup

import litellm

resp = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Reply with one short sentence."}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Each `litellm.completion` / `acompletion` / `embedding` / `aembedding` call now
emits a `litellm_request` span with `gen_ai.*` attributes, exported to ingest.

## Example: Anthropic Python client

```python
# Requires: export HARNESS_ENABLE_AI_ANTHROPIC=true
from harness_sdk.agent import Agent
Agent().instrument()  # do this first, at startup

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
msg = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply with one short sentence."}],
)
print(msg.content[0].text)
```

`Messages.create` / `stream` (sync and async) are wrapped automatically and emit
GenAI spans to ingest.

---

## Scope the instrumentation (optional)

Because instrumentation is opt-in, you scope it by enabling only the flags you
need — for example, set `HARNESS_ENABLE_AI_LITELLM=true` and leave the other
`HARNESS_ENABLE_*` flags unset to instrument LiteLLM only.

For finer control you can also exclude an otherwise-enabled library at the call
site; `skip_libraries` takes precedence over the enable flags:

```python
# HTTP/API enabled, but never instrument requests.
# export HARNESS_ENABLE_API=true
Agent().instrument(skip_libraries=["requests"])
```

## Useful toggles (optional)

| Variable | Effect |
|---|---|
| `HARNESS_GEN_AI_PAYLOAD_CAPTURE_ENABLED` | `false` to omit prompt/response bodies from spans |
| `HARNESS_SPAN_ATTRIBUTES` | extra attributes on every span, e.g. `env=prod,team=ai` |
| `HARNESS_OBSERVABILITY_PLUGINS` | `builtin_span_attributes` to keep instrumentation but disable the SDK's own OTLP exporter (if you already export spans yourself) |
| `HARNESS_ENABLED` | `false` to disable the SDK entirely (no code change) |

