# harness-sdk — agent orientation

## What this repo is

A Python SDK that instruments arbitrary Python applications with OpenTelemetry tracing and ships spans to a Harness OTLP ingest endpoint. Two extension points: **control plugins** (policy/blocking) and **observability plugins** (span processors/exporters).

## Local setup

```bash
bash scripts/fetch-vendor.sh
pip install -e ".[dev,anthropic,openai,litellm]"
./scripts/run-unit-tests.sh
```

Integration tests (need Docker):
```bash
cd test/externalServices && docker compose up -d --wait
cd ../..
RUN_SDK_INTEGRATION_TESTS=1 ./scripts/run-unit-tests.sh
```

## Environment variable naming

All SDK config uses the `HARNESS_` prefix. `HA_`, `AT_`, and `TA_` are legacy aliases accepted for backwards compatibility. When a setting is defined under multiple prefixes, precedence is `HARNESS_` > `HA_` > `AT_` > `TA_`. Resolution lives in `src/harness_sdk/env.py` (`get_env_value`, `is_env_var_present`, `is_harness_flag_enabled`).

Key variables:
| Variable | Purpose |
|---|---|
| `HARNESS_SERVICE_NAME` | Service name on all spans |
| `HARNESS_REPORTING_ENDPOINT` | OTLP endpoint URL |
| `HARNESS_REPORTING_TOKEN` | Auth token (`x-harness-service-token` header) |
| `HARNESS_REPORTING_TRACE_REPORTER_TYPE` | `OTLP` (gRPC) or `OTLP_HTTP` |
| `HARNESS_REPORTING_SECURE` | `true`/`false` for TLS |
| `HARNESS_REPORTING_COMPRESSION` | `gzip` or empty |
| `HARNESS_CONTROL_PLUGINS` | Comma-separated control plugin names |
| `HARNESS_OBSERVABILITY_PLUGINS` | Comma-separated observability plugin names |
| `HARNESS_ENABLE_CONSOLE_SPAN_EXPORTER` | Set to any value to dump spans to stdout |
| `HARNESS_CONFIG_FILE` | Path to YAML config file (overrides env) |
| `HARNESS_GEN_AI_PAYLOAD_CAPTURE_ENABLED` | Capture LLM prompt/response payloads |
| `HARNESS_GEN_AI_PAYLOAD_EVALUATION_ENABLED` | Run control plugins on GenAI spans |

### Instrumentation opt-in (strict `HARNESS_` prefix, no legacy aliases)

Instrumentation is opt-in: `Agent().instrument()` instruments nothing unless a flag below is set to `true`. Categorization and gating live in `src/harness_sdk/instrumentation/instrumentation_definitions.py` (`is_library_enabled`, `is_api_instrumentation_enabled`, `any_ai_provider_enabled`), enforced in `Agent.instrument()`.

| Variable | Enables |
|---|---|
| `HARNESS_ENABLE_API` | All non-AI instrumentation: HTTP servers/clients, gRPC, MySQL/PostgreSQL, botocore, and generic OTel contrib fallback |
| `HARNESS_ENABLE_AI_OPENAI` | OpenAI |
| `HARNESS_ENABLE_AI_ANTHROPIC` | Anthropic |
| `HARNESS_ENABLE_AI_LITELLM` | LiteLLM |
| `HARNESS_ENABLE_AI_GOOGLE_GENAI` | Google GenAI (Gemini / Vertex AI) |
| `HARNESS_ENABLE_AI_MCP` | Model Context Protocol |

The legacy `HA_GEN_AI_ENABLED` master switch no longer controls instrumentation. `skip_libraries=[...]` on `instrument()` still takes precedence over enable flags.

## Plugin system

### Entry point groups

| Group | Purpose |
|---|---|
| `harness_sdk_control_plugin` | Policy plugins — can block requests |
| `harness_sdk_observability_plugin` | Span processor plugins |

Built-in observability plugins (auto-loaded unless overridden):
- `builtin_pipeline` — wires OTLP exporter + DB span filter + attribute exclusion
- `builtin_span_attributes` — stamps `service.name` and custom span attributes

### Writing a plugin

**Control plugin** — implement the `ControlPlugin` protocol (`src/harness_sdk/plugins/control.py`):
- `on_init(config)` — called once at load time
- `evaluate(span, url, headers, body, is_grpc) -> ControlResult` — HTTP/gRPC spans
- `evaluate_agent_span(span, body) -> ControlResult` — GenAI spans
- `shutdown()` — called on SDK teardown
- Set `provides_blocking = True` on the class if `evaluate` may return `block=True`

**Observability plugin** — any class with:
- `on_init(config)`
- `create_span_processors(config) -> List[SpanProcessor]`
- `shutdown()`

Register via `pyproject.toml` entry points, then list the name in `HA_OBSERVABILITY_PLUGINS` or `HA_CONTROL_PLUGINS`.

## Key source files

| File | What it does |
|---|---|
| `src/harness_sdk/agent.py` | `Agent` singleton — entry point for all SDK users |
| `src/harness_sdk/agent_init.py` | OTel tracer provider + exporter setup |
| `src/harness_sdk/config/config.py` | Config loading (YAML file → env → protobuf) |
| `src/harness_sdk/config/environment.py` | Env var → protobuf field mapping |
| `src/harness_sdk/plugins/control.py` | `ControlRegistry` singleton + `ControlPlugin` protocol |
| `src/harness_sdk/plugins/loader.py` | Entry-point discovery + ordered plugin loading |
| `src/harness_sdk/plugins/builtin/pipeline.py` | Default OTLP pipeline observability plugin |
| `src/harness_sdk/db_control_span_processor.py` | Filters DB spans through control registry |
| `src/harness_sdk/instrumentation/litellm/` | LiteLLM wraps with pre-call policy evaluation |
| `src/harness_sdk/instrumentation/mcp/` | MCP SDK instrumentation + GenAI attribute mirroring |
| `src/harness_sdk/autoinstrumentation/sitecustomize.py` | Zero-touch init via `PYTHONPATH` |

## Span processor pipeline (default)

```
SamplingSpanProcessor (DbControlSpanProcessor)
  └─ ExcludeByAttributeSpanProcessor   (drops traceableai.span_type=nospan)
       └─ BatchSpanProcessor
            └─ OTLPSpanExporter
```

`DbControlSpanProcessor` — filters MySQL/PostgreSQL spans through control plugins.

## Build / vendor

```bash
bash scripts/fetch-vendor.sh   # download vendored deps into temporary-vendor/
python -m build --outdir dist  # produces wheel + sdist
```

The `scripts/bundle_vendor.py` script is used by CI to embed vendored packages into the wheel so the SDK can be installed in environments without network access.

## Test layout

Tests mirror source: `test/instrumentation/flask/` covers `src/harness_sdk/instrumentation/flask/`. File naming is mixed (`test_*.py` and `*_test.py`) — both are collected by pytest.

Integration tests are gated behind `RUN_SDK_INTEGRATION_TESTS=1` and require Docker services started from `test/externalServices/`.
