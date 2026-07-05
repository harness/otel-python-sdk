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

All SDK config uses the `HA_` prefix. `AT_` and `TA_` are legacy aliases accepted during migration.

Key variables:
| Variable | Purpose |
|---|---|
| `HA_SERVICE_NAME` | Service name on all spans |
| `HA_REPORTING_ENDPOINT` | OTLP endpoint URL |
| `HA_REPORTING_TOKEN` | Auth token (`x-harness-service-token` header) |
| `HA_REPORTING_TRACE_REPORTER_TYPE` | `OTLP` (gRPC) or `OTLP_HTTP` |
| `HA_REPORTING_SECURE` | `true`/`false` for TLS |
| `HA_REPORTING_COMPRESSION` | `gzip` or empty |
| `HA_CONTROL_PLUGINS` | Comma-separated control plugin names |
| `HA_OBSERVABILITY_PLUGINS` | Comma-separated observability plugin names |
| `HA_ENABLE_CONSOLE_SPAN_EXPORTER` | Set to any value to dump spans to stdout |
| `HA_CONFIG_FILE` | Path to YAML config file (overrides env) |
| `HA_GEN_AI_ENABLED` | Enable/disable GenAI instrumentation |
| `HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED` | Capture LLM prompt/response payloads |
| `HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED` | Run control plugins on GenAI spans |

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
