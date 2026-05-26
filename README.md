# harness-sdk (otel-python-sdk)

Generic Python agent SDK with OpenTelemetry instrumentation and a plugin architecture.

## Local development

```bash
bash scripts/fetch-vendor.sh
pip install -e ".[dev,anthropic,openai,litellm]"
./scripts/run-unit-tests.sh
```

Environment variables for SDK configuration use the `HA_` prefix (for example `HA_SERVICE_NAME`, `HA_REPORTING_ENDPOINT`).

### Integration tests (MySQL / PostgreSQL)

```bash
cd test/externalServices && docker compose up -d --wait
cd ../..
RUN_SDK_INTEGRATION_TESTS=1 ./scripts/run-unit-tests.sh
```

## CI

GitHub Actions workflows (public repo, `ubuntu-latest`):

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| [pr_build.yaml](.github/workflows/pr_build.yaml) | PR + push to `main` | Build manylinux wheels, lint, pytest (with Docker for DB tests) |
| [publish.yaml](.github/workflows/publish.yaml) | Tag `v*.*.*` | Build release artifacts and publish to PyPI / TestPyPI |
| [staticanalysis.yaml](.github/workflows/staticanalysis.yaml) | PR, `main`, weekly | Trivy filesystem scan |

## Publishing to PyPI

1. Configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) on PyPI (and TestPyPI for RCs):
   - **PyPI**: environment `pypi`, workflow `publish.yaml`, job `publish-pypi`
   - **TestPyPI**: environment `testpypi`, workflow `publish.yaml`, job `publish-testpypi`
2. Create and push a release tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. Release candidates (`v1.0.0-rc.1`) publish to **TestPyPI** only; stable tags publish to **PyPI**.

Version is taken from the tag (`v1.2.3` → `1.2.3`) via `scripts/build.sh`, which updates `src/harness_sdk/version.py` and `pyproject.toml`.

## Usage

```python
from harness_sdk.agent import Agent

agent = Agent()
agent.instrument()
```

Auto-instrumentation:

```bash
export HA_CONFIG_FILE=/path/to/config.yaml
harness-instrument python app.py
```

## Plugins

The SDK loads extensions via [setuptools entry points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html). Each plugin has a **name** (the entry-point key). Names are listed in config or environment variables; only installed plugins are loaded, in the order you configure.

| Type | Entry-point group | Config key / env |
|------|-------------------|------------------|
| Control | `harness_sdk_control_plugin` | `plugins.control` / `HA_CONTROL_PLUGINS` |
| Observability | `harness_sdk_observability_plugin` | `plugins.observability` / `HA_OBSERVABILITY_PLUGINS` |

Built-in observability plugins (shipped with `harness-sdk`):

- `builtin_pipeline` — OTLP export, sampling, exclusion processors
- `builtin_span_attributes` — service name and configured span attributes

Example `agent-config.yaml`:

```yaml
service_name: my-service
reporting:
  endpoint: http://localhost:4318
plugins:
  control:
    - my_policy          # order matters: first plugin runs first
  observability:
    - builtin_pipeline
    - builtin_span_attributes
    - my_exporter        # custom plugin after builtins
```

Or via environment (comma-separated, same order semantics):

```bash
export HA_CONTROL_PLUGINS=my_policy
export HA_OBSERVABILITY_PLUGINS=builtin_pipeline,builtin_span_attributes,my_exporter
```

### Create a control plugin

Control plugins evaluate HTTP/gRPC ingress and GenAI spans. They return a `ControlResult` (block, headers, span attributes, etc.). Plugins run in config order; the chain stops when one returns `block=True`.

1. **Implement the plugin** in your package (see `harness_sdk.plugins.control.ControlPlugin`):

```python
# my_company_policy/plugin.py
from typing import Any
from opentelemetry.trace import Span
from harness_sdk.plugins.control import ControlResult, ControlPlugin


class MyPolicyPlugin:
    name = "my_policy"
    provides_blocking = True  # set True if this plugin can block requests

    def on_init(self, config: Any) -> None:
        self._config = config

    def evaluate(
        self, span: Span, url: str, headers: dict, body, is_grpc: bool
    ) -> ControlResult:
        result = ControlResult()
        # result.block = True
        # result.response_status_code = 403
        return result

    def evaluate_agent_span(self, span: Span, body: str = "") -> ControlResult:
        return ControlResult()

    def shutdown(self) -> None:
        pass


def factory(config: Any) -> ControlPlugin:
    return MyPolicyPlugin()
```

2. **Register the entry point** in your package `pyproject.toml`:

```toml
[project.entry-points.harness_sdk_control_plugin]
my_policy = "my_company_policy.plugin:factory"
```

(Equivalent in `setup.py`: `entry_points={'harness_sdk_control_plugin': ['my_policy = ...']}`.)

3. **Install** your package in the same environment as the app (`pip install my-company-policy`).

4. **Enable** the plugin by name in config or `HA_CONTROL_PLUGINS` (see above).

For tests or one-off wiring you can also call `agent.register_control_plugin(plugin)` after `Agent()` is constructed.

Reference implementation: the Traceable agent ships a control plugin as `traceable` (`traceableai.plugins.traceable_control:factory`).

### Create an observability plugin

Observability plugins contribute OpenTelemetry `SpanProcessor` instances to the tracer provider. Processors are registered in config order (each plugin’s `create_span_processors` may return multiple processors).

1. **Implement the plugin** (see `harness_sdk.plugins.observability.ObservabilityPlugin`):

```python
# my_company_telemetry/plugin.py
from typing import Any, List
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


class MyExporterPlugin:
    name = "my_exporter"
    priority = 300  # informational; ordering is driven by config list

    def on_init(self, config: Any) -> None:
        self._config = config

    def create_span_processors(self, config: Any) -> List[SpanProcessor]:
        return [SimpleSpanProcessor(ConsoleSpanExporter())]

    def shutdown(self) -> None:
        pass


def factory(config: Any) -> MyExporterPlugin:
    return MyExporterPlugin()
```

2. **Register the entry point**:

```toml
[project.entry-points.harness_sdk_observability_plugin]
my_exporter = "my_company_telemetry.plugin:factory"
```

3. **Install** your package and **enable** it under `plugins.observability` or `HA_OBSERVABILITY_PLUGINS`.

If you omit observability plugins entirely, the SDK defaults to `builtin_pipeline` and `builtin_span_attributes`. Custom plugins typically keep those builtins and append your entry after them.

Reference implementations in this repo:

- `harness_sdk.plugins.builtin.pipeline`
- `harness_sdk.plugins.builtin.span_attributes`
