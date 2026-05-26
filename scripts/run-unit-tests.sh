#!/usr/bin/env bash
# Run harness-sdk unit tests (matches CI). Set RUN_SDK_INTEGRATION_TESTS=1 for DB tests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d "$ROOT/temporary-vendor/opentelemetry-util-genai" ]]; then
  echo "Using existing temporary-vendor"
else
  bash scripts/fetch-vendor.sh
fi

python3 scripts/bundle_vendor.py
pip install -q -e ".[dev,anthropic,openai,litellm]"

export HA_ENABLE_CONSOLE_SPAN_EXPORTER=true
export PYTHONUNBUFFERED=1

PYTEST_ARGS=(-ra --tb=short)
if [[ "${RUN_SDK_INTEGRATION_TESTS:-}" != "1" ]]; then
  echo "Skipping DB integration tests (set RUN_SDK_INTEGRATION_TESTS=1 to enable)"
  PYTEST_ARGS+=(
    --ignore=test/instrumentation/mysql/mysql_integration_test.py
    --ignore=test/instrumentation/postgresql/postgresql_integration_test.py
  )
fi

python3 -m pytest "${PYTEST_ARGS[@]}" "$@"
