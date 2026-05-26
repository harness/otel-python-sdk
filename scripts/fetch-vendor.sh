#!/usr/bin/env bash
# Fetches pre-release OpenTelemetry gen-ai packages not yet on PyPI.
set -euo pipefail

REPO="https://github.com/open-telemetry/opentelemetry-python-contrib.git"
SHA="d83761fe969727f6ee09985179f412871157732d"
DEST="$(cd "$(dirname "$0")/.." && pwd)/temporary-vendor"

PACKAGES=(
  "instrumentation-genai/opentelemetry-instrumentation-anthropic"
  "instrumentation-genai/opentelemetry-instrumentation-openai-v2"
  "util/opentelemetry-util-genai"
)

rm -rf "$DEST"
mkdir -p "$DEST"

CLONE_DIR=$(mktemp -d)
trap 'rm -rf "$CLONE_DIR" 2>/dev/null || true' EXIT

git clone --filter=blob:none --no-checkout "$REPO" "$CLONE_DIR"
git -C "$CLONE_DIR" sparse-checkout set --cone "${PACKAGES[@]}"
git -C "$CLONE_DIR" checkout "$SHA"

for pkg in "${PACKAGES[@]}"; do
  pkg_name=$(basename "$pkg")
  cp -r "$CLONE_DIR/$pkg" "$DEST/$pkg_name"
done

echo "Fetched vendor packages to $DEST"
python3 "$(dirname "$0")/bundle_vendor.py"
