#!/usr/bin/env python3
"""Copy pre-release OpenTelemetry gen-ai packages into src/opentelemetry for local/CI builds."""
from __future__ import annotations

import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEARCH_ROOTS = (
    os.path.join(_ROOT, "temporary-vendor"),
    os.path.join(os.path.dirname(_ROOT), "temporary-vendor"),
)

_VENDOR_PACKAGES = (
    ("opentelemetry-instrumentation-anthropic", "src", "opentelemetry"),
    ("opentelemetry-instrumentation-openai-v2", "src", "opentelemetry"),
    ("opentelemetry-util-genai", "src", "opentelemetry"),
)

_DEST = os.path.join(_ROOT, "src", "opentelemetry")


def _find_vendor_root() -> str | None:
    for root in _SEARCH_ROOTS:
        if os.path.isdir(root):
            return root
    return None


def bundle_vendor() -> None:
    vendor_root = _find_vendor_root()
    if vendor_root is None:
        print(
            "temporary-vendor not found; run scripts/fetch-vendor.sh first",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(_DEST, exist_ok=True)
    for pkg_dir, src_subdir, namespace in _VENDOR_PACKAGES:
        src = os.path.join(vendor_root, pkg_dir, src_subdir, namespace)
        if not os.path.isdir(src):
            print(f"warning: missing vendor source {src}", file=sys.stderr)
            continue
        shutil.copytree(src, _DEST, dirs_exist_ok=True)
        print(f"bundled {pkg_dir}")

    openai_v2 = os.path.join(_DEST, "instrumentation", "openai_v2")
    if not os.path.isdir(openai_v2):
        print("error: opentelemetry.instrumentation.openai_v2 not bundled", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    bundle_vendor()
