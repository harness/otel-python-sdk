#!/usr/bin/env bash
# Set package version from a release tag (e.g. scripts/build.sh 1.2.3).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:?usage: scripts/build.sh <version>}"

VERSION_FILE="$ROOT/src/harness_sdk/version.py"
PYPROJECT="$ROOT/pyproject.toml"

cat > "$VERSION_FILE" <<EOF
__version__ = "$VERSION"
EOF

if [[ -f "$PYPROJECT" ]]; then
  python3 - <<PY
from pathlib import Path
import re

path = Path("$PYPROJECT")
text = path.read_text()
text = re.sub(r'^version = ".*"$', 'version = "$VERSION"', text, count=1, flags=re.M)
path.write_text(text)
PY
fi

echo "Updated version to $VERSION"
cat "$VERSION_FILE"
