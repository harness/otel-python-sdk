#!/usr/bin/env python3
"""Call Bedrock Runtime directly with boto3 and write captured OTel spans to JSON."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            continue
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        os.environ[key] = value


def _default_env_path() -> Path:
    workspace_env = WORKSPACE_ROOT / ".work" / ".env"
    if workspace_env.exists():
        return workspace_env
    return ROOT / ".env"


def _resolve_model_id(explicit_model_id: str | None) -> str:
    if explicit_model_id:
        return explicit_model_id
    for key in ("BEDROCK_MODEL_ID", "BEDROCK_MODEL_ARN", "AWS_BEDROCK_MODEL_ID"):
        value = os.environ.get(key)
        if value:
            return value
    model_map = os.environ.get("BEDROCK_MODEL_MAP")
    if model_map:
        parsed = json.loads(model_map)
        first_model = next(iter(parsed.values()))
        if isinstance(first_model, dict):
            arn = first_model.get("arn")
            if arn:
                return arn
    raise ValueError("No Bedrock model found; pass --model-id or set BEDROCK_MODEL_MAP")


def _response_text(response: dict[str, Any]) -> str:
    content = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _span_to_dict(span: Any) -> dict[str, Any]:
    context = span.get_span_context()
    parent = span.parent
    return {
        "name": span.name,
        "context": {
            "trace_id": format(context.trace_id, "032x"),
            "span_id": format(context.span_id, "016x"),
        },
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "status": {
            "status_code": str(span.status.status_code),
            "description": span.status.description,
        },
        "attributes": _jsonable(dict(span.attributes or {})),
    }


class _StaticBearerToken:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_frozen_token(self) -> Any:
        from botocore.tokens import FrozenAuthToken  # pylint: disable=import-outside-toplevel

        return FrozenAuthToken(token=self._token)


def _configure_bedrock_bearer_auth(client: Any) -> None:
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        return
    client._request_signer._auth_token = _StaticBearerToken(token)  # pylint: disable=protected-access
    client.meta.events.register("choose-signer.bedrock-runtime", lambda **_kwargs: "bearer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=_default_env_path())
    parser.add_argument("--spans-file", type=Path, default=ROOT / "test-code" / "bedrock_spans.json")
    parser.add_argument("--model-id")
    parser.add_argument("--prompt", default="Reply with one short sentence about OpenTelemetry.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    _load_env_file(args.env_file)
    os.environ.setdefault("HA_GEN_AI_ENABLED", "true")
    os.environ.setdefault("HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED", "false")
    os.environ.setdefault("HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED", "false")

    import boto3  # pylint: disable=import-outside-toplevel
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # pylint: disable=import-outside-toplevel
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # pylint: disable=import-outside-toplevel
        InMemorySpanExporter,
    )

    from harness_sdk.agent import Agent  # pylint: disable=import-outside-toplevel

    model_id = _resolve_model_id(args.model_id)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("AWS_REGION or AWS_DEFAULT_REGION is required")

    agent = Agent()
    exporter = InMemorySpanExporter()
    agent.register_processor(SimpleSpanProcessor(exporter))
    agent.instrument(
        skip_libraries=[
            "flask",
            "django",
            "fastapi",
            "grpc:server",
            "grpc:client",
            "postgresql",
            "mysql",
            "requests",
            "httpx",
            "aiohttp:client",
            "anthropic",
            "openai",
            "litellm",
            "mcp",
        ]
    )

    client = boto3.client("bedrock-runtime", region_name=region)
    _configure_bedrock_bearer_auth(client)
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": args.prompt}]}],
        inferenceConfig={"maxTokens": args.max_tokens},
    )

    spans = [_span_to_dict(span) for span in exporter.get_finished_spans()]
    args.spans_file.parent.mkdir(parents=True, exist_ok=True)
    args.spans_file.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "region": region,
                "response_text": _response_text(response),
                "spans": spans,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(spans)} spans to {args.spans_file}")


if __name__ == "__main__":
    main()
