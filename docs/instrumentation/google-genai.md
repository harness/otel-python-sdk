# Google Gen AI (`google-genai`) instrumentation

Instruments the unified **Google Gen AI SDK** (`from google import genai`) for
both backends:

- **Gemini Developer API** — `genai.Client(api_key=...)` → `gen_ai.provider.name = gcp.gemini`
- **Vertex AI** — `genai.Client(vertexai=True, project=..., location=...)` → `gen_ai.provider.name = gcp.vertex_ai`

> The legacy `vertexai.generative_models` / `google-cloud-aiplatform` generative
> modules and `google-generativeai` are **not** instrumented. Google deprecated
> them (generative modules removal 2026-06-24); `google-genai` is the supported client.

Enable with `pip install "harness-sdk[google-genai]"`. Telemetry is produced via
`opentelemetry-util-genai` (`TelemetryHandler` → `LLMInvocation` / `EmbeddingInvocation`),
so spans, metrics, and attributes follow the OpenTelemetry GenAI semantic conventions,
identical in shape to the OpenAI/Anthropic wrappers.

## Covered call surface

| Method (class) | Span |
|---|---|
| `Models.generate_content` / `AsyncModels.generate_content` | `chat {model}` |
| `Models.generate_content_stream` / `AsyncModels.generate_content_stream` | `chat {model}` (streaming) |
| `Models.embed_content` / `AsyncModels.embed_content` | `embeddings {model}` |

Not covered: the Live API (`client.aio.live`), batch/prediction, and tuning. Chat
sessions (`client.chats`) are covered indirectly because they delegate to
`generate_content`.

## Configuration gates

| Config key | Env | Effect |
|---|---|---|
| `gen_ai.enabled` | `HA_GEN_AI_ENABLED` | Master switch; off = passthrough, no span |
| `gen_ai.payload_capture_enabled` | `HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED` | Enables prompt/response/tool **content** capture (sets OTEL experimental + `SPAN_ONLY`) |
| `gen_ai.payload_evaluation_enabled` | `HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED` | Enables pre-call control-plugin evaluation (blocking) |

---

## Captured fields, by confidence

### ✅ Will **definitely** be captured

Set on every span whenever `gen_ai.enabled = true`, independent of content capture
and independent of the response contents (they derive from the span lifecycle and
the request arguments the SDK requires):

| Attribute | Source |
|---|---|
| `gen_ai.operation.name` (`chat` / `embeddings`) | wrapper (fixed per method) |
| `gen_ai.provider.name` (`gcp.vertex_ai` \| `gcp.gemini` \| `gcp.gen_ai`) | client backend detection |
| `gen_ai.request.model` | `model` argument |
| `gen_ai.framework` = `google-genai` | wrapper (fixed) |
| `gen_ai.request.streaming` = `true` | streaming methods only |
| span name (`chat {model}`), span kind `CLIENT`, status/error | `TelemetryHandler` |

### 🟡 **Likely** captured

Present whenever the request/response actually carries the value — i.e. the normal
case for a successful call, but not guaranteed (a field may be omitted by the API,
the caller may not set a request param, or an error may end the span early):

| Attribute | Source | Why not guaranteed |
|---|---|---|
| `gen_ai.response.model` | `response.model_version` | Some responses omit `model_version` |
| `gen_ai.usage.input_tokens` | `usage_metadata.prompt_token_count` | Absent on failed calls |
| `gen_ai.usage.output_tokens` | `usage_metadata.candidates_token_count` | Absent on failed calls |
| `gen_ai.response.finish_reasons` | `candidate.finish_reason` | Absent if no candidates returned |
| `gen_ai.request.temperature` / `top_p` / `max_tokens` / `stop_sequences` / `seed` | request `config` | Only when the caller sets them |
| `gen_ai.input.messages` | `contents` | **Only when `payload_capture_enabled`** |
| `gen_ai.output.messages` | `response.candidates[*].content.parts` | **Only when `payload_capture_enabled`** |
| `gen_ai.system_instructions` | `config.system_instruction` | Only when set **and** content capture on |

Content attributes (`*.messages`, `system_instructions`) are serialized only in
OTEL experimental mode with capture mode `SPAN_ONLY`/`SPAN_AND_EVENT`, which the SDK
enables from `payload_capture_enabled`.

### 🟠 **Probable** (best-effort / conditional / version-sensitive)

Captured only under specific conditions or subject to SDK/backend variability:

| Attribute / behavior | Condition it depends on |
|---|---|
| **Tool calls** in `gen_ai.output.messages` (`ToolCallRequest`: name, arguments, id) | model actually returns `function_call` parts **and** content capture on |
| Tool results (`ToolCallResponse`) in `gen_ai.input.messages` | caller passes `function_response` parts back in `contents` |
| `gen_ai.response.id` | populated by response; the Gemini Developer API sometimes omits `response_id` |
| Streaming token usage / finish reasons | only if the **final** stream chunk carries `usage_metadata` (Gemini normally does; not contractual) |
| Streaming output text | reconstructed by concatenating `chunk.text`; non-text parts in streams are not reassembled |
| Embeddings `gen_ai.usage.input_tokens` | from `embeddings[0].statistics.token_count` — populated on Vertex, often absent on the Gemini Developer API |
| Embeddings dimension count | derived from `embeddings[0].values` length |
| Input message roles/parts for complex `contents` | best-effort mapping of str / `Part` / `Content`; exotic part types (blobs, files, URIs) are not fully expanded |

### ❌ Not captured (current limitations)

- **Automatic Function Calling (AFC):** a single `generate_content` that internally
  loops through tool calls is recorded as **one** span (the final response);
  intermediate turns / `automatic_function_calling_history` are not expanded into
  child spans.
- Cache token counts (`gen_ai.usage.cache_*`) and `tool_use_prompt_token_count`.
- `gen_ai.tool.definitions` (declared tools) — not currently mapped from `config.tools`.
- Live API bidirectional streaming.

---

## Blocking (control plugins)

When `payload_evaluation_enabled = true` and a blocking control plugin is
registered, each call is evaluated **before** the request is sent. A blocking
decision raises `ControlEvaluationBlocked`, marks the span as errored, and the
underlying Google API is never called — identical to the OpenAI/Anthropic/LiteLLM
GenAI wrappers.
