# Hugging Face provider

The Hugging Face connector lets you resolve Hub model metadata, validate
credentials, run inference, and attach a model configuration to an agent
genome. It is built on `httpx` (already a hard dependency), so **the SDK core
never requires `huggingface-hub`**. The optional extra is only for downstream
tooling that wants the official hub client.

```bash
pip install "agenomic[huggingface]"   # optional; not needed for the connector
```

## Provider name and aliases

The canonical provider name is `huggingface`. Accepted aliases are
case-insensitive and treat `-` and `_` as equivalent: `huggingface`, `hf`,
`hugging_face` (so `Hugging-Face`, `HF`, etc. all normalize).

```python
from agenomic.providers.huggingface import normalize_provider, is_huggingface

normalize_provider("Hugging-Face")  # -> "huggingface"
is_huggingface("hf")                # -> True
```

## Environment variables

| Variable                        | Purpose                                   | Default |
| ------------------------------- | ----------------------------------------- | ------- |
| `HUGGINGFACE_API_TOKEN`         | API token (preferred)                     | —       |
| `HF_TOKEN`                      | API token (fallback)                      | —       |
| `HUGGINGFACE_ENDPOINT_URL`      | Custom Inference Endpoint base URL        | public Inference API |
| `HUGGINGFACE_ORG`               | Default organization                      | —       |
| `HUGGINGFACE_DEFAULT_MODEL`     | Default model id                          | —       |
| `HUGGINGFACE_TIMEOUT_SECONDS`   | HTTP timeout (seconds)                     | `30`    |

```python
from agenomic.providers.huggingface import HuggingFaceConfig

config = HuggingFaceConfig.from_env()  # reads the variables above
```

## Security

- The token is stored privately and **excluded from `repr`**; it never appears
  in returned objects, traces, or error messages.
- All connector error strings are passed through redaction, which scrubs both
  the configured token value and any `hf_...`-shaped token.
- Endpoint URLs containing inline credentials (`user:pass@host`) are rejected.

```python
HuggingFaceConfig(_token="hf_secret").redact("oops hf_secret and hf_other")
# -> "oops *** and ***"
```

## Resolving metadata and running inference

```python
from agenomic.providers.huggingface import HuggingFaceClient, HuggingFaceConfig

client = HuggingFaceClient(HuggingFaceConfig.from_env())

client.validate_credentials()                      # whoami-v2; raises on 401/403
meta = client.resolve_model_metadata(
    "mistralai/Mistral-7B-Instruct-v0.3", revision="main"
)
# meta.resolved_commit, meta.task ("text-generation"), meta.private

client.generate_text("gpt2", "Once upon a time", parameters={"max_new_tokens": 32})
client.embeddings("sentence-transformers/all-MiniLM-L6-v2", "hello")
```

## Trace instrumentation

```python
from agenomic.integrations.huggingface import instrument_huggingface

client = instrument_huggingface(HuggingFaceClient(HuggingFaceConfig.from_env()))
# generate_text / embeddings now record ModelCall(provider="huggingface", model=...)
# on the active TraceRecorder — on success and on error, without the token.
```

For inference functions you call yourself (e.g. `huggingface_hub`'s
`InferenceClient`), use `trace_huggingface_call`:

```python
from agenomic.integrations.huggingface import trace_huggingface_call

trace_huggingface_call(
    some_inference_fn,
    model="gpt2",
    prompt="hello",
    parameters={"temperature": 0.2},
    # ...kwargs forwarded to some_inference_fn...
)
```

## Configuring a model on an agent genome

```python
from agenomic import Client

client = Client()
agent = client.agent.load("./path-to-agent")   # reads genome.yaml / genome.json
agent.configure_model(
    provider="huggingface",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)
```

This writes a `runtime.model` block into the genome. `configure_model` is
provider-agnostic (any provider string works); Hugging Face aliases are
normalized to `huggingface` and validated.

## Lockfile entries

`build_lockfile_model` produces a deterministic, token-free lockfile entry.
Content hashes are **SHA-256 over canonical JSON** (sorted keys, compact
separators, UTF-8). The `endpoint_ref` is reduced to `scheme://host[/path]`
with any credentials and query string stripped.

```python
from agenomic.providers.huggingface import build_lockfile_model

entry = build_lockfile_model(config=config, metadata=meta, parameters={"temperature": 0.7})
# {
#   "provider": "huggingface",
#   "model_id": ..., "revision": ..., "resolved_commit": ..., "task": ...,
#   "endpoint_ref": "https://api-inference.huggingface.co",
#   "endpoint_hash": "<sha256>", "metadata_hash": "<sha256>", "parameter_hash": "<sha256>",
# }
```
