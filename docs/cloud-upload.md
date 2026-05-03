# Cloud upload

Uploading to AgentLock Cloud is **optional**. Everything else in the SDK
works fully offline.

```python
from agentlock.client.client import AgentLockClient

client = AgentLockClient("https://cloud.example.com", api_key="sk-...")

await client.upload_traces(envelopes)
await client.upload_atep_segment("agent://acme/demo", segment_path)
await client.upload_bundle("agent://acme/demo", bundle_archive)
```

## Idempotency

Every POST automatically gets an `Idempotency-Key` header (a fresh ULID).
Retries reuse the same key, so the server can deduplicate.

## Retries

The client retries:

- network errors (connection refused, timeout, etc.)
- 429 / 502 / 503 / 504 responses

The default `RetryPolicy` is 3 retries with exponential backoff
(0.2s, 0.4s, 0.8s). When a `429` response includes `Retry-After`, the
header value overrides the exponential delay.

`401` raises `AuthenticationError` immediately (no retry). Other 4xx raise
`CloudError` immediately.

## Batched HTTP exporter

`HttpExporter` buffers up to `batch_size` envelopes (or
`batch_interval_ms` milliseconds) before calling `upload_traces`. Use
`await exporter.aclose()` to flush the queue cleanly.

## Sync wrapper

When async is not feasible, use `SyncAgentLockClient`. It wraps each
method with `asyncio.run`. Prefer the async client whenever possible.
