# Errors & exceptions

All SDK exceptions derive from `AgenomicError`:

```python
from agenomic.exceptions import (
    AgenomicError,        # base for everything below
    ValidationError,      # trace, event, or schema validation failed
    CryptoError,          # hashing, signing, or canonical encoding failed
    AtepError,            # ATEP segment integrity, format, or signature error
    ExportError,          # export to JSONL, ATEP, or HTTP failed
    CloudError,           # Agenomic Cloud HTTP error
    AuthenticationError,  # cloud auth failed — subclass of CloudError
    RedactionError,       # redaction rule could not be applied
)
```

```
AgenomicError
├── ValidationError
├── CryptoError
├── AtepError
├── ExportError
├── RedactionError
└── CloudError
    ├── AuthenticationError
    └── ToolExecutionError            (code, status; tool execution and Protect)
        ├── ToolCallDenied, ToolApprovalPending
        └── VaultError                (Agents Vault; request_id, action_id)
            ├── VaultNotEntitled, VaultApprovalRequired, VaultPolicyDenied
            ├── VaultGrantUnusable, VaultRevoked, VaultOutcomeUnknown
            ├── VaultExecutionFailed, VaultExecutionInProgress
            ├── VaultRateLimited, VaultValidationError, VaultAuthenticationError
            ├── VaultPermissionDenied, VaultNotFound, VaultConflict
            ├── VaultRefused → VaultFailClosed, VaultUnavailable
            ├── VaultServerError, VaultTransportError, VaultNotConfigured
            └── ReplayFixtureMissing, ReplayUnsupported
```

The Agents Vault errors and what each server code maps to are tabulated in
[vault.md](vault.md#errors).

## What raises what

| situation                                            | exception              |
| ---------------------------------------------------- | ---------------------- |
| Cloud request fails after retries / 4xx / 5xx        | `CloudError`           |
| Cloud returns 401                                    | `AuthenticationError`  |
| Agents Vault locked, pending, denied, unknown outcome | `VaultError` subclasses, see [vault.md](vault.md#errors) |
| ATEP segment bad magic, CRC mismatch, truncation     | `AtepError`            |
| Store `agent_id` mismatch                            | `AtepError`            |
| Bad PEM key file / not ed25519                       | `CryptoError`          |
| Redaction `HASH` on an unserializable value          | `RedactionError`       |
| Invalid `agent_id` / manifest / model fields         | `ValueError` (pydantic)|
| Export on a closed exporter, emit on stopped session | `RuntimeError`         |
| Unknown tracking event type                          | `ValueError`           |
| Optional integration package missing                 | `ImportError`          |

Two deliberate design points:

- **Exporter errors never reach your agent.** The decorator and
  `MultiExporter`/`HttpExporter` catch, log, and continue — telemetry
  failure must not take production down. Watch the `agenomic.*` loggers.
- **Verification returns, it doesn't raise.** `VerifyingKey.verify`,
  `AtepEvent.verify`, and `SegmentReader.verify_merkle_root` return
  `bool`; `AtepStore.verify_all` returns a `VerificationReport` with
  `ok`, counts, and a `failures` list. Only unreadable/corrupt files
  raise `AtepError`.

## Logging

Library code never prints. Named loggers: `agenomic.client`,
`agenomic.trace.decorator`, `agenomic.exporters.http`,
`agenomic.exporters.multi`, `agenomic.exporters.atep_local`,
`agenomic.crypto.signing`, `agenomic.vault` (route and status of each
request and each retry, never a body, an argument or a secret value).

```python
import logging
logging.getLogger("agenomic").setLevel(logging.DEBUG)
```
