# Managed prompts conformance vectors

These vectors are the parity contract for every implementation of RFC 0012
(managed prompts). An implementation that claims renderer version `"1"` and the
secret pattern set `agenomic-secrets/1` passes every vector whose `consumers`
list names it. The vectors pin four things byte for byte: canonical JSON and
sha256 digests, the prompt reference grammar, the `agenomic-fstring/v1`
template grammar with its renderer, and the portable secret patterns.

## Suites

| Suite | Directory | Ids | Vectors | What it pins |
|---|---|---|---|---|
| render | `render/` | R001 to R066 | 66 | typed rendering, fragments, placeholders, history, render errors |
| template | `template/` | T001 to T069 | 69 | tokenizer (T001 to T039) and content validation (T040 to T069) |
| digest | `digest/` | D001 to D028 | 28 | canonical JSON, digests of every hashed document, bundle loading by digest pin |
| ref | `ref/` | F001 to F054 | 54 | prompt reference parsing, context checks and formatting |
| secrets | `secrets/` | S001 to S014 | 14 | the `agenomic-secrets/1` patterns, scrubbing and the secret-shaped key rule |

`MANIFEST.json` lists every other file of this directory with the sha256 of its
raw bytes, keys sorted. It hashes bytes, not canonical JSON, because vendoring
integrity is about bytes. Regenerate it with
`node scripts/vectors.js --write-manifest` after any change here, this README
included.

## Consuming the vectors

Copy this directory into the implementation repository and record a
`SPEC_VECTORS.lock` next to it:

```json
{ "spec_commit": "<40 hex>", "manifest_sha256": "sha256:<hex of the MANIFEST.json bytes>" }
```

Every harness:

1. checks that sha256 of the vendored `MANIFEST.json` bytes equals the lock;
2. checks that the vendored file set and every file hash equal the manifest;
3. runs every vector whose `consumers` includes it (`rust-cloud`, `rust-cli`,
   `python`, `typescript`);
4. fails on an unknown suite.

A file named `<id>-<slug>.no-rust.json` holds an input that a Rust JSON parser
cannot represent (today only R060, a lone surrogate). Rust harnesses skip such
files before parsing them; their `consumers` never list a Rust consumer.

## File format

Every file is one `agenomic.conformance_vector/v1` document, validated by
`schemas/v0.4/conformance-vector.schema.json`:

```json
{
  "schema": "agenomic.conformance_vector/v1",
  "suite": "render",
  "id": "R001",
  "intent": "a text prompt with one string variable renders the value verbatim",
  "consumers": ["rust-cloud", "rust-cli", "python", "typescript"],
  "input": {},
  "expected": { "ok": true }
}
```

The file name is `<id>-<kebab-slug>.json`, `id` is unique, and its letter
matches the suite.

## Matching rule

A harness asserts exact equality on every member present in `expected`, and on
nothing else:

- `expected.ok` says whether the operation succeeds.
- Success members are compared with deep JSON equality. Lists keep their order.
- On failure, `expected.error` holds the members to assert: `code` (the
  top-level error code), `reason` (ref suite), `item` (the first error item)
  and `details` (bundle loading). Inside `item`, and inside each element of a
  `warnings` list, only the members present are compared; an implementation may
  report more members. A `warnings` list must have exactly the expected length
  and order.
- Item members are `code`, `syntax`, `variable`, `value_path`, `path`,
  `offset`, `line`, `column`, `pattern` and `fragment`. `path` is an RFC 6901
  JSON Pointer into the prompt content, `value_path` a JSON Pointer into the
  render input (`/variables/<name>/...` or `/history/<index>/...`) or, for an
  AJS failure in a document, into that document. `offset` is a 0-based count of
  Unicode code points, `line` is 1 plus the number of LF before the offset,
  `column` is 1 plus the code points since the last LF. CR is an ordinary
  character. `fragment` is a `prompt_id:version` ref.
- Members whose value would depend on implementation internals are left out on
  purpose: R052 and R061 assert only the code, R053 and R054 the code and the
  fragment ref.

## Suite forms

### render

Input: `{ method, content, fragments, variables, options }`.

- `method` is `render`, `text` (`render_text`) or `messages`
  (`render_messages`). `text` on a chat prompt and `messages` on a text prompt
  fail with `kind_mismatch`, checked after content validation and before
  variable binding.
- `fragments` maps `prompt_id:version` to `{ prompt_kind, content }`. It is the
  fragment source: a pin resolves only when its key is present and the digest
  of `content` equals the pinned `content_digest`.
- `options` is `{ strict, history, allow_duplicate_system, secret_policy }`.

Success: `{ ok, kind, text, messages, expanded_template, rendered_document,
rendered_hash, warnings, langchain_parity }`.

- `messages` is the full rendered list: template messages, spliced placeholder
  items and history items, in order. Placeholder and history items are passed
  through unchanged.
- `expanded_template` is the fully expanded template source (fragments spliced,
  braces re-escaped): a string for a text prompt, or an array mirroring the body
  where a role entry becomes `{ role, content }` and a placeholder entry stays
  `{ placeholder, optional }`.
- `rendered_document` is the `agenomic.rendered_prompt/v1` document and
  `rendered_hash` its digest. It never contains placeholder or history items:
  placeholders become `{ placeholder, count }` and history only counts in
  `history_count`.
- `langchain_parity` is `true` on success vectors whose content declares only
  `string` and `messages` variables, rendered in strict mode without the
  `history` option. The Python harness then also checks that `to_langchain()`
  produces the same text or messages.

Failure: `{ ok: false, error: { code: "prompt_render_error", item } }`.

### template

Tokenizer input `{ template }`. Success `{ ok, tokens, source }`, where a token
is `{ t: "literal", text }`, `{ t: "var", name, offset }` or
`{ t: "include", name, offset }` and `source` re-serializes the tokens (it
equals the input). Failure:
`{ ok: false, error: { code: "prompt_template_invalid", item: { code: "syntax_error", syntax, offset, line, column } } }`.

Content input `{ content, fragments }`. Success
`{ ok, validation: { variables: { declared, referenced, placeholders }, warnings, content_digest } }`,
each name list sorted by UTF-16 code units. Failure
`{ ok: false, error: { code, item } }` where `code` is
`prompt_secret_detected` when any secret finding exists and
`prompt_template_invalid` otherwise, and `item` is the first error.

Validation order, so that every implementation reports the same first error:

1. C1 shape: the content is an object; `schema` is present and supported; the
   AJS pre-pass over the whole content (object keys visited in UTF-16 order);
   missing members in the order `schema`, `template_format`,
   `renderer_version`, `kind`, `body`, `variables`, `partials`,
   `output_contract`, `fragments`; unknown members; member values in that same
   order; then the size limits.
2. C2 chat entries, in body order.
3. C3 syntax, one tokenizer error per template string, in body order.
4. C4 fragments: first every pin of the `fragments` map in UTF-16 key order
   (well formed, found, digest equal, prompt kind `fragment`, content kind
   `text`, no partials, no output contract), then the expansion of every
   template string in body order with one include counter shared by the whole
   content.
5. C5 variable use: items tied to the body come first, in body order and token
   order (undeclared, `messages` outside a placeholder, fragment type mismatch,
   placeholder type, duplicate placeholder); then items tied to the
   `variables` map in UTF-16 key order.
6. C6, C7, C8 and C9 in that order; within each rule, body items first, then
   map keys in UTF-16 order.

Warnings are reported in rule order: `fragment_unused` (C4),
`variable_unused` (C5), `secret_shaped_variable_name` (C9), then the render
warning `strict_disabled`.

### digest

Hash input `{ operation: "digest", document, projection? }`. Success
`{ ok, document_type, canonical, digest }`: `document_type` is
`document.schema`, `canonical` the canonical JSON and `digest` the sha256
digest. An optional `projection` `{ rule, from }` proves where the hashed
document comes from:

- `rule: "content"`: `document` is `from.content` and `from.content_digest`
  equals the digest (a prompt version hashes only its content).
- `rule: "without_plan_digest"`: `document` is `from` without its
  `plan_digest` member, and `from.plan_digest` equals the digest.

A document outside the Agenomic JSON Subset fails with
`{ ok: false, error: { item: { code, value_path } } }`. There is no top-level
code: the caller maps the AJS reason to its own error.

Bundle load input `{ operation: "bundle_load", bundle, expected_bundle_digest,
expected_workspace_id, expected_agent_id }` runs the offline load procedure with
digest pinning (the bundle is unsigned). Success
`{ ok, prompt_bundle_digest, prompt_manifest_digest, prompt_refs, managed_slots }`
lists the loaded prompt refs and the managed slots of the root manifest, both
sorted. Failure
`{ ok: false, error: { code: "prompt_digest_mismatch", details: { document: "artifact_set", expected, actual } } }`,
where `expected` is the pin and `actual` the recomputed artifact set digest.

### ref

Input `{ ref, context, workspace_id }`, where `context` is `management` or
`execution` and `workspace_id` is the current workspace, or `null` offline.
Success `{ ok, form, prompt_id, version, alias, workspace_id, canonical,
version_ref }`, absent parts written `null`. `version_ref` is `prompt_id:n`
for the `version` form, and for the `uri` form only when the URI names the
current workspace; otherwise `null`. Failure
`{ ok: false, error: { code: "prompt_ref_invalid", reason } }`,
`{ code: "prompt_ref_unversioned" }` or `{ code: "prompt_ref_cross_workspace" }`.
Vectors with `workspace_id: null` exercise the client-side check and are not
run by the cloud harness, which always knows the workspace.

### secrets

- `{ operation: "scan", text }` gives `{ ok, findings, scrubbed }`. Findings are
  `{ pattern, offset, length }` in code points, sorted by offset, then by the
  order of the pattern table.
- `{ operation: "secret_shaped", keys }` gives `{ ok, secret_shaped }`, one
  boolean per key.
- `{ operation: "scrub_json", value }` gives `{ ok, scrubbed }`.

## Authoring

`node scripts/vectors.js` validates every vector against its schema, checks
names, ids and the manifest, recomputes every digest with `node:crypto`
(digest documents, artifact sets, rendered documents, content digests) and
validates the contents and documents against their v0.4 schemas. The `ref` and
`secrets` suites are checked for shape only; the implementations are their
semantic checkers. `node scripts/vectors.js --compute <file>` prints the
canonical JSON and the digest of the hashable members of a vector, or of a
whole JSON document. A vector changes only together with the implementations
that consume it.
