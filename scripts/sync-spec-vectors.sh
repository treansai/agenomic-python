#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: scripts/sync-spec-vectors.sh <spec-commit> [<agenomic-spec checkout>]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC_DIR="${2:-${AGENOMIC_SPEC_DIR:-$ROOT/../agenomic-spec}}"
DEST="$ROOT/tests/fixtures/spec_vectors"
SOURCE_PATH="conformance/vectors/prompts"

COMMIT="$(git -C "$SPEC_DIR" rev-parse --verify "$1^{commit}")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git -C "$SPEC_DIR" archive --format=tar "$COMMIT" "$SOURCE_PATH" | tar -x -C "$WORK"

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$WORK/$SOURCE_PATH/." "$DEST/"

MANIFEST_SHA="$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$DEST/MANIFEST.json")"
printf '{ "spec_commit": "%s", "manifest_sha256": "sha256:%s" }\n' "$COMMIT" "$MANIFEST_SHA" > "$DEST/SPEC_VECTORS.lock"

echo "vendored $SOURCE_PATH at $COMMIT into tests/fixtures/spec_vectors"
