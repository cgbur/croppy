## Workflow Requirements

Before committing any changes in this repository:

1. Run formatting with nightly rustfmt:
`cargo +nightly fmt`
2. Run clippy against tests:
`cargo clippy --tests`

## Rotation Debug Sandbox

Use this flow to debug rotation on a single RAW without touching originals.

Template:

```bash
RAW_SRC="/mnt/c/Users/cburg/Pictures/YYYY/YYYY-MM-DD/FILE.ARW"
CASE_ID="FILE"
CASE_DIR="tmp/cases/${CASE_ID}"

mkdir -p "${CASE_DIR}/raw" "${CASE_DIR}/step01" "${CASE_DIR}/step02"
cp -f "${RAW_SRC}" "${CASE_DIR}/raw/${CASE_ID}.ARW"

cargo run --release --features debug-artifacts --example step01_prepare -- \
  "${CASE_DIR}/raw" \
  --raw "${CASE_DIR}/raw/${CASE_ID}.ARW" \
  --out-dir "${CASE_DIR}/step01"

cargo run --release --features debug-artifacts --example step02_detect_stub -- \
  --input "${CASE_DIR}/step01/next.json" \
  --out-dir "${CASE_DIR}/step02"
```
