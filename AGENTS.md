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

mkdir -p "tmp/raw-sandbox" "tmp/cases/${CASE_ID}/step01" "tmp/cases/${CASE_ID}/step02"
cp -f "${RAW_SRC}" "tmp/raw-sandbox/${CASE_ID}.ARW"

cargo run --release --example step01_prepare -- \
  tmp/raw-sandbox \
  --raw "tmp/raw-sandbox/${CASE_ID}.ARW" \
  --out-dir "tmp/cases/${CASE_ID}/step01"

cargo run --release --example step02_detect_stub -- \
  --input "tmp/cases/${CASE_ID}/step01/next.json" \
  --out-dir "tmp/cases/${CASE_ID}/step02"
```
