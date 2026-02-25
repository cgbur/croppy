# Contributing to Croppy

Thanks for contributing. This project is intentionally simple and pragmatic, and improvements are welcome.

## Required local checks

Before committing changes, run:

```bash
cargo +nightly fmt
cargo clippy --tests
```

Recommended before opening a PR:

```bash
cargo test
```

## Development run

Main app:

```bash
cargo run --release -- /path/to/raw/folder
```

## Rotation debug sandbox

Use this flow to debug rotation on a single RAW without touching originals.

```bash
RAW_SRC="/path/to/your/Pictures/YYYY/YYYY-MM-DD/FILE.ARW"
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

## Debug examples reference

### Step 01: prepare one RAW

```bash
cargo run --release --features debug-artifacts --example step01_prepare -- \
  /path/to/raw/folder \
  --recursive \
  --out-dir tmp/step01
```

Useful flags:

- `--raw <PATH>` choose a specific RAW instead of random
- `--invert <true|false>`
- `--flip-180 <true|false>`
- `--black-pct <f32>` and `--white-pct <f32>` (defaults `2`, `80`)
- `--knee-pct <f32>` soft roll-off width percent (default `2`)
- `--max-edge <u32>` resize long edge before preprocessing (default `1000`)
- `--keep-color` write `prepared_color.jpg`

### Step 02: boundary detect + rotation refine

```bash
cargo run --release --features debug-artifacts --example step02_detect_stub -- \
  --input tmp/step01/next.json
```

Useful flags:

- `--out-dir <PATH>` output directory (default: same directory as `--input`)
- `--refine-rotation <true|false>` run one deterministic rotation-refine pass (default `true`)
- `--max-refine-abs-deg <f32>` max applied correction in degrees (default `3.0`)

### Handoff JSON

`tmp/step01/next.json` contains paths and transform metadata:

```json
{
  "raw": "/absolute/or/mnt/path/file.ARW",
  "prepared": "tmp/step01/prepared.jpg",
  "preprocess": "bw+invert+levels+flip180",
  "transform": {
    "raw_width": 6336,
    "raw_height": 4224,
    "prepared_width": 1000,
    "prepared_height": 667,
    "flip_180": true
  }
}
```

Common step-2 outputs:

- `overlay_initial.jpg` (initial inner box)
- `overlay.jpg` (initial + refined/backprojected box)
- `overlay_cropped.jpg` (cropped rotated result when refine applies)
- `overlay.json` (overlay geometry)
- `result.json` (normalized bounds + confidence + rotation data)
- `debug.json` and profile plots/CSVs when debug dump is enabled

## CI

CI runs on PRs and pushes to `main`:

- `cargo +nightly fmt --all -- --check`
- `cargo +stable check --all-targets --locked`
- `cargo +stable clippy --tests --locked`
- `cargo +stable test --locked`

## Releases

Tagging `v*` triggers release automation. The workflow:

- verifies the tag commit is on `main`
- builds `croppy` for Linux, Windows, and macOS
- uploads archives and `SHA256SUMS.txt` to the GitHub Release

Example:

```bash
git checkout main
git pull
git tag v0.1.0
git push origin v0.1.0
```
