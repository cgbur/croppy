# Croppy (Rust-only Steps)

This repo has:
- a `croppy` TUI main binary (`cargo run --release -- ...`)
- debug step tools in `examples/` over the shared Rust library.

## Run

1. Prepare one RAW image (random or fixed), then preprocess:
- grayscale
- invert
- levels stretch
- optional 180-degree flip

```bash
cargo run --release --example step01_prepare -- \
  /path/to/raw/folder \
  --recursive \
  --out-dir tmp/step01
```

Useful flags:
- `--raw <PATH>` choose a specific RAW instead of random
- `--invert <true|false>`
- `--flip-180 <true|false>`
- `--black-pct <f32>` and `--white-pct <f32>` (defaults: `2`, `80`)
- `--knee-pct <f32>` soft roll-off width in percent (default `2`)
- `--max-edge <u32>` resize long edge before preprocessing (default `1000`)
- `--keep-color` write `prepared_color.jpg`

2. Run boundary detection on prepared image:

```bash
cargo run --release --example step02_detect_stub -- \
  --input tmp/step01/next.json \
  --out-dir tmp/step02
```

Sampling options:
- `--band-margin-pct <f32>` (default `0.22`)
- `--refine-rotation <true|false>` estimate from fixed inner-edge lines, rotate once, re-detect once (default `true`)
- `--max-refine-abs-deg <f32>` max applied correction in degrees (default `3.0`)

## Handoff JSON

`tmp/step01/next.json` contains paths plus minimal transform metadata:

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

Step 2 outputs:
- `tmp/step02/overlay_initial.jpg` (red outer + yellow initial inner)
- `tmp/step02/overlay.jpg` (same static image, plus cyan refined inner mapped back after rotation refine)
- `tmp/step02/result.json` (normalized bounds + confidence + deterministic 4-edge rotation estimate)
- `tmp/step02/debug.json` (raw vectors + chosen indices)
- `tmp/step02/vertical_profile.csv`, `tmp/step02/horizontal_profile.csv`
- `tmp/step02/vertical_profile_plot.jpg`, `tmp/step02/horizontal_profile_plot.jpg`

## Main TUI

```bash
cargo run --release -- .
```

TUI flow:
- scan RAWs from path
- choose actions (previews / sidecar writes)
- select subset of files
- run in parallel with live progress
- outputs previews to `previews/`

## Dev checks

```bash
cargo +nightly fmt
cargo clippy --tests
```
