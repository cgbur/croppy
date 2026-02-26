# Croppy

Croppy is a Rust TUI tool that batch-generates Lightroom crop sidecars for scanned film negatives.

## Why this exists

After scanning negatives (for example with holders like the VALOI easy35 or a custom camera scanning setup), frames are often slightly off-center, skewed, or inconsistent.  
Manually fixing crop on every frame in Lightroom before running Negative Lab Pro is slow.

Croppy is meant to remove most of that manual first-pass crop work.

## What it does

- Scans a folder of RAW files.
- Detects each frame boundary with lightweight signal-processing heuristics.
- Writes Lightroom-compatible `.xmp` crop sidecars next to each RAW.
- Optionally writes preview JPEGs to inspect results.

Croppy does **not** modify, delete, or rewrite your RAW files.

## Typical Lightroom + Negative Lab Pro flow

1. Bulk import your scanned RAW negatives into Lightroom Classic.
2. Run Croppy on the scan directory.
3. In Croppy, select files and enable `XMP Sidecars`.
4. Back in Lightroom, select the files and use `Metadata > Read Metadata from Files` (or right-click equivalent) to apply crops.
5. Do any needed flips/inversion/WB adjustments in Lightroom.
6. Run Negative Lab Pro in bulk, usually roll by roll.

## Install

### Option 1: Download a prebuilt release (recommended)

1. Open the latest release page: `https://github.com/cgbur/croppy/releases/latest`
2. Download the asset for your platform:
- Linux x86_64: `croppy-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz`
- macOS (Apple Silicon): `croppy-vX.Y.Z-aarch64-apple-darwin.tar.gz`
3. Extract the archive.
4. Move the binary into a directory on your `PATH`.

Windows release assets are temporarily disabled while `rsraw`/`rsraw-sys` Windows builds are being resolved.

Linux / macOS example:

```bash
mkdir -p "$HOME/.local/bin"
cp croppy "$HOME/.local/bin/croppy"
chmod +x "$HOME/.local/bin/croppy"
```

If needed, add to shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Option 2: Build from source

```bash
cargo build --release
```

Binary path:

```text
target/release/croppy
```

### TODO: one-line installer

A curl-only `install.sh` that auto-detects OS/arch and installs the latest release is planned, but not shipped yet.

## Run

Basic run (after install):

```bash
croppy /path/to/raw/folder
```

If you built from source but did not install into `PATH`:

```bash
./target/release/croppy /path/to/raw/folder
```

Useful options:

- `--max-edge <N>` downscale long edge before detection (default `1000`)
- `--out-dir <PATH>` where preview output folder is created (default current directory)
- `--recursive` recursive scan flag (scan is recursive in normal use)

Supported RAW extensions include: `ARW`, `CR2`, `CR3`, `NEF`, `NRW`, `RAF`, `ORF`, `RW2`, `DNG`, `PEF`.

## TUI controls

- File selection:
  - `Up/Down` move cursor
  - `Space` toggle one file
  - `a` select all
  - `u` select all files without existing `.xmp`
  - `n` clear selection
- Outputs and preview mode:
  - `x` toggle XMP sidecars
  - `v` toggle preview JPEGs
  - `p` cycle preview mode (`Overlay`, `Final Crop`, `Crop + Frame`)
- Crop tuning:
  - `-` / `=` adjust `Final Crop Scale` by `0.25%`
  - `0` reset crop scale
  - `,` / `.` adjust horizontal trim
  - `;` / `'` adjust vertical trim
  - `9` reset trim defaults
- Run/cancel:
  - `Enter` start
  - `y` / `n` confirm/cancel XMP overwrite prompt
  - `c` cancel active run
  - `q` quit

## Trim mode explained

`XMP Trim` is a normalized adjustment applied symmetrically on each axis:

- Horizontal trim moves both left and right crop edges.
- Vertical trim moves both top and bottom crop edges.
- Positive trim shrinks the crop inward.
- Negative trim expands it outward.

Defaults are:

- Horizontal trim: `+0.0000`
- Vertical trim: `+0.0050`

That default vertical trim is a small inward bias that often helps remove a little extra edge/border in real scans.

## Output behavior and safety

- XMP output: `<raw_basename>.xmp` next to each RAW.
- Preview output: `<out-dir>/previews/<raw_basename>.jpg`.
- If XMP already exists, Croppy prompts before overwriting.
- RAW source files are never modified.

## Limitations

- Crops can be wrong on difficult frames, especially high-contrast black and white scans.
- Crops are currently free-form and not constrained to a fixed `3:2` aspect ratio.
- You should still do a quick manual review pass.

## How it works (high level)

- RAW decode with `libraw` bindings (`rsraw`/`rsraw-sys`).
- Preprocess to improve edge contrast (grayscale, invert, levels, 180-degree normalization).
- Detect frame bounds from image profiles/edge scans.
- Attempts one deterministic rotation-refine pass from fitted edge lines, and only applies it when the estimate is valid/plausible.
- Write Lightroom crop metadata (`CropLeft`, `CropTop`, `CropRight`, `CropBottom`, `CropAngle`) to XMP sidecars.

No machine learning, no OpenCV, minimal dependencies.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
