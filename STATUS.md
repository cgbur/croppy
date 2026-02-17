# Croppy Session Handoff

Date: 2026-02-17

## Planning docs moved here

- `notes/croppy-plan.md`
- `notes/lightroom-crop-research.md`

## Implemented in this repo

- Added `AGENTS.md` with required pre-commit checks.
- Added initial Rust scaffold in `src/main.rs`:
  - `croppy detect <input>` command
  - RAW discovery for common extensions
  - skip-by-default filtering for:
    - existing `.xmp` sidecar
    - existing crop/rotation edits in `.lrcat` (if `--lrcat` is passed)
  - RAW decode via `rawloader`
  - OpenCV-based first-pass rectangle + angle detection
  - optional preview overlay export via `--preview-dir`
  - JSON report output via `--out`

- Updated dependencies in `Cargo.toml`:
  - `clap`, `anyhow`, `serde`, `serde_json`, `walkdir`, `rawloader`, `rusqlite`, `opencv`

## Current blockers in this shell

Build failed because toolchain/runtime deps are missing in this session:

- `cc` compiler not found
- `cargo +nightly fmt` not usable in this environment (no rustup-managed cargo)

## Next steps for next session

1. Enter a nix shell/devshell with toolchain + system deps (`rustc/cargo`, `clang` or `gcc`, OpenCV, SQLite).
2. Run:
   - `cargo +nightly fmt`
   - `cargo check`
   - `cargo clippy --tests`
3. Run a smoke test detect pass against your scan folder:
   - `cargo run -- detect /mnt/c/Users/cburg/Pictures/2026/2026-02-11 --lrcat /mnt/c/Users/cburg/Pictures/Lightroom/Lightroom\ Catalog.lrcat --preview-dir previews --max-files 20`
4. Tune detection quality using generated previews and report.

## Scope not yet implemented

- No XMP writing/apply flow yet.
- No TUI/review mode yet.
- No sequence-aware smoothing yet.
