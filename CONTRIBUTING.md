# Contributing to Croppy

Thanks for contributing. This project is intentionally simple and pragmatic, and improvements are welcome.

## Environment prerequisites

`croppy` depends on `rsraw-sys`, which uses `bindgen` during build. `bindgen` requires `libclang` at build time.

If `cargo build` fails with `Unable to find libclang`, set `LIBCLANG_PATH` to a directory containing `libclang.so`.

For Nix/Home-Manager users, configure it as a session variable in your Home-Manager module:

```nix
home.sessionVariables = {
  LIBCLANG_PATH = "${pkgs.llvmPackages_21.libclang.lib}/lib";
};
```

Then apply Home-Manager:

```bash
home-manager switch --flake <your-config>#<your-user>
```

One-off fallback (current shell only):

```bash
export LIBCLANG_PATH="$(nix eval --raw nixpkgs#llvmPackages.libclang.lib.outPath)/lib"
```

This repository does not currently ship a project `flake.nix`/`.envrc`. We may add one later so `direnv` can provide `libclang` and other build dependencies automatically per-project.

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

For RAW-focused debugging, use the helper under `scripts/`:

```bash
scripts/debug.sh /path/to/file.ARW
```

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
