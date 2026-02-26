## Workflow Requirements

When explicitly asked to commit changes, beforehand:

1. Run formatting with nightly rustfmt:
   `cargo +nightly fmt`
2. Run clippy against tests:
   `cargo clippy --tests`
