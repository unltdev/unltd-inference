# Contributing

Thanks for your interest in UNLTD Inference. All contributions are made under the **Apache License 2.0** (see [LICENSE](LICENSE)); third-party attribution is retained in [NOTICE](NOTICE).

## Getting started

```powershell
git clone https://github.com/unltdev/unltd-inference.git
cd unltd-inference
```

Requirements:

- **Rust 1.80+** (MSRV declared in `Cargo.toml`)
- **Windows x86-64** is the only platform validated end-to-end; Linux is not yet validated

Create a branch for your work:

```powershell
git checkout -b feat/my-change
```

## Build and validate

```powershell
cargo build --release -p unltd-cli
cargo fmt --check
cargo check --workspace
cargo test --workspace
```

All checks must pass before opening a Pull Request (the CI runs the same checks on every PR).

## What must NOT be committed

- GGUF model files or any model weights
- logs, crash dumps, or large generated artifacts
- credentials, tokens, `.env` files, or any other secrets

## Pull Requests

1. Keep PRs small and focused on one change.
2. Changes that modify behavior must include or update tests.
3. Changes to architecture, model support, or the memory design must come with reproducible evidence (commands run, measured numbers, comparison against the oracle where applicable).
4. There is **no requirement** of bit-exactness against llama.cpp outside the numerical contract documented in the README — read it before changing numeric paths.
5. Link any related issue.

Main is protected: changes land via Pull Request with a green CI (`Rust checks`), merged with **squash**.

## Reporting issues

Use the issue templates. Security vulnerabilities must **never** be reported in public issues — see [SECURITY.md](SECURITY.md).
