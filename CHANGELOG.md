# Changelog

## [1.0.0] — 2026-08-18

First public release. UNLTD Inference v1.0.0 — functional MVP, CPU-first, disk-first.

- **GGUF loader**: hand-written parser (header, metadata, tensor index) + whole-file mmap with tensor views — weights never copied to heap
- **Qwen3.5 / Ornith inference**: full 32-layer forward (24 recurrent GatedDeltaNet layers + 8 full-attention layers, M-RoPE), validated against llama.cpp oracle
- **Quantized kernels**: scalar reference path for Q4_K/Q6_K/Q8_0 (and F32/F16) — no SIMD yet
- **Tokenizer**: GPT-2 byte-level BPE built from GGUF metadata, bit-exact vs llama.cpp for the validated prompt (qwen35 pre-tokenizer)
- **Greedy generation**: deterministic temperature-0 decode with incremental KV cache and recurrent state
- **Memory budget** (`--memory-budget`): pre-allocation planning, accounting of controlled memory (KV + scratch + runtime state), `REFUSING TO RUN` with concrete numbers (exit 2); mapped bytes ≠ controlled bytes — model > budget is valid
- **Disk-first execution**: OS-backed mmap demand paging (not explicit layer streaming)
- **Validation suite**: 107 workspace tests; oracle fixtures and comparison workflows (`min-forward`, `forward-oracle`)
- **CLI**: `inspect`, `tokenize`, `run` (user-facing) + `min-forward`, `forward-oracle` (validation)

Known limitations are documented in `README.md` §20 (CPU-only, Windows-validated, no sampling, slower than llama.cpp, documented numerical divergence vs optimized oracle after longer contexts).
