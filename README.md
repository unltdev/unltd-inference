# UNLTD Inference

Disk-first, CPU-first LLM inference runtime written in Rust.

Version: 1.0.0 · Language: Rust · Platform validated: Windows x86-64 · Execution: CPU-first · Model format: GGUF · License: Apache-2.0

> Developed by UNLTD as an experimental / research runtime. Correctness was prioritized over performance throughout the project; see [Current limitations](#20-current-limitations) for what this means in practice.

---

## 1. What is UNLTD Inference?

UNLTD Inference is an experimental research runtime in Rust for running GGUF models locally, built around one central idea:

```
model_size > available_or_configured_memory
```

The memory limit is an **explicit feature** of the runtime, not a failure mode: the engine is designed so that a model larger than the configured memory budget can still run, because model weights live in an OS-backed memory map while the runtime controls only a small, bounded amount of memory.

The project was conceptually inspired by the disk-first inference work investigated in `kimi-k3-in-c`, but:

- it is **not a fork** of that project;
- it is **not a port**;
- it contains **no Kimi K3 code**;
- it has its own architecture (workspace, IR, kernels, CLI);
- it targets extensible models/architectures through its own internal representation.

## 2. Why does this project exist?

The conceptual difference:

```
Traditional:
model → RAM → compute

UNLTD Inference:
model on disk
     ↓
mmap / OS demand paging
     ↓
bounded runtime-controlled memory
     ↓
compute
```

Note: version 1.0 uses **OS-backed mmap demand paging** — it does *not* implement explicit layer streaming (see [Disk-first execution](#13-disk-first-execution)).

## 3. Current capabilities

An honest list of what v1.0.0 implements and validates:

- GGUF parsing (header, metadata, tensor index)
- mmap-backed model access (weights are views into the map — no heap copies)
- Qwen3.5-compatible forward path used by the validated model (hybrid recurrent/full-attention layers, GatedDeltaNet, M-RoPE)
- Q4_K quantized dot kernels required by the validated model (scalar reference path)
- GPT-2 byte-level BPE tokenizer built from GGUF metadata (qwen35 pre-tokenizer)
- greedy generation (temperature 0, deterministic)
- incremental KV cache (full-attention layers) + recurrent state (conv + GDN)
- configurable memory budget with pre-allocation planning
- memory accounting (KV, scratch, runtime state, controlled caches)
- refusal of impossible configurations (`REFUSING TO RUN`, exit code 2)
- deterministic generation under the documented numerical contract
- Windows working-set / I/O measurement of the running process (via PowerShell)
- CLI tools for inspection, tokenization, and inference

## 4. Validated model

The single model validated end-to-end for v1.0.0:

```
Ornith 1.0 9B
Q4_K_M
GGUF
~5.63 GB
```

> GGUF is a container format, not an architecture guarantee. Only the architecture/model actually exercised in this project is declared validated; there is no claim of universal GGUF support.

## 5. Architecture

```
                GGUF Model
                    │
                    ▼
             Model Loader
                    │
             mmap-backed data
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
 Architecture                Tokenizer
       │                         │
       ▼                         ▼
 Tensor / Quant Kernels      Token IDs
       │                         │
       └────────────┬────────────┘
                    ▼
              Forward Engine
                    │
                  KV Cache
                    │
                    ▼
                  Logits
                    │
                 Argmax
                    │
                    ▼
              Generated Text
```

The Memory Manager is transversal to the whole pipeline: it plans the configured memory budget (KV + scratch + runtime state + controlled caches) **before** any allocation and refuses with concrete numbers if the configuration cannot fit.

## 6. Workspace structure

Eight Rust crates, dependency order: `tensor ← core ← {model-loader, architectures, memory, tokenizer} ← generation ← cli`.

| Crate | Purpose |
|---|---|
| `unltd-tensor` | Tensor data types and quantized kernels (Q4_K/Q6_K/Q8_0 dots, scalar reference path) |
| `unltd-core` | Shared error type (`LoadError`) with refuse-rather-than-guess semantics, formatting helpers |
| `unltd-model-loader` | Hand-written GGUF parser and `MappedWeights` (whole-file mmap, tensor views) |
| `unltd-architectures` | Internal IR (attention/FFN/norm kinds, layer/model specs) and the Qwen3.5 config |
| `unltd-memory` | Memory budget parser and accounting component (Fase 9), residency policy scaffolding |
| `unltd-tokenizer` | GPT-2 byte-level BPE tokenizer built from GGUF metadata (qwen35/qwen2/gpt2 pre-tokenizers) |
| `unltd-generation` | Forward engine (`Qwen35Forward`, session/KV cache) and the greedy loop |
| `unltd-cli` | Command-line interface: `inspect`, `min-forward`, `forward-oracle`, `tokenize`, `run` |

## 7. Building

Windows PowerShell:

```powershell
cargo build --release -p unltd-cli
```

Executable: `target\release\unltd.exe`

Requirements (as actually used and validated):

- Rust stable (MSRV declared in `Cargo.toml`: 1.80)
- Windows x86-64 (the only platform validated end-to-end; Linux is **not** validated yet)
- a CPU compatible with the scalar kernels (no SIMD requirement at v1.0)
- enough storage for the model file (the model is never copied to RAM)

## 8. CLI

Real commands:

```
unltd.exe inspect        # model header, metadata, tensor table
unltd.exe tokenize       # tokenize text with the model's tokenizer
unltd.exe run            # greedy generation
unltd.exe min-forward    # validation-only: embed + norms + output head vs oracle bins
unltd.exe forward-oracle # validation-only: full 32-layer prefill dump vs oracle bins
```

User-facing: `inspect`, `tokenize`, `run`.
Validation/development: `min-forward`, `forward-oracle`.

The model path is a **positional** argument in every command (there is no `--model` flag). `--help` on each subcommand is the source of truth for syntax.

## 9. Inspect a model

```powershell
.\target\release\unltd.exe inspect "path\to\ornith-1.0-9b-Q4_K_M.gguf"
```

Prints: file size, GGUF version, metadata key/values, the tensor table (name, dims, type, offset, exact bytes, % of file), a summary of total tensor bytes and top tensors. `--no-tensors` skips the per-tensor table.

## 10. Tokenize

```powershell
.\target\release\unltd.exe tokenize "path\to\ornith-1.0-9b-Q4_K_M.gguf" --text "The capital of France is"
```

Prints the tokenizer kind and pre-tokenizer, vocab size, BOS/EOS, and the per-token table (id, piece, decoded bytes) plus the full decoded text. Tokenization is raw (no BOS), matching the validation mode.

## 11. Generate text

```powershell
.\target\release\unltd.exe run `
  "path\to\ornith-1.0-9b-Q4_K_M.gguf" `
  --prompt "The capital of France is" `
  --max-tokens 3 `
  --temperature 0 `
  --memory-budget 4G
```

The model is the **positional** argument. `--temperature` only accepts 0 (deterministic greedy; sampling is not implemented). The run prints a per-step table against the recorded oracle sequence, decoded text, timings, and (with `--memory-budget`) a memory plan before allocating and a measured memory report at the end.

## 12. Memory budget

```powershell
--memory-budget 4G
```

Accepted sizes: `512M`, `1G`, `2G`, `4G`, `8G`, `512MB`, `4GB`, or raw bytes (e.g. `536870912`). Prefixes are binary (1G = 1024³), case-insensitive. Empty, zero, unknown suffix, overflow, and non-numeric values are rejected with a typed error.

What the budget controls — memory UNLTD explicitly and deliberately keeps resident:

- KV cache (exact formula from context and attention config)
- scratch (peak per-step activations + decode-loop logits)
- runtime state (tokenizer structures, GGUF index, recurrent session state: conv ring + GDN)
- controlled weight buffers/caches (zero in v1.0: weights are mmap views)

What it does **not** control, and must not be confused with:

```
memory budget != process RSS
memory budget != mapped virtual bytes
```

The whole model file is mapped as virtual address space (5.63 GB mapped for a 4G or even 113 MB budget — this is valid and intended). Pages the OS keeps resident after touching them belong to the OS page cache / process working set; the measured peak working set can therefore exceed the budget without violating it. If the minimum mandatory controlled memory exceeds the budget, the runtime prints `REFUSING TO RUN` with a concrete table and exits with code 2 — before allocating anything.

## 13. Disk-first execution

What v1.0 actually implements, precisely:

```
GGUF file
   ↓
memory map
   ↓
OS demand paging
   ↓
tensor views
   ↓
bounded runtime allocations
```

The runtime does **not** implement explicit layer streaming, prefetch, or double buffering. The correct description is **disk-first / mmap-backed execution**: the model never enters heap memory; the OS pages the file on demand, and the runtime's own allocations are bounded and independent of model size.

## 14. Real validation results

Measured during the Fase 9-10 campaign (2026-08-18), Ornith 1.0 9B Q4_K_M:

| Metric               | Result                     |
| -------------------- | -------------------------- |
| Model                | Ornith 1.0 9B Q4_K_M       |
| Model size           | 5.63 GB                    |
| Memory budget tested | 4 GB                       |
| Lower budget tested  | 3 GB                       |
| Minimum budget tested| 113.06 MB (exact mandatory)|
| Controlled memory    | ~113.19 MB (ctx 8)         |
| Peak working set     | ~5.24 GB (incl. OS page cache) |
| KV cache             | ~524 KB for tested context |
| Tests                | 107 passed                 |
| Generation           | PASS                       |
| Model > budget       | PASS                       |

## 15. Performance

From the recorded campaign (warm OS cache, CPU scalar path):

```
Prefill ~73 s        (14.6 s/token for the 5-token prompt)
Decode  ~14.3 s/token
```

The engine is scalar/reference oriented: correctness was prioritized over optimization, llama.cpp is much faster on the same machine, and v1.0 does **not** aim to compete on throughput. These numbers document what the runtime does today, not a performance claim.

## 16. Validation against llama.cpp

llama.cpp was used as the oracle throughout the project (same model, same prompt, greedy, temperature 0):

- the tokenizer is **bit-exact** for the validated prompt;
- the first **11** greedy generation steps match the oracle exactly;
- a later divergence (first at generated token 12) is known and documented: scalar reference kernels vs the oracle's optimized AVX2/repacked kernels accumulate numerical differences that flip an argmax near-tie after ~16 context tokens;
- there is **no guarantee of bit-exact internal tensors**;
- the accepted contract prioritizes runtime correctness and determinism (same prompt + weights → same sequence, run after run).

This divergence is documented openly in `docs/PHASE-7-8-CHECKPOINT.md`, including the measured decision logits.

## 17. Memory experiment

The central result of the project:

```
Model file:          5.63 GB
Configured budget:   3–4 GB
Controlled memory:   ~113 MB
Generation:          successful
```

The model runs under a budget smaller than the model itself. Peak RSS may exceed the budget because of mmap / OS page cache — that is expected and is not a budget violation (see [Memory budget](#12-memory-budget)).

## 18. Testing

```powershell
cargo fmt --check
cargo check --workspace
cargo test --workspace
```

107 tests pass at v1.0.0 (tokenizer 27, generation 11, model-loader 17, tensor 35, architectures 4, memory 13).

## 19. Design principles

```
Correctness before performance
Measurements before assumptions
Refuse rather than guess
Model files stay on disk
Memory is an explicit resource
Scalar path acts as numerical reference
```

## 20. Current limitations

Honest and specific:

- only the validated architecture is declared supported (no universal GGUF claim)
- CPU-only, scalar kernels
- Windows x86-64 is the only end-to-end validated platform (Linux not yet)
- much slower than llama.cpp (~14 s/token decode vs ~230 ms/token)
- no sampling (temperature 0 / greedy only)
- no GPU, no SIMD, no multithreading
- no MoE / expert routing support
- no explicit expert streaming
- no explicit layer prefetch or double buffering
- known numerical divergence vs the optimized oracle after longer contexts (documented, see §16)
- mmap page cache is governed by the OS, not by `--memory-budget`

## 21. Roadmap

Possible future lines — **not part of v1.0**:

```
SIMD / AVX2 kernels
multithreading
additional architectures
explicit streaming experiments
MoE / expert cache
Linux validation
performance instrumentation
```

## 22. Documentation

| Document | Contents |
|---|---|
| `docs/AUDIT.md` | Design audit and lessons inherited from the kimi-k3-in-c investigation |
| `docs/ARCHITECTURE.md` | Workspace layout, dependency direction, engine design decisions |
| `docs/MEMORY-DESIGN.md` | Memory management design: budgets, residency policy, disk readers, expert cache |
| `docs/MODEL-SUPPORT.md` | Model/architecture support policy and criteria |
| `docs/MODEL-CANDIDATES.md` | Candidate models evaluated for the project |
| `docs/ROADMAP.md` | Phase roadmap (Fases 0-10 completed; future work) |
| `docs/PHASE-6-CHECKPOINT.md` | Fase 6 checkpoint: full forward, numerical contract, oracle comparisons |
| `docs/PHASE-7-8-CHECKPOINT.md` | Fase 7-8 checkpoint: tokenizer bit-exactness, greedy generation, measured divergence |
| `docs/PHASE-9-10-CHECKPOINT.md` | Fase 9-10 checkpoint: memory budget, disk-first execution, real gate results |
| `docs/QWEN35-FORWARD.md` | Qwen3.5 forward path reference notes |

## 23. License

UNLTD Inference is open source, licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full license text.

Portions of this project (tokenizer splitter logic and regex patterns, scalar quantized kernels, Qwen3.5 forward structure) are implemented from llama.cpp / ggml's MIT-licensed algorithms. Their MIT notice is retained in [NOTICE](NOTICE).

Contributions are welcome under the terms of the Apache License 2.0.

## 24. Credits / References

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — used as the validation oracle, GGUF format reference, and benchmark; portions of the tokenizer and scalar kernels follow its MIT-licensed reference algorithms (see [NOTICE](NOTICE)); no affiliation
- The GGUF ecosystem — model container format
- [kimi-k3-in-c](https://github.com/MoonshotAI/kimi-k3-in-c) — conceptual inspiration for disk-first inference; not a fork, not a port, contains no K3 code; no affiliation

*This project is not affiliated with llama.cpp, Moonshot AI, Kimi, or any of the above projects.*
