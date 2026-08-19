<div align="center">

# 🚀 UNLTD Inference

### Disk-first, CPU-first LLM inference runtime written in Rust

<p>
  <strong>Run GGUF models locally with explicit memory budgeting.</strong><br/>
  Built for constrained hardware, correctness-first validation, and mmap-backed disk-first execution.
</p>

<p>
  <img src="https://img.shields.io/badge/version-1.0.0-1D4ED8?style=for-the-badge" alt="version" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-0B1220?style=for-the-badge" alt="license" />
  <img src="https://img.shields.io/badge/platform-Windows%20x86--64-06B6D4?style=for-the-badge" alt="platform" />
  <img src="https://img.shields.io/badge/runtime-Rust-E2E8F0?style=for-the-badge&logo=rust&logoColor=black" alt="rust" />
</p>

<p>
  <a href="https://github.com/unltdev/unltd-inference/actions/workflows/ci.yml"><img src="https://github.com/unltdev/unltd-inference/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/github/stars/unltdev/unltd-inference?style=flat-square&color=1D4ED8" alt="stars" />
  <img src="https://img.shields.io/github/forks/unltdev/unltd-inference?style=flat-square&color=06B6D4" alt="forks" />
  <img src="https://img.shields.io/github/issues/unltdev/unltd-inference?style=flat-square&color=0B1220" alt="issues" />
  <img src="https://img.shields.io/github/last-commit/unltdev/unltd-inference?style=flat-square&color=16A34A" alt="last commit" />
  <img src="https://img.shields.io/badge/tests-107%20passing-16A34A?style=flat-square" alt="tests" />
</p>

<p>
  <a href="#-what-is-unltd-inference">What is it?</a> •
  <a href="#-why-this-project-exists">Why?</a> •
  <a href="#-how-it-works">How it works</a> •
  <a href="#-validated-models">Validated models</a> •
  <a href="#-quick-start">Quick start</a> •
  <a href="#-real-results">Real results</a> •
  <a href="#-limitations">Limitations</a>
</p>

</div>

---

## ✨ What is UNLTD Inference?

**UNLTD Inference** is an open-source LLM runtime written in **Rust** for running **GGUF** models locally.

Its core idea is simple:

> **A model can be larger than the configured runtime memory budget and still run successfully.**

Instead of copying model weights into heap memory, UNLTD Inference uses:

- **memory-mapped GGUF files**
- **OS demand paging**
- **bounded runtime-controlled memory**
- **explicit memory accounting**
- **deterministic greedy inference**

This makes the project especially useful for studying and validating **disk-first inference under constrained memory conditions**.

> Developed by **UNLTD** as an experimental / research runtime.  
> **Correctness was prioritized over performance** throughout the project.

---

## 🎯 Why this project exists

Most local inference runtimes assume a simpler execution model:

```text
model → RAM → compute
```

UNLTD Inference explores a different design:

```text
model on disk
     ↓
mmap / OS demand paging
     ↓
bounded runtime-controlled memory
     ↓
compute
```

The goal is not to beat highly optimized runtimes in throughput.

The goal is to prove, measure, and document that:

```text
model_size > configured_memory_budget
```

can still work correctly.

This makes UNLTD Inference useful as both:

* a **practical research runtime**, and
* a **technical reference** for memory-constrained local inference.

---

## 🧠 How it works

UNLTD Inference follows this execution model:

```text
GGUF model on disk
        ↓
memory map (mmap)
        ↓
OS demand paging
        ↓
tensor views
        ↓
bounded runtime memory
        ↓
forward pass
        ↓
KV cache / recurrent state
        ↓
greedy decoding
        ↓
generated text
```

### In short

* The **model stays on disk**
* The runtime controls only a **small, explicit memory budget**
* The model file may be **larger than the configured budget**
* The engine still performs real inference and generates text

> ⚠️ **Important:** v1.0.0 uses **mmap-backed disk-first execution**.
> It does **not** yet implement explicit layer streaming, prefetch, or double buffering.

---

## ✅ Current capabilities

An honest summary of what **v1.0.0** implements and validates:

* ✅ GGUF parsing (header, metadata, tensor index)
* ✅ mmap-backed model access (weights are views into the map — no heap copies)
* ✅ Qwen3.5-compatible forward path used by the validated model
* ✅ scalar quantized kernels (Q4_K / Q6_K / Q8_0; Q4_K validated path)
* ✅ GPT-2 byte-level BPE tokenizer built from GGUF metadata
* ✅ deterministic greedy generation (`temperature = 0`)
* ✅ incremental KV cache for full-attention layers
* ✅ recurrent state support used by the validated architecture
* ✅ configurable memory budget with pre-allocation planning
* ✅ memory accounting (KV, scratch, runtime state, controlled caches)
* ✅ refusal of impossible configurations (`REFUSING TO RUN`, exit code 2)
* ✅ deterministic generation under the documented numerical contract
* ✅ Windows working-set / I/O measurement of the running process
* ✅ CLI tools for inspection, tokenization, and inference

---

## 🧪 Validated models

UNLTD Inference **does not claim universal GGUF support**.

The model validated end-to-end for **v1.0.0** is:

| Model             |        Format |        Size | Status            |
| ----------------- | ------------: | ----------: | ----------------- |
| **Ornith 1.0 9B** | GGUF / Q4_K_M | **5.63 GB** | ✅ Fully validated |

### Architecture path validated

* **Qwen3.5-compatible forward path**
* hybrid recurrent + full-attention layers
* GatedDeltaNet / M-RoPE path used by Ornith
* GPT-2 byte-level BPE tokenizer with qwen35 pre-tokenizer behavior

> **Important:** GGUF is a **container format**, not an architecture guarantee.
> v1.0.0 only declares support for the architecture/model actually exercised and validated.

---

## ⚡ Quick start

### Build

```powershell
cargo build --release -p unltd-cli
```

### Check the CLI

```powershell
.\target\release\unltd.exe --version
.\target\release\unltd.exe --help
.\target\release\unltd.exe run --help
```

### Run inference

```powershell
.\target\release\unltd.exe run `
  "path\to\ornith-1.0-9b-Q4_K_M.gguf" `
  --prompt "The capital of France is" `
  --max-tokens 3 `
  --temperature 0 `
  --memory-budget 4G
```

> ℹ️ The model path is a **positional argument**.
> There is **no `--model` flag**.

### Tokenize

```powershell
.\target\release\unltd.exe tokenize `
  "path\to\ornith-1.0-9b-Q4_K_M.gguf" `
  --text "The capital of France is"
```

### Inspect a model

```powershell
.\target\release\unltd.exe inspect `
  "path\to\ornith-1.0-9b-Q4_K_M.gguf"
```

---

## 📊 Real results

Validated in the **v1.0.0** release campaign:

| Metric                           | Result                       |
| -------------------------------- | ---------------------------- |
| Model                            | Ornith 1.0 9B Q4_K_M         |
| Model size                       | **5.63 GB**                  |
| Budget tested                    | **4 GB**                     |
| Lower budget tested              | **3 GB**                     |
| Minimum viable controlled budget | **113.06 MB**                |
| Controlled runtime memory        | **~113.19 MB**               |
| Peak working set                 | **~5.24 GB**                 |
| KV cache                         | **~524 KB** (tested context) |
| Generation                       | ✅ PASS                       |
| Model > budget                   | ✅ PASS                       |
| Tests                            | **107 passed**               |

### Example release smoke test

* Prompt: `"The capital of France is"`
* Output token: **`Paris`**
* Oracle match: **1/1**
* Result: ✅ PASS

### What this means

UNLTD Inference successfully runs a:

```text
5.63 GB model
```

with a configured budget of:

```text
3 GB or 4 GB
```

because the runtime’s **controlled memory** stays around:

```text
~113 MB
```

while the model itself remains **mmap-backed on disk**.

---

## 🏗️ Architecture

```text
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
         KV Cache / Recurrent State
                    │
                    ▼
                  Logits
                    │
                 Argmax
                    │
                    ▼
              Generated Text
```

The **Memory Manager** is transversal to the whole pipeline: it plans the configured memory budget (**KV + scratch + runtime state + controlled caches**) **before** any allocation and refuses with concrete numbers if the configuration cannot fit.

---

## 📦 Workspace structure

UNLTD Inference is organized as a Rust workspace with **8 crates**:

| Crate                 | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| `unltd-tensor`        | Tensor data types and quantized kernels                   |
| `unltd-core`          | Shared error types and refuse-rather-than-guess semantics |
| `unltd-model-loader`  | Hand-written GGUF parser and memory-mapped tensor views   |
| `unltd-architectures` | Internal IR and validated architecture configuration      |
| `unltd-memory`        | Memory budget parser and accounting                       |
| `unltd-tokenizer`     | GPT-2 byte-level BPE tokenizer built from GGUF metadata   |
| `unltd-generation`    | Forward engine, session state, KV cache, greedy loop      |
| `unltd-cli`           | Command-line interface                                    |

Dependency direction:

```text
tensor ← core ← {model-loader, architectures, memory, tokenizer} ← generation ← cli
```

---

## 🛠️ Build requirements

Validated environment:

* **Rust stable** (MSRV declared in `Cargo.toml`: **1.80**)
* **Windows x86-64**
* CPU execution (scalar kernels only in v1.0.0)
* enough storage for the model file

> Linux is **not** yet validated end-to-end in v1.0.0.

---

## 🧾 CLI overview

Real commands:

```text
unltd.exe inspect
unltd.exe tokenize
unltd.exe run
unltd.exe min-forward
unltd.exe forward-oracle
```

### User-facing commands

* `inspect`
* `tokenize`
* `run`

### Validation / development commands

* `min-forward`
* `forward-oracle`

---

## 💾 Memory budget

You can run with an explicit memory budget:

```powershell
--memory-budget 4G
```

Accepted formats:

* `512M`
* `1G`
* `2G`
* `4G`
* `8G`
* `512MB`
* `4GB`
* raw bytes (e.g. `536870912`)

### What the memory budget controls

The budget applies to memory **explicitly controlled by UNLTD**, including:

* KV cache
* scratch buffers
* runtime state
* controlled caches / buffers

### What it does **not** control

```text
memory budget != process RSS
memory budget != mapped virtual bytes
```

The whole model file may still be mapped as virtual address space.

That is expected and valid.

### Rejection behavior

If the minimum required controlled memory exceeds the budget, the runtime refuses **before** running:

```text
REFUSING TO RUN
```

with a detailed breakdown and exit code `2`.

Example tested in v1.0.0:

* `--memory-budget 32M`
* required minimum: **113.06 MB**
* result: ✅ correctly refused

---

## 💿 Disk-first execution

What **v1.0.0** actually implements:

```text
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

This is best described as:

> **disk-first / mmap-backed execution**

### What v1.0.0 does **not** implement yet

* explicit layer streaming
* explicit prefetching
* double buffering
* expert streaming
* MoE execution

---

## 🔍 Validation against llama.cpp

Throughout development, **llama.cpp** was used as the **oracle**:

* same model
* same prompt
* same greedy setup
* same temperature (`0`)

### Validated behavior

* ✅ tokenizer is **bit-exact** for the validated prompt
* ✅ first **11** greedy generation steps match the oracle exactly
* ⚠️ later divergence is known and documented

### Known divergence

The first mismatch appears later in longer generation because:

* UNLTD Inference uses a scalar/reference-oriented path
* the oracle uses optimized AVX2/repacked kernels
* small numerical differences accumulate
* eventually an argmax near-tie flips

This is documented openly in:

* `docs/PHASE-7-8-CHECKPOINT.md`
* `docs/PHASE-6-CHECKPOINT.md`

UNLTD Inference **does not** claim bit-exact internal tensors.

The accepted contract in v1.0.0 prioritizes:

* correctness,
* determinism,
* and transparent documentation of known differences.

---

## 📈 Performance

Measured on the validated v1.0.0 setup:

```text
Prefill ~73 s
Decode  ~14.3 s/token
```

This runtime is:

* CPU-only
* scalar
* correctness-first
* validation-oriented

It is **not** intended to compete with `llama.cpp` in throughput at v1.0.0.

> These numbers document the current implementation honestly.
> They are not presented as a performance claim.

---

## ✅ Testing

```powershell
cargo fmt --check
cargo check --workspace
cargo test --workspace
```

### v1.0.0 test status

* **107 passed**
* **0 failed**

Breakdown:

| Area          | Tests |
| ------------- | ----: |
| tokenizer     |    27 |
| generation    |    11 |
| model-loader  |    17 |
| tensor        |    35 |
| architectures |     4 |
| memory        |    13 |

---

## 🧭 Design principles

```text
Correctness before performance
Measurements before assumptions
Refuse rather than guess
Model files stay on disk
Memory is an explicit resource
Scalar path acts as numerical reference
```

---

## ⚠️ Limitations

UNLTD Inference v1.0.0 is a **functional MVP**, not a production-grade high-performance runtime.

Current limitations:

* only the validated architecture/model path is declared supported
* CPU-only
* scalar kernels only
* Windows x86-64 is the only end-to-end validated platform
* much slower than llama.cpp
* no sampling (`temperature = 0` only)
* no GPU
* no SIMD / AVX2 kernels
* no multithreading
* no MoE / expert routing support
* no explicit expert streaming
* no explicit layer prefetching or double buffering
* known numerical divergence vs optimized oracle after longer contexts
* OS page cache is not governed by `--memory-budget`

---

## 🗺️ Roadmap

Possible future lines — **not part of v1.0.0**:

* SIMD / AVX2 kernels
* multithreading
* additional architectures
* explicit streaming experiments
* MoE / expert cache
* Linux validation
* more performance instrumentation
* broader model validation

---

## 📚 Documentation

| Document                        | Contents                                                               |
| ------------------------------- | ---------------------------------------------------------------------- |
| `docs/AUDIT.md`                 | Design audit and lessons inherited from the kimi-k3-in-c investigation |
| `docs/ARCHITECTURE.md`          | Workspace layout, dependency direction, engine design decisions        |
| `docs/MEMORY-DESIGN.md`         | Memory budgets, residency policy, disk-first execution design          |
| `docs/MODEL-SUPPORT.md`         | Model / architecture support policy                                    |
| `docs/MODEL-CANDIDATES.md`      | Candidate models evaluated for the project                             |
| `docs/ROADMAP.md`               | Phase roadmap and future work                                          |
| `docs/PHASE-6-CHECKPOINT.md`    | Full forward validation and numerical contract                         |
| `docs/PHASE-7-8-CHECKPOINT.md`  | Tokenizer exactness, generation, measured divergence                   |
| `docs/PHASE-9-10-CHECKPOINT.md` | Memory budget, disk-first execution, real gate results                 |
| `docs/QWEN35-FORWARD.md`        | Qwen3.5 forward path reference notes                                   |
| `docs/README.es.md`             | Spanish companion overview                                             |

---

## 📜 License

UNLTD Inference is open source, licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for the full license text.

Portions of this project (including tokenizer splitter logic, scalar quantized kernels, and parts of the forward structure) are implemented from **llama.cpp / ggml** MIT-licensed reference algorithms. Their MIT attribution is retained in [NOTICE](NOTICE).

Contributions are welcome under the terms of the Apache License 2.0.

---

## 🙏 Credits / References

* [llama.cpp](https://github.com/ggml-org/llama.cpp)
  Used as the validation oracle, GGUF format reference, and benchmark baseline.
  Portions of tokenizer / scalar-kernel behavior follow its MIT-licensed reference algorithms (see [NOTICE](NOTICE)).

* The **GGUF ecosystem**
  Model container format and surrounding tooling ecosystem.

* [kimi-k3-in-c](https://github.com/MoonshotAI/kimi-k3-in-c)
  Conceptual inspiration for disk-first inference research.
  **UNLTD Inference is not a fork, not a port, and contains no K3 code.**

> This project is **not affiliated** with llama.cpp, ggml, Moonshot AI, Kimi, or any of the above projects.

---

## 🏁 Status

### **UNLTD Inference v1.0.0 — Functional MVP**

Release highlights:

* ✅ GGUF loading
* ✅ Qwen3.5 / Ornith forward
* ✅ tokenizer
* ✅ greedy generation
* ✅ KV cache
* ✅ memory budget
* ✅ disk-first mmap-backed execution
* ✅ model > budget validation
* ✅ 107 passing tests
* ✅ Apache-2.0 open source release

---

<div align="center">

**Built by UNLTD**
Experimental infrastructure for local inference, memory-aware runtime design, and applied systems research.

</div>
