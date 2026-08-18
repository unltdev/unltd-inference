# unltd-inference

Runtime de inferencia LLM en Rust: **CPU-first y disk-first**. La premisa central es
que `model_size > available_RAM` es una característica de diseño, no un caso
excepcional: los pesos se streamean desde disco con políticas de memoria explícitas
(prefijo pineado + ring para el tronco denso, cache LRU de expertos para MoE,
presupuestos configurables), y la misma corrida produce salida idéntica en cualquier
presupuesto.

Inspirado en [kimi-k3-in-c](https://github.com/…) (motor C que corre Kimi K3, 2,78 T
parámetros, desde 8 GB de RAM en CPU). **No es un port**: hereda los mecanismos
generalizables documentados en la auditoría y descarta lo específico de la arquitectura
K3.

## Estado

Fase de diseño. Los documentos de `docs/` son la fuente de verdad:

| Documento | Contenido |
|---|---|
| `docs/AUDIT.md` | Auditoría completa de kimi-k3-in-c: qué se hereda y qué no |
| `docs/MODEL-CANDIDATES.md` | Modelos open-weight candidatos, clasificados A/B/C/D |
| `docs/ARCHITECTURE.md` | Diseño del runtime: pipeline, crates, IR, contrato numérico |
| `docs/ROADMAP.md` | Etapas de implementación y Proof of Concept |
| `docs/MEMORY-DESIGN.md` | Estrategias de memoria/disco y matriz de plataformas |
| `docs/MODEL-SUPPORT.md` | Soporte por arquitectura + comparación con llama.cpp |

## Requisitos

- Rust estable (`rustup`, `rust-toolchain.toml` ya lo fija). `cargo check --workspace`
  y `cargo test --workspace` verificados en verde con rustc 1.97.1 (Windows MSVC).
  Si `cargo` no está en el PATH del shell, usar la ruta completa
  `%USERPROFILE%\.cargo\bin\cargo.exe`.
- CPU x86-64 con AVX2 (el backend scalar funciona sin él).
- Linux o WSL2 para las políticas Direct I/O y la escalera de medición con cgroups;
  Windows nativo para uso diario con fallback buffered (ver MEMORY-DESIGN.md §6).

## Estructura

```
crates/
  unltd-tensor/         dtypes, vistas, buffers alineados, kernels scalar + AVX2
  unltd-core/           config estricta, errores/refusals, scratch sizing
  unltd-model-loader/   parsers GGUF + safetensors, índice de tensores
  unltd-architectures/  adapters (llama, qwen2/3, mixtral, deepseek-mla) → IR
  unltd-memory/         MemoryManager, políticas, DiskReader, ExpertCache, trace
  unltd-tokenizer/      BPE byte-level (tokenizer.json, tiktoken.model)
  unltd-generation/     forward, KV cache, decode loop, sampler
  unltd-cli/            binario `unltd`
tests/ benchmarks/ tools/ docs/
```

Ver `docs/ARCHITECTURE.md` §3 para el detalle de cada crate y la dirección de
dependencias.
