# Soporte de modelos y comparación honesta con llama.cpp

Este documento responde dos preguntas: (1) qué arquitecturas soporta el runtime y en qué fase, y (2) qué le aporta al mundo un runtime nuevo cuando llama.cpp ya existe — respondida con lo que llama.cpp hace mejor, dicho claramente.

## 1. Fases de soporte de arquitecturas

### v1 (Stages 0–4 del roadmap)

| Familia | Adapter | Piezas de IR que ejercita | Tokenizador |
|---|---|---|---|
| Qwen2.5 / Qwen3 denso | `qwen3` | GQA, RoPE θ=1e6, SwiGLU, RMSNorm, tie ≤4B. Qwen2.5: SWA residual (no-op) | BPE tiktoken-style 151,936 |
| Llama 3.1/3.2 | `llama` | GQA, θ=500k, tie en 3.2, RoPE scale factor 32 (>8k ctx) | BPE tiktoken-style 128,256 |
| SmolLM2 (135M–1.7B) | `llama` | MHA o GQA, θ=100k–130k, tie | BPE 49,152 |
| Phi-4 / Phi-3 | `phi4` | GQA (10 KV), θ=250k, untied | BPE 100,352 |
| Mistral-7B-v0.3 · Small 3.x · Ministral 3 | `mistral` | GQA, θ=1e6–1e9, untied | BPE / Tekken 32k–131k |
| **Qwen3-MoE (30B-A3B, Coder-30B)** | `qwen3_moe` | `FfnKind::Moe` (128e top-8, sin shared, sigmoid+renorm, scaling) + todo lo de `qwen3` | BPE 151,936 |
| **OLMoE-1B-7B** | `olmoe` | `FfnKind::Moe` (64e top-8, softmax, aux-loss), MHA | BPE 50,304 |
| Mixtral-8x7B | `mixtral` | `FfnKind::Moe` (8e top-2 softmax), GQA | SentencePiece 32k (etapa tokenizador posterior) |

### v2 (post Stage 4)

- **Qwen3.5/3.6-35B-A3B**: mismo MoE + shared expert + MTP + tokenizador nuevo 248k. Exige extensiones de IR: `partial_rotary` (0.25), M-RoPE, `attn_output_gate`.
- **Phi-4-mini**: partial rotary 0.75 + LongRoPE — el caso más simple de partial rotary.
- **Gemma 2/3 densas**: GeGLU-tanh, soft-capping, patrón 5:1 local/global con RoPE dual por capa, SentencePiece 256k/262k.
- **SmolLM3-3B**: capas NoPE (intervalo cada 4ª) + YaRN — primer contacto con RoPE por-capa.
- **Gemma-4-12B / 31B densos** (Apache-2.0): misma línea Gemma sin licencia.
- **DeepSeek-V2-Lite**: `AttnKind::MlaDeepSeek` completo + 64e top-6 + shared + YaRN 40 — el paso de Pregunta A más cercano a K3.

### Futuro / experimental (Stage 5+)

- **DeepSeek-V3/R1/V3.2**: MLA + top-8/256 + shared + MTP + YaRN 16.
- **Gemma-4-26B-A4B**: FFN dual-path (MLP denso + MoE en paralelo) — no expresable en la IR actual.
- **Llama-4-Scout**: qk-norm, top-1 de 16, todas las capas MoE.
- **Qwen3-Next / 3.5 chicas**: GatedDeltaNet (recurrencia por chunks, decay, conv) — la IR necesita una clase de capa nueva (`LayerKind::LinearAttention`), no un campo.
- **Mistral-Small-4**: MLA con rope_interleave + top-4 de 128 + YaRN 1M.
- **DeepSeek-V4-Flash**: FP4 experts (e4m3/ue8m0) — no soportado ni por llama.cpp upstream.
- Fuera de alcance permanente: Gemma-3n (MatFormer/PLE), Phi-4-mini-flash (SSM híbrido), modelos multimodales completos.

### Extensiones de IR que la investigación ya detectó (documentadas para no mentir)

| Necesidad | Familias | Estado |
|---|---|---|
| partial rotary (0.25/0.5/0.75) | Qwen3-Next/3.5, Phi-4-mini, Phi-3.5 | pendiente en `RoPeKind` |
| M-RoPE (mrope_section) | Qwen3.5 | pendiente |
| soft-capping (attn 50 / logits 30) | Gemma | pendiente |
| RoPE dual por capa (θ local/global) | Gemma-3 | pendiente |
| QK-norm | Llama-4, OLMo-2, Gemma | ya en IR (`NormKind::QkRmsNorm`) |
| FFN dual-path | Gemma-4-MoE | pendiente (fuera de v1) |
| ReLU² | PhiMoE | ya en IR (`FfnKind::Relu2`) — NO es de Qwen3, mito corregido |
| Sigmoid gate (con/sin bias, con/sin renorm) | DeepSeek-V3, Qwen3-MoE | ya en IR (`Moe.sigmoid_gate`, `Moe.norm_topk`) |
| Capa densa inicial (first_k_dense_replace) | DeepSeek, K3 | ya en IR (`LayerSpec.attn: None`) |
| MTP (multi-token prediction) | DeepSeek-V3, Qwen3.5 | fuera de v1 |

## 2. Cobertura de tokenizadores

| Tokenizador | Vocab típico | Familias | Fase |
|---|---|---|---|
| BPE byte-level (`tokenizer.json`) | 49k–152k | Qwen2.5/3, Llama-3.1/3.2, Mistral, Phi, SmolLM, DeepSeek | **v1** |
| tiktoken (`tiktoken.model`) | 152k | Qwen, Kimi-linaje | **v1** (loader ya auditado en K3) |
| SentencePiece unigram | 32k–262k | Mixtral, Gemma, TinyLlama, MiniCPM | v2 (Gemma la exige; Mixtral puede esperar) |
| BPE 248k (Qwen2Tokenizer nuevo) | 248,320 | Qwen3.5/3.6 | v2 |

La regla de carga sigue a `unltd-tokenizer`: se mira QUÉ archivo existe, nunca se adivina por nombre de modelo.

## 3. Números de referencia del ecosistema (objetivos de benchmark)

Todos verificados durante la investigación (2026-08); sirven de techo y de piso:

| Referencia | Número | Fuente |
|---|---|---|
| Qwen3-30B-A3B Q4_K_M, llama.cpp | **42.15 t/s** decode en EPYC 9554 · 17.28 GiB en disco · >10 t/s en PC de 16 GB | llama-bench público |
| DeepSeek-V3-0324 Q4_K_M (378 GiB), en-RAM CPU | **8.45 t/s** decode | issue llama.cpp #14201 |
| 8B Q4_K_M, CPU AVX2 de escritorio | 5–20 t/s (ancho de banda manda, no FLOPS) | guías 2026 |
| GLM-5.2 de 220 GB sobre 128 GB (mayormente disco) | 1.4 t/s | PR llama.cpp #26003 |
| Windows nativo vs Linux, modelo 671B-clase | Windows ~70% más lento | issue #14201 |
| WSL2 vs Windows nativo, MoE grande | WSL2: ~25% pérdida en PP + ~15% overhead I/O (9P) | benchmarks 2026 |
| K3 en su régimen (1.56 TB → 8.24 GB) | ~14.7 s/token, I/O share 40–60% | `docs/PERFORMANCE.md` de kimi-k3-in-c |

El régimen intermedio (16 GB, modelos de 20–400 GB en disco) está SIN medir en público — es exactamente el espacio donde este proyecto publica datos.

## 4. El estado del arte en streaming (lo que ya existe)

- **llama.cpp**: mmap-first, sin presupuesto explícito; el page cache del kernel ES la política (documentado y deliberado). `--lazy-experts` (el único intento de streaming de expertos a nivel runtime) sigue **sin mergear** tras un año (PR #26003, abierto al 2026-08-17).
- **kimi-k3-in-c**: el otro extremo — presupuesto explícito, pin+ring, cache LRU de expertos con prefetch, todo medido con cgroups.
- **KTransformers**: colocación de expertos CPU/GPU por frecuencia — requiere GPU; filosofía cercana, historia de hardware opuesta.
- **runNburn**: "memory-aware GGUF runtime", paging sparse de expertos file-backed bajo presupuesto host (orientado CUDA) — el competidor conceptual más cercano.
- **oxillama**: reimplementación de llama.cpp en Rust puro (~165k LOC, 20 arquitecturas, 25 quants). **Existe: "llama.cpp en Rust" ya está hecho.** Este proyecto es otra cosa: investigación de políticas de memoria con instrumentación.

## 5. Comparación honesta: dónde llama.cpp gana (dicho claro)

1. **Amplitud**: 146 arquitecturas GGUF registradas (incluida `LLM_ARCH_KIMI_K3` — correr K3 ya no es un diferenciador). Soporta todo lo de este roadmap y más, desde el día uno.
2. **Madurez de cuantización**: los K-quants (Q4_K_M recomendado para CPU) y el ecosistema IQ/imatrix son años de ingeniería; los IQ son consistentemente más lentos en CPU que su ancho de bit — conocimiento que nos ahorra experimentos.
3. **Rendimiento ingenieril**: kernels AVX-512/AMX, `ik_llama.cpp` como fork de referencia, años de micro-optimización. Los números de §3 son el piso que hay que alcanzar en régimen residente.
4. **Ecosistema**: 28,500+ GGUFs publicados, todos los quants de todos los modelos candidatos. Por eso el diseño lo usa como ORÁCULO (validar salida contra llama.cpp) y como formato de entrada.
5. **Ejecución sobre-RAM por mmap**: funciona hoy, sin configuración, aprovechando el page cache — y sus mediciones (mmap le gana a O_DIRECT para MoE sobre-RAM en su régimen) son un dato real que este proyecto respeta: el camino Direct I/O será opt-in y medido contra mmap, no dogma.

## 6. Y dónde está el espacio (los gaps verificados)

Cada punto fue verificado contra llama.cpp master (2026-08-17); ninguno es especulación:

1. **Políticas de memoria explícitas (pin prefix + ring, presupuesto como dial)**: no existen en llama.cpp. Solo hints (madvise) + `--mlock` de modelo entero. El único experimento (PR #26003) midió pinning "neutral debajo de 24 GB, peor arriba" — pero es UN experimento, en presupuestos ALTOS, donde el page cache ya sostiene parte del set caliente. El régimen de K3 es el opuesto (8–32 GB para 1.56 TB: el page cache no sostiene NADA del tronco), y ahí la asignación explícita midió 1.69× de ventaja. **El punto de cruce entre ambos regímenes está sin medir — medirlo requiere la instrumentación que este proyecto construye.**
2. **Cache de expertos LRU con prefetch batch como política visible**: llama.cpp delega la política al kernel (y su propio PR concluyó "el page cache ya gana" — de nuevo, en su régimen). No hay slots EMPTY/INFLIGHT, ni prefetch en 3 fases, ni hit rate verdadero, ni política configurable. Para INVESTIGAR políticas de evicción no hay nada sobre lo cual construir.
3. **Trace de accesos + replay offline**: no existe. Sin trace no hay Belady/LRU/pinned comparables, ni techo de compulsory misses, ni justificación de pinning por histograma.
4. **Determinismo bit a bit entre presupuestos de memoria**: concepto inexistente en llama.cpp (el determinismo depende de build/threads/backend; la escalera de presupuestos no existe como eje). Es la aserción central de este proyecto: misma salida en 4 GB y en 32 GB.
5. **Config que se niega en vez de degradar**: llama.cpp "hace que funcione por default" — incluye el warmup WILLNEED del modelo ENTERO, que thrash-ea activamente en sistemas sobre-RAM (mitigado con flags opt-in). Un runtime disk-first puede negarse ante configs imposibles (modelo > RAM con `--no-mmap`, warmup mayor que el presupuesto) con ambos números impresos.
6. **O_DIRECT / io_uring como camino diseñado**: `--direct-io` es experimental, sensible al filesystem, y su propio dato dice que mmap gana en su régimen; io_uring no existe. Aquí es un camino medido contra mmap, con alineación y fallback, no una bandera.
7. **Instrumentación por-tensor del costo de I/O**: no hay modelo de costo por tensor/capa ni programación de prefetch por capa en el runtime (la investigación de layout vive en herramientas offline). El I/O share por token de K3 (40–60%) no tiene contraparte publicable en llama.cpp.

**La frase que resume la posición**: llama.cpp optimizó la ejecución y confía en el kernel para la memoria; este proyecto instrumenta la memoria y confía en la simplicidad para la ejecución. Los dos datos se complementan — y la única forma de saber cuál gana en 16 GB es medir ambos con el mismo protocolo.

## 7. Implicaciones para el diseño (por qué las decisiones de ARCHITECTURE.md se mantienen)

- **GGUF como formato primario** (no competir con el ecosistema: consumirlo y validarse contra él).
- **pread y no mmap como camino investigable** (RSS medible, igual que K3) — con mmap disponible como modo alternativo para la comparación A/B exigida por §5.5.
- **El oráculo es llama.cpp**, y los tests exigen coincidencia token a token sobre el mismo GGUF. Esto convierte la amplitud de llama.cpp en un activo del proyecto, no en un competidor.
- **El diferenciador nunca será "soporta más arquitecturas"** — será "publica la curva de memoria con salida idéntica".
