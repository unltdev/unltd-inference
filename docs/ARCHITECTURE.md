# ARCHITECTURE.md — Diseño del runtime `unltd-inference`

> Runtime de inferencia en Rust, CPU-first y disk-first. Inspirado en `kimi-k3-in-c` (ver AUDIT.md), NO es un port: es un diseño nuevo que hereda los mecanismos generalizables y descarta lo específico de Kimi K3.
> Principio rector: `model_size > available_RAM` es una característica central. La RAM es un dial, no un piso.

---

## 1. Pipeline conceptual

```
┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐
│ Model files  │ → │ Model Loader  │ → │ Architecture Adapter │
│ GGUF / safet.│   │ índice+lectura│   │ llama/qwen/mixtral/   │
│              │   │ (unltd-model- │   │ deepseek-mla → IR     │
│              │   │  loader)      │   │ (unltd-architectures) │
└──────────────┘   └───────┬───────┘   └──────────┬───────────┘
                           │                      │
                     ┌─────▼──────────────────────▼─────┐
                     │        Memory Manager            │
                     │ RAM / mmap / ring+pin / expert   │
                     │ cache / O_DIRECT / presupuestos  │
                     │        (unltd-memory)            │
                     └─────┬───────────────────┬────────┘
                           │                   │
                     ┌─────▼──────┐      ┌─────▼──────────┐
                     │ Inference  │      │  CPU backends  │
                     │ Engine     │      │ scalar ≡ AVX2  │
                     │ kernels,   │      │ (≡ AVX-512*)   │
                     │ attention, │      │ contrato bit-  │
                     │ RoPE, KV,  │      │ idéntico       │
                     │ MoE router │      │ (unltd-tensor) │
                     │ (unltd-gen)│      └────────────────┘
                     └─────┬──────┘
                           │
                     ┌─────▼──────────┐    ┌───────────────┐
                     │ Tokenizer      │    │ Generation    │
                     │ BPE + tiktoken │    │ decode loop,  │
                     │ (unltd-        │    │ sampler, spec.│
                     │  tokenizer)    │    │ decode        │
                     └────────────────┘    └───────────────┘
```

La dirección de dependencias es estricta: `tensor ← core ← {model-loader, architectures, memory, tokenizer} ← generation ← cli`. Nadie importa hacia arriba.

---

## 2. Evaluación de crates (pedida explícitamente)

Regla: **ningún framework pesado** (candle, burn, tch, ggml están fuera por decisión). El engine se entiende desde abajo.

| Crate | Veredicto | Razón |
|---|---|---|
| `memmap2` | **Adoptar** | Es la política "mmap" del MemoryManager (una de varias). Maduro, sin deps, seguro. |
| `safetensors` (crate oficial) | **No adoptar; implementar propio** | El formato es trivial (8 B length + JSON header + data). El lector hecho a mano es una de las mejores partes de K3 (validación, negarse ante offsets/EOF inconsistentes, FNV sobre nombres hostiles). Con `serde_json` + `HashMap` alcanza para ≤100k tensores. El crate oficial queda como escape hatch documentado. |
| `tokenizers` (HF) | **No adoptar** | Pesado (dependencias C++/onig para algunos paths), y para BPE byte-level alcanzan ~300 líneas propias (K3 lo demuestra con `tok.h` vendored + `k3_tok.h`). SentencePiece unigram queda como etapa posterior documentada. |
| `rayon` | **Adoptar, con disciplina** | Reemplaza a OpenMP. Solo paralelismo de datos sobre filas de salida independientes (matmul, router, MoE); orden de reducción fijo DENTRO de cada fila. Contrato: scalar ≡ rayon ≡ SIMD, bit-idéntico, testeado. |
| `std::simd` / `portable_simd` | **Diferir** | Nightly (salvo que se estabilice). El camino estable hoy: `core::arch::x86_64` intrinsics detrás de `#[cfg]`, con el backend scalar como referencia y el mismo contrato numérico. `portable_simd` se evalúa cuando sea stable. |
| `half` | **Adoptar** | Conversión f16↔f32 correcta (subnormales incluidas). Micromódulo sin deps. |
| `bytemuck` | **Adoptar** | Vistas zero-copy de datos empaquetados (bloques Q4_K/IQ, cabeceras GGUF) con checks. |
| Alineación (`aligned-vec` o propia) | **Propia (~50 líneas)** | `AlignedBuf<T, ALIGN>` sobre `std::alloc` con `Layout::align_to`. Se necesita para el camino Direct I/O (buffers 4 KB). |
| I/O asíncrono (tokio / async-std) | **RECHAZADO, explícitamente** | Este workload no multiplexa: un solo lector con orden de acceso mayormente conocido. El patrón K3 (1 reader thread + ring + `Condvar`/`Notify`) da todo el solape que existe (1,70× medido) sin runtime, sin dependencias, y con determinismo más fácil de razonar. `tokio` se reintroduce SOLO si algún día hay un servidor HTTP. |
| `serde` + `serde_json` | **Adoptar** | Metadata y config. La política "campo ausente = error" es NUESTRA capa encima (deserialización sin defaults; ver §8). |
| `thiserror` / `anyhow` | **Adoptar** | `thiserror` en librerías (errores tipados, refusal con TODAS las claves faltantes acumuladas); `anyhow` solo en el CLI. |
| `clap` | **Adoptar** | CLI ergonómica. (K3 parsea a mano; clap es la elección idiomática y no es "framework pesado".) |
| `candle` / `burn` / `tch` / `ggml` | **Excluidos** | El proyecto existe para NO usarlos. |

---

## 3. Estructura del workspace

```
unltd-inference/
  Cargo.toml                    # workspace, resolver 2, perfil release optimizado
  crates/
    unltd-tensor/               # DType, TensorView, AlignedBuf, kernels scalar + AVX2, contrato FP
    unltd-core/                 # ModelCfg estricto, errores/refusals, scratch sizing, human-readable sizes
    unltd-model-loader/         # parsers GGUF y safetensors, índice de tensores, WeightIndex
    unltd-architectures/        # adapters → IR: llama, qwen2, qwen3, qwen3-moe, mixtral, deepseek2-mla
    unltd-memory/               # MemoryManager, presupuestos, políticas (resident/mmap/pin+ring),
                                # DiskReader (buffered/direct), ExpertCache + trace, sim_cache
    unltd-tokenizer/            # BPE byte-level (tokenizer.json) + tiktoken.model
    unltd-generation/           # forward pass, KV cache, decode loop, sampler, speculative decode
    unltd-cli/                  # binario `unltd`
  tests/                        # integración por etapa (stage0..stage5)
  benchmarks/                   # bench_kernels, memory-ladder, sim_cache
  tools/                        # scripts (doctor, download, verify)
  docs/                         # estos documentos
```

(Diferencias con la propuesta original del usuario: kernels viven en `unltd-tensor` en vez de `core`; `core` queda como config/errores/scratch — la parte "de contrato" del motor; se agrega el par DiskReader/ExpertCache dentro de `memory` porque ambos SON memoria; todo lo demás igual.)

### Crate por crate

**`unltd-tensor`** — sin dependencias salvo `half`/`bytemuck`.
- `DType`: F32, F16, BF16, I8, U8, y los bloques GGUF (Q4K, Q6K, Q8_0, IQ...) como tipos de vista, nunca copiados.
- `TensorView<'a, D>`: shape + strides + datos tomados prestados (de arena propia, de mmap, o de un slot del ring).
- Kernels con contrato numérico declarado en el doc de cada kernel:
  - `rmsnorm(out, x, w, eps)` — acumulador f64, eps DENTRO de la raíz (como K3; DeepSeek/Qwen también lo definen así).
  - `matmul_f32(acc, a, b)` — referencia scalar con orden de reducción fijo `((a0+a1)+(a2+a3))...`, acumulador f64; `matmul_f32_avx2` con `_mm256_fmadd_pd` por lane (misma op IEEE que el scalar), misma partición.
  - `matmul_q4k` y amigos — GEMV desde bloques empaquetados sin dequantizar (el principio "nunca desquantizar" heredado).
  - `rope_apply` / `rope_apply_scaled` (YaRN) — variantes Llama y NeoX.
  - `swiglu`, `gelu_tanh`, `relu2` (Qwen3), `softmax` con escala explícita y orden fijo.
- Regla de oro: **todo kernel con fast path SIMD tiene un test que exige igualdad bit a bit con el scalar.**

**`unltd-core`**
- `ModelCfg`: deserialización estricta — `deny_unknown_fields` NO alcanza (hay campos legítimos que no usamos), entonces: deserializar a `serde_json::Value`, extraer solo lo necesario, y **rechazar si falta algo necesario**, acumulando TODAS las claves ausentes en un solo error (patrón `k3cfg_miss`). Campos genuinamente opcionales declarados con default EXPLÍCITO y comentado.
- Checks estructurales post-parse (layers>0, topk≤n_experts, rango del mapa de capas one-based, etc.).
- `Refusal` — el tipo de error "el motor se niega a correr" con dos números cuando aplica ("necesita X, hay Y").
- `human()` para bytes.
- `ScratchPlanner`: tamaños de scratch calculados por el motor, nunca por el usuario (lección K3: el off-by-one silencioso).

**`unltd-model-loader`**
- `GgufReader`: parseo del header (magia, versión, pares KV, infos de tensores con offset/nbytes/quant), validación de rango contra tamaño de archivo, negarse ante archivo truncado. Los offsets GGUF ya son contiguos por tensor — la capa es la unidad de stream natural.
- `SafetensorsReader`: header JSON (serde_json), validación dtype/shape/offsets/EOF (la lista de checks de `k3_st.c` completa), índice `HashMap` con nombres como vienen del checkpoint.
- `WeightIndex`: trait unificado sobre ambos: `find(name) -> Option<TensorMeta>`, `read(t, buf)`, `read_aligned(...)`.
- Detección de dtype por sufijo de nombre cuando el formato no lo declara (safetensors sí lo declara; cuidado con los `weight_scale`).

**`unltd-architectures`** — el corazón extensible. Cada adapter:
1. valida config contra lo que la arquitectura necesita (refusal con lista de faltantes),
2. construye el mapa `nombre de tensor → (rol, forma esperada)` — el análogo de `plan_layer` de K3, con chequeo de element count ANTES de leer bytes,
3. emite el `ModelSpec` (la IR) — ver §4.

**`unltd-memory`** — ver MEMORY-DESIGN.md. Componentes: `MemoryManager` (plan/allocate/refuse), políticas (`ResidentPolicy`, `MmapPolicy`, `PinRingPolicy`, `ExpertLruPolicy`), `DiskReader` (trait + `Buffered`/`Direct` por plataforma), `ExpertCache` (3 fases, INFLIGHT, dedup, trace), `sim_cache` (replay LRU/Belady/pinned).

**`unltd-tokenizer`** — BPE byte-level desde `tokenizer.json` (vocab + merges + specials + regex de pre-tokenización tomado del archivo, nunca re-implementado de memoria — lección K3 con `tokenization_kimi.py`); loader de `tiktoken.model` para los modelos que lo publiquen; decode byte-exacto (los bytes parciales UTF-8 se acumulan, no se imprimen mojibake — K3 lo documenta).

**`unltd-generation`** — forward por capa sobre la IR, KV cache (pre-dimensionada, negarse antes de correr), decode loop incremental, sampler greedy (default; la salida determinista entre presupuestos es una propiedad que los tests dependen — sampling es opt-in, lección del ROADMAP de K3), y speculative decoding (n-gram + verificación batcheada exacta) como etapa posterior.

**`unltd-cli`** — el binario. Presets de memoria, `--budget auto`, banner de plan ANTES de asignar, reporte de PEAK RSS al final, y el reporte de share I/O con la advertencia de >100% heredada.

---

## 4. La IR: lo que hace extensible al runtime

La decisión central del diseño. Los adapters NO ejecutan: describen. La IR es lo único que el motor de ejecución conoce.

```rust
// unltd-architectures (esquema, no código final)
enum AttnKind {
    Mha,                                        // Llama-3.2-1B/3B, Gemma-3-1B
    Gqa { kv_groups: u32 },                     // Llama-3.1-8B, Mistral, Qwen3
    Mla {                                      // DeepSeek-V2/V3 (semántica DeepSeek:
        kv_lora: u32, qk_rope: u32,            //  latente comprimido + RoPE real;
        qk_nope: u32, v_head: u32,             //  NO es la MLA-NoPE de K3)
    },
}
enum RoPeKind {
    Llama { theta: f32, dims: u32 },
    LlamaYaRn { theta: f32, dims: u32, factors: Vec<f32>, base_scale: f32 },
    NeoX,                                       // el otro orden de intercalado
}
enum FfnKind {
    SwiGlu, GeGluTanh, Relu2,                   // Qwen3: ReLU^2 en lugar de GLU
    Moe { n_experts: u32, topk: u32, n_shared: u32,
          inter: u32, norm_topk: bool, sigmoid_gate: bool, routed_scale: f32 },
}
struct LayerSpec {
    input_norm: NormKind,                       // RmsNorm{eps} | LayerNorm | (Qwen2.5: QkNorm antes de q/k)
    attn: Option<AttnSpec>,                     // dense layer 0 no tiene en K3; en Llama todas tienen
    post_norm: NormKind,
    ffn: FfnSpec,
    residual: ResidualStyle,                    // PreNorm universal; K3-style attn-res SOLO si se portara K3
}
struct ModelSpec {
    embed: WeightRef, final_norm: WeightRef, lm_head: WeightRef,
    tie_embeddings: bool,                       // Qwen/Mistral/Gemma: true (o lm_head compartido)
    eos_ids: Vec<u32>,
    layers: Vec<LayerSpec>,
    rope_global: RoPeKind,                      // dims máximos; por-capa si difieren
}
```

Reglas de la IR:
- **Sin ramas por arquitectura en el motor.** El motor hace `match` sobre enums; agregar una arquitectura = agregar un adapter, no tocar el motor.
- **Los pesos son `WeightRef`** (índice + clase de residencia). El motor pide pesos al MemoryManager, nunca abre archivos.
- **El adapter valida con refusal**: un tensor con element count distinto al que implica la config es un error ANTES de leer un byte (lección K3: A_log de 128 para 96 heads; q_conv1d rank-3).
- La semántica MLA queda explícita: la IR distingue la MLA de DeepSeek (comprime K/V al latente, rota la parte rope, cachea el latente comprimido) de la de K3 (NoPE, cachea k/v expandidos). Si algún día se soporta K3, se agrega un `AttnKind::MlaK3NoPe` — no se fuerza una fusión falsa.

---

## 5. Motor de ejecución y contrato numérico

El forward es: `for layer in spec.layers { layer.execute(&mut state, &memory, &scratch) }` donde `state` lleva la activación corriente y el KV cache. Detalles heredados de la auditoría:

- **Atención**: scale = 1/sqrt(head_dim) calculado en f32 (o f64 acumulado); softmax con max-subtraction; orden de reducción fijo para determinismo. GQA: cache por KV head (expandido) para simplicidad; optimización de cachear por grupo queda anotada como TODO medible.
- **RoPE**: intercalado Llama y NeoX en el mismo kernel con switch; escalado YaRN para DeepSeek.
- **KV cache**: fp32, pre-dimensionada `bytes/pos × posiciones`, negarse up-front (lección K3: es el único término que crece con contexto).
- **MoE**: router (softmax+bias o sigmoid+renorm según IR) → `ExpertSource::getmany(topk)` → contribuciones con pesos de ruteo; shared experts sumados; RMSNorm del agregado solo si la IR lo pide (K3) — DeepSeek/Mixtral no lo hacen.
- **Contrato numérico** (la herencia más subestimada de K3):
  1. acumuladores f64 en reducciones largas; f32 solo donde el modelo lo define (softmax, normas);
  2. orden de reducción fijo y documentado por kernel;
  3. `mul_add` explícito donde se quiere FMA — nunca FMA del autovectorizador (Rust no tiene `-ffp-contract=off`; la única forma de controlarlo es no dejarlo al compilador en el camino de referencia);
  4. backend scalar = referencia; AVX2 debe ser bit-idéntico (tests dedicados);
  5. paralelismo solo sobre filas de salida independientes (rayon), jamás dentro de una reducción.
  - Meta aspiracional (heredada): logits bit-idénticos entre backends y entre presupuestos de memoria; como mínimo, max-diff acotado y testeado.

---

## 6. Tokenizer

Solo BPE byte-level en v1 (`tokenizer.json` + `tiktoken.model`). Eso cubre: Qwen2.5/3 (tiktoken-style BPE en tokenizer.json), Mistral, DeepSeek (tokenizer.json propio del linaje), Phi-3/4, SmolLM2/3, Llama-3.1+ (tiktoken BPE). Quedan fuera en v1: SentencePiece unigram (Llama-3.2-1B/3B, Gemma, TinyLlama) — etapa posterior con el mismo patrón que K3 usó para tiktoken: loader directo del `.model` de SentencePiece (formato público) o vocab+merges cuando el modelo los publique como JSON. **Esto pesa en la selección del PoC** (§9): el primer modelo debe usar BPE.

---

## 7. Formatos de pesos: GGUF primario, safetensors secundario

- **GGUF es el formato de carga del runtime** (ver MODEL-CANDIDATES.md: es el estándar de facto de CPU, con quants maduros y metadata). El parser es propio (~300 líneas) y validante.
- **Safetensors** se soporta para experimentos de precisión completa (bf16) y como formato "fuente" para tooling de verificación contra referencias (el análogo de `ref_forward.py`).
- El runtime **no re-quantiza** pesos en v1 (decisión K3: el formato empaquetado del modelo es sagrado); `tools/` puede usar `llama-quantize` del ecosistema para derivaciones, y el motor consume lo que le den.
- **"Nunca desquantizar"**: los bloques Q4_K/IQ se multiplican directamente desde sus vistas empaquetadas; la cache de expertos guarda los bloques tal cual llegan del disco.

---

## 8. Refuse rather than guess, en Rust

- Config: extracción estricta con acumulación de faltantes (§3, unltd-core). Un modelo con config a medias produce texto fluido y equivocado — el peor fallo posible, y el default-friendly serde lo hace fácil. **Prohibido `#[serde(default)]`** salvo los campos con default REAL documentado.
- Formato de pesos: GGUF truncado → error con offset vs tamaño. Safetensors con `data_offsets` que se salen del archivo → error. dtype inesperado → error.
- Expertos: un experto que no se pudo leer es un fallo de corrida (exit code distinto + mensaje "RUN INVALID"), nunca un drop silencioso (K3: exit 4).
- Estado de conversación: fingerprint de config en el header; restauración de arquitectura distinta → negarse ("produciría salida fluida y equivocada").
- Prompt/KV: pedir más contexto del que entra → negarse con ambos números antes de correr 40 minutos.

---

## 9. Progresión de tests (las etapas pedidas)

| Etapa | Qué se valida | Modelo | Qué se hereda de K3 |
|---|---|---|---|
| **Stage 0** | Motores de kernels, IR, forward completo contra una referencia propia naive (sin torch en CI) | Sintético: pesos aleatorios sembrados, ~13 capas (el número de K3: ≥2 bloques attn-res… aquí ≥2 capas MoE/dense para probar ramas) | `make_tiny_checkpoint.py` / oráculo tiny; fixtures adversariales |
| **Stage 1** | Pipeline real completo: GGUF, tokenizer.json, adapter, decode | 0,5–1,5B denso (PoC, §10) | GATE 1-3 del oráculo: teacher forcing, greedy, incremental≡recompute, todo EXACTO |
| **Stage 2** | GQA a escala, KV cache real, velocidad razonable residente | 3–4B denso | `scale_test` en dimensiones reales |
| **Stage 3** | El caso "justo en 16 GB", Q4_K_M residente | 7–8B denso | Plan de memoria completo, negarse si no entra |
| **Stage 4** | Expert streaming + cache LRU + trace/replay | MoE pequeño/mediano (MiniCPM-MoE / Qwen3-MoE-30B-A3B) | `k3_cache.c` completo: 3 fases, INFLIGHT, hit rate verdadero, sim_cache |
| **Stage 5** | `model_size >> RAM` como caso central | MoE o denso > 16 GB (GGUF > RAM con margen) | Pin+ring, O_DIRECT, presupuestos, escalera con cgroups |

Cada etapa agrega una clase de residencia o un backend; ninguna etapa cambia el contrato numérico. La escalera de memoria (salida idéntica en todos los escalones) corre desde Stage 1 como test de integración.

---

## 10. El Proof of Concept inicial

**Qwen3-0.6B** (ver justificación completa en ROADMAP.md §PoC y MODEL-CANDIDATES.md). Razones, en una línea cada una: Apache-2.0; arquitectura vainilla (RMSNorm + SwiGLU... ReLU² + GQA + RoPE + tie-embeddings — el adapter más simple posible, y Qwen3 es la familia que probablemente termine en las etapas 2-4); tokenizer BPE (entra en nuestro tokenizer v1); GGUF oficial con quants Q4_K_M (~0,5 GB — cabe 25 veces en la RAM objetivo, ideal para validar corrección contra llama.cpp con verificación barata); utilidad real (multilingüe, decente para su tamaño); y deja la puerta abierta a Qwen3-4B/8B/30B-A3B como etapas siguientes dentro de la MISMA familia (menos adapters nuevos por etapa).

La verificación del PoC es la parte que lo hace "el mejor primer objetivo": 0,6B corre en segundos en CPU, permite comparar logits elementwise contra `llama.cpp` (o contra una referencia en Python) en CI, y hace barato el gate de "salida idéntica en todos los presupuestos de memoria" desde el día 1.

---

## 11. Comparación honesta con llama.cpp

(Desarrollada en MODEL-SUPPORT.md §"llama.cpp"; resumen aquí.)

Lo que llama.cpp ya hace mejor y **no intentamos superar**: kernels maduros a mano para cada arquitectura, ecosistema GGUF (conversión, quants, comunidad), mmap para el caso residente, cobertura de arquitecturas, tooling de serving.

Lo que `unltd-inference` ofrece que llama.cpp no tiene como característica central:
1. **Políticas de memoria explícitas** (pin+ring, cache de expertos con presupuesto, O_DIRECT, techo RSS) en lugar de confiar en el page cache del OS.
2. **El caso `model >> RAM` como producto**, con instrumentación de investigación: trace de accesos, replay offline (LRU/Belady/pinned), hit rate verdadero, GB/token por clase.
3. **Determinismo bit a bit entre backends y entre presupuestos**, testeado en CI.
4. **"Refuse rather than guess"** como filosofía de producto.
5. **Arquitectura didáctica**: cada mecanismo en su crate, cada política un trait, sin capas de compatibilidad heredadas.

Si en alguna etapa llama.cpp resuelve mejor una parte concreta (p. ej. kernels AVX-512 de cierta arquitectura), este documento lo dirá explícitamente en vez de reimplementarlo peor. El objetivo no es reemplazar llama.cpp; es investigar la frontera disk-first con una arquitectura limpia.
