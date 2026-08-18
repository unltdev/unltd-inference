# AUDIT.md — Auditoría de `kimi-k3-in-c`

> Fecha de la auditoría: 2026-08-17. Repo auditado: `D:\AI\projects\kimi-k3-in-c` (commit `ff11dce`, release v1.0.0).
> Método: lectura completa de las fuentes del motor (no del README). Cada afirmación de este documento se verificó contra el código citado entre paréntesis.

---

## 1. Resumen ejecutivo

`kimi-k3-in-c` es un motor de inferencia de **~5.000 líneas de C99** (sin contar CLI, tests y herramientas Python) que ejecuta el checkpoint completo de Kimi K3 (2,78 T parámetros, 1,56 TB en disco, 96 shards safetensors, 497.220 tensores) en una sola CPU, **desde 8,24 GB de RAM hasta 224 GB**, produciendo **salida byte-idéntica en todo el rango**. Velocidad medida: de 32,69 s/token (8 GB) a 10,66 s/token (128+ GB), en una EPYC 7763 con NVMe de 3,2 GB/s (docs/PERFORMANCE.md, docs/data/memory-ladder.tsv).

El diseño se apoya en cuatro reducciones sucesivas de memoria:

| # | Reducción | De | A | Técnica |
|---|---|---|---|---|
| 1 | Precisión nativa del checkpoint | 5,56 TB (bf16) | 1,56 TB | Expertos en MXFP4 (0,53125 B/peso), sin desquantizar nunca |
| 2 | Residencia | 1,56 TB | 113,49 GB | Expertos ruteados nunca residentes: streaming por demanda |
| 3 | Streaming del tronco | 113,49 GB | ~8,2 GB | Tronco empaquetado + prefijo pineado + ring de 2 slots |
| 4 | Presupuesto | fijo | dial | El presupuesto de RAM se reparte entre tronco y cache de expertos; la salida no cambia |

Ninguna de las cuatro depende de una GPU. Ninguna de las cuatro pierde precisión: los bytes que se calculan son los del checkpoint.

Lo más valioso de este repo **no son los kernels** (que en su mayoría son específicos de la arquitectura K3), sino:

1. la **arquitectura de memoria** (pinned prefix + ring, cache de expertos con prefetch en 3 fases, O_DIRECT, presupuestos como dial);
2. la **disciplina de medición** (33% de piso de ruido, replicación obligatoria, contadores antes que relojes);
3. la **filosofía de fallo** ("refuse rather than guess": nunca sustituir un default, nunca correr un modelo que se entiende a medias);
4. la **escalera de testing** (fixtures adversariales → oráculo tiny → conformancia por capa → logits elementwise contra torch).

---

## 2. Inventario del código

### 2.1 Motor (todo C99, OpenMP + pthreads)

| Archivo | Líneas | Función | Clasificación (ver §4) |
|---|---|---|---|
| `include/k3/k3.h` | 522 | API pública: K3Cfg, structs de pesos, tags de dtype, invariantes | 2 + 5 |
| `include/k3/k3_cfg.h` | 254 | Lector de config JSON (2 shapes, "ausente = error, nunca default") | 2 + 3 |
| `src/core/k3_ops.c` | 1259 | Todos los kernels: RMSNorm, SiTU-GLU, ShortConv, KDA, MLA, AttnRes, MoE, matmul bf16/AVX2, matmul MXFP4 | 1 (la mayoría) |
| `src/io/k3_st.c` | 523 | Lector safetensors hecho a mano: scanner JSON, hash FNV-1a, tabla open-chaining, doble fd (buffered + O_DIRECT) | 2 |
| `src/io/k3_load.c` | 141 | Resolución de los 6 tensores de un experto; una sola pread coalescida de 17,55 MB | 2 (patrón) + 5 (formato) |
| `src/io/k3_trunk.c` | 537 | Streaming del tronco: pin + ring, lector asíncrono, prefetch | 3 |
| `src/cache/k3_cache.c` | 408 | Cache LRU de expertos, prefetch batch en 3 fases, trace | 3 |
| `src/model/k3_bind.c` | 455 | Binding nombre→structs de kernels, widen de vectores chicos, plan de memoria | 2 (patrón) + 5 (nombres) |
| `src/cli/k3_run.c` | 1424 | CLI: presets, plan de memoria, decode loop, estado incremental, spec. decoding | 3 (patrón) |
| `src/tokenizer/k3_tok.h` | 207 | Loader directo de `tiktoken.model` + `tokenizer_config.json` | 2 (formato tiktoken) + 5 (flag kimi) |
| `src/io/k3_portable_io.h` | 61 | Shims Linux↔Darwin (O_DIRECT ↔ F_NOCACHE, fadvise) | 3 |
| `third_party/json.h` + `tok.h` | — | JSON y tokenizer vendored | 2 |

### 2.2 Herramientas Python (todas de validación/tooling, no parte del runtime)

| Herramienta | Función | Clasificación |
|---|---|---|
| `tools/pack_trunk.py` | Empaqueta el tronco en `trunk.bin` (una corrida contigua por capa, padding 4 KB) | 3 (patrón) |
| `tools/sim_cache.py` | Replay offline de traces: LRU vs Belady vs pinned | 3 |
| `tools/budget.py` | Suma bytes reales del checkpoint por clase (residente vs streamable) desde headers | 3 |
| `tools/make_tiny_checkpoint.py` | Genera el checkpoint tiny (mismo grafo, dtype-exacto) + referencia torch | 3 |
| `tools/emit_fixtures.py`, `make_k3_oracle.py`, `make_cache_fixture.py`, `make_st_fixture.py` | Generadores de fixtures adversariales | 3 |
| `tools/verify_*.py`, `ref_forward.py`, `cmp_logits.py` | Verificación contra torch (logits elementwise, max diff 7,87e-6) | 3 |
| `tools/devbw.py`, `budget.py` | Medición de ancho de banda del dispositivo | 3 |
| `tools/qdq_trunk.py`, `int8_trunk.py` | Derivación cuantizada del tronco para el draft model híbrido | 3 |
| `scripts/download-model.sh` | Descarga 1,56 TB + verificación por tamaños publicados + checksums | 3 |
| `scripts/k3-doctor.sh` | Diagnóstico de máquina: preset + ancho de banda de storage | 3 |
| `benchmarks/memory-ladder.sh`, `split-sweep.sh` | Harness de medición con cgroups (MemoryMax + MemorySwapMax=0) | 3 |

### 2.3 Tests

`make test` corre **sin pesos del modelo** (pico RSS ~1,7 GB): `test_ops` (fixtures adversariales con tolerancia declarada en MANIFEST), `test_cache` (shard sintético con tamaño de experto **deliberadamente no conforme a 4096**), `test_st` (dtypes, offsets, colas, nombres escapados, no-finitos), `test_cfg` (3 configs malformados que **debe** rechazar), `scale_test` (dimensiones reales 7168/93/96/896, una sola capa completa = 1,77 GB), `test_tok` (roundtrip byte-exacto + paridad con tokenizer de referencia), `k3_model` (oráculo tiny de 13 capas — 13 elegidas para que existan ≥2 bloques attn-res: 3 gates exactos: teacher forcing, greedy, incremental≡recompute).

Los tests dependientes del checkpoint (`test_expert`, `test_real_layer`, `conform_all.py` — 93 capas individuales contra referencia) requieren `SHARD_DIR`.

---

## 3. Las cinco clasificaciones pedidas

### 3.1 Acoplado exclusivamente a Kimi K3

Estas partes **no se portan**: son la arquitectura del modelo, no del motor. Sirven como referencia de cómo se implementa un bloque exótico con kernels elementales, y nada más.

| Componente | Por qué es K3-específico |
|---|---|
| **KDA** (Kimi Delta Attention), `k3_kda_decay`/`k3_kda_step`/`k3_kda_layer` | Recurrencia lineal con estado fijo S (H×D×D), olvido por head `g = g_min·σ(e^A·z)` con A_log **por head** (no por canal), ShortConv k=4 causal + SiLU fusionado, L2Norm (suma de cuadrados) solo sobre q,k, gate alfa por canal en (e^-5, 1], beta por head, y recurrencia en 4 pasos **ordenados** (decay → read → delta-write → read-updated). Ningún otro modelo open-weight usa esto. |
| **Gated MLA con NoPE**, `k3_mla_cached` | Las 64 dims rope **existen, se cachean, y nunca se rotan** (NoPE); softmax escala sobre 192 (no 128); slot rope compartido entre heads; k/v expandidos cacheados en fp32 (2,37 MB/pos); output gate. La MLA de DeepSeek (V2/V3) es *otra cosa*: comprime el latente y SÍ rota. Implementar "MLA" genérica requiere decidir cuál de las dos semánticas se sigue — no hay una sola MLA. |
| **Stable LatentMoE**, `k3_moe` | Ruteo sobre ancho completo → baja a latente 3584 → expertos **en el espacio latente** → RMSNorm sobre el **agregado** (no por experto) → up-project → 2 expertos compartidos de ancho completo sumados **sin peso**. La MoE de DeepSeek/Mixtral/Qwen es estándar (expertos a ancho completo, softmax+bias o sigmoid, sin latente). |
| **SiTU-GLU** | β1=4, β2=25; sigmoid sobre la puerta **sin capar** (el cap 100 aplica solo al output), soft caps saturables. Es la activación del linaje K2/K3. |
| **Block Attention Residuals**, `k3_attn_res` | Bloques de 12 capas; snapshot+clear del residual corriendo en cada frontera; cada capa atiende sobre las salidas de bloque; aggregador a nivel modelo (norm⊗proj plegado). Único. |
| **Tokenizer con flag `kimi=1`** | El formato `tiktoken.model` es compartido (GPT-4o), pero el pre-tokenizer de K3 agrega una regla líder de corridas Han y excluye Han de las clases de letras; `rankbpe=1`; los 3 invariantes silenciosos documentados en `k3_tok.h` (bytemap antes de construir claves, rankbpe, kimi) son K3-específicos. |
| Nombres de config y tensores | `text_config.*`, `linear_attn_config.*`, `activation_situ_beta`, `language_model.model.layers.%d.*`, `full_attn_layers` one-based. |
| Valores de dimensión | hidden 7168, 93 capas (69 KDA + 24 MLA en posiciones 4,8,…,92,93), 896 expertos top-16, vocab 163.840, etc. Obviamente. |

### 3.2 Reutilizable para otros Transformers (con trabajo de adaptación)

| Componente | Qué se lleva | Qué hay que adaptar |
|---|---|---|
| **`k3_st`** (safetensors) | El lector entero: scanner JSON a mano, hash FNV-1a con mezcla de bytes completos (comentario explícito sobre nombres hostiles con prefijos largos compartidos), pool de nombres con resolución cuando el pool deja de moverse, tabla hash open-chaining con load factor < 0,5 y detección de duplicados, validación por shard (dtype/offsets presentes, nbytes==shape×elemsize, fin-pasado-EOF, cola con bytes), doble fd (buffered + O_DIRECT con fallback), `k3_st_read_aligned` (ensanchado hacia afuera a ventana de 4 KB, short-read en EOF tolerado), `k3_st_read_f32` (chunks de 4 MB, subnormales f16). Safetensors es **el** formato de HuggingFace: todo modelo open-weight relevante lo usa. | Agregar dtypes que falten (p. ej. I64/F8 para modelos futuros). El formato K3 usa U8 para MXFP4 — un lector genérico trata U8 como bytes crudos y ya. |
| **`k3_bind`** (patrón Req/Plan) | El mecanismo `Req{name,tensor,want,take,narrow,dest,off}` → `plan_resolve` (chequeo de element count **antes** de leer) → `plan_load`. La idea de validar forma-contra-config antes de tocar bytes, el rechazo de takes parciales de tensores estrechos, y el split widen/narrow (matrices grandes quedan en bf16 con dispatch por tag; vectores leídos elemento-a-elemento se ensanchan a fp32 para que un tipo equivocado sea **error de compilación**, no salida plausible). | El mapa de nombres concreto es K3; hay que escribir un mapa por arquitectura (como `k3_bind.c` plan_layer). |
| **Loader de `tiktoken.model`** | `k3_tok.h` es un loader directo de `tiktoken.model` (base64 → byte-level string → hash) + `tokenizer_config.json` para especiales. Sirve tal cual para cualquier modelo que publique tiktoken.model (Qwen3 publica `tokenizer.json`; hay modelos que publican tiktoken). | Para el grueso de modelos HF hay que leer `tokenizer.json` (que es lo que el `tok.h` vendored ya hace — el loader K3 es el caso alternativo). |
| **Estado incremental + fingerprint** | `k3_state_save/load`: header con magic+version+fingerprint de 12 campos de config; restauración **se niega** ante arquitectura distinta ("produciría salida fluida y equivocada"). Patrón directamente portable. | El contenido del estado (S de KDA, KV MLA) es específico; el patrón no. |
| **Speculative decoding exacto** | Draft por n-gramas con gate de evidencia (los sufijos deben **coincidir** en la continuación; medida: draft ávido fue 0,91× en código), verificación greedy por batch, rollback de estado recurrente (snapshot + replay del prefijo aceptado), aceptación por construcción = salida idéntica al decode serial. Medición de acuerdo teacher-forced (94,2% draft int8 vs techo 96,2%). Todo genérico. | Nada conceptual; el snapshot de estado recurrente es KDA-específico, en un transformer estándar es el KV cache. |
| **Medición de I/O vs cómputo** | Contadores de carga (device rate vs bind wall, bytes leídos por token, requests/evictions) y el reporte de share I/O con advertencia de >100% (overlap real, no bug). Genérico. | — |

### 3.3 Generalizable (conceptos, mecánica y disciplina — lo más valioso)

| Concepto | Dónde vive | Por qué generaliza |
|---|---|---|
| **Pinned prefix + ring para recorridos fijos** | `k3_trunk.c`, `k3_trunk.h` | Todo decoder recorre sus capas en el MISMO orden fijo (0..N-1) por token. Eso hace el prefetch perfecto (no hay nada que predecir) y convierte un barrido cíclico — la patología clásica de LRU (hit rate 0% con N<93 slots) — en hit rate determinístico = K/N con K capas pineadas. Cada GB extra compra su parte justa. |
| **Asignación de presupuesto por tráfico garantizado** | `k3_run.c` presets, `k3_trunk.h` | Por token se relee el tronco ENTERO (108,81 GB) pero solo ~25,8 GB de expertos. Un GB en el tronco elimina ~1,17 GB/token de tráfico **garantizado**; un GB en cache de expertos, por debajo de la rodilla, no elimina nada medible. Medido: 1,69× más rápido a presupuesto total idéntico (Spearman ρ = −0,886 a 128 GB y −0,714 a 32 GB, doce puntos). **Método general: contar el tráfico por token de cada clase de peso y asignar por valor marginal.** |
| **Cache de expertos LRU + prefetch batch de 3 fases** | `k3_cache.c` | Fase 1 serial (reserva de slots, dedup intra-batch), fase 2 lecturas paralelas **ordenadas por offset de disco**, fase 3 publicación solo de lo que llegó. Estado INFLIGHT como tercer estado de slot para que pick_victim no entregue un slot cuya lectura no aterrizó. Aplica a cualquier MoE (DeepSeek, Mixtral, Qwen3-MoE) sin cambios conceptuales. |
| **Trace + replay offline** | `k3_cache.c` trace, `tools/sim_cache.py` | Como el ruteo no depende de la cache, UNA corrida produce TODA la curva: se graba (layer, expert) en orden (12 KB/token) y se simula cualquier política a cualquier capacidad. LRU vs Belady vs pinned. Hallazgo transferible: LRU plana de 8→64 GB de arena mientras Belady sube → hay localidad explotable que LRU no alcanza; la planitud de LRU era del workload (Quantile Balancing de K3), no universal — pero **el método de medirlo sí lo es**. |
| **Métrica de hit rate VERDADERA** | `k3_cache.h`, `k3_run.c` | `hits` cuenta expertos que el prefetcher trajo del disco microsegundos antes → "hits - prefetch_reads" y "retention = requests - evictions". Sin esto el reporte miente (hit rate sube mientras el I/O no baja). Lección de instrumentación universal. |
| **"Campo ausente = error, nunca default"** | `k3_cfg.h` | El peor fallo es un modelo fluido y equivocado. La demostración del doc: defaults en situ_beta (4,0/25,0 son CORRECTOS) + lista de capas vacía → el motor carga, streamea y decodifica la arquitectura equivocada sin ningún síntoma. Acumula TODAS las claves faltantes en una corrida. En Rust esto se traduce en `serde` con `deny_unknown_fields`... y sin `default()` salvo campos genuinamente opcionales. |
| **Contrato de punto flotante** | Makefile `-ffp-contract=off`, `k3_ops.c` | Dobles acumuladores, orden de reducción fijo ((a0+a1)+(a2+a3)), fma explícito = misma op IEEE por lane, scalar ≡ OpenMP ≡ AVX2 bit-idénticos, OpenMP solo sobre filas de salida independientes. En Rust: control explícito de `mul_add`, ordenes de suma fijos, y tests de igualdad bit a bit entre backends. |
| **Nunca desquantizar** | `k3_matmul_mxfp4`, `k3_cache.h` | El matmul lee nibbles directamente (LUTs E2M1[256][2] y E8M0[256], wf[64] en stack). Desquantizar un experto lo lleva de 17,55 MB a 132 MB (7,5×) sin beneficio porque el GEMV es memory-bound. Generaliza a cualquier formato empaquetado (Q4_K, IQ, etc.): **cachear y multiplicar en formato empaquetado**. |
| **O_DIRECT + hugepages para streaming** | `k3_st.c`, `k3_trunk.c` | Bajo cap de 32 GB, buffered midió 1.247 MB/s en un disco de 6.400; O_DIRECT lo recupera. Un experto streameado se lee una vez y se desaloja: buffered solo lo copia dos veces y expulsa algo útil. Ventana ensanchada a 4 KB + fallback a buffered. 2 MB posix_memalign + MADV_HUGEPAGE para targets O_DIRECT. A/B con env vars (K3_NOHUGE, K3_NOPREFETCH). Aplica a cualquier runtime de streaming en Linux. |
| **pread, no mmap** | `k3_st.h` | A 1,56 TB, mmap haría que el RSS contara todo el checkpoint contra el proceso y la medición perdería sentido. Para un runtime *disk-first* la decisión pread-vs-mmap es central (ver MEMORY-DESIGN.md — llama.cpp usa mmap y funciona, pero con trade-offs distintos). |
| **Presupuesto como dial** | `k3_run.c` | El mismo modelo corre en 8 GB y en 224 GB con salida byte-idéntica (verificado: `17374, 20829, 10, 427, 414, 1008, 606, 142957` en los 12 escalones). Los presets son presupuestos, no pisos. `--preset auto` dimensiona desde MemAvailable con reserva explícita y un techo de 55% RSS para evitar el hazard de pinning parcial (medido: pinear 51/109 GB fue 14% MÁS LENTO que no pinear nada por reclaim del kernel). |
| **Disciplina de medición** | `docs/BENCHMARKING.md`, `benchmarks/` | Piso de ruido medido: 33%. Regla: 3 repeticiones mínimo, reportar todas, nunca la mejor. Cgroups MemoryMax+MemorySwapMax=0 (sin swap la escalera mide banda de swap, no la config bajo prueba). Contadores (GB leídos, requests, evictions) antes que relojes: los contadores **no se mueven** y no necesitan replicación. Registro de máquina (`machine.txt`). |
| **Escalera de testing** | `docs/TESTING.md`, fixtures | Per-op con fixtures **adversariales** (un test que no puede fallar no es un test: el fixture del router reordena el top-2 en 5 de 6 filas; SiTU-GLU satura el cap analítico; el shard de cache tiene tamaño no múltiplo de 4096 para que la coincidencia 17.547.264=4284×4096 no oculte bugs) → oráculo tiny con el MISMO grafo de tensores (13 capas para ≥2 bloques attn-res) → conformancia por capa contra referencia → logits elementwise contra torch (max diff 7,87e-6). Todo sin pesos: si la corrección dependiera del checkpoint de 1,56 TB, no se chequearía nunca. |
| **"Refuse rather than clamp"** | todo el código | Prompt ambiguo → error; gen fuera de rango → error; topk > K3_MAX_TOPK → error; KV cache que no entra → negarse con ambos números; expert drop → exit 4 ("RUN INVALID", la salida es CORRUPTA); estado con fingerprint distinto → negarse; download con bytes que no cuadran → FAIL. La regla: **un fallo que se anuncia es gratis; un fallo silencioso produce texto plausible.** |
| **Slot sizing desde el checkpoint, no desde aritmética** | `k3_cache.c` | El tamaño de slot se mide del archivo real (probe), no se deriva; el fixture no conforme existe para gatear la coincidencia aritmética. Igual: `budget.py` clasifica tensores por bytes REALES (el error que corrige: los shared experts son 6144 de ancho y bf16, parecen expertos, pero corren cada token → residentes; la aritmética a mano los clasificaba como streameables). |
| **Lector asíncrono con 2 slots mínimo** | `k3_trunk.c` | Con 1 slot + reader thread hay **corrupción silenciosa** que produce tokens equivocados plausibles (32609 2329... vs 17374 20829...). El reader nunca publica un nombre de capa antes de que la lectura termine; bind espera la condición antes de consumir. Lección universal para cualquier diseño de streaming asíncrono: la corrección exige ≥2 slots o publicación explícita post-éxito. |
| **Dedup de expertos en prefill batch** | `k3_moe_prefill` | Chunks de 64 tokens, contribución [T][K][latent], fetch expert-major, dedup de expertos únicos → 3-4× menos I/O de prefill, bit-idéntico (cola seriada). Generaliza a cualquier MoE. |
| **Híbrido draft cuantizado + verificación exacta** | `k3_run.c` | Un segundo tronco cuantizado (I8R per-row) propone; el modelo exacto verifica en sweeps batcheados. La exactitud es estructural (lo que se emite es lo que el modelo exacto habría elegido). El draft solo comparte lo idéntico (embed, lm_head, ruteo de expertos con cache_only). Generaliza. |
| **Shims de portabilidad I/O** | `k3_portable_io.h` | O_DIRECT (Linux) ↔ F_NOCACHE (Darwin) tras open(); fadvise como no-op seguro. En Rust: traits `DiskReader` con impls por plataforma. |

### 3.4 Diseñado específicamente para reducir RAM

| Mecanismo | Reducción concreta |
|---|---|
| Expertos nunca residentes | 1,45 TB → 0 residentes; ~25,8 GB/token leídos por demanda |
| Tronco streameado (pin + ring) | 108,81 GB → desde ~2,4 GB de ring (slot de 2,34 GB por la capa 0 densa; 2 slots + prefetch = 71,75 → 42,27 s/token, 1,70×) |
| Cachear MXFP4 empaquetado, no floats | slot de 17,55 MB vs 132 MB (7,5× más expertos por GB) |
| O_DIRECT en expertos y tronco | evita el doble buffering del page cache bajo presión (1.247 → 6.400 MB/s medidos bajo cap) |
| Widen selectivo | Solo vectores leídos elemento-a-elemento van a fp32 (2,43 GB en 93 capas, de los cuales 1,18 GB es el router gate); matrices grandes quedan bf16 (108,81 GB vs 227 GB todo-fp32) |
| Pin con asignación exacta por capa | La capa 0 densa mide 2,34 GB vs 1,27 GB normal; slots uniformes desperdiciarían la mitad del presupuesto |
| Plan de memoria antes de asignar | Suma TODO (tronco, modelo, cache, estado, buffers, KV) vs MemAvailable×0,95; negarse con ambos números en vez de morir por OOM a la hora de ejecución |
| KV cache chequeado up-front | 2,37 MB/pos × posiciones vs disponible; negarse temprano |
| Embed por fila, nunca ensanchada | 2,35 GB bf16 tal cual, sin copia fp32 |
| Auto budget | MemAvailable − reserva fija (2 GB + 2% + 4,70 + 1,70) − cache mínima; techo RSS 0,55×MemTotal si no cabe el tronco completo |
| pico RSS como cifra autoritativa | getrusage (con la corrección KB-vs-bytes Linux/Darwin); el banner de plan se declara explícitamente un pronóstico que SOBREESTIMA |

### 3.5 Dependiente del formato particular de pesos de Kimi K3

| Componente | Detalle |
|---|---|
| MXFP4/E2M1/E8M0 | Nibble bajo = elemento PAR (convención, no regla — invertirlo da una matriz con estadísticas correctas y modelo equivocado; hay fixture para eso); escala por grupo de 32; 255=NaN→0; 0,53125 B/peso; contrato de exactitud ~1e-16 vs dequant |
| Layout de experto | 6 tensores (`w1/w2/w3 × weight_packed/weight_scale`) contiguos en una corrida de exactamente 17.547.264 bytes; nombre `language_model.model.layers.%d.block_sparse_moe.experts.%d.%s.%s`; fallback a 6 preads si un checkpoint futuro intercala distinto |
| Tags de dtype | `K3_WF32/K3_WBF16/K3_WI8` + dispatch `k3_mmw`; bf16→f32 = shift-left-16 (propiedad del formato bf16, sin tabla ni redondeo) |
| I8R draft | Formato del tronco draft: `[f32 scale][int8*cols]` por fila, dequant en widen buffer |
| Formato tronco empaquetado | `trunk.bin` (corridas por capa, padding 4 KB en cabeza y cola) + `trunk.json` (file_off, nbytes redondeado a ALIGN, tensores con off relativo) — el CONTENIDO es K3; el patrón (un archivo, corridas por capa, una pread por capa) es generalizable |
| Quirks de datos | A_log embarcado como [128] para 96 heads (tomar los primeros 96 — leer los 128 "corre sin queja y da otro modelo"); `q_conv1d.weight` rank-3 [H*D][1][conv_k] que el kernel quiere rank-2 (mismos bytes, el chequeo de element count lo acepta sin repack) |
| Config | Dos shapes JSON (nested/flat) con aliases; one-based layer list; `activation_situ_beta` vs `situ_beta` |
| Tokenizer | `tiktoken.model` + flag `kimi=1` + `rankbpe=1` |

---

## 4. Hallazgos medidos que el diseño nuevo debe heredar

(Con sus fuentes en `docs/data/` — los números exactos, no los redondeados, están en los archivos.)

1. **El piso de ruido es 33%.** Tres corridas idénticas: 14,78 / 14,67 / 20,14 s/token. Todo el diseño de benchmarking del runtime nuevo hereda la regla de las 3 repeticiones y el reporte de todas.
2. **La asignación le gana a la capacidad.** A 128 GB fijos, tronco-primero corre 1,69× más rápido que cache-primero. La configuración más rápida tiene 0,0% de retención de expertos. El método (tráfico garantizado por token) es la herencia, no el factor exacto.
3. **La cache de expertos es plana hasta la rodilla.** Retención 0,0% de 28 a 1.344 slots (0,49→23,59 GB de arena); bytes/token clavados en 25,83 GB en siete presupuestos. En replay offline, LRU es plana de 8→64 GB mientras Belady sube → localidad real que LRU no alcanza. Lección: **para modelos con ruteo balanceado, no regalarle RAM a la cache de expertos; para modelos con hot set, el trace lo dirá.** Y el trace se graba barato.
4. **Pinning parcial es un hazard.** 51/109 GB pineados = 14% MÁS LENTO que 0 pineados (reclaim del kernel). El auto-budget lo documenta y lo evita con techo de RSS.
5. **O_DIRECT importa bajo presión.** 1.247 vs 6.400 MB/s bajo cap de 32 GB; ~6× en un disco rápido. A escala de 135 GB/token, el storage es el techo usual.
6. **El tronco NO se cuantiza a 4 bits.** int4 post-hoc: 17,4% de error medio relativo de pesos (filas peores 65%) vs int8 0,96% — en tensores de atención que el reporte técnico de K3 mantiene en precisión alta a propósito. Streaming pierde cero exactitud; cuantizar pierde lo que nada recupera. (Esto es específico de K3, pero la lección general es: **clasificar los pesos por sensibilidad antes de decidir el formato de streaming.**)
7. **El share de I/O es 40,9–60,6%** en la escalera, y puede superar 100% con solape real (reportado y explicado, no escondido).
8. **Long runs amortizan el cold start.** La escalera (8 tokens) subestima: 19,21 vs 10,66–11,79 s/token en generaciones sostenidas de 16–32 tokens.

---

## 5. Qué NO heredar (decisión explícita)

1. **Los kernels K3** (§3.1). El runtime nuevo implementa arquitecturas estándar (Llama/Qwen/DeepSeek/Mistral/Mixtral); los kernels K3 quedan como referencia de estilo (comentarios que documentan INVARIANTES, órdenes de pasos numerados, contratos numéricos).
2. **OpenMP.** En Rust, rayon para paralelismo de datos a nivel de filas (con el mismo principio: paralelizar solo sobre filas de salida independientes, orden de reducción fijo dentro de cada fila). El contrato "scalar ≡ SIMD ≡ paralelo" se conserva.
3. **El formato MXFP4 como formato propio.** El runtime nuevo no inventa formatos: consume GGUF (el estándar de facto para CPU, ver MODEL-CANDIDATES.md) y safetensors. MXFP4 queda como caso de estudio de "matmul que consume formato empaquetado sin dequantizar" — el principio se aplica a los quants GGUF.
4. **El patrón O(T²) full-recompute como default.** K3 lo tiene como camino validado por el oráculo; el runtime nuevo hace incremental (KV cache) el camino principal y conserva recompute como test de equivalencia.
5. **La aritmética de 8,2 GB.** Es K3. El runtime nuevo tiene su propia escalera (ver MEMORY-DESIGN.md), pero la *estructura* de la escalera (cgroup cap, salida idéntica en cada escalón, contadores) se copia.

---

## 6. La pregunta honesta: ¿qué hace llama.cpp mejor?

(Detalle completo en MODEL-SUPPORT.md; aquí lo esencial para el contraste.)

- **Ecosistema GGUF**: cientos de miles de modelos convertidos, quants estandarizados (Q4_K_M, IQ1..IQ4), `llama-quantize` maduro, `convert-*.py` por arquitectura. Un runtime nuevo que NO lea GGUF se excluye del 95% de los pesos disponibles para CPU.
- **Performance pulida por años**: kernels AVX2/AVX-512/NEON optimizados a mano, batch scheduling maduro, soporte multi-backend (CPU, CUDA, Metal, Vulkan, SYCL...).
- **mmap + page cache**: para modelos que CABEN en RAM, llama.cpp es esencialmente óptimo; el OS hace el trabajo de cachear y el código es simple.
- **Soporte de arquitecturas**: decenas, incluidas las difíciles (MLA de DeepSeek, MoE de todo tipo).

**Dónde el proyecto nuevo puede diferenciarse legítimamente** (lo que llama.cpp NO hace, verificado por el agente de investigación):

- Políticas de memoria **explícitas y configurables** (pin prefix + ring, cache de expertos con presupuesto propio, O_DIRECT, techo RSS) en lugar de "mmap y confiar en el page cache".
- El caso `model_size >> RAM` tratado como característica central con instrumentación de investigación: trace de accesos, replay offline (LRU vs Belady), hit rate verdadero, bytes/token por clase de peso.
- Output **bit-idéntico** a través de presupuestos de memoria (K3 lo garantiza y lo testea; llama.cpp no hace esta promesa).
- "Refuse rather than guess" como filosofía de producto (config, formatos, estados).
- Arquitectura didáctica en Rust: adapters por arquitectura explícitos, traits de storage, backends SIMD intercambiables — para aprender el motor desde abajo, que es el objetivo declarado del usuario.

**Dónde NO intentar competir**: kernels más rápidos que llama.cpp en el caso residente, cobertura de arquitecturas, tooling de conversión. La honestidad aquí es parte del diseño: ver MODEL-SUPPORT.md §"llama.cpp".
