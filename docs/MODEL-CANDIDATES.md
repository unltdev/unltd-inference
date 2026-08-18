# Candidatos de modelos para unltd-inference

**Máquina objetivo**: 16 GB RAM · x86-64 AVX2 · Windows principal + WSL2 · CPU-first · ≤ 1 TB de almacenamiento.
**Fecha de la investigación**: 2026-08-17/18, contra configs verificados en vivo (config.json de HuggingFace/ModelScope), tamaños GGUF reales donde se cita el repo, y el estado de llama.cpp master (b10472).

## 1. Resumen ejecutivo

| Decisión | Modelo | Por qué |
|---|---|---|
| **PoC (Stage 1)** | **Qwen3-0.6B** | Vanilla verificado, Apache-2.0, GGUF oficial, tokenizador compartido por toda la familia, camino directo al MoE estrella. Justificación completa en `ROADMAP.md` §3. |
| MoE de arranque | **OLMoE-1B-7B** | 64 expertos top-8, Apache-2.0, GGUF oficial: el test de cache de expertos más barato que existe. |
| **MoE estrella** | **Qwen3-30B-A3B** | 30.5B totales / 3.3B activos, Q4_K_M 18.6 GB en disco, ~2.2 GB residentes streameado, llama.cpp maduro para comparar. |
| Showcase denso > RAM | **Qwen3-32B** | Q4 no cabe en 16 GB (19.3 GB) pero streameado son ~8.2 GB: el caso "model_size > available_RAM" denso. |
| Meta final (Stage 5+) | **DeepSeek-V3/R1 Q4** | 671B / 37B activos, 378 GB en disco, MIT: la validación definitiva de streaming extremo. |
| Denso de calidad MIT | **Phi-4** | 14.7B, Q4 8.9 GB cabe cómodo, arquitectura estándar, MIT. |

## 2. Metodología y criterios

**Qué se verificó vs qué se estima**: `[V]` = verificado en config.json / API de HF / repo GGUF citado durante esta investigación. `[E]` = estimación por conteo de parámetros (BF16 = 2.0 B/param, INT8 ≈ 1.06, INT4 ≈ 0.61). Los "GB streameado" son estimaciones del equipo de diseño: pesos residentes (embeddings + una capa + activaciones) + KV cache.

**Definición de clases**:

- **A — muy buen candidato**: arquitectura dentro del alcance v1 (transformer denso estándar o MoE simple), licencia Apache/MIT, GGUF disponible como referencia numérica, utilidad real, y cabe en 16 GB ya sea residente (mmap) o por streaming del tronco.
- **B — posible, con optimización**: cabe con cuantización agresiva o streaming, o tiene UNA traba puntual (licencia no-OSI, una operación extra como partial RoPE, tokenizador fuera de v1, sin GGUF de referencia).
- **C — experimental**: requiere streaming/offloading como la razón de existir (modelo claramente > RAM), arquitectura exótica (MLA, DeltaNet, FP4 propietario) o es un objetivo futuro.
- **D — no recomendado**: no entra ni streameado, licencia no-comercial, repo eliminado, o calidad obsoleta sin valor de benchmark.

**Reglas que aplicamos**: (1) el almacenamiento de 1 TB se chequea contra el tamaño BF16 del checkpoint y el Q4 en disco; (2) "cabe en 16 GB" significa pesos + KV + ~2 GB de overhead del proceso; (3) un modelo delisted (Qwen2-MoE-A2.7B) no es candidato aunque la arquitectura sea bonita — no hay referencia contra la cual validar.

## 3. Pregunta A vs Pregunta B

Antes de las tablas, la separación pedida explícitamente:

### Pregunta A — modelos que podrían reusar parte significativa de kimi-k3-in-c

Son los modelos cuya **forma de ejecución** coincide con lo que K3 resuelve: tronco denso de recorrido fijo + expertos ruteados + (en su caso) MLA.

1. **DeepSeek-V2-Lite, V3/R1, V3.2, V4** — el linaje arquitectónicamente más cercano a K3: MLA (K3 usa MLA NoPE), MoE con expertos compartidos (K3: 2 compartidos), YaRN, RMSNorm. La maquinaria K3 (cache LRU de expertos con prefetch en 3 fases, ring de tronco, trace/replay, presupuestos) **mapea directo** sobre esta familia. El propio K3 comparte ADN: `first_k_dense_replace` de DeepSeek = capa 0 densa de K3.
2. **Qwen3-MoE (30B-A3B), Qwen2-57B-A14B, Mixtral, OLMoE, Gemma-4-MoE** — atención distinta (GQA/MHA), pero el problema de streaming es el mismo: tronco de orden fijo + expertos de acceso dependiente de datos. Todo el subsistema de memoria de K3 se reusa conceptualmente; solo cambia la atención.

### Pregunta B — modelos NO directamente compatibles, pero buenos para un runtime inspirado en el proyecto

1. **Qwen3/Qwen2.5 densos, Llama 3.x, Phi, Mistral densos, SmolLM** — transformers estándar sin nada K3-específico. Nada del código de K3 se reusa línea a línea; lo que se hereda es la **filosofía completa**: disk-first, presupuesto como dial, streaming de tronco, determinismo entre presupuestos, negarse con números. Estos modelos son los que validan que el runtime no es "K3 en Rust" sino un motor general.

**Conclusión**: la respuesta correcta es las dos a la vez — y el roadmap está ordenado para eso: primero Qwen denso (Pregunta B, arquitectura trivial), después MoE (Pregunta A, donde el valor del streaming explota).

## 4. Familia Qwen (Alibaba)

La familia central del proyecto: Apache-2.0 en TODOS los tamaños, GGUF oficiales en toda la línea densa, y el tokenizador tiktoken-style BPE (vocab 151,936) compartido de 0.5B hasta 30B-A3B — el trabajo de tokenizador se amortiza sobre todos los stages.

| Modelo | Total/Activos | BF16 GB | INT8/INT4 GB | GGUF | Clase | Nota |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.49B | 1.00 [V] | 0.5 / 0.25 [E] | sí | **A** | SWA = ctx completo (no-op de facto) |
| Qwen3-0.6B | 0.76B [V] | 1.52 [V] | 0.8 / 0.4 [E] | sí, oficial [V] | **A** | **PoC**. `use_sliding_window: false` verificado |
| Qwen3-1.7B | 2.04B [V] | 4.08 [V] | 2.0 / 1.0 [E] | sí | **A** | Stage 1–2 |
| Qwen3-4B | 4.03B [V] | 8.06 [V] | 4.0 / 2.0 [E] | sí | **A** | Stage 2 |
| Qwen3-8B | 8.2B [V] | 16.40 [V] | 8.2 / **5.0 Q4_K_M** [V] | oficial [V] | **A** | Stage 3; 42 t/s es el régimen de referencia |
| Qwen3-14B | 14.78B [V] | 29.55 [V] | 14.8 / ~7.4 [E] | sí | **A** | Q4 ~8.8 GB cabe |
| Qwen3-32B | 32.77B [V] | 65.54 [V] | 32.8 / ~16.4 [E] | sí | **B** | Q4_K_M ~19.3 GB NO cabe mmap; streameado ~8.2 GB: showcase denso |
| Qwen3-30B-A3B | 30.5B / 3.3B [V] | 61.08 [V] | 30.5 / **18.6 Q4_K_M** [V] | oficial [V] | **A** | **MoE estrella.** 128e top-8, sin shared, sigmoid+renorm. Streameado ~2.2 GB. llama.cpp: 42 t/s en EPYC, >10 t/s en PC de 16 GB |
| Qwen3-Coder-30B-A3B | 30.5B / 3.3B [V] | 61.1 [V] | 18.6 Q4 / 16.4 IQ4_XS [V] | sí [V] | **A** | Mismo arch, θ=1e7, 262K ctx. Flagship de código |
| Qwen2-57B-A14B | 57.4B / 14B [V] | 114.83 [V] | 57 / 29 [E] | fino | **B** | Softmax router + shared expert; calidad 2024 |
| Qwen3-Coder-480B-A35B | 480B / 35B [V] | 960.3 [V] | 480 / 240 [E] | escaso | **C** | BF16 roza el TB; streameado ~7 GB viable. Experimental |
| Qwen3-Next-80B-A3B | 80B / ~3B [V] | 162.7 [V] | IQ4_XS 43.1 [V] | sí [V] | **C** | GatedDeltaNet híbrido + partial RoPE: costo alto, streameado ~2.5 GB |
| Qwen3.5/3.6-35B-A3B | 35B / ~3B [V] | 71.9 [V] | UD-IQ4_XS 17.5 [V] | sí [V] | **B** | 256e top-8 + shared, MTP, tokenizador NUEVO 248k. El upgrade 2026; streameado ~3 GB |
| Qwen3.5-27B | 27.8B [V] | 55.6 [V] | ~14 [E] | sí | **B** | Denso híbrido; streameado ~7 GB |
| Qwen3.5-9B/4B/2B/0.8B | 0.9–9.7B [V] | 1.8–19.3 [V] | 0.5–5.5 [E] | emergente | **C** | DeltaNet + vision tower: fuera de v1 |
| Qwen3.5-122B/397B | 122B/397B [V] | 250/807 [V] | — | — | **D** | Para 16 GB no |
| Qwen2-MoE-A2.7B | (14.3B) | — | — | no | **D** | Repo delisted, sin referencia |

**Arquitectura Qwen verificada** (correcciones importantes vs folklore): Qwen3 denso = GQA sin bias, sin QK-norm, sin SWA, θ=1e6, **silu** (el mito "ReLU²" es falso — ningún config de Qwen3 lo usa), RoPE completo sin partial. Qwen2.5 = igual salvo SWA en las capas finales con ventana igual al contexto máximo (no-op en la práctica). El delta real Qwen2-MoE → Qwen3-MoE es el router: softmax+bias → **sigmoid+renorm** (`norm_topk_prob: true`).

## 5. Familia DeepSeek

| Modelo | Total/Activos | BF16 GB | GGUF | Clase | Nota |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 15.7B / 2.4B | 31.4 | Q4_K_M 9.65 GB [V] | **B** | MLA (kv_lora 512, qk_rope 64), 64e top-6 + 2 shared, YaRN 40, vocab 102k. Licencia DeepSeek (no MIT). Streameado residente ≈ 8.2 GB BF16: el sweet spot de streaming MLA |
| Coder-V2-Lite | 15.7B / 2.4B | 31.4 | sí | **B** | Mismo config que V2-Lite |
| DeepSeek-V3 / R1 | 671B / 37B | 1343 | **Q4_K_M 378 GB** [V] | **C** | MIT. Sigmoid noaux_tc, top-8/256 + 1 shared, MTP, YaRN 16. Streaming INT3 ~11 GB: la meta Stage 5. Referencia en-RAM llama.cpp: 8.45 t/s decode |
| R1-Distill-Qwen-1.5B/7B/14B | denso Qwen2.5 | 3/15/30 | sí | **A** | MIT sobre arquitectura Qwen2.5: regalo de compatibilidad |
| R1-Distill-Qwen-32B | 32B | 65 | Q4 ~19 GB | **B** | Qwen2.5-32B denso: no cabe mmap, streameable |
| DeepSeek-V3.2 | 685B / ~40B | 1370 | emergente | **C** | MIT, DSA indexer. Futuro |
| DeepSeek-V4-Flash-0731 | 304B / 13B | 155.4 en disco (FP8+FP4) | **no en llama.cpp upstream** | **C** | MIT, FP4 experts (e4m3/ue8m0), 1M ctx. Experimental puro |
| V4-Pro / R2 | 1.6T / — | — | — | **D** | Fuera de alcance; R2 no existe (jul 2026) |

La familia DeepSeek es la respuesta de la **Pregunta A**: K3 y DeepSeek comparten el linaje MLA+MoE, y la maquinaria de memoria del proyecto de referencia se hereda casi 1:1. La traba es la licencia de V2-Lite (DeepSeek License, no OSI) — V3/R1 son MIT.

## 6. Llama (Meta)

| Modelo | Total/Activos | Q4 GGUF | Clase | Nota |
|---|---|---|---|---|
| Llama-3.2-1B / 3B | 1.23 / 3.21B [V] | 0.83 / 2.0 GB | **B** | Arq trivial, tied, tiktoken BPE 128k. Licencia Community (no-OSI), repos gated |
| Llama-3.1-8B | 8.03B | 4.92 GB | **B** | El denso 8B de referencia del ecosistema; misma licencia |
| Llama-4-Scout-17B-16E | 109B / 17B [V] | ~66 GB | **C** | 48 capas TODAS MoE, top-1 de 16, qk-norm, 10M ctx declarado. Streaming viable (~1.4 GB Q4/capa) pero licencia + reputación 2025 |
| Llama-4-Maverick-17B-128E | ~402B / 17B [V] | ~244 GB | **D** | ~5.1 GB Q4/capa: apenas streameable y mala fama |

Veredicto honesto: como familia de soporte, Llama da el "ecosistema gold standard" (cada runtime lo benchemarkea) pero su licencia no-OSI la relega a tests de compatibilidad. "Llama 5" no existe (los rumores 2026 fueron retractados).

## 7. Mistral / Mixtral

| Modelo | Total/Activos | BF16/Q4 GB | Clase | Nota |
|---|---|---|---|---|
| Mistral-7B-v0.3 | 7.24B | 14.5 / 4.4 | **B** | Apache-2.0, trivial (SWA null en v0.3), pero calidad obsoleta: benchmark legacy |
| Mistral-Small-3.1/3.2-24B | 24.2B | 48.4 / **14.7** | **A** | Apache-2.0, θ=1e9, 128K. Q4 cabe JUSTO en 16 GB: el denso grande residente |
| Ministral 3 (3B/8B/14B) | 3.5/8.8/~14B | 17.6–28 | sí | **A** | Apache-2.0, 256K, Base/Instruct/Reasoning. La generación 2025 |
| Mixtral-8x7B | 46.7B / 12.9B | 93 / 28.7 | **B** | Apache, 8e top-2, softmax. SentencePiece (fuera de v1). Clásico de compatibilidad |
| Mixtral-8x22B | 141B / 39B | 282 / 85 | **C** | Apache; 39B activos pesan. Streaming 1.5 GB Q4/capa |
| Mistral-Small-4-119B | 119B / ~6.5B | 238 / ~72 | **C** | Apache. MLA + 128e top-4 + shared + YaRN 1M: el "DeepSeek de Mistral". Futuro |
| Mistral-Large-3-675B | 675B / 41B | **1350** | **D** | BF16 excede el TB de almacenamiento |

## 8. Gemma (Google)

Licencia: Gemma 2/3/3n = Gemma Terms; **Gemma 4 = Apache-2.0** (primer cambio de la familia).

| Modelo | Total/Activos | BF16/Q4 GB | Clase | Nota |
|---|---|---|---|---|
| Gemma-2-2B / 9B | 2.6 / 9.2B | 5.2/1.6 · 18.5/5.6 | **B** | GeGLU, soft-capping, SWA intercalada, SentencePiece 256k, Terms |
| Gemma-3-4B / 12B | 4.4 / 12.2B | 8.8/2.7 · 24.4/7.4 | **B** | Patrón 5:1 local/global, RoPE dual por capa (θ=10k/1M). Mejor calidad-per-RAM de su generación, pero stack no-vanilla + Terms |
| Gemma-3-27B | 27.4B | 54.8 / 17.1 | **C** | Q4 pasa de 16 GB; Q3 o streaming + quirks |
| Gemma-4-12B | ~12B | ~24 / ~7.4 | **B** | Apache. Denso con windowing: candidato denso 2026 |
| Gemma-4-26B-A4B | 25.2B / 3.8B | 50 / 15.5 | **C** | Apache. 128e top-8 + shared, FFN dual-path, atención híbrida. Streaming ~0.5 GB Q4/capa: el MoE moderno más atractivo a futuro |
| Gemma-4-31B | 30.7B | 61 / 18.6 | **C** | Apache. Q4 no cabe; streameable |
| Gemma-3n E2B/E4B | 5.1–8.1B eff | — | **D** | MatFormer/PLE/LAuReL: exótico sin retorno |

## 9. Phi (Microsoft) — MIT

| Modelo | Params | Q4 GB | Clase | Nota |
|---|---|---|---|---|
| Phi-3-mini / medium | 3.8 / 14.2B | 2.3 / 8.6 | **B** | Datados pero fáciles |
| **Phi-4** | 14.7B | **8.9** | **A** | Arquitectura estándar (θ=250k, sin partial rotary en config), MIT, excelente en math/código. El denso MIT de referencia |
| Phi-4-mini | 3.76B | 2.3 | **B** | partial_rotary_factor 0.75 + LongRoPE: un quirks |
| Phi-4-mini-flash-reasoning | 3.8B | — | **D** | Híbrido SSM (SambaY) + differential attention: fuera de alcance |

## 10. Modelos chicos y MoE chicos (la escalera de testing)

| Modelo | Total/Activos | Q4 GB | Clase | Nota |
|---|---|---|---|---|
| SmolLM2-135M / 360M / 1.7B | 0.135 / 0.36 / 1.7B | 0.1 / 0.2 / 1.0 | **A** | `model_type: llama` puro, Apache-2.0, GGUF pleno. Smoke tests ultra-rápidos (135M corre en segundos) |
| SmolLM3-3B | 3.0B | 1.9 | **B** | Un quirks: capas NoPE (cada 4ª sin RoPE) + YaRN. Apache |
| TinyLlama-1.1B | 1.1B | 0.66 | **B** | Vanilla pero 2023 y SentencePiece |
| OLMo-2-0425-1B | 1.48B | 0.9 | **B** | QK-norm (una operación extra), Apache, datos 100% abiertos |
| **OLMoE-1B-7B** | 6.9B / **1.1B** | **4.0** | **A** | **MoE de test ideal**: 64e top-8, Apache, GGUF OFICIAL. 4.6 GB Q4 residente; el top-8 exige cache de expertos en serio |
| MiniCPM-MoE-8x2B | 13.9B / ~4.3B | ~8 | **B** | 8e top-2, Apache, sin shared. PERO: sin GGUF ni referencia llama.cpp — sin oráculo contra el cual validar |
| MiniCPM3-4B | 4.0B | 2.4 | **C** | MLA + escala emb/depth + licencia de pesos con registro |
| MiniCPM-2B | 2.4B | 1.4 | **C** | scale_emb/scale_depth + calidad 2024 |
| Pythia / GPT-Neo | 0.16–2.8B | escaso | **C** | SOLO como corpus de arch-test (parallel residual, partial rotary): no utilidad |
| StarCoder2-3B | 3.0B | 1.8 | **C** | LayerNorm+GELU+bias+SWA, OpenRAIL: niche de código |
| EXAONE-3.5-2.4B | 2.14B | 1.2 | **D** | Licencia NO-comercial: bloqueada |

## 11. Tabla maestra de clasificación

**A (muy buenos)**: Qwen3-0.6B · Qwen2.5-0.5B · Qwen3-1.7B/4B/8B/14B · Qwen3-30B-A3B · Qwen3-Coder-30B-A3B · R1-Distill-Qwen-1.5B/7B/14B · Mistral-Small-3.1/3.2-24B · Ministral 3 · Phi-4 · SmolLM2 (3 tamaños) · OLMoE-1B-7B

**B (posibles)**: Qwen3-32B · Qwen3.5/3.6-35B-A3B · Qwen3.5-27B · Qwen2-57B-A14B · DeepSeek-V2-Lite · Coder-V2-Lite · R1-Distill-32B · Llama-3.2-1B/3B · Llama-3.1-8B · Mistral-7B-v0.3 · Mixtral-8x7B · Gemma-2-2B/9B · Gemma-3-4B/12B · Gemma-4-12B · Phi-4-mini · Phi-3-mini/medium · SmolLM3-3B · TinyLlama-1.1B · OLMo-2-0425-1B · MiniCPM-MoE-8x2B

**C (experimentales/futuro)**: DeepSeek-V3/R1 · V3.2 · V4-Flash · Qwen3-Coder-480B-A35B · Qwen3-Next / Coder-Next-80B-A3B · Qwen3.5-9B/4B/2B/0.8B · Llama-4-Scout · Mixtral-8x22B · Mistral-Small-4-119B · Gemma-3-27B · Gemma-4-26B-A4B · Gemma-4-31B · MiniCPM3-4B · MiniCPM-2B · Pythia/GPT-Neo · StarCoder2-3B

**D (no recomendados)**: Qwen2-MoE-A2.7B (delisted) · Qwen3.5-122B/397B · DeepSeek V4-Pro/R2 · Llama-4-Maverick · Mistral-Large-3-675B (disco > 1 TB) · Gemma-3n · Phi-4-mini-flash · Ministral-2024 (licencia) · EXAONE (NC) · OLMo-1B (discrepancia de params)

## 12. Régimen de memoria a 16 GB (lo que cada clase exige del runtime)

| Régimen | Modelos | Qué usa el runtime |
|---|---|---|
| **Residente cómodo** (< 6 GB Q4) | Qwen3-0.6B→8B, Phi-4, SmolLM2, Llama-3.2, Gemma-3-4B, Ministral-3B | Carga completa en arena propia + KV. Mide el piso de rendimiento sin streaming |
| **Residente justo** (6–15 GB Q4) | Qwen3-14B, Mistral-Small-24B (14.7 GB), Gemma-4-12B | Presupuesto como dial: plan de memoria obligatorio, negarse con números si el KV no entra |
| **Streaming denso** | Qwen3-32B (~8.2 GB), Gemma-3-27B, Gemma-4-31B | Pin prefix + ring de 2 slots sobre el tronco denso — exactamente la maquinaria K3 sin MoE |
| **Streaming MoE (estrella)** | Qwen3-30B-A3B (~2.2 GB), OLMoE (~1 GB), MiniCPM-MoE (~2 GB) | Cache LRU de expertos + prefetch batch + trace/replay. El caso donde el diseño gana |
| **Streaming extremo** | DeepSeek-V3 Q4 (378 GB disco), Qwen2-57B, Mixtral-8x22B, Llama-4-Scout, Qwen3-Coder-480B | Todo lo anterior + medición de tráfico por token e I/O share. Stage 5 |

**Nota de régimen MoE**: a 16 GB el mmap clásico de Qwen3-30B-A3B Q4 (18.6 GB) es marginal y vive del page cache; el streaming explícito lo baja a ~2.2 GB residentes. Y al revés: DeepSeek-V2-Lite streameado en BF16 son ~8.2 GB — cabe sin cuantizar. Ambas direcciones son features del runtime.

## 13. Almacenamiento (≤ 1 TB)

Todos los candidatos A y B caben holgados: el más pesado, Qwen3-Coder-480B en BF16 (960 GB), roza el límite — en INT8 (480 GB) o Q4 (~240 GB) deja margen. DeepSeek-V3 Q4_K_M: 378 GB. Llama-4-Scout Q4: ~66 GB. Un solo modelo queda fuera por disco: Mistral-Large-3 en BF16 (1350 GB).

## 14. Qué quedó descartado y por qué (conclusiones de investigación)

- **No existe Llama 5** (rumores retractados) ni Phi-5 ni Qwen4; R2 no existe.
- **Qwen3.5-0.8B/2B** son híbridos DeltaNet + vision: fuera del PoC, dentro del roadmap futuro.
- **Gemma-3n, Phi-4-mini-flash**: arquitecturas exóticas sin retorno para un motor from-scratch en v1.
- **EXAONE**: licencia no-comercial — bloqueada como producto, útil solo como benchmark privado.
- **Qwen2-MoE-A2.7B**: delisted. Sin checkpoint no hay validación posible.
- La familia **Qwen es la espina dorsal** porque cubre los cinco stages con UN tokenizador y UN adaptador base: 0.6B denso → 32B denso streameado → 30B-A3B MoE.
