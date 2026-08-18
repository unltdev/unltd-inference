# Roadmap de unltd-inference

Los stages siguen la progresión pedida: sintético → 0.5–1.5B → 3–4B → 7–8B → MoE → modelo claramente mayor que la RAM. Cada stage tiene criterios de salida que se VERIFICAN con tests y mediciones — un stage no se cierra por "funciona", se cierra por "la escalera de tests pasa y la curva de memoria se midió".

## 1. Principios del roadmap

1. **El streaming es el destino, pero no el punto de partida.** Un engine streameado sobre un engine incorrecto es corrupción rápida. Los stages 0–3 corren con todo residente y validan contra oráculos; el streaming entra cuando el modelo lo exige (Stage 3+, Qwen3-32B) y la maquinaria de presupuestos ya existe desde el Stage 0.
2. **Cada stage corre en la escalera de memoria.** Desde el Stage 1, cada modelo se corre en 2+ presupuestos (p. ej. total y total/4) y la salida debe ser idéntica token a token. El determinismo entre presupuestos es la firma del proyecto — no es un bonus, es la aserción que valida todo lo demás (ver `benchmarks/README.md`).
3. **La familia Qwen es la espina dorsal** (un tokenizador, un adaptador base, un MoE estrella); los modelos de otras familias entran como puntos de datos de generalidad, no como desvíos.
4. **Medir antes que optimizar.** Cada stage produce: s/token, MB leídos/token, hit rate verdadero, PEAK RSS. Sin esos números, "mejoró" es una opinión.
5. **Negarse es una feature.** El refusal (config incompleta, plan que no cabe) se construye en el Stage 0 y se testea en todos los stages — nunca se delega a "ya lo agregamos".

## 2. Los stages

### Stage 0 — Sintético, sin modelo real

**Objetivo**: el motor existe y es correcto a nivel de kernels y contratos.

- GGUF reader (header, metadata, índice de tensores) con refusal ante archivo truncado / tensor ausente.
- Adapter `qwen3` mínimo (config sintética) emitiendo la IR completa.
- Kernels scalar de referencia: RMSNorm, matmul f32, RoPE, SwiGLU, softmax — cada uno con su orden de reducción documentado y tests de fixtures adversarios (`tools/make_tiny.py` genera el checkpoint de 1 capa).
- `MemoryManager.plan()`: suma el plan entero, se niega con números, y funciona con `--no-alloc`.
- Tokenizer BPE byte-level sobre un `tokenizer.json` sintético (vocab de juguete).
- `unltd-cli` corre end-to-end sobre el modelo sintético: 1 capa, greedy, salida de texto.
- **Fixture opcional ultra-rápido**: SmolLM2-135M (0.27 GB, `model_type: llama` puro) como smoke test real de 30 segundos — NO es el PoC, es el canario.

**Sale cuando**: kernels scalar ≡ referencia naive bit a bit; refusal de config lista TODAS las claves ausentes; `cargo test --workspace` verde; el plan de memoria se niega correctamente en 3 escenarios hostiles.

### Stage 1 — PoC: Qwen3-0.6B (0.76B)

**Objetivo**: primer modelo real, todo el pipeline validado contra referencias externas.

- Adapter qwen3 completo (validación de config real con refusal).
- Tokenizador real: BPE tiktoken-style, vocab 151,936, tokens especiales.
- KV cache con GQA (16/8), prefill + decode incremental, greedy.
- Validación numérica en DOS frentes: (a) logits vs torch (oráculo en Python, max diff < 1e-5); (b) salida vs llama.cpp con el MISMO GGUF (Q4_K_M 0.48 GB) — los tokens deben coincidir.
- Primer arnés de la escalera de memoria: corrida con presupuesto total y total/4, salida idéntica.

**Sale cuando**: salida token a token igual a llama.cpp sobre el GGUF oficial; el test de equivalencia incremental ≡ full-recompute pasa; la escalera de 2 presupuestos da salida idéntica.

### Stage 2 — 3–4B: generalidad del adaptador

**Objetivo**: probar que la IR no es "un adapter disfrazado de qwen".

- **Qwen3-4B** (4.03B, Q4 2.0 GB): escala del mismo adaptador.
- **SmolLM2-1.7B** (vocab 49,152, MHA, θ=130k): SEGUNDO adapter (llama) — la IR demuestra que el motor no conoce arquitecturas.
- **Llama-3.2-1B** (tiktoken BPE 128k, tied): compatibilidad con el estándar del ecosistema (licencia no-OSI → solo tests, no producto).
- AVX2 en los kernels calientes (matmul, RMSNorm) con tests de bit-identidad scalar ≡ AVX2.
- rayon sobre filas de salida independientes, con el contrato de determinismo de thread count fijo.

**Sale cuando**: los 3 modelos decodifican con salida validada; scalar ≡ AVX2 byte a byte; primer número de rendimiento publicado con protocolo de medición (mediana de 3, machine.txt).

### Stage 3 — 7–8B y el primer "no cabe": streaming denso

**Objetivo**: el presupuesto deja de ser una comodidad y pasa a ser el mecanismo central.

- **Qwen3-8B** (Q4_K_M 5.0 GB): residente cómodo — el modelo "serio" de referencia. **Phi-4** (14.7B, MIT, Q4 8.9 GB) como segundo denso de otra familia.
- **Qwen3-32B streameado**: Q4_K_M 19.3 GB NO entra en 16 GB; streaming del tronco denso lo baja a ~8.2 GB residentes. Este es el primer modelo que convierte `model_size > available_RAM` en feature: pin prefix + ring de 2 slots + reader thread.
- `DiskReader` buffered completo + fadvise; camino Direct I/O (O_DIRECT + hugepages) en Linux/WSL2.
- PEAK RSS como veredicto (getrusage) vs plan como pronóstico — el banner de memoria de k3_run.c.
- Estado de decode serializable (save/load de KV + posición + fingerprint de config de 12 campos).

**Sale cuando**: Qwen3-32B corre streameado con salida idéntica a la corrida residente (mismo modelo, dos presupuestos); la curva s/token vs presupuesto se publica (8/12/16 GB); el 2-slot ring pasa el test de corrupción silenciosa del reader thread.

### Stage 4 — MoE: cache de expertos, trace, replay

**Objetivo**: la maquinaria que hereda de K3 donde K3 brilla.

- **OLMoE-1B-7B primero** (6.9B/1.1B activos, 64e top-8, Apache, GGUF oficial): chico, con referencia llama.cpp, y el top-8 exige cache de verdad.
- **Qwen3-30B-A3B después** (128e top-8, sin shared, sigmoid+renorm): el objetivo real. Streameado ~2.2 GB; en disco Q4_K_M 18.6 GB.
- `ExpertCache` completo: slots EMPTY/INFLIGHT, prefetch batch en 3 fases (reserva serial + lecturas paralelas ordenadas por offset + publicar solo lo que llegó), hit rate verdadero `(hits − prefetch_reads) / requests`.
- Trace de accesos (8 B/request) + `tools/sim_cache.py` para replay offline: LRU vs Belady vs pinned_lru, techo de compulsory misses.
- MoE prefill: dedup de expertos por chunk (la optimización de K3 que evita re-leer el mismo experto en el mismo chunk de prefill).
- Router con refusal: un experto caído = corrida inválida con exit code propio, nunca drop silencioso.

**Sale cuando**: Qwen3-30B-A3B decodifica con salida validada contra llama.cpp; el replay offline reproduce el hit rate del runtime; la escalera de presupuestos (2/4/8 GB de cache) muestra la curva de degradación con salida idéntica; I/O share por token medido.

### Stage 5 — Modelo claramente > RAM: streaming extremo

**Objetivo**: la validación definitiva del principio fundacional.

- **Fase 5a**: Qwen3-30B-A3B bajo presupuesto artificial (p. ej. 4 GB totales) — forzar el régimen de K3 (tráfico de disco por token, cache en undershoot severo) con un modelo ya validado.
- **Fase 5b**: los modelos grandes reales de la tabla `MODEL-CANDIDATES.md` §11: **DeepSeek-V3/R1 Q4** (671B, 378 GB en disco, streaming ~11 GB en INT3 — cabe en el TB de almacenamiento), Qwen2-57B-A14B, Mixtral-8x22B, Llama-4-Scout, Qwen3-Coder-480B-A35B (INT8 480 GB).
- Prefetch con solapamiento de cómputo (el 1.70× de K3), Direct I/O como camino primario, trace/replay para elegir políticas de pinning (Quantile Balancing), medición del piso de ruido del sistema.
- Especulación (n-gram o draft híbrido) si los números lo piden — no antes.

**Sale cuando**: un modelo de 100B+ decodifica en 16 GB con la curva completa publicada (s/token, MB/token, hit rate, PEAK RSS, piso de ruido) y salida determinista entre presupuestos.

## 3. Por qué Qwen3-0.6B es el mejor primer objetivo (PoC)

Los nueve argumentos, en orden de peso:

1. **Arquitectura vanilla verificada, no folklore.** `use_sliding_window: false`, sin attention bias, sin QK-norm, GQA 16/8, θ=1e6, silu, RoPE completo — verificado contra el config.json real. Compárese con los competidores del mismo tamaño: SmolLM3-3B (capas NoPE), Phi-4-mini (partial rotary 0.75), Gemma-3-1B (GELU + ventana 5:1 + RoPE dual + SentencePiece), Qwen3.5-0.8B (DeltaNet híbrido). El PoC correcto es el que tiene CERO quirks de arquitectura: todo lo que falle será culpa del motor, no del modelo.
2. **Un GGUF oficial con ecosistema maduro = oráculo numérico.** `Qwen/Qwen3-0.6B-GGUF` publica la gama completa (Q4_K_M 0.48 GB). La salida de unltd debe coincidir token a token con llama.cpp sobre el MISMO archivo — un oráculo independiente, gratis, que ningún modelo sin GGUF puede dar (esa es la trampa de MiniCPM-MoE-8x2B, descartado como primer MoE por eso).
3. **El tokenizador se amortiza sobre toda la escalera.** BPE tiktoken-style de vocab 151,936, compartido por Qwen2.5-0.5B, Qwen3-0.6B→32B, Qwen3-30B-A3B. El trabajo de tokenizador del Stage 1 se reusa literalmente hasta el Stage 5.
4. **Camino directo al MoE estrella sin cambiar de familia.** Qwen3-30B-A3B difiere del 0.6B en UN enum de la IR (`FfnKind::SwiGlu` → `FfnKind::Moe`) más el router sigmoid+renorm. Aislar "qué aporta el MoE" requiere comparar denso y MoE de la MISMA familia — Qwen es la única familia que ofrece ese par en todos los tamaños con la misma licencia.
5. **Apache-2.0.** A diferencia de Llama-3.2 (Community License, gated), Gemma 2/3 (Terms), DeepSeek-V2 (licencia propia): sin restricciones para el proyecto ni para terceros.
6. **Utilidad real.** La variante Instruct tiene thinking mode y es un asistente usable — no es un toy de laboratorio. El PoC produce un binario que ya sirve para algo.
7. **Ciclo de desarrollo en segundos.** 0.76B = corrida completa de prefill + decode en segundos, oráculo torch incluido. La escalera de tests corre en tiempo de CI. Un PoC de 8B arranca el proyecto con 10× más fricción en cada iteración.
8. **Cabe entero en RAM — y está bien que así sea.** El Stage 1 valida el motor contra oráculos con todo residente; el streaming llega en el Stage 3 con Qwen3-32B, cuando el motor ya está demostrado. No se puede testear un engine streameado contra un engine que todavía no es correcto.
9. **Las alternativas consideradas pierden por una razón concreta, no por gusto.** SmolLM2-135M: aún más simple, pero sin el camino familiar al MoE y sin el mismo ecosistema de GGUFs oficiales — queda como canario del Stage 0. Qwen2.5-0.5B: misma familia pero calidad inferior y SWA residual. TinyLlama: obsoleto y SentencePiece. Llama-3.2-1B: licencia y gating. OLMo-2-1B: QK-norm, un twist innecesario para el primer modelo.

## 4. Qué NO está en este roadmap (explícito)

- **GPU**: no está planificado, igual que en el roadmap de K3. El valor del proyecto es CPU+disco.
- **Diálogo de precisión de tronco**: K3 lo explícitamente no planea; acá tampoco.
- **Serving / API HTTP, multimodal, entrenamiento**: fuera de alcance.
- **Nuevos formatos de cuantización propios**: se consumen los quants GGUF existentes; no se inventan.
- **Reimplementar llama.cpp**: si el objetivo fuera compatibilidad total, llama.cpp ya existe (ver `MODEL-SUPPORT.md` §5). El objetivo es la investigación de políticas de memoria — el soporte de arquitecturas es el medio.

## 5. Riesgos principales y cómo se mitigan

| Riesgo | Mitigación |
|---|---|
| MLA de DeepSeek subestima su complejidad (decompresión de latente, YaRN) | Postergada al Stage 5; Qwen3-MoE cubre el Stage 4 sin MLA. El `AttnKind::MlaDeepSeek` de la IR documenta los campos verificados |
| Qwen3.5 cambia el tokenizador (vocab 248k) | Aislado: solo afecta los modelos 3.5+, fuera de los stages 1–4 |
| El camino Direct I/O es frágil en Windows nativo | WSL2 como entorno de medición (cgroups + O_DIRECT); Windows nativo con mmap como camino práctico (evidencia en `MEMORY-DESIGN.md` §6) |
| El ruido de medición domina los deltas (33% en K3) | Protocolo de benchmarks obligatorio desde el Stage 2 (`benchmarks/README.md`) |
| Streaming correcto pero lento en exceso | El techo es conocido: llama.cpp corre Qwen3-30B-A3B a 42 t/s en EPYC / >10 t/s en PC de 16 GB, y DeepSeek-V3 Q4 en-RAM a 8.45 t/s. Si unltd no se acerca a esos números en régimen residente, hay un bug de rendimiento antes que una limitación de diseño |

## 6. Orden de implementación sugerido (dependencias de crates)

```
Stage 0:  unltd-core (refusal) → unltd-tensor (kernels scalar + tests)
          → unltd-model-loader (GGUF) → unltd-architectures (IR + adapter sintético)
          → unltd-tokenizer (BPE) → unltd-memory (plan, sin streaming)
          → unltd-generation (forward denso) → unltd-cli
Stage 3:  unltd-memory (ring + DiskReader direct)      [streaming denso]
Stage 4:  unltd-memory (ExpertCache + trace)           [MoE]
          tools/sim_cache.py + tools/budget.py
Stage 5:  prefetch solapado, políticas de pinning, especulación
```

El workspace completo (8 crates + tests/benchmarks/tools) ya existe como esqueleto compilable con las firmas de arriba — el primer commit del Stage 0 es `cargo check --workspace` en verde sobre una máquina con Rust 1.80+.
