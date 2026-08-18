# Fases 7-8 — Checkpoint: tokenizer real + generación greedy multi-token

Fecha: 2026-08-18. Modelo: `ornith-1.0-9b-Q4_K_M.gguf` (Ornith-1.0-9B). Prompt: "The capital of France is".

## Tokenizer (Fase 7)

- **Modelo:** `D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf` (no movido, no modificado).
- **Tokenizer detectado:** `tokenizer.ggml.model` = **gpt2** (BPE byte-level); `tokenizer.ggml.pre` = **qwen35**. Vocab 248320 piezas, 247587 merges, histograma de tipos: 248044 NORMAL, 243 UNUSED, 27 CONTROL, 6 USER_DEFINED, 0 BYTE.
- **BOS/EOS:** sin BOS (`bos_token_id` ausente en el GGUF); EOS = **248046** (`<|im_end|>`, CONTROL → suprimido en decode).
- **Implementación:** réplica EXACTA del pipeline de llama.cpp — split del texto CRUDO con los splitters custom (`unicode_regex_split_custom_qwen35/qwen2/gpt2`, transcritos rama a rama de unicode.cpp; el regex crate solo clasifica \p{L}/\p{N}/\p{M}/\s por char), byte-encoding POR PALABRA (tabla gpt2), BPE merge por (rank, left) como el priority queue de llama-vocab.cpp, fallback por char. Negativas donde llama.cpp silencia: pieza o char fuera del vocab, char fuera de la tabla byte-to-unicode, pre-tokenizador desconocido.
- **Comando UNLTD:**
  ```
  ./target/release/unltd.exe tokenize "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" --text "The capital of France is"
  ```
- **Comando oráculo:** llama-server Prism b1-9fcaed7 `/completion` con `logprobs:1, n_predict 20, -ngl 0`, modo raw `-no-cnv` (fuente del fixture: `benchmarks/reference/ornith-greedy-tokens.txt` + `ornith-llama-server-completion.json`; el JSON documenta el comando exacto del server).
- **Validación en vivo del fixture (esta sesión):**
  ```
  "D:\AI\models\Bonsai-demo\bin\cuda\llama-completion.exe" \
    -m "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" \
    -no-cnv -p "The capital of France is" -n 20 --temp 0 -ngl 0 -t 6 -c 64
  ```
  Produce EXACTAMENTE la secuencia del fixture (20/20): "The capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe". Nota: sin `-c 64` el build pide un buffer de KV de 8 GB (contexto default 32768) y falla por memoria; con template (sin `-no-cnv`) emite el thinking mode documentado en el fixture.
- **IDs:** UNLTD `[760, 6511, 314, 9338, 369]` == oráculo `[760, 6511, 314, 9338, 369]`.
- **Resultado:** **PASS** — puerta `UNLTD token IDs == oracle token IDs` (bit a bit, incluido decode ida y vuelta al texto original).

## Generation (Fase 8)

- **Prompt:** "The capital of France is" → 5 tokens `[760, 6511, 314, 9338, 369]` (raw, sin BOS — mismo modo que el oráculo). n = 20 tokens, temperatura 0 (greedy determinista).
- **Comando UNLTD:**
  ```
  ./target/release/unltd.exe run "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" --prompt "The capital of France is" --max-tokens 20 --temperature 0
  ```
- **Secuencia UNLTD (20):** `[11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 3710, 369, 279, 6511, 314, 9338, 30, 198, 760]`
  → " Paris.\nThe capital of France is Paris.\n**What is the capital of France?\nThe**" (en negrita, la divergencia).
- **Secuencia oráculo (20):** `[11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760]`
  → " Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe".
- **Tabla por paso:** 11/11 MATCH hasta el paso 10; **primer mismatch en el token generado Nº 12 (índice 11): UNLTD 3710 (" What") vs oráculo 760 (" The")**. Desde ahí las secuencias divergen (continuaciones distintas plausibles); los pasos 18-19 vuelven a coincidir por casualidad (ambos emitieron `\nThe` desde contextos distintos).
- **Coincidencias:** 13/20 (65%).

### Primer mismatch — causa (medida, no supuesta)

- **Logits de la decisión (UNLTD, paso 11):** top-5 = [(3710, 14.501498), **(760, 14.023525)**, (5236, 13.060937), (3742, 12.718761), (20810, 12.544708)]. El token del oráculo (760, " The") es el **Nº 2 del top-5 del motor, a gap 0.477973 del argmax** (3710, " What").
- **Logprob del oráculo en ese paso:** 760 con ln(softmax) = **-0.964508** (fixture) — cerca del tope de la distribución, consistente con un near-tie genuino del modelo entre "What is the capital…" y "The capital…" (las dos continuaciones son plausibles: el modelo sin template repite la frase o reformula la pregunta).
- **Conclusión:** flip de argmax por la **divergencia numérica documentada en la Fase 6** (oráculo AVX2/repacked vs motor réplicas scalares de ggml; a 5 tokens ya había pin de capa 31 = 2.79e-1 y Δlogits top-1 ≈ 0.16), amplificada por 16 tokens de contexto. Un gap de 0.478 está dentro del orden de esa amplificación. No es bug de tokenizer (Fase 7 bit-exacta) ni de arquitectura: los 11 primeros pasos son idénticos al oráculo y, tras la divergencia, los pasos 18-19 (contextos `\n` → "The") vuelven a coincidir cuando el argmax es decisivo (gap 1.20 en el último paso).
- **Determinismo verificado:** dos corridas completas del motor producen la MISMA secuencia de 20 (tablas idénticas, mismo DIFF en el paso 11; solo cambian los tiempos por cache de OS: prefill 111.5 s frío vs 84.2 s tibio).

## Performance

- **Modo:** CPU, greedy, foreground, sin optimizaciones (directiva: solo medir). Comparación de referencia del oráculo en la misma máquina (llama-completion, CPU -ngl 0): eval 230.65 ms/token (4.34 t/s) — el motor está a ~17× del oráculo, esperable sin optimizaciones y sin kernels SIMD.
- **Prefill:** 5 tokens en 111.5 s (22.30 s/token, cache frío) / 84.2 s (16.85 s/token, cache tibio).
- **TTFT:** ~= prefill (el primer token generado sale del último logit del prefill; sin forward adicional).
- **Decode:** 19 forwards en 284.6 s (frío) / 322.0 s (tibio, con varianza de máquina) → **14.98-16.95 s/token** (coherente con los ~14 s/token marginales medidos en Fase 6).
- **Total:** 396-406 s.
- **RAM aproximada (analítica):** 5.63 GB modelo (mmap paginado por demanda) + 1.64 MB KV cache (ctx 25 × 8 capas full-attn × 4 KV heads × 128 dim × 2 × f32) + activaciones O(n_embd=4096). No hay API de RSS en el proceso (sin dependencias nuevas); el plan impreso es el pronóstico, como en Fase 6.

## KV cache

- **Implementado: sí** — la sesión (`Qwen35Session`) ES el KV cache incremental de las 8 capas full-attn (K/V en f32, pre-dimensionado a ctx = prompt + max_tokens, posición avanzada por `step`), más los estados recurrentes (conv ring + GDN) de las 24 capas GatedDeltaNet. Reusado del forward de Fases 5-6, sin cambios en esta sesión.
- **Full-recompute vs incremental:** NO comparados (directiva: "full recompute aceptable inicialmente; primero greedy multi-token correcto, después evaluar KV cache"). Queda como test de equivalencia pendiente, documentado en ROADMAP.
- **Resultado:** el incremental genera los 11 primeros tokens idénticos al oráculo — la acumulación del estado es la correcta dentro del contrato numérico; no hay indicio de bug de KV en este tramo.

## Tests

- **Cantidad:** 94 tests verdes en el workspace (unltd-tokenizer 27, unltd-generation 11, unltd-model-loader 17, unltd-tensor 35, unltd-architectures 4). 0 fallos. `cargo check --workspace` sin warnings.
- **Nuevos (Fase 7):** encode del prompt del oráculo, roundtrip encode/decode, fallback por char, split de contracciones (case-insensitive qwen35 / case-sensitive gpt2), palabra con espacio líder, pieza de newline, doble espacio en medio (rama `\s+(?!\S)`), run de whitespace final, supresión de CONTROL en decode, USER_DEFINED crudo, id fuera de rango, determinismo de encode, splitters qwen35/qwen2/gpt2 equivalentes en el caso del oráculo, split de símbolos y dígitos, marcas de combinación (`\p{M}` gluing en qwen35, ausente en qwen2), símbolo que absorbe `\r\n` final (qwen35) vs no (gpt2), contracción solo al inicio, pin de los strings de regex del oráculo (verbatim de llama-vocab.cpp), tabla byte-to-unicode, `from_gguf` (fixture en disco): parse, pre "default", negativa ante modelo desconocido / pre desconocido / pieza corrupta (U+FFFD) / token_type ausente. Fix de infra: escritor de fixtures GGUF (elem type de arrays de strings = 8 STRING, no 12 F64 — bug real encontrado y corregido en el test writer) y nombres de archivo temporales únicos por test (antes compartían nombre y pisaban en paralelo).
- **Nuevos (Fase 8):** argmax con empates (primero gana), stop por EOS (EOS incluido, no se re-envía el forward), stop por max_tokens (sin forward del último token), determinismo, max_tokens 0, prompt vacío (negativa), logits visibles tras prefill.

## Resultado

**PHASE 7-8 RESULT: PARTIAL — primer mismatch en el token generado Nº 12 (índice 11): UNLTD 3710 (" What") vs oráculo 760 (" The"); 11/11 correctos hasta ahí (13/20 en total).** Causa medida: flip de argmax por la divergencia numérica F32/AVX2 ya documentada (Fase 6), amplificada por 16 tokens de contexto — el token del oráculo es el Nº 2 del top-5 del motor, a gap 0.477973 del argmax (3710 14.501498 vs 760 14.023525; logprob del oráculo para 760: -0.964508). Fase 7 (tokenizer) en verde completo; Fase 8 genera texto correcto y determinista, con el primer mismatch aislado, medido y reproducible.

## Próximo paso

1. Decisión de la directiva, ahora con datos: el gate ideal "20/20 == oráculo" exige cerrar la divergencia numérica interna (portar los kernels AVX2/repacked al motor — gemm Q4_K/Q6_K 8x8, gemv repacked, `quantize_mat_q8_K_4x8`, reducciones F32 — ya validados por transcripción en Fase 6), o aceptar el contrato numérico actual como gate (argmax estable hasta donde la amplificación lo permite).
2. Test de equivalencia full-recompute vs KV incremental (diferido por directiva).
3. NO abrir Fases 9-10 (memory budget, streaming) sin cerrar 1-2.
