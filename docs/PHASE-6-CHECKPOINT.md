# Fase 6 — Checkpoint: validación de forward qwen3_5 contra oráculo

Fecha: 2026-08-18. Modelo: `ornith-1.0-9b-Q4_K_M.gguf` (Ornith-1.0-9B). Prompt: 5 tokens `[760, 6511, 314, 9338, 369]` ("The capital of France is").

## Estado

- **Puerta token-level: PASS.** El primer token greedy del motor (`argmax = 11751`, " Paris") coincide con el oráculo (dump prec9 argmax = 11751, y llama-server Prism generó 11751 como primer token).
- **Mismatch `ffn_gate-0[token 1] col 8322`: RESUELTO CON CAUSA DEFINITIVA.** El valor del oráculo (0.18729219) se reproduce **bit a bit** con una transcripción exacta del gemm AVX2 que realmente corrió en el oráculo (`ggml_gemm_q4_K_8x8_q8_K`). La diferencia contra el motor es una **divergencia legítima de kernels SIMD** (distinta estructura de redondeo f32), no un bug de matemática.
- La puerta layer-level bit-exacta exigiría portar los kernels AVX2/repacked al motor (trabajo acotado y diferido); la puerta viable hoy es token-level (verde).

## Causa raíz definitiva (cerrada en esta sesión)

Corrige la conclusión previa de ed73c65 ("el oráculo corrió scalar"): **el oráculo es un build AVX2**. Header del dump prec9: `CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1`.

1. **Dispatch** (`ggml-cpu/repack.cpp:4204`, `forward_mul_mat_one_chunk`): `if (nrows > 3) gemm(..., nrows - nrows%4, ...); for (iter = nrows - nrows%4; iter < nrows; iter++) gemv(...)`. Con 5 tokens: **tokens 0–3 → GEMM, token 4 → GEMV**.
2. **Kernel que corrió** para los mul_mats Q4_K: `ggml_gemm_q4_K_8x8_q8_K` (`ggml-cpu/arch/x86/repack.cpp:2042`), sobre pesos repacked `block_q4_Kx8` (`make_block_q4_Kx8`) y activación `ggml_quantize_mat_q8_K_4x8` (AVX2, misma línea de archivo:290).
3. **Contrato aritmético por elemento de salida** (m = fila src1, c = col src0):
   - `iacc = Σ_{e∈64sb..64sb+63} v(e,c)·s_{e/32}(c)·q8(e,m)` — entero i32 exacto (≤ 2^24 → `cvtepi32_ps` exacto);
   - `acc = fma32(iacc, f32mul(d_col[c], d_row[m]), acc)` — 2 redondeos por par de sub-bloques (fma + mul);
   - `minacc = bs_{2sb}(m)·m_{2sb}(c) + bs_{2sb+1}(m)·m_{2sb+1}(c)` — i32 exacto (el layout de bsums quedó **confirmado empíricamente**: la hipótesis semántica por-fila reproduce el dump bit a bit);
   - `acc_min = fma32(minacc, f32mul(dmin_col[c], d_row[m]), acc_min)`;
   - final: `out = f32sub(acc, acc_min)`.
   El gemm generic (`repack.cpp:1905`, que replica el motor vía `ggml_vec_dot_q4_K_q8_K_generic`) acumula con **3 redondeos por término y por k** — estructura de redondeo distinta → 1–62 ulps en `z-0` con entrada bit-exacta.
4. **Mecanismo de amplificación** (documentado en ed73c65, sigue válido): los 1–62 ulps de `z-0` se propagan a `attn_post_norm-0` (~1e-7) y cruzan límites de redondeo de la cuantización Q8_K → flips de qs → desplazamiento W·d por fila ~1e-4 → pin 2.98e-4 en col 8322. La cuantización en sí es idéntica en valor (`quantize_q8_k` ≡ `quantize_mat_q8_K_4x8`); los flips son consecuencia de la diferencia de kernel aguas arriba, no de la cuantización.
5. Los nodos F32 pre-FFN (convs linear_attn, gates) muestran pins 1e-7..4e-6 con entrada bit-exacta: mismas familias de kernels AVX2/repacked (reducciones y mul_mats F32) vs réplicas escalares — misma causa, no re-investigado (fuera del alcance del objetivo).

## Validación bit a bit (transcripción del gemm AVX2)

Script: `benchmarks/reference/gemm-avx2-validate.py` (importa el parseador GGUF/dump de diag; `--parse` cachea el dump de 11.9 GB a npz; fma32 exacto con camino rápido f64 + chequeo de doble redondeo via `nextafter` + fallback entero exacto — 800k casos unit-testeados contra referencias exactas).

- **`z-0`: 0/16384 elementos difieren** (4 tokens × 4096 cols, transcripción vectorizada) — **BIT-EXACTO** contra el dump.
- **`ffn_gate-0` cols 8320..8327 × tokens 0..3: 32/32 BIT-EXACTO** (camino entero exacto elemento a elemento). Col 8322 token 1 = **0.18729219** (oráculo) ✓.
- **Contraste (prueba de identidad del kernel):** la misma col con la aritmética del gemm generic (3 redondeos) da 0.18729217 ≠ 0.18729219 (1 ulp) → confirma que el oráculo corrió el kernel AVX2, no el generic.
- Unit tests de la aritmética exacta (int2_to_f32 / fma32_int / fma32_vec): 0 fallos, incl. casos borde (subnormal mínimo, f32 max, overflow→inf, ties-to-even).

## Q4_K

### Layout natural (GGUF, el que consume el motor)

`block_q4_K` = 144 B: `d` f16 (2 B) + `dmin` f16 (2 B) + `scales` u8[12] + `qs` u8[128].

- **Escalas/mins:** sub-bloques t = 0..3 usan `s_t = scales[t] & 63`, `m_t = scales[t+6] & 63`; sub-bloques t = 4..7 reutilizan los 2 bits altos de los slots 0..5: `s_t = ((scales[t-4] & 192) >> 2) | (scales[t] & 15)`, `m_t = ((scales[t+2] & 192) >> 2) | (scales[t+6] & 15)`.
- **Orden de nibbles:** cada sub-bloque t tiene 32 quants en `qs[16t : 16t+16]`; el quant i está en el byte `qs[16t + i/2]`, nibble bajo si i es par, alto si es impar.
- **Dequant:** `x = d * (q * s_t - m_t)` (dmin = d * min global, no se usa en el dot).

### Layout REPACK x8 (el que consume el oráculo)

`block_q4_Kx8` = 1152 B: `d` f16[8] + `dmin` f16[8] + `scales` u8[96] + `qs` u8[1024] (`repack.h:43`).

- **qs (interleave):** `out.qs[i*8 + r] = in[i % 8].qs[(i/8)*8 + r]`, i = 0..127, r = 0..7 (`make_block_q4_Kx8`, `ggml-cpu/repack.cpp:2836`) — los 128 bytes de qs de cada una de las 8 filas se intercalan byte a byte.
- **scales:** 96 B = 8 slots × 12 B. Primeros 48 B para sub-bloques 0-3 (`out.scales[i*12] = (s[0]&63)+((s[4]&48)<<2)` … `out.scales[i*12+8] = (s[4]&15)+((m[4]&15)<<4)`); segundos 48 B para sub-bloques 4-7 con `s[j] = ((in[j].scales[i]&192)>>2) | (in[j].scales[i+8]&15)`.
- **Acumulación (gemm 8x8 AVX2, la que corrió en el oráculo):** por bloque l (256 columnas) × sb (4 pares de sub-bloques), `iacc` entero (i32) sobre `v0·a0·s0 + v1·a1·s1`, epílogo float por sb con `fma(iacc, d_col·d_row)` y `fma(minacc, dmin_col·d_row)`, resta final `acc − acc_min`. Ver contrato aritmético en "Causa raíz definitiva".
- **Cuantización de la activación del oráculo:** `quantize_mat_q8_K_4x8` (AVX2; 4 filas a la vez, escala conjunta por columna, layout x4 interleaved), no `quantize_row_q8_K`. Idéntica en valor a la del motor.

### Diferencias vs nuestro motor (corregida)

| | Motor unltd | Oráculo llama.cpp |
|---|---|---|
| Layout pesos | Q4_K natural directo desde GGUF (mmap) | REPACK `block_q4_Kx8` |
| Gemm Q4_K | dot por fila (réplica del generic `vec_dot_q4_K_q8_K`) | gemm 8x8 **AVX2/FMA** repacked (tokens 0–3) + gemv repacked (token 4) |
| Cuantización activación | `quantize_q8_k` por columna (≡ en valor) | `quantize_mat_q8_K_4x8` (x4) |
| Normas/act. no lineales | réplicas escalares exactas de ggml (commit ed73c65) | ggml AVX2 (reducciones vectorizadas) |

La parte entera del dot es exacta; la divergencia es la **estructura de redondeo f32 del kernel AVX2** (fma en el epílogo del gemm, orden de acumulación de las reducciones) vs la del generic escalar.

## Errores (rama "diferencia legítima" de la directiva)

- **`z-0`:** 3708/4096 elementos difieren (token 1), 1–62 ulps; pin en valor = 3.83e-6.
- **`ffn_gate-0[token 1] col 8322`:** motor 0.187590 vs oráculo 0.18729219 → err_abs = 2.98e-4, err_rel = 1.59e-3. (Causa: 2 flips de qs en la cuantización de `attn_post_norm`, documentados en ed73c65.)
- **Logits:** top-1 (11751): motor 16.190655 vs oráculo 16.352005 (Δ = 0.161); result_norm max \|diff\| = 0.400 en [2955] vs prec9.
- **Estabilidad del argmax:** top-1 = 11751 en motor, oráculo prec9, oráculo %.12.4f y llama-server → **argmax estable** a través de todos los impls.
- **Greedy:** primer token motor = 11751 == oráculo ✓ (confirmado en la re-ejecución de esta sesión).

## Validación

### Comandos oráculo

1. **Dumps por nodo/capa (llama.cpp, `D:\AI\runtimes\llama.cpp`, builds con evalcallback `common_debug_cb_eval`):**
   - `benchmarks/reference/ornith-evalcallback-prompt5.txt` — build 0.46.645, formato %.12.4f, REPACK=1, n_threads=6 (volcado por capa).
   - `benchmarks/reference/ornith-prec9-ffn0.txt` — build 0.10.596, formato %.9g, REPACK=1, n_threads=6 (nodos FFN de capa 0).
   - `benchmarks/reference/ornith-final-prec9/` — bins del build %.9g: result_norm / result_output (argmax 11751, top-1 v=16.352005).
   - Ambos builds: **AVX2** (header `AVX2 = 1 | FMA = 1 | REPACK = 1` — corrige ed73c65, que los creía scalar).
2. **Token-level:** llama-server Prism b1-9fcaed7 `/completion` con logprobs:1, n_predict 20, -ngl 0 → `benchmarks/reference/ornith-greedy-tokens.txt` + `ornith-llama-server-completion.json`.

### Comando motor (UNLTD)

```
./target/release/unltd.exe forward-oracle \
  "D:/AI/models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf" \
  --out-dir <tmp>/fwd-final --oracle-dir benchmarks/reference/ornith-final-prec9 \
  --debug-nodes <tmp>/fwd-final/engine-nodes.f32.bin
```

5 tokens en 98.5 s. Prompt: `[760, 6511, 314, 9338, 369]` ("The capital of France is").

### Resultado esperado vs obtenido

- **Token esperado: 11751** (" Paris"). **Token obtenido: 11751** → PASS (gate greedy). Top-5 motor: [(11751, 16.190655), (264, 13.940932), (3750, 13.669998), (198, 13.2436695), (1259, 12.991763)] vs oráculo top-1 (11751, 16.352005).

### Resultados intermedios (puerta layer-level)

- `compare-nodes-oracle.py --all benchmarks/reference/ornith-prec9-ffn0.txt <tmp>/fwd-final/engine-nodes.f32.bin` → **21 PASS, 603 FAIL**. Todos los nodos pre-FFN de capa 0 en verde (attn_norm, q/k/v_conv, linear_attn_*, gate, beta, conv_output_*, attn_residual: pins 1e-7..4e-6); `z-0` PASS pin=3.83e-6 (token 1 col 166: -10.130692 vs -10.130688).
- **Primer mismatch:** `ffn_gate-0` pin=2.98e-4 (token 1 col 8322: motor 0.187590 vs oráculo 0.187292, sum_d 4.02e-2); después ffn_up-0 (2.66e-4), ffn_swiglu-0 (1.64e-4), ffn_out-0 (2.53e-4), l_out-0 (2.53e-4).
- `compare-layers-oracle.py benchmarks/reference/ornith-evalcallback-prompt5.txt <tmp>/fwd-final/engine-layers.f32.bin` → **0/32 capas**. Capa 0: pin=1.97e-4 (l_out-0[token 1]); capa 31: pin=2.79e-1 (attn_residual-31[token 3]), sum 2.73e+1. (tol pin 1e-4, sum 1e-2.)
- `forward-oracle` gate: **FAIL result_norm** max |diff| = 0.400251389 en [2955] > 1e-4 → "RUN INVALID".
- Divergencia entre los tres impls en result_norm `[0]`: volcado 0.9599 (sum -43.646568) vs prec9 1.0531 (sum -47.653786) vs motor 1.0730 (sum -52.313553) — los dos builds del oráculo difieren ~0.1 entre sí (amplificación recurrente de diferencias ~1e-6 a través de 32 capas × 5 tokens).

## Código

- **Nuevo:** `benchmarks/reference/gemm-avx2-validate.py` — transcripción validada del gemm repacked AVX2 (Q4_K 8x8) con fma32 exacto: reproduce `z-0` (0/16384) y `ffn_gate-0` cols 8320–8327 (32/32) bit a bit contra el dump, e incluye el contraste con el gemm generic que prueba la identidad del kernel. Cache del dump regenerable con `--parse` (no commitado, 11.9 GB de origen).
- **Sin cambios en el motor** en esta sesión (investigación + documentación): el motor sigue replicando el dot generic de ggml — matemáticamente equivalente, redondeo distinto.
- **Tests:** `cargo check --workspace` verde; `cargo test --workspace` verde (60 tests: unltd-architectures 4, unltd-generation 4, unltd-model-loader 17, unltd-tensor 35).
- **Doc corregido:** la afirmación "el oráculo corrió scalar" (ed73c65) queda reemplazada por la causa definitiva AVX2 de este documento.
- **Pendiente de decisión (no es código temporal):** si el motor debe portar los kernels AVX2/repacked (gemm q4_K/q6_K 8x8, gemv repacked, 4x8, reducciones F32) para perseguir bit-exactitud layer-level — ver Próximo paso.

## Resultado final

PHASE 6 RESULT: CLOSED — ffn_gate-0[token 1] col 8322 explicado definitivamente: **diferencia legítima de kernel SIMD** (oráculo AVX2: gemm repacked `ggml_gemm_q4_K_8x8_q8_K` con fma; motor: réplica del dot generic), reproducida bit a bit por transcripción exacta (z-0 0/16384, ffn_gate cols 32/32). Primer token greedy 11751 == oráculo (argmax estable en los 4 impls).

## Próximo paso

1. Decisión (diferida, directiva de la sesión): si se exige bit-exactitud layer-level, portar al motor las réplicas escalares de los kernels AVX2/repacked — gemm q4_K/q6_K 8x8 (contrato fma ya validado para q4_K; q6_K es la misma familia), gemv repacked (token 4), `quantize_mat_q8_K_4x8`, y las reducciones F32 de las capas pre-FFN. La transcripción Python validada es la referencia de tests.
2. Fase 7: implementar `tokenize` + `run` greedy sobre la puerta token-level (verde hoy para el token 1).
3. NO abrir Fases 9-10 (memory budget, streaming) hasta cerrar 1-2.
