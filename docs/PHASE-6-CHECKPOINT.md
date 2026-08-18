# Fase 6 — Checkpoint: validación de forward qwen3_5 contra oráculo

Fecha: 2026-08-18. Modelo: `ornith-1.0-9b-Q4_K_M.gguf` (Ornith-1.0-9B). Prompt: 5 tokens `[760, 6511, 314, 9338, 369]` ("The capital of France is").

## Estado

- **Puerta token-level: PASS.** El primer token greedy del motor (`argmax = 11751`, " Paris") coincide con el oráculo (dump prec9 argmax = 11751, y llama-server Prism generó 11751 como primer token).
- **Puerta layer-level: BLOCKED.** El primer mismatch reproducible está en `ffn_gate-0[token 1]` col 8322: motor 0.187590 vs oráculo 0.187292, pin = 2.98e-4 > tol 1e-4. Todas las capas divergen crecientemente hasta result_norm (max |diff| = 0.400 en [2955] vs prec9).
- **Causa raíz identificada (commit ed73c65):** el motor replica las kernels escalares de ggml, pero diferencias residuales ~1e-7 en las activaciones pre-cuantización caen en lados opuestos de límites de redondeo Q8_K → "flips" de qs → desplazamiento W·d por fila ~1e-4 (piso real ~1e-4 por flip, no 5e-5). La entrada del gemm (`z-0`) está en verde (pin 3.83e-6), el primer nodo cuantizado (`ffn_gate-0`) es el primero en rojo.
- **Los dos builds del oráculo difieren entre sí** ~0.1 en result_norm (`[0]`: volcado %12.4f = 0.9599 vs prec9 = 1.0531 vs motor = 1.0730): no existe un único target bit-exacto; la única puerta consistente entre builds es token-level.

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
- **Acumulación (gemm 8x8 generic, la que corre el oráculo):** por bloque l (256 columnas) × sb (4 pares de sub-bloques), `iacc` entero (i32) sobre `v0*a0*s0 + v1*a1*s1`, epílogo float por sb con `fma(iacc, col_scale_d8 * row_scale_d4)` y resta del término min (`hsum` de pares de bsums i16 × mins interleaved). El oráculo corrió **scalar** (GGML_SIMD indefinido: build sin /arch, verificado en el vcxproj — commit ed73c65).
- **Cuantización de la activación del oráculo:** `quantize_mat_q8_K_4x8` (4 columnas, escala conjunta por columna, layout x4 interleaved), no `quantize_row_q8_K`.

### Diferencias vs nuestro motor

| | Motor unltd | Oráculo llama.cpp |
|---|---|---|
| Layout pesos | Q4_K natural directo desde GGUF (mmap) | REPACK `block_q4_Kx8` (dll llama.cpp) |
| Gemm | dot por fila (estilo `vec_dot_q4_K_q8_K`), acumulación i32, aritmética escalar ggml-exacta | gemm 8x8 generic (8 filas × 4 cols, `ggml_gemm_q4_K_8x8_q8_K`) |
| Cuantización activación | `quantize_q8_k` por columna | `quantize_mat_q8_K_4x8` (x4) |
| Normas/act. no lineales | réplicas escalares exactas de ggml (commit ed73c65) | ggml escalar |

La parte entera del dot es exacta; la fuente de la divergencia no es el gemm sino los **inputs**: diferencias f32 ~1e-7 aguas arriba que cruzan límites de redondeo de la cuantización Q8_K.

## Validación

### Comandos oráculo

1. **Dumps por nodo/capa (llama.cpp, `D:\AI\runtimes\llama.cpp`, builds con evalcallback `common_debug_cb_eval`):**
   - `benchmarks/reference/ornith-evalcallback-prompt5.txt` — build 0.46.645, formato %.12.4f, REPACK=1, n_threads=6 (volcado por capa).
   - `benchmarks/reference/ornith-prec9-ffn0.txt` — build 0.10.596, formato %.9g, REPACK=1, n_threads=6 (nodos FFN de capa 0).
   - `benchmarks/reference/ornith-final-prec9/` — bins del build %.9g: result_norm / result_output (argmax 11751, top-1 v=16.352005).
   - Ambos builds: **scalar** (sin /arch → GGML_SIMD indefinido, commit ed73c65).
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

### Hipótesis más probable (no re-investigar sin evidencia nueva)

Los 1-2 elementos de activaciones pre-cuantización que difieren ~1e-7 del oráculo (documentado en ed73c65 para attn_post_norm) cruzan un límite de redondeo de la cuantización Q8_K → flip de un qs → desplazamiento W·d por fila ~1e-4. z-0 verde (3.83e-6) pero no bit-exacto es consistente con esto. Evidencia: el primer nodo en rojo es exactamente el primero que depende de una cuantización Q8_K; los nodos f32 puros previos están en verde.

## Código

- **Modificados:** `benchmarks/reference/compare-layers-oracle.py` (fix del header de 12 bytes y stride en bytes del dump del motor — sin esto la puerta corría corrida 3 floats).
- **Eliminados:** 13 scripts de diagnóstico `benchmarks/reference/tmp-*.py` (exploración de layout x8/AVX2/flips, superados por este documento y por ed73c65). Se conservan los comparadores (`compare-layers-oracle.py`, `compare-nodes-oracle.py`, `extract-logits-oracle.py`, `ornith-decode-crosscheck.py`) y todos los fixtures (bins, txt, JSON) como puertas reproducibles.
- **Doc actualizado:** referencia de `crates/unltd-generation/src/qwen35_forward.rs` a `tmp-flips.py` → `docs/PHASE-6-CHECKPOINT.md`.
- **Tests:** sin tests nuevos en este checkpoint; las puertas reproducibles son los scripts `compare-*` + fixtures commitados. `cargo check --workspace` y `cargo test --workspace` verdes.
- **Pendiente de decisión (no es código temporal):** si el motor debe replicar `quantize_mat_q8_K_4x8` en lugar de `quantize_q8_k` por columna — ver Próximo paso.

## Resultado final

PHASE 6 RESULT: BLOCKED — first reproducible mismatch at ffn_gate-0[token 1] col 8322 (pin 2.98e-4 > 1e-4; primer token greedy 11751 == oráculo)

## Próximo paso

1. Comparar bit a bit la cuantización de la activación sobre `z-0` t1: motor `quantize_q8_k` vs oráculo `quantize_mat_q8_K_4x8` (fixture `q8x4.bin` ya existe en `<job-tmp>/dlltest/`); localizar los elementos donde el qs difiere.
2. Si la cuantización coincide, rastrear los elementos divergentes de `z-0` (pin 3.83e-6, 1-2 elementos) hasta la operación f32 previa (attn_post_norm) y replicar su aritmética exacta.
3. Formalizar la puerta de aceptación: con dos builds del oráculo divergiendo ~0.1 entre sí, la igualdad bit-exacta contra ambos es imposible — la puerta viable es token-level (greedy argmax), hoy verde para el token 1.
4. Fase 7: implementar `tokenize` + `run` greedy sobre la puerta token-level.
5. NO abrir Fases 9-10 (memory budget, streaming) hasta cerrar 1-3.
