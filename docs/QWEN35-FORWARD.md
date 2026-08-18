# Forward qwen3_5 (Ornith 9B) — referencia de implementación (Fase 6)

Fuentes primarias (leídas y verificadas contra el volcado del oráculo):

- `llama.cpp/src/models/qwen35.cpp` — grafo por capa;
- `llama.cpp/src/models/delta-net-base.cpp` — delta net (AR / chunked / fused);
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp:10613` — kernel FUSED `GATED_DELTA_NET`
  (el que usó el oráculo: el volcado contiene `__fgdn_ch__-{il}`, 72 hits,
  y NO contiene los cbs no-fused `q_in`/`dnet_add_ch_lhs`);
- `benchmarks/reference/ornith-evalcallback-prompt5.txt` — volcado por capa
  del prefill de 5 tokens (targets de validación);
- `benchmarks/reference/ornith-tensor-table.txt` — nombres/dtypes GGUF.

## 1. Config (ya validada en `unltd-architectures::qwen35`)

n_layer=32, n_embd=4096, n_ff=12288, n_head=16, n_head_kv=4,
head_dim_k=head_dim_v=256, n_rot=64, freq_base=1e7,
rms_eps=1e-6, conv_kernel=4, state_size=128, group_count=16,
time_step_rank=32, d_inner=4096, full_attn_interval=4,
rope_sections=[11,11,10,0], context=262144.

Derivadas: head_dim_linear=128, n_v_heads=32, d_conv=8192,
kq_scale=1/√256=0.0625, theta_scale=1e7^(−2/64) (f32 en ggml, f64 en el motor).

Capas recurrentes (GDN): todas menos (il+1)%4==0 → il ∈ {0,1,2,4,5,6,…}.
Capas full-attention: il ∈ {3,7,11,15,19,23,27,31}.

## 2. Pesos por capa (nombres GGUF exactos, de ornith-tensor-table.txt)

Recurrente:
| tensor | forma | dtype |
|---|---|---|
| blk.{il}.attn_norm.weight | 4096 | F32 |
| blk.{il}.attn_qkv.weight | 4096×8192 | Q6_K |
| blk.{il}.attn_gate.weight | 4096×4096 | Q4_K |
| blk.{il}.ssm_conv1d.weight | 4×8192 | F32 |
| blk.{il}.ssm_dt.bias | 32 | F32 |
| blk.{il}.ssm_a | 32 | F32 (SIN sufijo .weight) |
| blk.{il}.ssm_beta.weight | 4096×32 | Q4_K |
| blk.{il}.ssm_alpha.weight | 4096×32 | Q4_K |
| blk.{il}.ssm_norm.weight | 128 | F32 |
| blk.{il}.ssm_out.weight | 4096×4096 | Q4_K |
| blk.{il}.post_attention_norm.weight | 4096 | F32 |
| blk.{il}.ffn_gate.weight | 4096×12288 | Q4_K |
| blk.{il}.ffn_up.weight | 4096×12288 | Q4_K |
| blk.{il}.ffn_down.weight | 12288×4096 | Q6_K |

Atención: attn_norm / attn_q (4096×8192, Q4_K) / attn_q_norm (256 F32) /
attn_k (4096×1024, Q4_K) / attn_k_norm (256 F32) / attn_v (4096×1024,
Q4_K o Q6_K — MIXTO por capa) / attn_output (4096×4096 Q4_K) /
post_attention_norm / FFN igual (ffn_down Q4_K o Q6_K, mixto).

Global: token_embd.weight 4096×248320 Q4_K; output_norm.weight 4096 F32;
output.weight 4096×248320 Q6_K. No hay pesos atados ni nextn/MTP.

## 3. Capa recurrente (GatedDeltaNet) — algoritmo EXACTO

Por token (x = fila de n_embd tras attn_norm):

1. `qkv = gemv(attn_qkv, x)` → 8192; `z = gemv(attn_gate, x)` → 4096.
2. `beta = sigmoid(gemv(ssm_beta, x))` → 32 (escalar por v-head).
3. `alpha = gemv(ssm_alpha, x)` → 32;
   `gate = softplus(alpha + ssm_dt.bias) * ssm_a` → 32.
   (ssm_a = −exp(−A_log): gate ≤ 0, exp(gate) ≤ 1 = decaimiento.)
4. Convolución depthwise temporal sobre qkv (estado rodante de 3 filas):
   ventana `[estado(3); actual(1)]` × 8192 canales, kernel 4×8192
   (`ssm_conv`, orden f32 secuencial = ggml, bit-idéntico) → `conv_raw` 8192;
   `mix = silu(conv_raw)`.
   Split de `mix`: `q_conv = mix[0..2048]` (16 k-heads × 128),
   `k_conv = mix[2048..4096]`, `v_conv = mix[4096..8192]` (32 v-heads × 128).
   Estado rodante tras el paso: última fila de qkv empujada al anillo.
5. `q_conv = l2_norm_128(q_conv)`, `k_conv = l2_norm_128(k_conv)`
   (norma por head de 128, eps = 1e-6, sin peso — `l2_norm_rows` bit-idéntico
   a ggml). `v_conv` NO se normaliza.
6. Núcleo GDN (replica de `ggml_compute_forward_gated_delta_net_one_chunk`,
   el kernel FUSED que usó el oráculo) — por v-head `h` (32), estado
   S[128][128] propio:
   - **mapeo de heads: q/k head = h % 16** (repetición periódica de ggml,
     NO pares consecutivos; en atención GQA sí es h/4 — no confundir);
   - `S *= expf(gate[h])` — escalar f32 exp, toda la matriz;
   - por columna j: `d[j] = (v[j] − dot(S[:,j], k)) * beta[h]`
     (dot sobre 128; ggml usa vec_dot SIMD f32, el motor f64 pairwise —
     desviación documentada ~1e-7);
   - `S[:,j] += d[j] * k` (mul_add — ggml usa FMA);
   - `out[j] = dot(S_nueva[:,j], q) * scale` con scale = 1/√128 (f32) —
     **la escala va sobre la SALIDA** (el camino no-fused escala q primero;
     el oráculo es el fused: escala en salida).
7. `out` [4096] (head-major: head h en [128h, 128h+128)).
   Norma con compuerta: `y = rmsnorm_128(out, ssm_norm.weight) * silu(z)`
   (rmsnorm por head de 128 CON peso; z es el gate de 4096, elementwise).
8. `attn_out = gemv(ssm_out, y)` → 4096. Residual: `r = x_in + attn_out`.
9. `p = rmsnorm(r, post_attention_norm)`; FFN SwiGLU paralelo:
   `f = gemv(ffn_down, silu(gemv(ffn_gate, p)) * gemv(ffn_up, p))`;
   `l_out = r + f`.

## 4. Capa full-attention

1. `attn_norm` (rmsnorm 4096 con peso).
2. `Qfull = gemv(attn_q, x)` → 8192: por q-head h (16):
   `q = Qfull[512h .. 512h+256]`, `gate = Qfull[512h+256 .. 512h+512]`
   (primera mitad Q, segunda G, por head).
3. `q = rmsnorm_256(q, attn_q_norm)`.
4. `K = gemv(attn_k, x)` → 1024 (4 kv-heads × 256); `rmsnorm_256(attn_k_norm)`.
   `V = gemv(attn_v, x)` → 1024, sin norma.
5. IMROPE sobre q y k (por head, primeros 64 dims, pares (i, i+32)):
   el par `ic` rota SOLO si `ic % 3 == 0` (secciones t/h/w con pos_h=pos_w=0;
   para ornith sections=[11,11,10,0] las cotas 3·sections[i] nunca muerden),
   ángulo `pos · theta_scale^ic`, trigonometría f64 (`rope_apply_imrope`).
   msale = 1.0 (sin rope.scaling.type ni attn_factor en el GGUF — verificado
   en llama-hparams.h:124 y llama-context.cpp:181).
6. Atención causal: por q-head h → kv-head = h/4 (GQA consecutiva):
   `scores[p] = dot(q_h, k_kv[p])` para p ≤ t; softmax con scale kq_scale
   (mi kernel: `((v−max)·scale).exp()` / Σ — ggml soft_max_ext igual);
   `attn = Σ_p soft[p] · V_kv[p]` (256-dim, f64 pairwise).
7. `attn_gated = attn * sigmoid(gate)`; `attn_out = gemv(attn_output, attn_gated)`.
8. Residual + post_norm + FFN: igual que la recurrente.

## 5. Cola

`result_norm = rmsnorm(l_out, output_norm.weight)`; logits = output.weight ·
result_norm (Fase 5 ya validada). SSM y conv state iniciales = ceros.

## 6. Targets de validación por capa (volcado prompt5)

Recurrente: attn_norm, norm, linear_attn_qkv_mixed, z, beta, beta_sigmoid,
alpha, a_softplus, gate, conv_states(_reshaped), qkv_mixed_transposed,
conv_input, conv_output_raw, conv_output_silu, q_conv, k_conv,
q_conv_predelta, k_conv_predelta, v_conv_predelta, __fgdn_ch__,
attn_output, new_state, final_output, linear_attn_out, attn_residual,
attn_post_norm, ffn_gate, ffn_up, ffn_swiglu, ffn_out, post_ffn, l_out.
(NOTA: NO existe `v_conv-{il}` en el volcado.)

Atención: Qcur_full, Qcur_reshaped, Qcur_normed, Kcur, Kcur_normed, Vcur,
gate_reshaped, Qcur, Kcur, __fattn__, attn_pregate, gate_sigmoid,
attn_gated, attn_output, attn_residual, attn_post_norm, ffn_*, l_out.

Gate por capa: `l_out` y `attn_residual`/`linear_attn_out` ≤ 1e-4 abs (el
oráculo del volcado es %12.4f → piso 5e-5 de redondeo de impresión + orden
de reducción); el gate FINAL es result_norm/result_output contra los bins
prec9 y los tokens greedy == oráculo.

## 7. Desviaciones numéricas documentadas del motor

- dots de 128/256 en f64 pairwise (ggml: SIMD f32 con FMA) → ~1e-7 por dot;
- trigonometría f64 con ángulo f64 (ggml: cos/sin f32 con theta stepping f32);
- softplus en f64 (ggml: log1pf(expf(x)) f32);
- rmsnorm pairwise-f64 (validado: 3.8e-6 en 4096-dims);
- l2_norm y ssm_conv bit-idénticos (excepciones ya implementadas y testeadas).
