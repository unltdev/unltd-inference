//! Kernels scalar de referencia. Ver `docs/ARCHITECTURE.md` §3 ("unltd-tensor")
//! y el contrato numérico en `docs/AUDIT.md` §3.3.
//!
//! Reglas que TODO kernel de este módulo cumple:
//! 1. reducciones largas en `f64`; `f32` solo donde el modelo lo define (softmax,
//!    normas); la salida se redondea a f32 al final;
//! 2. orden de reducción FIJO y documentado: partición por pares
//!    `((a0+a1)+(a2+a3))...` (ver [`pairwise_sum_f64`]);
//! 3. `mul_add` explícito donde se quiere FMA — nunca FMA del autovectorizador;
//! 4. el backend scalar es la referencia: AVX2 debe ser bit-idéntico (tests dedicados
//!    cuando exista);
//! 5. el paralelismo (futuro) ocurre solo sobre filas de salida independientes,
//!    jamás dentro de una reducción.
//!
//! Excepción documentada: donde la referencia ggml define EXPLÍCITAMENTE otro orden
//! (l2-norm con suma secuencial en f64; ssm_conv con suma secuencial en f32 — ver
//! `ggml-cpu/ops.cpp`), el kernel replica la referencia para bit-identidad con el
//! oráculo y lo documenta en su propio doc. La partición por pares sigue siendo la
//! regla por defecto de TODA reducción que la referencia no define.

/// Suma por pares en f64: reduce `vals` nivel a nivel con `((a0+a1)+(a2+a3))...`.
/// La cola impar se arrastra al nivel siguiente sin pareja, así que para n=5 el
/// árbol es `((a0+a1)+(a2+a3))+a4`. Suma vacía = 0.0.
///
/// Esta función ES el orden de reducción del contrato: cada kernel que suma en
/// f64 lo hace a través de ella (o replica el árbol en su test para fijarlo).
pub(crate) fn pairwise_sum_f64(vals: &[f64]) -> f64 {
    let mut level = vals.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i + 1 < level.len() {
            next.push(level[i] + level[i + 1]);
            i += 2;
        }
        if i < level.len() {
            next.push(level[i]); // cola impar: se arrastra sin pareja
        }
        level = next;
    }
    level.first().copied().unwrap_or(0.0)
}

/// RMSNorm + multiplicación por pesos: `out[i] = x[i] * scale * w[i]` con
/// `scale = 1/sqrtf(mean(x²) + eps)` — réplica BIT-EXACTA del camino del
/// oráculo ggml (`ggml_compute_forward_rms_norm_f32` + MUL separado; la
/// variante fusionada `RMS_NORM+MUL` tiene la MISMA aritmética):
///
/// ```c
/// ggml_float sum = 0.0;
/// for (i) sum += (ggml_float)(x[i] * x[i]); // producto EN F32 → suma f64 secuencial
/// const float mean  = sum / n;              // división f64 → redondea a f32
/// const float scale = 1.0f / sqrtf(mean + eps); // suma, raíz y división en f32
/// norm[i] = x[i] * scale;                   // mul f32
/// out[i]  = norm[i] * w[i];                 // mul f32 (el MUL separado)
/// ```
///
/// NOTA: nada de suma pairwise ni f64 en los productos — el oráculo redondea
/// cada paso a f32. Una diferencia de 1 ulp aquí produce "flips" de qs en la
/// cuantización Q8_K de los gemvs aguas abajo (~1e-4 de ruido por flip).
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len(), "rmsnorm: len(out) != len(x)");
    assert_eq!(x.len(), w.len(), "rmsnorm: len(x) != len(w)");
    let n = x.len();
    if n == 0 {
        return;
    }
    let mut sum: f64 = 0.0;
    for &v in x {
        sum += (v * v) as f64;
    }
    let mean = (sum / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    for i in 0..n {
        let norm = x[i] * scale;
        out[i] = norm * w[i];
    }
}

/// MatMul f32×f32: `acc[m,n] += a[m,k] @ b[k,n]` (a row-major [m,k], b row-major
/// [k,n], acc row-major [m,n]).
///
/// Contrato por elemento de salida: productos `a*b` en f64, suma por pares
/// ([`pairwise_sum_f64`]), y `acc_nuevo = (acc_viejo as f64) + suma` también en f64;
/// el resultado se redondea a f32. Sin FMA en la referencia scalar: el FMA
/// explícito llega con AVX2 y su test de bit-identidad (contrato §3-4).
pub fn matmul_f32_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "matmul: len(a) != m*k");
    assert_eq!(b.len(), k * n, "matmul: len(b) != k*n");
    assert_eq!(acc.len(), m * n, "matmul: len(acc) != m*n");
    let mut prods = vec![0.0f64; k];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                prods[l] = (a[i * k + l] as f64) * (b[l * n + j] as f64);
            }
            let sum = pairwise_sum_f64(&prods);
            acc[i * n + j] = ((acc[i * n + j] as f64) + sum) as f32;
        }
    }
}

/// RoPE intercalado estilo Llama: pares `(2i, 2i+1)`. `freqs` precalculadas en f32
/// (`freqs[i] = base^(-2i/dim)`), `pos` es la posición del token.
///
/// Contrato: trigonometría en f64 sobre `pos * freqs[i]`; la rotación usa
/// `mul_add` explícito con orden fijo (todo en f64, salida redondeada a f32):
/// `x0' = c.mul_add(x0, -(s * x1));  x1' = s.mul_add(x0, c * x1)`.
pub fn rope_apply_llama(x: &mut [f32], freqs: &[f32], pos: usize) {
    let n_pairs = x.len() / 2;
    assert_eq!(x.len() % 2, 0, "rope_apply_llama: len(x) debe ser par");
    assert!(
        freqs.len() >= n_pairs,
        "rope_apply_llama: freqs cortas ({} < {})",
        freqs.len(),
        n_pairs
    );
    let pos = pos as f64;
    for i in 0..n_pairs {
        let (s, c) = (pos * (freqs[i] as f64)).sin_cos();
        let x0 = x[2 * i] as f64;
        let x1 = x[2 * i + 1] as f64;
        x[2 * i] = c.mul_add(x0, -(s * x1)) as f32;
        x[2 * i + 1] = s.mul_add(x0, c * x1) as f32;
    }
}

/// RoPE estilo NeoX (GPT-NeoX): `x[i]` rota junto con `x[i + n/2]` usando la
/// misma frecuencia `freqs[i]` (sin intercalado). Mismo contrato que
/// [`rope_apply_llama`] (mul_add explícito, trigonometría en f64).
pub fn rope_apply_neox(x: &mut [f32], freqs: &[f32], pos: usize) {
    let half = x.len() / 2;
    assert_eq!(x.len() % 2, 0, "rope_apply_neox: len(x) debe ser par");
    assert!(
        freqs.len() >= half,
        "rope_apply_neox: freqs cortas ({} < {})",
        freqs.len(),
        half
    );
    let pos = pos as f64;
    for i in 0..half {
        let (s, c) = (pos * (freqs[i] as f64)).sin_cos();
        let x0 = x[i] as f64;
        let x1 = x[i + half] as f64;
        x[i] = c.mul_add(x0, -(s * x1)) as f32;
        x[i + half] = s.mul_add(x0, c * x1) as f32;
    }
}

/// Softmax con max-subtraction y escala explícita (ARCHITECTURE §5):
/// `out[i] = exp(scale * (x[i] - max)) / Σ_j exp(scale * (x[j] - max))`.
///
/// Contrato: el max se toma en f32 (comparación exacta, sin orden); el exponente
/// `((x - max) * scale).exp()` en f32 (el modelo define softmax en f32); el
/// denominador se suma en f64 con partición por pares; la división en f64 y se
/// redondea a f32.
pub fn softmax(out: &mut [f32], x: &[f32], scale: f32) {
    assert_eq!(out.len(), x.len(), "softmax: len(out) != len(x)");
    let n = x.len();
    if n == 0 {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| ((v - max) * scale).exp()).collect();
    let sum = pairwise_sum_f64(&e.iter().map(|&v| v as f64).collect::<Vec<_>>());
    for i in 0..n {
        out[i] = (e[i] as f64 / sum) as f32;
    }
}

/// SwiGLU: `out[i] = silu(gate[i]) * up[i]` — réplica f32 de
/// `ggml_vec_swiglu_f32` (vec.cpp): `y = ggml_silu_f32(x) * g` con
/// `silu(x) = x / (1.0f + expf(-x))`, TODO en f32 (expf, suma, división, mul).
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    assert_eq!(out.len(), gate.len(), "swiglu: len(out) != len(gate)");
    assert_eq!(gate.len(), up.len(), "swiglu: len(gate) != len(up)");
    for i in 0..gate.len() {
        let t = gate[i];
        let s = t / (1.0f32 + (-t).exp());
        out[i] = s * up[i];
    }
}

/// GELU tanh (Gemma): `0.5x (1 + tanh(sqrt(2/π) (x + 0.044715 x³)))`, en f64.
pub fn gelu_tanh(out: &mut [f32], x: &[f32]) {
    assert_eq!(out.len(), x.len(), "gelu_tanh: len(out) != len(x)");
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    const C: f64 = 0.044_715;
    for i in 0..x.len() {
        let t = x[i] as f64;
        let inner = SQRT_2_OVER_PI * (t + C * t * t * t);
        out[i] = (0.5 * t * (1.0 + inner.tanh())) as f32;
    }
}

/// Sigmoid elementwise — réplica f32 de ggml (`ggml_sigmoid_f32`):
/// `1.0f / (1.0f + expf(-x))`, todo en f32.
pub fn sigmoid(out: &mut [f32], x: &[f32]) {
    assert_eq!(out.len(), x.len(), "sigmoid: len(out) != len(x)");
    for i in 0..x.len() {
        out[i] = 1.0f32 / (1.0f32 + (-x[i]).exp());
    }
}

/// SiLU elementwise — réplica f32 de ggml (`ggml_silu_f32`, vec.h):
/// `x / (1.0f + expf(-x))`, todo en f32.
pub fn silu(out: &mut [f32], x: &[f32]) {
    assert_eq!(out.len(), x.len(), "silu: len(out) != len(x)");
    for i in 0..x.len() {
        out[i] = x[i] / (1.0f32 + (-x[i]).exp());
    }
}

/// Softplus elementwise — réplica f32 de ggml (`op_softplus`, unary-ops.cpp):
/// `(x > 20.0f) ? x : logf(1.0f + expf(x))`, todo en f32. (No log1pf: ggml usa
/// `logf(1+expf(x))`.)
pub fn softplus(out: &mut [f32], x: &[f32]) {
    assert_eq!(out.len(), x.len(), "softplus: len(out) != len(x)");
    for i in 0..x.len() {
        let t = x[i];
        out[i] = if t > 20.0f32 {
            t
        } else {
            (1.0f32 + t.exp()).ln()
        };
    }
}

/// L2-norm de filas con la fórmula EXACTA de la referencia (`ggml_compute_forward_l2_norm_f32`):
/// - cuadrados `xi * xi` redondeados a f32 y ensanchados, acumulados en f64 SECUENCIAL
///   (la referencia NO usa la partición por pares aquí — excepción documentada);
/// - `scale = 1 / max(sqrtf(sum), eps)` con la raíz en f32;
/// - `y = x * scale` en f32.
///
/// `data` son `rows` filas contiguas de largo `dim`. Bit-idéntico al oráculo.
pub fn l2_norm_rows(data: &mut [f32], dim: usize, eps: f32) {
    assert_eq!(data.len() % dim, 0, "l2_norm_rows: len no es múltiplo de dim");
    let rows = data.len() / dim;
    for r in 0..rows {
        let row = &mut data[r * dim..(r + 1) * dim];
        let mut sum = 0.0f64;
        for &v in row.iter() {
            sum += (v * v) as f64; // cuadrado en f32, acumulación f64 secuencial
        }
        let scale = 1.0f32 / (sum as f32).sqrt().max(eps);
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

/// Convolución depthwise del camino recurrente (referencia: `ggml_compute_forward_ssm_conv_f32`):
/// `out[t*channels + c] = Σ_{τ=0..d_conv-1} x[(t+τ)*channels + c] * kernel[τ*channels + c]`.
///
/// Suma SECUENCIAL en f32 (4 términos) — la referencia lo define así con un comentario
/// explícito de que NO usa `ggml_vec_dot_f32` porque ese acumula en f64 (excepción
/// documentada al contrato). Bit-idéntico al oráculo.
///
/// `x`: `[ncs, channels]` ventana deslizante (`ncs = d_conv - 1 + n_t`, estado de
/// 3 + tokens nuevos). `out`: `[channels, n_t]` → largo `channels * n_t`.
pub fn ssm_conv(out: &mut [f32], x: &[f32], kernel: &[f32], d_conv: usize, channels: usize) {
    let ncs = x.len() / channels;
    assert_eq!(x.len(), ncs * channels, "ssm_conv: len(x) no es múltiplo de channels");
    assert!(ncs >= d_conv, "ssm_conv: ventana {ncs} más corta que el kernel {d_conv}");
    assert_eq!(kernel.len(), d_conv * channels, "ssm_conv: len(kernel) != d_conv*channels");
    let n_t = ncs - (d_conv - 1);
    assert_eq!(out.len(), channels * n_t, "ssm_conv: len(out) != channels*n_t");
    for t in 0..n_t {
        for c in 0..channels {
            let mut sumf = 0.0f32;
            for tau in 0..d_conv {
                sumf += x[(t + tau) * channels + c] * kernel[tau * channels + c];
            }
            out[t * channels + c] = sumf;
        }
    }
}

/// Un paso del núcleo GatedDeltaNet FUSED (referencia: el bucle interno de
/// `ggml_compute_forward_gated_delta_net_one_chunk`, ggml-cpu/ops.cpp — el kernel
/// que usó el oráculo, rama `kda=false` de gate escalar; build SIN SIMD, verificado
/// en el vcxproj: sin /arch → ggml_vec_dot_f32/mad_f32 scalar):
///
/// 1. `S *= expf(gate)` — toda la matriz por el escalar (f32 expf, multiplicación f32);
/// 2. por columna `j`: `d[j] = (v[j] - dot(S[:,j], k)) * beta` — el dot acumula
///    productos f32 en f64 SECUENCIAL (ggml_vec_dot_f32 scalar) y redondea a f32;
///    la resta y el producto por beta en f32;
/// 3. `S[i][j] += d[j] * k[i]` con mul + add SEPARADOS (ggml_vec_mad_f32 scalar,
///    SIN FMA — el build del oráculo no tiene /arch y MSVC no contrae);
/// 4. `out[j] = dot(S_nueva[:,j], q) * scale` — **la escala va sobre la SALIDA**
///    (así lo hace el camino fused; el no-fused escala q primero).
///
/// Layout: `state` es la matriz 128×128 del v-head en orden de filas `S[i][j]`
/// (`i` = dim del producto con k/q, `j` = dim de v/out). El mapeo q/k-head →
/// v-head (`h % 16`) es responsabilidad del llamador, igual que en ggml.
/// `dim = q.len() == k.len() == v.len() == out.len()`, `state.len() == dim*dim`.
pub fn gdn_fused_step(
    state: &mut [f32],
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: f32,
    beta: f32,
    scale: f32,
) {
    let dim = q.len();
    assert_eq!(k.len(), dim, "gdn_fused_step: len(k) != dim");
    assert_eq!(v.len(), dim, "gdn_fused_step: len(v) != dim");
    assert_eq!(out.len(), dim, "gdn_fused_step: len(out) != dim");
    assert_eq!(state.len(), dim * dim, "gdn_fused_step: len(state) != dim*dim");

    // Réplica exacta de ggml_compute_forward_gated_delta_net_one_chunk (build
    // sin SIMD): decaimiento ggml_vec_scale_f32; dots con ggml_vec_dot_f32
    // SCALAR (productos f32 → acumulación f64 SECUENCIAL → redondeo f32); y
    // mad con ggml_vec_mad_f32 scalar (`y += x*v`, mul + add SIN FMA).
    //
    // 1) decaimiento: S *= expf(gate)
    let g_exp = gate.exp();
    for s in state.iter_mut() {
        *s *= g_exp;
    }

    // 2) delta por columna j: d[j] = (v[j] - dot(S[:,j], k)) * beta
    let mut d = vec![0.0f32; dim];
    for j in 0..dim {
        let mut sum: f64 = 0.0;
        for i in 0..dim {
            sum += (state[i * dim + j] * k[i]) as f64;
        }
        let sum = sum as f32;
        d[j] = (v[j] - sum) * beta;
    }
    // 3) actualización: S[i][j] += d[j] * k[i]  (mul + add, sin FMA)
    for j in 0..dim {
        for i in 0..dim {
            state[i * dim + j] += k[i] * d[j];
        }
    }
    // 4) salida: out[j] = dot(S_nueva[:,j], q) * scale
    for j in 0..dim {
        let mut sum: f64 = 0.0;
        for i in 0..dim {
            sum += (state[i * dim + j] * q[i]) as f64;
        }
        out[j] = sum as f32 * scale;
    }
}

/// RoPE IMROPE de qwen3.5 (referencia: `GGML_ROPE_TYPE_IMROPE` en `ggml_mrope_cache_init`):
/// los primeros `n_rot` dims rotan en pares `(i, i + n_rot/2)`; el par `ic` (0..n_rot/2)
/// rota SOLO si `ic % 3 == 0` (sección temporal; las secciones h/w de texto tienen
/// posición 0 → identidad). El ángulo del par `ic` es `pos * theta_scale^ic` con
/// `theta_scale = base^(-2/n_rot)` — la constante de ggml se computa en f32 y aquí se
/// ensancha a f64, para que el ángulo coincida con la referencia hasta donde la
/// trigonometría f64 lo permite (contrato: trigonometría en f64 con `mul_add`).
/// Los dims `>= n_rot` se copian sin tocar.
pub fn rope_apply_imrope(x: &mut [f32], n_rot: usize, theta_scale: f64, pos: usize) {
    let half = n_rot / 2;
    assert_eq!(n_rot % 2, 0, "rope_apply_imrope: n_rot debe ser par");
    assert!(n_rot <= x.len(), "rope_apply_imrope: n_rot > len(x)");
    let pos = pos as f64;
    let mut angle = pos; // pos * theta_scale^ic, multiplicado por par
    for ic in 0..half {
        if ic % 3 == 0 {
            let (s, c) = angle.sin_cos();
            let x0 = x[ic] as f64;
            let x1 = x[ic + half] as f64;
            x[ic] = c.mul_add(x0, -(s * x1)) as f32;
            x[ic + half] = s.mul_add(x0, c * x1) as f32;
        }
        angle *= theta_scale;
    }
}

/// Lookup de embeddings: copia las filas `ids` de `table` [n_vocab, dim] (f32,
/// row-major) a `out` (len = ids.len() * dim). Copia bit a bit, sin aritmética:
/// no hay contrato numérico que fijar, solo checks de rango.
pub fn embedding_lookup(out: &mut [f32], table: &[f32], ids: &[u32], dim: usize) {
    assert_eq!(
        table.len() % dim,
        0,
        "embedding_lookup: len(table) no es múltiplo de dim"
    );
    assert_eq!(
        out.len(),
        ids.len() * dim,
        "embedding_lookup: len(out) != len(ids) * dim"
    );
    for (r, &id) in ids.iter().enumerate() {
        let src = (id as usize) * dim;
        assert!(
            src + dim <= table.len(),
            "embedding_lookup: id {id} fuera de la tabla (vocab {})",
            table.len() / dim
        );
        out[r * dim..(r + 1) * dim].copy_from_slice(&table[src..src + dim]);
    }
}

// ---------------------------------------------------------------------------
// Tests: cada kernel fija su contrato con valores exactos o, donde la
// aritmética no es exacta, replica el árbol de operaciones explícitamente
// (sin llamar al helper) para clavar el ORDEN.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_partition_is_the_documented_tree() {
        // n = 5: ((a0+a1)+(a2+a3))+a4
        let v = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(pairwise_sum_f64(&v), 15.0);
        // Adversario: el orden DEBE ser el pairwise, no el serial.
        // 1e16 + 1 se redondea a 1e16 en f64 (1 ulp = 2).
        // Serial: ((1e16+1)+(-1e16))+1 = 0+1 = 1.  Pairwise: (1e16+1)+(-1e16+1) = 1e16-1e16 = 0.
        let v = [1e16f64, 1.0, -1e16, 1.0];
        assert_eq!(pairwise_sum_f64(&v), 0.0);
        // cola impar se arrastra: [a0+a1, a2] → (a0+a1)+a2
        assert_eq!(pairwise_sum_f64(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(pairwise_sum_f64(&[]), 0.0);
    }

    #[test]
    fn rmsnorm_exact_cases() {
        // x todo 1, w todo 1, eps 0 → out todo 1
        let x = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &x, 0.0);
        assert_eq!(out, [1.0; 4]);

        // x = [3, 4], eps = 0, w = 1: réplica ggml — cuadrados exactos aquí;
        // mean y scale en f32, DOS muls f32 ((x*scale)*w). El contrato viejo
        // (triple producto f64 con una sola redondez) daría OTROS bits.
        let x = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        rmsnorm(&mut out, &x, &[1.0, 1.0], 0.0);
        let mean = ((9.0f64 + 16.0) / 2.0) as f32;
        let scale = 1.0f32 / (mean + 0.0f32).sqrt();
        assert_eq!(out[0].to_bits(), (3.0f32 * scale).to_bits());
        assert_eq!(out[1].to_bits(), (4.0f32 * scale).to_bits());

        // eps DENTRO de la raíz: x = 0, eps = 1e-6 → out = 0
        let mut out = [0.0f32; 2];
        rmsnorm(&mut out, &[0.0, 0.0], &[2.0, 3.0], 1e-6);
        assert_eq!(out, [0.0, 0.0]);

        // cuadrados EN F32 (no (v as f64)²): 1e8² en f32 no es 1e16 exacto.
        // La suma f64 secuencial de 4 términos iguales es exacta → mean = sq.
        let x = [1e8f32, 1e8, 1e8, 1e8];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &[1.0f32; 4], 0.0);
        let sq = 1e8f32 * 1e8f32;
        assert_ne!(sq as f64, 1e16f64, "sanity: el cuadrado f32 pierde precisión");
        let scale = 1.0f32 / (sq + 0.0f32).sqrt();
        for i in 0..4 {
            assert_eq!(out[i].to_bits(), (x[i] * scale).to_bits(), "elem {i}");
        }
    }

    #[test]
    fn matmul_exact_and_acc_semantics() {
        // m=2, k=3, n=2 con enteros chicos: todo exacto en f64/f32
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // [k=3, n=2]
        let mut acc = [0.0f32; 4];
        matmul_f32_acc(&mut acc, &a, &b, 2, 3, 2);
        assert_eq!(acc, [58.0, 64.0, 139.0, 154.0]); // a mano

        // acc += : segunda llamada duplica
        matmul_f32_acc(&mut acc, &a, &b, 2, 3, 2);
        assert_eq!(acc, [116.0, 128.0, 278.0, 308.0]);

        // Adversario de orden: productos [H, 1, -H, 1] con H = 1e19 (f32).
        // En f64, H+1 se redondea a H (ulp ≈ 2^11). Serial: H+1→H; -H→0; +1→1.
        // Pairwise: (H+1)+(-H+1) = H-H = 0. La partición DEBE dar 0: la
        // cancelación de signos hace el orden observable en la salida f32.
        let a = [1e19f32, 1.0, -1e19f32, 1.0];
        let b = [1.0f32, 1.0, 1.0, 1.0];
        let mut acc = [0.0f32];
        matmul_f32_acc(&mut acc, &a, &b, 1, 4, 1);
        assert_eq!(acc[0], 0.0);
    }

    #[test]
    fn rope_llama_identity_cases() {
        // pos = 0 → sin = 0, cos = 1 → identidad bit a bit
        let mut x = [1.5f32, -2.25, 3.0, 4.0];
        let freqs = [0.5f32, 0.25];
        let orig = x;
        rope_apply_llama(&mut x, &freqs, 0);
        assert_eq!(x, orig);

        // freqs = 0 → theta = 0 → identidad para cualquier pos
        let mut x = [1.5f32, -2.25, 3.0, 4.0];
        rope_apply_llama(&mut x, &[0.0f32, 0.0], 7);
        assert_eq!(x, orig);

        // rotación de π/2: freqs[0] = π/2 (f32), pos = 1, x = [1, 0]
        // cos(π/2) ≈ 6.12e-17, sin(π/2) = 1 exacto en f64.
        // OJO: el kernel usa `freqs[i] as f64`, así que el theta del test DEBE
        // ser el π/2 de f32 ensanchado, no el de f64 (difieren en ulps).
        let mut x = [1.0f32, 0.0];
        rope_apply_llama(&mut x, &[std::f32::consts::FRAC_PI_2], 1);
        let (s, c) = (std::f32::consts::FRAC_PI_2 as f64).sin_cos();
        let e0 = c.mul_add(1.0f64, -(s * 0.0f64)) as f32;
        let e1 = s.mul_add(1.0f64, c * 0.0f64) as f32;
        assert_eq!(x[0].to_bits(), e0.to_bits());
        assert_eq!(x[1].to_bits(), e1.to_bits());
        assert_eq!(x[1], 1.0);
    }

    #[test]
    fn rope_neox_identity_and_rotation() {
        // pos = 0 → identidad; el par que rota es (x[0], x[2]) para freqs[0]
        let mut x = [1.5f32, -2.25, 3.0, 4.0];
        let orig = x;
        rope_apply_neox(&mut x, &[0.5f32, 0.25], 0);
        assert_eq!(x, orig);

        // π/2 con x = [1, 99, 0, 99]: rota (x[0], x[2]); x[1], x[3] no se tocan.
        // theta del test = f32::FRAC_PI_2 ensanchado a f64 (como en el kernel).
        let mut x = [1.0f32, 99.0, 0.0, 99.0];
        rope_apply_neox(&mut x, &[std::f32::consts::FRAC_PI_2, 0.0], 1);
        let (s, c) = (std::f32::consts::FRAC_PI_2 as f64).sin_cos();
        let e0 = c.mul_add(1.0f64, -(s * 0.0f64)) as f32;
        let e2 = s.mul_add(1.0f64, c * 0.0f64) as f32;
        assert_eq!(x[0].to_bits(), e0.to_bits());
        assert_eq!(x[2].to_bits(), e2.to_bits());
        assert_eq!(x[1], 99.0);
        assert_eq!(x[3], 99.0);
    }

    #[test]
    fn softmax_max_subtraction_and_partition() {
        // [1000, 1001, 1002] ≡ [-2, -1, 0] bit a bit (max-subtraction)
        let mut o1 = [0.0f32; 3];
        let mut o2 = [0.0f32; 3];
        softmax(&mut o1, &[1000.0, 1001.0, 1002.0], 1.0);
        softmax(&mut o2, &[-2.0, -1.0, 0.0], 1.0);
        assert_eq!(o1, o2);

        // árbol explícito del denominador para n = 3: (e0+e1)+e2
        let e0 = (-2.0f32).exp() as f64;
        let e1 = (-1.0f32).exp() as f64;
        let e2 = (0.0f32).exp() as f64;
        let sum = (e0 + e1) + e2;
        let exp0 = (e0 / sum) as f32;
        let exp1 = (e1 / sum) as f32;
        let exp2 = (e2 / sum) as f32;
        assert_eq!(o1[0].to_bits(), exp0.to_bits());
        assert_eq!(o1[1].to_bits(), exp1.to_bits());
        assert_eq!(o1[2].to_bits(), exp2.to_bits());

        // la suma de salidas ≈ 1 (cada salida es f32 ya redondeada)
        let s = (o1[0] as f64 + o1[1] as f64) + o1[2] as f64;
        assert!((s - 1.0).abs() < 1e-7, "suma de salidas = {s}");

        // escala explícita: softmax(x, 2) == softmax(2x, 1)
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        softmax(&mut a, &[-2.0, -1.0, 0.0], 2.0);
        softmax(&mut b, &[-4.0, -2.0, 0.0], 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn swiglu_hand_values() {
        // gate = 0 → silu(0) = 0·0.5 = 0 → out 0
        let mut out = [9.9f32; 2];
        swiglu(&mut out, &[0.0, 3.0], &[7.0, 0.0]);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0); // up = 0

        // gate = 1, up = 2: t=1; s = 1/(1+e^-1); out = (t*s)*up
        let mut out = [0.0f32];
        swiglu(&mut out, &[1.0], &[2.0]);
        let t = 1.0f64;
        let s = 1.0 / (1.0 + (-t).exp());
        let e = (t * s * 2.0f64) as f32;
        assert_eq!(out[0].to_bits(), e.to_bits());
        assert!((out[0] - 1.4621172).abs() < 1e-6); // silu(1)·2, no silu(1)
    }

    #[test]
    fn gelu_tanh_hand_values() {
        let mut out = [0.0f32; 2];
        gelu_tanh(&mut out, &[0.0, 1.0]);
        assert_eq!(out[0], 0.0); // 0.5·0·(1+0)
        const S: f64 = 0.797_884_560_802_865_4;
        const C: f64 = 0.044_715;
        let t = 1.0f64;
        let e = (0.5 * t * (1.0 + (S * (t + C * t * t * t)).tanh())) as f32;
        assert_eq!(out[1].to_bits(), e.to_bits());
        assert!((out[1] - 0.84119199).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_silu_softplus_f64_formulas() {
        // sigmoid(0) = 0.5 exacto; silu(0) = 0; softplus(0) = ln(2)
        let mut s = [9.9f32; 3];
        sigmoid(&mut s, &[0.0, 0.0, 0.0]);
        assert_eq!(s, [0.5; 3]);
        let mut s = [9.9f32; 3];
        silu(&mut s, &[0.0, 0.0, 0.0]);
        assert_eq!(s, [0.0; 3]);
        let mut s = [9.9f32];
        softplus(&mut s, &[0.0]);
        assert_eq!(s[0], (2.0f64.ln()) as f32);

        // t = 1: replicar la fórmula f64 exacta (mismo árbol que swiglu)
        let t = 1.0f64;
        let sig = 1.0 / (1.0 + (-t).exp());
        let mut s = [0.0f32];
        sigmoid(&mut s, &[1.0]);
        assert_eq!(s[0].to_bits(), ((sig) as f32).to_bits());
        let mut s = [0.0f32];
        silu(&mut s, &[1.0]);
        assert_eq!(s[0].to_bits(), ((t * sig) as f32).to_bits());
        let mut s = [0.0f32];
        softplus(&mut s, &[1.0]);
        assert_eq!(s[0].to_bits(), (((1.0 + t.exp()).ln()) as f32).to_bits());
        assert!((s[0] - 1.3132617).abs() < 1e-6);

        // lado negativo: sigmoid(-1) + sigmoid(1) = 1 en el árbol exacto
        let mut s = [0.0f32; 2];
        sigmoid(&mut s, &[-1.0, 1.0]);
        let t1 = 1.0f64;
        let e_neg = (1.0 / (1.0 + t1.exp())) as f32;
        let e_pos = (1.0 / (1.0 + (-t1).exp())) as f32;
        assert_eq!(s[0].to_bits(), e_neg.to_bits());
        assert_eq!(s[1].to_bits(), e_pos.to_bits());
    }

    #[test]
    fn l2_norm_rows_reference_formula() {
        // [3, 4]: suma = 25 exacta, sqrtf(25) = 5 exacto, scale = 0.2 exacto
        let mut x = [3.0f32, 4.0];
        l2_norm_rows(&mut x, 2, 1e-6);
        assert_eq!(x, [0.6, 0.8]);

        // ceros + eps: scale = 1/eps finito, salida sigue siendo 0
        let mut x = [0.0f32, 0.0];
        l2_norm_rows(&mut x, 2, 0.1);
        assert_eq!(x, [0.0, 0.0]);

        // caso con raíz inexacta: replicar la fórmula de la referencia
        // (cuadrado f32 → suma f64 SECUENCIAL → sqrtf → 1/sqrtf → producto f32).
        // OJO: con términos positivos el orden de la suma f64 no es observable a
        // través de la salida f32 (mismo argumento que en rmsnorm_exact_cases);
        // el orden secuencial queda fijado por la doc + revisión contra ops.cpp.
        let x = [1.0f32, 2.0, 3.0];
        let mut out = x;
        l2_norm_rows(&mut out, 3, 1e-6);
        let sum = 1.0f64 + 4.0 + 9.0; // cuadrados f32 exactos aquí
        let scale = 1.0f32 / (sum as f32).sqrt().max(1e-6f32);
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(out[i].to_bits(), (v * scale).to_bits(), "elem {i}");
        }

        // varias filas: cada fila se normaliza de forma independiente
        let mut x = [3.0f32, 4.0, 6.0, 8.0];
        l2_norm_rows(&mut x, 2, 1e-6);
        assert_eq!(x, [0.6, 0.8, 0.6, 0.8]);
    }

    #[test]
    fn ssm_conv_hand_values_and_f32_order() {
        // d_conv=2, channels=2, kernel = [[1,2],[3,4]] (τ-major), ventana ncs=3
        let kernel = [1.0f32, 2.0, 3.0, 4.0];
        let x = [1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut out = [0.0f32; 4];
        ssm_conv(&mut out, &x, &kernel, 2, 2);
        assert_eq!(out, [7.0, 100.0, 11.0, 160.0]);

        // Adversario del orden: la referencia SUMA SECUENCIAL EN F32 (no usa
        // ggml_vec_dot_f32/f64). x = [1e16, 1, -1e16, 1], kernel = 1:
        // secuencial: ((1e16+1)-1e16)+1 = 0+1 = 1. Pairwise daría 0.
        let x = [1e16f32, 1.0, -1e16, 1.0];
        let kernel = [1.0f32; 4];
        let mut out = [0.0f32];
        ssm_conv(&mut out, &x, &kernel, 4, 1);
        assert_eq!(out[0], 1.0);

        // productos inexactos: replicar la suma secuencial f32 explícitamente
        let kernel = [0.1f32, 0.2, 0.3, 0.4];
        let x = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5]; // ncs=3, n_t=2
        let mut out = [0.0f32; 4];
        ssm_conv(&mut out, &x, &kernel, 2, 2);
        let e = |t: usize, c: usize| -> f32 {
            let mut s = 0.0f32;
            for tau in 0..2 {
                s += x[(t + tau) * 2 + c] * kernel[tau * 2 + c];
            }
            s
        };
        for t in 0..2 {
            for c in 0..2 {
                assert_eq!(out[t * 2 + c].to_bits(), e(t, c).to_bits());
            }
        }
    }

    #[test]
    fn rope_imrope_interleaved_pairs() {
        // n_rot=4, theta_scale=0.01, pos=1: par ic=0 rota (x[0], x[2]) por 1 rad;
        // par ic=1 NO rota (1 % 3 != 0); dims >= 4 no se tocan.
        let mut x = [1.0f32, 99.0, 0.0, 99.0, 7.0, 8.0];
        rope_apply_imrope(&mut x, 4, 0.01, 1);
        let (s, c) = 1.0f64.sin_cos();
        let e0 = c.mul_add(1.0f64, -(s * 0.0f64)) as f32;
        let e2 = s.mul_add(1.0f64, c * 0.0f64) as f32;
        assert_eq!(x[0].to_bits(), e0.to_bits());
        assert_eq!(x[2].to_bits(), e2.to_bits());
        assert_eq!(x[1], 99.0);
        assert_eq!(x[3], 99.0);
        assert_eq!(x[4], 7.0);
        assert_eq!(x[5], 8.0);

        // pos = 0 → identidad bit a bit (sin = 0, cos = 1)
        let mut x = [1.5f32, -2.25, 3.0, 4.0];
        let orig = x;
        rope_apply_imrope(&mut x, 4, 0.01, 0);
        assert_eq!(x, orig);

        // patrón %3 sobre n_rot=12 (half=6): rotan pares ic=0 (ángulo 1) e ic=3
        // (ángulo 1·0.01³); ic=1,2,4,5 identidad. El ángulo se multiplica por
        // theta_scale en CADA par, rote o no (replica el bucle de ggml).
        let mut x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let orig = x.clone();
        rope_apply_imrope(&mut x, 12, 0.01, 1);
        let rotate = |x0: f32, x1: f32, a: f64| -> (f32, f32) {
            let (s, c) = a.sin_cos();
            (
                c.mul_add(x0 as f64, -(s * x1 as f64)) as f32,
                s.mul_add(x0 as f64, c * x1 as f64) as f32,
            )
        };
        let (r0, r6) = rotate(orig[0], orig[6], 1.0);
        let (r3, r9) = rotate(orig[3], orig[9], 0.01f64.powi(3));
        for i in 0..12 {
            let e = match i {
                0 => r0,
                3 => r3,
                6 => r6,
                9 => r9,
                _ => orig[i],
            };
            assert_eq!(x[i].to_bits(), e.to_bits(), "dim {i}");
        }
    }

    #[test]
    fn embedding_lookup_copies_rows() {
        // tabla 4×3: fila id = [id*10, id*10+1, id*10+2]
        let table: Vec<f32> = (0..12).map(|i| i as f32 * 10.0 + (i % 3) as f32).collect();
        let mut out = [0.0f32; 9];
        embedding_lookup(&mut out, &table, &[0, 3, 1], 3);
        assert_eq!(&out[0..3], &table[0..3]);
        assert_eq!(&out[3..6], &table[9..12]);
        assert_eq!(&out[6..9], &table[3..6]);
    }

    #[test]
    #[should_panic(expected = "fuera de la tabla")]
    fn embedding_lookup_rejects_out_of_range_id() {
        let table = vec![0.0f32; 6];
        let mut out = [0.0f32; 4];
        embedding_lookup(&mut out, &table, &[0, 99], 2);
    }

    #[test]
    fn gdn_fused_step_hand_values() {
        // dim=2, estado identidad, gate=0 (exp=1), beta=1, scale=1.
        // d[j] = v[j] - dot(S[:,j], k): d[0] = 3-1 = 2, d[1] = 4-1 = 3.
        // S += d⊗k: [[1+2, 0+3], [0+2, 1+3]] = [[3,3],[2,4]].
        // out[j] = dot(S[:,j], q), q=[1,0]: out = [3, 3].
        let mut state = [1.0f32, 0.0, 0.0, 1.0]; // S[i][j] con i=fila
        let mut out = [0.0f32; 2];
        gdn_fused_step(&mut state, &mut out, &[1.0, 0.0], &[1.0, 1.0], &[3.0, 4.0], 0.0, 1.0, 1.0);
        assert_eq!(state, [3.0, 3.0, 2.0, 4.0]);
        assert_eq!(out, [3.0, 3.0]);

        // decay: gate = ln(0.5) -> expf = 0.5 escala TODO el estado primero;
        // con el estado ya final del paso anterior:
        // S' = 0.5*[[3,3],[2,4]]; d = v - dot(S'[:,j], k); el update y la salida
        // siguen la misma cadena f32.
        let prev = state;
        let mut state2 = prev;
        let mut out2 = [0.0f32; 2];
        let g = 0.5f32.ln(); // expf(g) = 0.5 exacto en f32? ln(0.5) no es exacto -> tolerancia
        gdn_fused_step(&mut state2, &mut out2, &[1.0, 0.0], &[1.0, 1.0], &[3.0, 4.0], g, 1.0, 1.0);
        let half = 0.5f32;
        // replicación manual en f32 (mismo orden documentado)
        let mut rep = prev;
        let g_exp = g.exp();
        for s in rep.iter_mut() {
            *s *= g_exp;
        }
        let d0 = (3.0f32 - (rep[0] + rep[2]) * 1.0) * 1.0;
        let d1 = (4.0f32 - (rep[1] + rep[3]) * 1.0) * 1.0;
        rep[0] = d0.mul_add(1.0, rep[0]);
        rep[2] = d0.mul_add(1.0, rep[2]);
        rep[1] = d1.mul_add(1.0, rep[1]);
        rep[3] = d1.mul_add(1.0, rep[3]);
        let o0 = (rep[0] * 1.0 + rep[2] * 0.0) * 1.0;
        let o1 = (rep[1] * 1.0 + rep[3] * 0.0) * 1.0;
        assert!((state2[0] - rep[0]).abs() < 1e-6 && (out2[0] - o0).abs() < 1e-6);
        assert!((out2[1] - o1).abs() < 1e-6);
        _ = half;
    }

    #[test]
    fn gdn_fused_step_scale_is_on_output_not_q() {
        // scale=2: out = dot(S,q)*2. Si la escala fuera sobre q, el dot usaría 2q.
        // Estado final del test anterior es asimétrico: [[3,3],[2,4]] -> dot(S[:,0],q)
        // con q=[2,0] es 6; con q=[1,0] y scale=2 también es 6 — usar q=[1,1]:
        // dot(S[:,0],[1,1]) = 5, dot(S[:,1],[1,1]) = 7. Con scale=2 -> [10,14].
        let mut state = [3.0f32, 3.0, 2.0, 4.0];
        let mut out = [0.0f32; 2];
        gdn_fused_step(&mut state, &mut out, &[1.0, 1.0], &[1.0, 1.0], &[3.0, 4.0], 0.0, 1.0, 2.0);
        // d = v - dot(S,q... k): dot(S[:,0],k)=5, dot(S[:,1],k)=7 -> d=[-2,-3]
        // S += d⊗k: S[0][0]-=2 ... S = [[1,0],[0,1]]; out = dot*scale = [1,1]*2 = [2,2]
        assert_eq!(state, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(out, [2.0, 2.0]);
    }

    #[test]
    fn gdn_fused_step_replicates_documented_f32_chain() {
        // Valores arbitrarios fijos (dim=3), sin decaimiento exacto: gate=0.
        let q = [0.25f32, -0.5, 0.75];
        let k = [-0.125f32, 0.5, 0.375];
        let v = [1.0f32, -2.0, 0.5];
        let beta = 0.8f32;
        let scale = 0.0625f32; // 1/sqrt(256) real
        let mut state = vec![0.0f32; 9];
        for i in 0..3 {
            state[i * 3 + i] = 1.0;
        }
        let mut out = [0.0f32; 3];
        let mut rep_state = state.clone();
        let mut rep_out = [0.0f32; 3];
        gdn_fused_step(&mut state, &mut out, &q, &k, &v, 0.0, beta, scale);
        // replicación independiente del MISMO orden documentado (productos f32
        // → f64 secuencial → f32; mad sin FMA)
        for j in 0..3 {
            let mut sum: f64 = 0.0;
            for i in 0..3 {
                sum += (rep_state[i * 3 + j] * k[i]) as f64;
            }
            let d = (v[j] - sum as f32) * beta;
            for i in 0..3 {
                rep_state[i * 3 + j] += k[i] * d;
            }
        }
        for j in 0..3 {
            let mut sum: f64 = 0.0;
            for i in 0..3 {
                sum += (rep_state[i * 3 + j] * q[i]) as f64;
            }
            rep_out[j] = sum as f32 * scale;
        }
        // bit a bit: la réplica sigue exactamente el mismo orden f32
        assert_eq!(state, rep_state);
        assert_eq!(out, rep_out);
    }
}
