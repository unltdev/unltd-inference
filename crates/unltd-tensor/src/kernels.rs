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

/// RMSNorm: `out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)`, con eps DENTRO de la
/// raíz (como K3; DeepSeek/Qwen también lo definen así).
///
/// Contrato: Σ x² en f64 con partición por pares ([`pairwise_sum_f64`]); la salida
/// se computa en f64 con orden fijo `((x * w) * inv_rms)` y se redondea a f32.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f64) {
    assert_eq!(out.len(), x.len(), "rmsnorm: len(out) != len(x)");
    assert_eq!(x.len(), w.len(), "rmsnorm: len(x) != len(w)");
    let n = x.len();
    if n == 0 {
        return;
    }
    let sq: Vec<f64> = x.iter().map(|&v| (v as f64) * (v as f64)).collect();
    let mean_sq = pairwise_sum_f64(&sq) / n as f64;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..n {
        out[i] = ((x[i] as f64) * (w[i] as f64) * inv_rms) as f32;
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

/// SwiGLU: `out[i] = silu(gate[i]) * up[i]`, con `silu(x) = x / (1 + e^(-x))`.
///
/// Contrato: cómputo en f64 con orden fijo: `t = gate; s = 1/(1+exp(-t));
/// out = (t * s) * up`, redondeado a f32.
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    assert_eq!(out.len(), gate.len(), "swiglu: len(out) != len(gate)");
    assert_eq!(gate.len(), up.len(), "swiglu: len(gate) != len(up)");
    for i in 0..gate.len() {
        let t = gate[i] as f64;
        let s = 1.0 / (1.0 + (-t).exp());
        out[i] = (t * s * (up[i] as f64)) as f32;
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

        // x = [3, 4], eps = 0: mean_sq = (9+16)/2 = 12.5 EXACTO en f64,
        // esperado = ((x * w) * (1/sqrt(12.5))) redondeado a f32
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        rmsnorm(&mut out, &x, &w, 0.0);
        let e0 = ((3.0f64) * (1.0f64) * (1.0 / 12.5f64.sqrt())) as f32;
        let e1 = ((4.0f64) * (1.0f64) * (1.0 / 12.5f64.sqrt())) as f32;
        assert_eq!(out[0].to_bits(), e0.to_bits());
        assert_eq!(out[1].to_bits(), e1.to_bits());

        // eps DENTRO de la raíz: x = 0, eps = 1e-6 → out = 0
        let mut out = [0.0f32; 2];
        rmsnorm(&mut out, &[0.0, 0.0], &[2.0, 3.0], 1e-6);
        assert_eq!(out, [0.0, 0.0]);

        // squares con rango grande: la suma f64 de cuadrados es exacta en este
        // régimen (1e16 tiene ulp 65536, y +1 ni siquiera redondea). Con entradas
        // todas positivas el orden no es observable a través de la salida f32; el
        // orden queda clavado por el test del helper y, en Fase 4, por el test de
        // bit-identidad scalar ≡ AVX2.
        let x = [1e8f32, 1.0, 1e8, 1.0];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &[1.0f32; 4], 0.0);
        let mean_sq_pairwise = 2e16f64 / 4.0; // = 5e15 exacto
        let inv = 1.0 / mean_sq_pairwise.sqrt();
        for (i, &v) in x.iter().enumerate() {
            let e = ((v as f64) * (1.0f64) * inv) as f32;
            assert_eq!(out[i].to_bits(), e.to_bits(), "elem {i}");
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
}
