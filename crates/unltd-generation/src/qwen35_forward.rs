//! Forward COMPLETO qwen3_5 (Fase 6) — las 32 capas del modelo real.
//!
//! Referencia de diseño: `docs/QWEN35-FORWARD.md` (algoritmo exacto por capa,
//! mapeos de heads, targets del volcado). El grafo replica `qwen35.cpp` +
//! `delta-net-base.cpp` de llama.cpp con los kernels de unltd-tensor ya
//! validados contra el oráculo (Fases 3-5).
//!
//! Sobre el prefill: el oráculo prefilló los 5 tokens en UN ubatch; el kernel
//! fused GDN itera los tokens en orden con estado compartido (y el conv ring
//! se actualiza token a token), así que este forward secuencial por token es
//! equivalente por capa. La atención causal con cache K/V también lo es
//! (mismo orden de suma por fila).
//!
//! Estado persistente ([`Qwen35Session`], pre-dimensionado en `new_session`):
//! - conv ring: (conv_kernel-1) × d_conv por capa recurrente;
//! - matrices GDN: n_v_heads × state_size² por capa recurrente;
//! - K/V cache: ctx × (n_head_kv × head_dim) por capa de atención.
//!
//! Política (heredada de k3_st.c, docs/AUDIT.md §3.2): `open` valida CADA
//! tensor de las 32 capas (nombre, forma, dtype) antes de servir un byte; un
//! dtype que el motor no conoce es una negativa, nunca un default. `step` es
//! puramente secuencial — Fase 6 es correctness; el paralelismo por filas
//! llega después y medido (contrato §5).
//!
//! Réplicas del oráculo (build ggml SIN SIMD — el vcxproj no lleva /arch, así
//! que todos los vec_kernels corrieron scalar): los gemvs cuantizados replican
//! los kernels GENERIC q4_K/q6_K×q8_K; rmsnorm/silu/sigmoid/softplus/swiglu en
//! f32 con las fórmulas exactas de ggml; los dots del GDN con productos f32 →
//! f64 secuencial (ggml_vec_dot_f32 scalar) y mad sin FMA. Queda documentado
//! en docs/QWEN35-FORWARD.md §7: la fuente residual de diferencias ~1e-7 en
//! las entradas produce "flips" de qs en la cuantización Q8_K de los gemvs
//! aguas abajo (piso real ~1e-4 por flip, no 5e-5) — ver el mecanismo en
//! benchmarks/reference/tmp-flips.py.

use unltd_architectures::qwen35::Qwen35Config;
use unltd_core::LoadError;
use unltd_model_loader::{GgufReader, MappedWeights, WeightIndex};
use unltd_tensor::{
    dot_f32, gdn_fused_step, gemv_quant_q8k, l2_norm_rows, rope_apply_imrope, rmsnorm, sigmoid,
    silu, softmax, softplus, ssm_conv, swiglu, DType,
};

use crate::min_forward::{dtype_of, MinForward};

/// Modelo qwen3.5 completo, validado tensor por tensor.
pub struct Qwen35Forward {
    pub cfg: Qwen35Config,
    /// Cabeza Fase 5 (embeddings + pesos + output): reusada, no re-mapeada.
    head: MinForward,
    /// Slot en conv/gdn por capa recurrente (`usize::MAX` = capa de atención).
    recr_slot: Vec<usize>,
    /// Slot en k/v cache por capa de atención (`usize::MAX` = recurrente).
    attn_slot: Vec<usize>,
    /// Dtype real de `attn_v.weight` por capa de atención (mixto Q4_K/Q6_K).
    attn_v_dtype: Vec<DType>,
    /// Dtype real de `ffn_down.weight` por capa (mixto Q4_K/Q6_K según la capa).
    ffn_down_dtype: Vec<DType>,
    /// Dtype real de `attn_qkv.weight` por capa recurrente (mixto Q4_K/Q6_K).
    qkv_dtype: Vec<DType>,
}

/// Estado persistente entre tokens: conv ring + matrices GDN + K/V cache.
/// Todos los buffers nacen en cero (igual que los estados iniciales de ggml).
pub struct Qwen35Session {
    conv: Vec<Vec<f32>>,
    gdn: Vec<Vec<f32>>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    /// Tokens procesados; `step` escribe el K/V en este slot.
    position: usize,
    /// Largo máximo de secuencia, fijado al crear la sesión.
    ctx: usize,
}

/// Salidas por capa de UN token, para comparar contra el volcado del oráculo.
/// `linear_attn_out` solo existe en capas recurrentes (ceros en atención).
#[derive(Default, Debug)]
pub struct LayerDump {
    /// `attn_residual-{il}`: n_layer × n_embd, una fila por capa.
    pub attn_residual: Vec<f32>,
    /// `l_out-{il}`.
    pub l_out: Vec<f32>,
    /// `linear_attn_out-{il}` (capas recurrentes).
    pub linear_attn_out: Vec<f32>,
}

impl LayerDump {
    fn push(&mut self, il: usize, l_out: &[f32], residual: &[f32], linear: &[f32]) {
        let n = l_out.len();
        let base = il * n;
        self.l_out[base..base + n].copy_from_slice(l_out);
        self.attn_residual[base..base + n].copy_from_slice(residual);
        self.linear_attn_out[base..base + n].copy_from_slice(linear);
    }
}

/// Captura de nodos intermedios del camino recurrente (nombres = cbs del
/// volcado: `attn_norm`, `linear_attn_qkv_mixed`, `z`, `beta`, …). Solo se
/// llena cuando `step` recibe `Some`: es el gancho de debugging de la Fase 6
/// contra `compare-nodes-oracle.py`, no parte del camino normal.
#[derive(Default, Debug)]
pub struct NodeCapture {
    /// Un `Vec<(nombre, valores)>` por token, en orden de grafo.
    pub per_token: Vec<Vec<(String, Vec<f32>)>>,
}

/// Valida un tensor del índice: presencia, element count EXACTO y dtype en la
/// lista permitida. Devuelve el dtype real (útil para los mixtos).
fn check(r: &GgufReader, name: &str, elems: usize, allowed: &[DType]) -> Result<DType, LoadError> {
    let meta = r
        .find(name)
        .ok_or_else(|| LoadError::MissingTensor { name: name.to_string() })?;
    let got = meta.n_elements as usize;
    if got != elems {
        return Err(LoadError::ElementCount { name: name.to_string(), got, want: elems });
    }
    let dt = dtype_of(meta)?;
    if !allowed.contains(&dt) {
        return Err(LoadError::corrupt(format!(
            "tensor '{name}': dtype {dt:?}, se esperaba uno de {allowed:?}"
        )));
    }
    Ok(dt)
}

impl Qwen35Forward {
    /// Abre y valida TODO antes de servir un byte: config (vía MinForward),
    /// embeddings + cabeza, y los ~15 tensores de cada una de las 32 capas.
    pub fn open(weights: MappedWeights) -> Result<Self, LoadError> {
        let head = MinForward::open(weights)?;
        let cfg = head.cfg.clone();
        let r = head.weights().reader();

        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let n_v = cfg.n_v_heads();
        let hd_lin = cfg.head_dim_linear();
        let d_conv = cfg.d_conv();
        let nhkv = cfg.n_head_kv;
        let hdv = cfg.head_dim_v;

        check(r, "output_norm.weight", n, &[DType::F32])?;

        let mut recr_slot = vec![usize::MAX; cfg.n_layer];
        let mut attn_slot = vec![usize::MAX; cfg.n_layer];
        let mut attn_v_dtype = vec![DType::F32; cfg.n_layer];
        let mut ffn_down_dtype = vec![DType::F32; cfg.n_layer];
        let mut qkv_dtype = vec![DType::F32; cfg.n_layer];
        let (mut n_recr, mut n_attn) = (0usize, 0usize);

        for il in 0..cfg.n_layer {
            let pre = format!("blk.{il}.");
            // Comunes a ambos tipos de capa.
            check(r, &format!("{pre}attn_norm.weight"), n, &[DType::F32])?;
            check(r, &format!("{pre}post_attention_norm.weight"), n, &[DType::F32])?;
            check(r, &format!("{pre}ffn_gate.weight"), n * n_ff, &[DType::Q4K])?;
            check(r, &format!("{pre}ffn_up.weight"), n * n_ff, &[DType::Q4K])?;
            ffn_down_dtype[il] =
                check(r, &format!("{pre}ffn_down.weight"), n_ff * n, &[DType::Q4K, DType::Q6K])?;

            if cfg.is_full_attn(il) {
                attn_slot[il] = n_attn;
                n_attn += 1;
                check(r, &format!("{pre}attn_q.weight"), n * 2 * n, &[DType::Q4K])?;
                check(r, &format!("{pre}attn_q_norm.weight"), hdv, &[DType::F32])?;
                check(r, &format!("{pre}attn_k.weight"), n * nhkv * hdv, &[DType::Q4K])?;
                check(r, &format!("{pre}attn_k_norm.weight"), hdv, &[DType::F32])?;
                attn_v_dtype[il] = check(
                    r,
                    &format!("{pre}attn_v.weight"),
                    n * nhkv * hdv,
                    &[DType::Q4K, DType::Q6K],
                )?;
                check(r, &format!("{pre}attn_output.weight"), n * n, &[DType::Q4K])?;
            } else {
                recr_slot[il] = n_recr;
                n_recr += 1;
                qkv_dtype[il] = check(
                    r,
                    &format!("{pre}attn_qkv.weight"),
                    n * d_conv,
                    &[DType::Q4K, DType::Q6K],
                )?;
                check(r, &format!("{pre}attn_gate.weight"), n * n, &[DType::Q4K])?;
                check(
                    r,
                    &format!("{pre}ssm_conv1d.weight"),
                    cfg.conv_kernel * d_conv,
                    &[DType::F32],
                )?;
                check(r, &format!("{pre}ssm_dt.bias"), n_v, &[DType::F32])?;
                check(r, &format!("{pre}ssm_a"), n_v, &[DType::F32])?;
                check(r, &format!("{pre}ssm_beta.weight"), n * n_v, &[DType::Q4K])?;
                check(r, &format!("{pre}ssm_alpha.weight"), n * n_v, &[DType::Q4K])?;
                check(r, &format!("{pre}ssm_norm.weight"), hd_lin, &[DType::F32])?;
                check(r, &format!("{pre}ssm_out.weight"), n * n, &[DType::Q4K])?;
            }
        }

        Ok(Self { cfg, head, recr_slot, attn_slot, attn_v_dtype, ffn_down_dtype, qkv_dtype })
    }

    /// Sesión con todo el estado recurrente a cero y cache K/V pre-dimensionado.
    pub fn new_session(&self, ctx: usize) -> Qwen35Session {
        let n_v = self.cfg.n_v_heads();
        let hd = self.cfg.head_dim_linear();
        let mut conv = Vec::new();
        let mut gdn = Vec::new();
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for il in 0..self.cfg.n_layer {
            if self.cfg.is_full_attn(il) {
                let kv_len = self.cfg.n_head_kv * self.cfg.head_dim_v;
                k_cache.push(vec![0.0f32; ctx * kv_len]);
                v_cache.push(vec![0.0f32; ctx * kv_len]);
            } else {
                conv.push(vec![0.0f32; (self.cfg.conv_kernel - 1) * self.cfg.d_conv()]);
                gdn.push(vec![0.0f32; n_v * hd * hd]);
            }
        }
        Qwen35Session { conv, gdn, k_cache, v_cache, position: 0, ctx }
    }

    pub fn vocab(&self) -> usize {
        self.head.vocab()
    }

    /// Lookup de embeddings (delegado a la cabeza Fase 5).
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, LoadError> {
        self.head.embed(tokens)
    }

    /// Cabeza de salida (delegada): `logits = output.weight · result_norm`.
    pub fn output_logits(&self, result_norm: &[f32]) -> Result<Vec<f32>, LoadError> {
        self.head.output_logits(result_norm)
    }

    /// Norma final: `result_norm = rmsnorm(l_out, output_norm.weight)`.
    pub fn output_norm(&self, x: &[f32]) -> Result<Vec<f32>, LoadError> {
        let n = self.cfg.n_embd;
        assert_eq!(x.len(), n, "output_norm: len(x) != n_embd");
        let w = self
            .head
            .weights()
            .tensor_checked("output_norm.weight", n)
            .expect("validado en open");
        let w: &[f32] = bytemuck::cast_slice(w);
        let mut out = vec![0.0f32; n];
        rmsnorm(&mut out, x, w, self.cfg.rms_eps);
        Ok(out)
    }

    /// Un token por las 32 capas. `x_in` = fila de embedding (o `l_out` previo).
    /// Devuelve el `l_out` de la capa 31 y, con `dump`, las salidas por capa
    /// para la comparación contra el oráculo. Avanza `session.position`.
    pub fn step(
        &self,
        s: &mut Qwen35Session,
        x_in: &[f32],
        mut dump: Option<&mut LayerDump>,
        mut nodes: Option<&mut NodeCapture>,
    ) -> Result<Vec<f32>, LoadError> {
        let n = self.cfg.n_embd;
        assert_eq!(x_in.len(), n, "step: len(x_in) != n_embd");
        if s.position >= s.ctx {
            return Err(LoadError::corrupt(format!(
                "contexto lleno: {} tokens procesados, ctx = {}",
                s.position, s.ctx
            )));
        }
        if let Some(ref mut c) = nodes {
            c.per_token.push(Vec::new());
        }
        let mut x = x_in.to_vec();
        for il in 0..self.cfg.n_layer {
            let (l_out, residual, linear) = if self.cfg.is_full_attn(il) {
                let r = self.attn_layer(s, il, &x)?;
                (r.l_out, r.residual, vec![0.0f32; n])
            } else {
                let r = self.recr_layer(
                    s,
                    il,
                    &x,
                    nodes.as_mut().map(|c| c.per_token.last_mut().expect("token pushed")),
                )?;
                (r.l_out, r.residual, r.linear_out)
            };
            if let Some(ref mut d) = dump {
                d.push(il, &l_out, &residual, &linear);
            }
            x = l_out;
        }
        s.position += 1;
        Ok(x)
    }

    /// Capa recurrente (GatedDeltaNet), algoritmo de docs/QWEN35-FORWARD.md §3.
    fn recr_layer(
        &self,
        s: &mut Qwen35Session,
        il: usize,
        x_in: &[f32],
        mut nodes: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Result<RecrOut, LoadError> {
        let cfg = &self.cfg;
        let n = cfg.n_embd; // 4096
        let d_conv = cfg.d_conv(); // 8192
        let hd = cfg.head_dim_linear(); // 128
        let n_qk = cfg.n_qk_heads(); // 16
        let n_v = cfg.n_v_heads(); // 32
        let eps = cfg.rms_eps;
        let slot = self.recr_slot[il];
        assert_ne!(slot, usize::MAX, "capa {il} no es recurrente");
        let w = self.head.weights();
        let pre = format!("blk.{il}.");

        // 1) attn_norm
        let mut x = vec![0.0f32; n];
        {
            let wn: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}attn_norm.weight"), n).expect("validado en open"),
            );
            rmsnorm(&mut x, x_in, wn, eps);
        }
        push_node(&mut nodes, il, "attn_norm", x.clone());

        // 2) qkv [8192] y z [4096]
        let mut qkv = vec![0.0f32; d_conv];
        gemv_quant_q8k(&mut qkv, &x, wq(&w, &pre, "attn_qkv.weight"), n, d_conv, self.qkv_dtype[il]);
        let mut z = vec![0.0f32; n];
        gemv_quant_q8k(&mut z, &x, wq(&w, &pre, "attn_gate.weight"), n, n, DType::Q4K);
        push_node(&mut nodes, il, "linear_attn_qkv_mixed", qkv.clone());
        push_node(&mut nodes, il, "z", z.clone());

        // 3) beta = sigmoid(ssm_beta·x); gate = softplus(ssm_alpha·x + dt_bias)·ssm_a
        let mut beta_raw = vec![0.0f32; n_v];
        gemv_quant_q8k(&mut beta_raw, &x, wq(&w, &pre, "ssm_beta.weight"), n, n_v, DType::Q4K);
        let mut alpha = vec![0.0f32; n_v];
        gemv_quant_q8k(&mut alpha, &x, wq(&w, &pre, "ssm_alpha.weight"), n, n_v, DType::Q4K);
        let mut beta = vec![0.0f32; n_v];
        sigmoid(&mut beta, &beta_raw);
        let mut biased = vec![0.0f32; n_v];
        let mut sp = vec![0.0f32; n_v];
        {
            let dt_b: &[f32] =
                bytemuck::cast_slice(w.tensor_checked(&format!("{pre}ssm_dt.bias"), n_v).unwrap());
            let a: &[f32] = bytemuck::cast_slice(w.tensor_checked(&format!("{pre}ssm_a"), n_v).unwrap());
            for h in 0..n_v {
                biased[h] = alpha[h] + dt_b[h];
            }
            softplus(&mut sp, &biased);
            for h in 0..n_v {
                biased[h] = sp[h] * a[h]; // biased ahora guarda gate
            }
        }
        push_node(&mut nodes, il, "beta", beta_raw.clone());
        push_node(&mut nodes, il, "beta_sigmoid", beta.clone());
        push_node(&mut nodes, il, "alpha", alpha.clone());
        push_node(&mut nodes, il, "a_softplus", sp.clone());
        push_node(&mut nodes, il, "gate", biased.clone());

        // 4) conv depthwise sobre el anillo de 3 filas + la actual
        let mut conv_out = vec![0.0f32; d_conv];
        {
            let ring = &mut s.conv[slot];
            let mut win = Vec::with_capacity(cfg.conv_kernel * d_conv);
            win.extend_from_slice(ring);
            win.extend_from_slice(&qkv);
            // GGUF guarda ssm_conv1d.weight como [channels, d_conv] con τ contiguo
            // (ne0 = 4, ver volcado del oráculo: {4, 8192, 1, 1}); ssm_conv indexa
            // kernel[τ*channels + c] → transponer aquí.
            let kraw: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}ssm_conv1d.weight"), cfg.conv_kernel * d_conv)
                    .unwrap(),
            );
            let mut kernel = vec![0.0f32; cfg.conv_kernel * d_conv];
            for c in 0..d_conv {
                for tau in 0..cfg.conv_kernel {
                    kernel[tau * d_conv + c] = kraw[c * cfg.conv_kernel + tau];
                }
            }
            ssm_conv(&mut conv_out, &win, &kernel, cfg.conv_kernel, d_conv);
            // anillo: [r1, r2, x_t]
            ring.copy_within(d_conv.., 0);
            ring[(cfg.conv_kernel - 2) * d_conv..].copy_from_slice(&qkv);
        }
        // mix = silu(conv_out); split q/k/v
        let mut mix = vec![0.0f32; d_conv];
        silu(&mut mix, &conv_out);
        push_node(&mut nodes, il, "conv_output_raw", conv_out.clone());
        push_node(&mut nodes, il, "conv_output_silu", mix.clone());
        let (qk_part, v_part) = mix.split_at_mut(2 * n_qk * hd);
        let (q_conv, k_conv) = qk_part.split_at_mut(n_qk * hd);
        let v_conv = v_part; // sin norma
        push_node(&mut nodes, il, "q_conv", q_conv.to_vec());
        push_node(&mut nodes, il, "k_conv", k_conv.to_vec());
        push_node(&mut nodes, il, "v_conv", v_conv.to_vec());
        l2_norm_rows(q_conv, hd, cfg.rms_eps);
        l2_norm_rows(k_conv, hd, cfg.rms_eps);
        push_node(&mut nodes, il, "q_conv_predelta", q_conv.to_vec());
        push_node(&mut nodes, il, "k_conv_predelta", k_conv.to_vec());
        push_node(&mut nodes, il, "v_conv_predelta", v_conv.to_vec());

        // 5) núcleo GDN fused por v-head: qk-head = h % n_qk (mapeo periódico)
        let mut gdn_out = vec![0.0f32; n];
        {
            let scale = 1.0f32 / (hd as f32).sqrt(); // 1/sqrt(128), sobre la SALIDA
            let state = &mut s.gdn[slot];
            for h in 0..n_v {
                let qk = h % n_qk;
                let (a0, a1) = (h * hd * hd, (h + 1) * hd * hd);
                let (b0, b1) = (h * hd, (h + 1) * hd);
                let (c0, c1) = (qk * hd, (qk + 1) * hd);
                gdn_fused_step(
                    &mut state[a0..a1],
                    &mut gdn_out[b0..b1],
                    &q_conv[c0..c1],
                    &k_conv[c0..c1],
                    &v_conv[b0..b1],
                    biased[h],
                    beta[h],
                    scale,
                );
            }
        }
        push_node(&mut nodes, il, "attn_output", gdn_out.clone());

        // 6) norma con compuerta: rmsnorm_128(gdn_out, ssm_norm) · silu(z)
        let mut y = vec![0.0f32; n];
        {
            let wn: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}ssm_norm.weight"), hd).unwrap(),
            );
            for h in 0..n_v {
                let (a0, a1) = (h * hd, (h + 1) * hd);
                let mut tmp = vec![0.0f32; hd]; // rmsnorm no declara aliasing
                rmsnorm(&mut tmp, &gdn_out[a0..a1], wn, eps);
                y[a0..a1].copy_from_slice(&tmp);
            }
        }
        let mut z_silu = vec![0.0f32; n];
        silu(&mut z_silu, &z);
        for i in 0..n {
            y[i] *= z_silu[i]; // ggml MUL f32
        }
        push_node(&mut nodes, il, "final_output", y.clone());

        // 7) proyección de salida + residual
        let mut attn_out = vec![0.0f32; n];
        gemv_quant_q8k(&mut attn_out, &y, wq(&w, &pre, "ssm_out.weight"), n, n, DType::Q4K);
        let mut residual = vec![0.0f32; n];
        for i in 0..n {
            residual[i] = x_in[i] + attn_out[i];
        }
        push_node(&mut nodes, il, "linear_attn_out", attn_out.clone());
        push_node(&mut nodes, il, "attn_residual", residual.clone());

        // 8) post_attention_norm + FFN SwiGLU paralela + residual
        let f = self.ffn(w, il, &residual, nodes.as_mut().map(|n| &mut **n))?;
        let mut l_out = vec![0.0f32; n];
        for i in 0..n {
            l_out[i] = residual[i] + f[i];
        }
        push_node(&mut nodes, il, "l_out", l_out.clone());
        Ok(RecrOut { l_out, residual, linear_out: attn_out })
    }

    /// Capa de atención completa, algoritmo de docs/QWEN35-FORWARD.md §4.
    fn attn_layer(
        &self,
        s: &mut Qwen35Session,
        il: usize,
        x_in: &[f32],
    ) -> Result<RecrOut, LoadError> {
        let cfg = &self.cfg;
        let n = cfg.n_embd; // 4096
        let nh = cfg.n_head; // 16
        let nhkv = cfg.n_head_kv; // 4
        let hdv = cfg.head_dim_v; // 256 (== head_dim, validado en config)
        let eps = cfg.rms_eps;
        let slot = self.attn_slot[il];
        assert_ne!(slot, usize::MAX, "capa {il} no es de atención");
        let w = self.head.weights();
        let pre = format!("blk.{il}.");

        // 1) attn_norm
        let mut x = vec![0.0f32; n];
        {
            let wn: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}attn_norm.weight"), n).expect("validado en open"),
            );
            rmsnorm(&mut x, x_in, wn, eps);
        }

        // 2) Qfull [8192]: por head h, q en [512h..512h+256), gate en [512h+256..512h+512)
        let mut qfull = vec![0.0f32; 2 * n];
        gemv_quant_q8k(&mut qfull, &x, wq(&w, &pre, "attn_q.weight"), n, 2 * n, DType::Q4K);
        let mut q = vec![0.0f32; n];
        let mut gate = vec![0.0f32; n];
        {
            let qnw: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}attn_q_norm.weight"), hdv).unwrap(),
            );
            for h in 0..nh {
                let (a0, a1) = (h * hdv, (h + 1) * hdv);
                let (s0, s1) = (h * 2 * hdv, h * 2 * hdv + hdv);
                let (g0, g1) = (h * 2 * hdv + hdv, (h + 1) * 2 * hdv);
                let mut tmp = vec![0.0f32; hdv];
                rmsnorm(&mut tmp, &qfull[s0..s1], qnw, eps);
                q[a0..a1].copy_from_slice(&tmp);
                gate[a0..a1].copy_from_slice(&qfull[g0..g1]);
            }
        }

        // 3) K [1024] con norma por kv-head; V [1024] sin norma
        let mut k = vec![0.0f32; nhkv * hdv];
        gemv_quant_q8k(&mut k, &x, wq(&w, &pre, "attn_k.weight"), n, nhkv * hdv, DType::Q4K);
        let mut kn = vec![0.0f32; nhkv * hdv];
        {
            let knw: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}attn_k_norm.weight"), hdv).unwrap(),
            );
            for hkv in 0..nhkv {
                let (a0, a1) = (hkv * hdv, (hkv + 1) * hdv);
                let mut tmp = vec![0.0f32; hdv];
                rmsnorm(&mut tmp, &k[a0..a1], knw, eps);
                kn[a0..a1].copy_from_slice(&tmp);
            }
        }
        let mut v = vec![0.0f32; nhkv * hdv];
        gemv_quant_q8k(
            &mut v,
            &x,
            wq(&w, &pre, "attn_v.weight"),
            n,
            nhkv * hdv,
            self.attn_v_dtype[il],
        );

        // 4) IMROPE sobre q y k (primeros 64 dims por head) y cache K/V en pos
        let theta = cfg.theta_scale();
        let pos = s.position;
        for h in 0..nh {
            rope_apply_imrope(&mut q[h * hdv..(h + 1) * hdv], cfg.n_rot, theta, pos);
        }
        for hkv in 0..nhkv {
            rope_apply_imrope(&mut kn[hkv * hdv..(hkv + 1) * hdv], cfg.n_rot, theta, pos);
        }
        let kv_len = nhkv * hdv;
        s.k_cache[slot][pos * kv_len..(pos + 1) * kv_len].copy_from_slice(&kn);
        s.v_cache[slot][pos * kv_len..(pos + 1) * kv_len].copy_from_slice(&v);

        // 5) atención causal por q-head: kv-head = h/4 (GQA consecutiva)
        let n_ctx = pos + 1;
        let mut attn = vec![0.0f32; n];
        {
            let mut scores = vec![0.0f32; n_ctx];
            let mut soft = vec![0.0f32; n_ctx];
            for h in 0..nh {
                let kv = h / (nh / nhkv);
                let (a0, a1) = (h * hdv, (h + 1) * hdv);
                for p in 0..n_ctx {
                    let (s0, s1) = (p * kv_len + kv * hdv, p * kv_len + (kv + 1) * hdv);
                    scores[p] = dot_f32(&q[a0..a1], &s.k_cache[slot][s0..s1]) as f32;
                }
                softmax(&mut soft, &scores, cfg.kq_scale());
                let mut acc = vec![0.0f64; hdv];
                for i in 0..hdv {
                    let mut sum = 0.0f64;
                    for p in 0..n_ctx {
                        sum += (soft[p] as f64)
                            * (s.v_cache[slot][p * kv_len + kv * hdv + i] as f64);
                    }
                    acc[i] = sum;
                }
                for i in 0..hdv {
                    attn[a0 + i] = acc[i] as f32;
                }
            }
        }

        // 6) compuerta: attn · sigmoid(gate)
        let mut gated = vec![0.0f32; n];
        {
            let mut gs = vec![0.0f32; n];
            sigmoid(&mut gs, &gate);
            for i in 0..n {
                gated[i] = attn[i] * gs[i];
            }
        }

        // 7) proyección de salida + residual + FFN (idéntico a la recurrente)
        let mut attn_out = vec![0.0f32; n];
        gemv_quant_q8k(&mut attn_out, &gated, wq(&w, &pre, "attn_output.weight"), n, n, DType::Q4K);
        let mut residual = vec![0.0f32; n];
        for i in 0..n {
            residual[i] = x_in[i] + attn_out[i];
        }
        let f = self.ffn(w, il, &residual, None)?;
        let mut l_out = vec![0.0f32; n];
        for i in 0..n {
            l_out[i] = residual[i] + f[i];
        }
        Ok(RecrOut { l_out, residual, linear_out: vec![0.0f32; n] })
    }

    /// post_attention_norm + FFN SwiGLU paralela: `f = ffn_down · (silu(ffn_gate·p) * ffn_up·p)`.
    fn ffn(
        &self,
        w: &MappedWeights,
        il: usize,
        residual: &[f32],
        mut nodes: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Result<Vec<f32>, LoadError> {
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let eps = cfg.rms_eps;
        let pre = format!("blk.{il}.");
        let mut p = vec![0.0f32; n];
        {
            let wn: &[f32] = bytemuck::cast_slice(
                w.tensor_checked(&format!("{pre}post_attention_norm.weight"), n)
                    .expect("validado en open"),
            );
            rmsnorm(&mut p, residual, wn, eps);
        }
        push_node(&mut nodes, il, "attn_post_norm", p.clone());
        let mut g = vec![0.0f32; n_ff];
        gemv_quant_q8k(&mut g, &p, wq(w, &pre, "ffn_gate.weight"), n, n_ff, DType::Q4K);
        let mut u = vec![0.0f32; n_ff];
        gemv_quant_q8k(&mut u, &p, wq(w, &pre, "ffn_up.weight"), n, n_ff, DType::Q4K);
        let mut sw = vec![0.0f32; n_ff];
        swiglu(&mut sw, &g, &u);
        push_node(&mut nodes, il, "ffn_gate", g.clone());
        push_node(&mut nodes, il, "ffn_up", u.clone());
        push_node(&mut nodes, il, "ffn_swiglu", sw.clone());
        let mut f = vec![0.0f32; n];
        gemv_quant_q8k(&mut f, &sw, wq(w, &pre, "ffn_down.weight"), n_ff, n, self.ffn_down_dtype[il]);
        push_node(&mut nodes, il, "ffn_out", f.clone());
        Ok(f)
    }
}

struct RecrOut {
    l_out: Vec<f32>,
    residual: Vec<f32>,
    linear_out: Vec<f32>,
}

/// Peso cuantizado empaquetado: bytes crudos del mmap (validado en `open`).
fn wq<'a>(w: &'a MappedWeights, pre: &str, name: &str) -> &'a [u8] {
    w.tensor(&format!("{pre}{name}")).expect("validado en open")
}

/// Empuja un nodo capturado con sufijo de capa (`nombre-{il}`, igual que los
/// cbs del volcado): `compare-nodes-oracle.py` empareja por nombre exacto.
fn push_node(
    nodes: &mut Option<&mut Vec<(String, Vec<f32>)>>,
    il: usize,
    name: &str,
    v: Vec<f32>,
) {
    if let Some(ns) = nodes {
        ns.push((format!("{name}-{il}"), v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_kind_pattern_is_3_of_4_recurrent() {
        // Sin modelo: el patrón viene de Qwen35Config (ya testeado en
        // unltd-architectures); aquí se fija el uso que hace el forward.
        let cfg = dummy_cfg();
        let mut n_recr = 0;
        let mut n_attn = 0;
        for il in 0..cfg.n_layer {
            if cfg.is_full_attn(il) {
                n_attn += 1;
            } else {
                n_recr += 1;
            }
        }
        assert_eq!((n_recr, n_attn), (24, 8));
    }

    #[test]
    fn layer_dump_push_layout_is_layer_major() {
        let n = 4usize;
        let mut d = LayerDump {
            attn_residual: vec![0.0f32; 3 * n],
            l_out: vec![0.0f32; 3 * n],
            linear_attn_out: vec![0.0f32; 3 * n],
        };
        let l1 = vec![1.0f32; n];
        let r1 = vec![2.0f32; n];
        let c1 = vec![3.0f32; n];
        d.push(2, &l1, &r1, &c1);
        assert_eq!(&d.l_out[2 * n..3 * n], &l1[..]);
        assert_eq!(&d.attn_residual[2 * n..3 * n], &r1[..]);
        assert_eq!(&d.linear_attn_out[2 * n..3 * n], &c1[..]);
        // capas 0 y 1 intactas
        assert!(d.l_out[..2 * n].iter().all(|&v| v == 0.0));
    }

    /// Config con los valores de ornith para tests que no tocan el archivo.
    fn dummy_cfg() -> Qwen35Config {
        Qwen35Config {
            n_layer: 32,
            n_embd: 4096,
            n_ff: 12288,
            n_head: 16,
            n_head_kv: 4,
            head_dim: 256,
            head_dim_v: 256,
            n_rot: 64,
            freq_base: 1e7,
            rms_eps: 1e-6,
            conv_kernel: 4,
            state_size: 128,
            group_count: 16,
            time_step_rank: 32,
            d_inner: 4096,
            full_attn_interval: 4,
            rope_sections: vec![11, 11, 10, 0],
            context_length: 262144,
        }
    }
}
