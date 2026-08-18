//! Config del checkpoint qwen3.5 ("qwen35", modelo Ornith 9B): híbrido de
//! capas GatedDeltaNet (recurrentes, 3 de cada 4) y atención completa (1 de cada 4).
//!
//! Fuente de verdad de la arquitectura: `src/models/qwen35.cpp` + `delta-net-base.cpp`
//! de llama.cpp (revisión completa, 2026-08). Cada campo de abajo mapea a una clave
//! GGUF verificada contra el archivo real (`unltd-cli inspect`), y cada valor derivado
//! mapea a la fórmula exacta de la fuente.
//!
//! Política (heredada de k3_cfg.h, docs/AUDIT.md §3.3): campo ausente = ERROR, nunca
//! default. Todas las claves ausentes se acumulan en un solo `MissingConfig`; los
//! checks estructurales corren después del parseo y antes de tocar un byte de pesos.
//!
//! NOTA: esto es la CONFIG cruda (Fase 5). El `Adapter` completo que emite la IR
//! (`ModelSpec` con capas SSM/lineales) llega en Fase 6: la IR actual solo describe
//! atención estándar y debe crecer con un `AttnKind::GatedDeltaNet`.

use unltd_core::LoadError;
use unltd_model_loader::gguf::{GgufArray, GgufReader, GgufValue};

pub const ARCH: &str = "qwen35";

/// Sufijos de clave GGUF (sin el prefijo `{ARCH}.`), con su tipo esperado.
const REQUIRED: &[(&str, &str)] = &[
    ("block_count", "u32"),
    ("context_length", "u32"),
    ("embedding_length", "u32"),
    ("feed_forward_length", "u32"),
    ("attention.head_count", "u32"),
    ("attention.head_count_kv", "u32"),
    ("attention.key_length", "u32"),
    ("attention.value_length", "u32"),
    ("attention.layer_norm_rms_epsilon", "f32"),
    ("rope.dimension_count", "u32"),
    ("rope.dimension_sections", "i32[4]"),
    ("rope.freq_base", "f32"),
    ("ssm.conv_kernel", "u32"),
    ("ssm.state_size", "u32"),
    ("ssm.group_count", "u32"),
    ("ssm.time_step_rank", "u32"),
    ("ssm.inner_size", "u32"),
    ("full_attention_interval", "u32"),
];

/// Hiperparámetros verificados del checkpoint qwen3.5.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    /// Cabezas de la ATENCIÓN COMPLETA (capas (i+1) % interval == 0).
    pub n_head: usize,
    /// KV heads de la atención completa (GQA: 16 q → 4 kv).
    pub n_head_kv: usize,
    /// Head dim de la atención completa = `attention.key_length`.
    pub head_dim: usize,
    /// `attention.value_length`: head dim de V de la atención completa.
    pub head_dim_v: usize,
    /// Dims que rota el IMROPE (dentro de cada head): `rope.dimension_count`.
    pub n_rot: usize,
    /// `rope.freq_base` (f32 del GGUF, 1e7 en ornith).
    pub freq_base: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `ssm.conv_kernel` (4): ventana de la conv1d del estado recurrente.
    pub conv_kernel: usize,
    /// `ssm.state_size` (128): lado del estado GatedDeltaNet.
    pub state_size: usize,
    /// `ssm.group_count` (16): heads de q/k lineales.
    pub group_count: usize,
    /// `ssm.time_step_rank` (32): rango de la proyección alpha/dt.
    pub time_step_rank: usize,
    /// `ssm.inner_size` (4096): ancho interno del camino lineal (v = d_inner).
    pub d_inner: usize,
    /// `full_attention_interval` (4): capa i es completa sii (i+1) % interval == 0.
    pub full_attn_interval: usize,
    /// `rope.dimension_sections` (i32[4]): partición de dims por sección IMROPE
    /// (t, h, w, e). n_pos_per_embd = len (4).
    pub rope_sections: Vec<i32>,
    pub context_length: usize,
}

impl Qwen35Config {
    pub fn from_gguf(r: &GgufReader) -> Result<Self, LoadError> {
        // 1. Arquitectura: un adapter qwen3.5 sobre otro archivo es un ERROR.
        match r.get_str("general.architecture") {
            Some(ARCH) => {}
            Some(other) => {
                return Err(LoadError::corrupt(format!(
                    "architecture '{other}': este adapter solo acepta '{ARCH}'"
                )));
            }
            None => {
                return Err(LoadError::MissingConfig {
                    n: 1,
                    fields: "general.architecture".to_string(),
                });
            }
        }

        // 2. Claves requeridas: acumular TODAS las ausentes y TODOS los tipos
        //    equivocados antes de fallar (una corrida, un error).
        let mut missing: Vec<&str> = Vec::new();
        let mut bad_type: Vec<(String, String)> = Vec::new();
        for (suffix, want) in REQUIRED {
            let key = format!("{ARCH}.{suffix}");
            match r.get(&key) {
                None => missing.push(suffix),
                Some(v) => {
                    let got = type_name(v);
                    if got != *want {
                        bad_type.push((key, got));
                    }
                }
            }
        }
        if !missing.is_empty() {
            return Err(LoadError::MissingConfig {
                n: missing.len(),
                fields: missing
                    .iter()
                    .map(|s| format!("{ARCH}.{s}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        if !bad_type.is_empty() {
            let first = &bad_type[0];
            return Err(LoadError::corrupt(format!(
                "clave '{}': esperaba u32/i32[4]/f32, hay {}",
                first.0, first.1
            )));
        }

        // 3. Extracción (ya verificada la presencia y el tipo).
        let u = |suffix: &str| match r.get(&format!("{ARCH}.{suffix}")) {
            Some(GgufValue::U32(v)) => *v as usize,
            _ => unreachable!("validado arriba"),
        };
        let f = |suffix: &str| match r.get(&format!("{ARCH}.{suffix}")) {
            Some(GgufValue::F32(v)) => *v,
            _ => unreachable!("validado arriba"),
        };
        let sections = match r.get(&format!("{ARCH}.rope.dimension_sections")) {
            Some(GgufValue::Arr(GgufArray::I32(v))) => v.clone(),
            _ => unreachable!("validado arriba"),
        };

        let cfg = Qwen35Config {
            n_layer: u("block_count"),
            n_embd: u("embedding_length"),
            n_ff: u("feed_forward_length"),
            n_head: u("attention.head_count"),
            n_head_kv: u("attention.head_count_kv"),
            head_dim: u("attention.key_length"),
            head_dim_v: u("attention.value_length"),
            n_rot: u("rope.dimension_count"),
            freq_base: f("rope.freq_base"),
            rms_eps: f("attention.layer_norm_rms_epsilon"),
            conv_kernel: u("ssm.conv_kernel"),
            state_size: u("ssm.state_size"),
            group_count: u("ssm.group_count"),
            time_step_rank: u("ssm.time_step_rank"),
            d_inner: u("ssm.inner_size"),
            full_attn_interval: u("full_attention_interval"),
            rope_sections: sections,
            context_length: u("context_length"),
        };

        // 4. Checks estructurales (refuse rather than guess).
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<(), LoadError> {
        let c = |cond: bool, msg: String| {
            if cond { Ok(()) } else { Err(LoadError::StructCheck(msg)) }
        };
        c(self.n_layer >= 1, "block_count = 0".into())?;
        c(self.n_embd > 0 && self.n_ff > 0, "dims de embedding/FFN nulos".into())?;
        c(self.n_head >= 1, "attention.head_count = 0".into())?;
        c(
            self.n_head_kv >= 1 && self.n_head % self.n_head_kv == 0,
            format!("GQA inválida: n_head={} n_head_kv={}", self.n_head, self.n_head_kv),
        )?;
        c(
            self.n_head * self.head_dim == self.n_embd,
            format!(
                "wq tiene {} columnas Q, n_embd={} (head_count × key_length debe ser exacto)",
                self.n_head * self.head_dim,
                self.n_embd
            ),
        )?;
        c(
            self.head_dim == self.head_dim_v,
            format!(
                "attention.key_length={} != attention.value_length={} (este adapter asume heads k/v iguales)",
                self.head_dim, self.head_dim_v
            ),
        )?;
        c(
            self.n_rot % 2 == 0 && self.n_rot <= self.head_dim,
            format!("n_rot={} inválido (par y ≤ head_dim={})", self.n_rot, self.head_dim),
        )?;
        c(self.freq_base > 0.0, "rope.freq_base <= 0".into())?;
        c(self.rms_eps > 0.0, "layer_norm_rms_epsilon <= 0".into())?;
        c(self.conv_kernel >= 2, "ssm.conv_kernel < 2".into())?;
        c(self.group_count >= 1, "ssm.group_count = 0".into())?;
        c(
            self.d_inner % (2 * self.group_count) == 0,
            format!("d_inner={} no es múltiplo de 2×group_count={}", self.d_inner, self.group_count),
        )?;
        c(self.time_step_rank >= 1, "ssm.time_step_rank = 0".into())?;
        c(
            self.state_size == self.head_dim_linear(),
            format!(
                "state_size={} != head dim lineal {} (este adapter asume estado cuadrado por head)",
                self.state_size,
                self.head_dim_linear()
            ),
        )?;
        c(self.full_attn_interval >= 1, "full_attention_interval = 0".into())?;
        c(
            self.rope_sections.len() == 4,
            format!(
                "rope.dimension_sections tiene {} secciones, IMROPE exige 4 (t,h,w,e)",
                self.rope_sections.len()
            ),
        )?;
        c(self.context_length >= 1, "context_length = 0".into())?;
        Ok(())
    }

    // ---- Derivados (fórmulas exactas de qwen35.cpp / delta-net-base.cpp) ----

    /// Heads de q/k lineales = group_count; v expande al doble.
    pub fn n_qk_heads(&self) -> usize {
        self.group_count
    }

    /// Head dim del camino lineal: q_conv/k_conv = d_inner/2 con group_count
    /// heads → d_inner / (2·group_count). Ornith: 4096/32 = 128.
    pub fn head_dim_linear(&self) -> usize {
        self.d_inner / (2 * self.group_count)
    }

    /// Heads de v lineales: d_inner / head_dim = 2·group_count. Ornith: 32.
    pub fn n_v_heads(&self) -> usize {
        self.d_inner / self.head_dim_linear()
    }

    /// Ancho de la conv1d: concat(q, k, v) = d_inner/2 + d_inner/2 + d_inner.
    pub fn d_conv(&self) -> usize {
        2 * self.d_inner
    }

    /// Escala del softmax de la atención completa: 1/sqrt(head_dim).
    /// (f_attention_scale ausente en el GGUF → derivada, como en llama.cpp.)
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Constante de frecuencia del IMROPE: base^(-2/n_rot), computada en f32
    /// como la referencia (luego se ensancha a f64 en el kernel).
    pub fn theta_scale(&self) -> f64 {
        self.freq_base.powf(-2.0f32 / self.n_rot as f32) as f64
    }

    /// La capa i (0-based) usa atención completa sii (i+1) % interval == 0.
    /// Ornith: 3, 7, 11, 15, 19, 23, 27, 31.
    pub fn is_full_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attn_interval == 0
    }
}

// ---------------------------------------------------------------------------
// Tests: fixtures GGUF sintéticas (solo metadata, 0 tensores) con los valores
// REALES de ornith; negativas de refusal (ausencias acumuladas, tipo erróneo,
// arquitectura errónea) y checks estructurales.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use unltd_model_loader::gguf::parse;

    /// Escritor GGUF mínimo (little-endian, solo metadata). Tipo de valor: 4=u32,
    /// 5=i32, 6=f32, 8=string, 9=array (según la spec GGUF v3).
    struct W(Vec<u8>);

    impl W {
        fn new() -> Self {
            W(Vec::new())
        }
        fn u32(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(mut self, v: u64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn raw(mut self, b: &[u8]) -> Self {
            self.0.extend_from_slice(b);
            self
        }
        fn str(mut self, s: &str) -> Self {
            self = self.u64(s.len() as u64);
            self.raw(s.as_bytes())
        }
        fn kv_u32(self, key: &str, v: u32) -> Self {
            self.str(key).u32(4).u32(v)
        }
        fn kv_f32(self, key: &str, v: f32) -> Self {
            self.str(key).u32(6).raw(&v.to_le_bytes())
        }
        fn kv_i32_arr(self, key: &str, v: &[i32]) -> Self {
            let mut w = self.str(key).u32(9).u32(5).u64(v.len() as u64);
            for x in v {
                w = w.raw(&x.to_le_bytes());
            }
            w
        }
        /// Rellena hasta el inicio de datos (align 32) — archivo sin tensores.
        fn finish(mut self) -> Vec<u8> {
            let data_start = (self.0.len() as u64 + 31) / 32 * 32;
            self.0.resize(data_start as usize, 0);
            self.0
        }
    }

    /// Valores reales de ornith; mutables para tests de checks estructurales.
    #[derive(Clone, Copy)]
    struct Fx {
        n_layer: u32,
        n_embd: u32,
        n_head: u32,
        key_len: u32,
        val_len: u32,
        n_rot: u32,
        state: u32,
        group: u32,
        d_inner: u32,
        interval: u32,
    }

    impl Default for Fx {
        fn default() -> Self {
            Fx {
                n_layer: 32,
                n_embd: 4096,
                n_head: 16,
                key_len: 256,
                val_len: 256,
                n_rot: 64,
                state: 128,
                group: 16,
                d_inner: 4096,
                interval: 4,
            }
        }
    }

    fn fixture_with(f: Fx) -> Vec<u8> {
        W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0) // n_tensors
            .u64(19) // n_kv
            .str("general.architecture")
            .u32(8)
            .str("qwen35")
            .kv_u32("qwen35.block_count", f.n_layer)
            .kv_u32("qwen35.context_length", 262144)
            .kv_u32("qwen35.embedding_length", f.n_embd)
            .kv_u32("qwen35.feed_forward_length", 12288)
            .kv_u32("qwen35.attention.head_count", f.n_head)
            .kv_u32("qwen35.attention.head_count_kv", 4)
            .kv_u32("qwen35.attention.key_length", f.key_len)
            .kv_u32("qwen35.attention.value_length", f.val_len)
            .kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
            .kv_u32("qwen35.rope.dimension_count", f.n_rot)
            .kv_i32_arr("qwen35.rope.dimension_sections", &[11, 11, 10, 0])
            .kv_f32("qwen35.rope.freq_base", 1e7)
            .kv_u32("qwen35.ssm.conv_kernel", 4)
            .kv_u32("qwen35.ssm.state_size", f.state)
            .kv_u32("qwen35.ssm.group_count", f.group)
            .kv_u32("qwen35.ssm.time_step_rank", 32)
            .kv_u32("qwen35.ssm.inner_size", f.d_inner)
            .kv_u32("qwen35.full_attention_interval", f.interval)
            .finish()
    }

    fn parse_cfg(b: &[u8]) -> Result<Qwen35Config, LoadError> {
        let r = parse(&mut Cursor::new(b), b.len() as u64).unwrap();
        Qwen35Config::from_gguf(&r)
    }

    #[test]
    fn parses_ornith_values_and_derived() {
        let cfg = parse_cfg(&fixture_with(Fx::default())).unwrap();
        assert_eq!(cfg.n_layer, 32);
        assert_eq!(cfg.n_embd, 4096);
        assert_eq!(cfg.n_ff, 12288);
        assert_eq!(cfg.n_head, 16);
        assert_eq!(cfg.n_head_kv, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.head_dim_v, 256);
        assert_eq!(cfg.n_rot, 64);
        assert_eq!(cfg.freq_base, 1e7f32);
        assert_eq!(cfg.rms_eps, 1e-6f32);
        assert_eq!(cfg.conv_kernel, 4);
        assert_eq!(cfg.state_size, 128);
        assert_eq!(cfg.group_count, 16);
        assert_eq!(cfg.time_step_rank, 32);
        assert_eq!(cfg.d_inner, 4096);
        assert_eq!(cfg.full_attn_interval, 4);
        assert_eq!(cfg.rope_sections, vec![11, 11, 10, 0]);
        assert_eq!(cfg.context_length, 262144);

        // Derivados (fórmulas de qwen35.cpp)
        assert_eq!(cfg.n_qk_heads(), 16);
        assert_eq!(cfg.head_dim_linear(), 128);
        assert_eq!(cfg.n_v_heads(), 32);
        assert_eq!(cfg.d_conv(), 8192);
        assert_eq!(cfg.kq_scale(), 0.0625); // 1/sqrt(256) exacto
        let theta = (1e7f32).powf(-2.0f32 / 64.0f32) as f64; // f32 como la referencia
        assert_eq!(cfg.theta_scale().to_bits(), theta.to_bits());

        // Patrón de capas: (i+1) % 4 == 0 → atención completa
        for i in [0, 1, 2, 4, 5, 6, 8] {
            assert!(!cfg.is_full_attn(i), "capa {i} debe ser lineal");
        }
        for i in [3, 7, 11, 15, 19, 23, 27, 31] {
            assert!(cfg.is_full_attn(i), "capa {i} debe ser completa");
        }
    }

    #[test]
    fn missing_keys_accumulate_in_one_error() {
        // Solo la arquitectura presente: TODAS las claves deben listarse juntas.
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0)
            .u64(1)
            .str("general.architecture")
            .u32(8)
            .str("qwen35")
            .finish();
        match parse_cfg(&b) {
            Err(LoadError::MissingConfig { n, fields }) => {
                assert_eq!(n, 18);
                assert!(fields.contains("qwen35.block_count"), "fields: {fields}");
                assert!(fields.contains("qwen35.ssm.conv_kernel"), "fields: {fields}");
                assert!(fields.contains("qwen35.rope.dimension_sections"), "fields: {fields}");
                assert!(fields.contains("qwen35.full_attention_interval"), "fields: {fields}");
            }
            other => panic!("esperaba MissingConfig, obtuve {other:?}"),
        }

        // Dos claves ausentes → n=2 y las dos listadas.
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0)
            .u64(17)
            .str("general.architecture")
            .u32(8)
            .str("qwen35")
            // sin qwen35.ssm.conv_kernel ni qwen35.rope.freq_base
            .kv_u32("qwen35.block_count", 32)
            .kv_u32("qwen35.context_length", 262144)
            .kv_u32("qwen35.embedding_length", 4096)
            .kv_u32("qwen35.feed_forward_length", 12288)
            .kv_u32("qwen35.attention.head_count", 16)
            .kv_u32("qwen35.attention.head_count_kv", 4)
            .kv_u32("qwen35.attention.key_length", 256)
            .kv_u32("qwen35.attention.value_length", 256)
            .kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
            .kv_u32("qwen35.rope.dimension_count", 64)
            .kv_i32_arr("qwen35.rope.dimension_sections", &[11, 11, 10, 0])
            .kv_u32("qwen35.ssm.state_size", 128)
            .kv_u32("qwen35.ssm.group_count", 16)
            .kv_u32("qwen35.ssm.time_step_rank", 32)
            .kv_u32("qwen35.ssm.inner_size", 4096)
            .kv_u32("qwen35.full_attention_interval", 4)
            .finish();
        match parse_cfg(&b) {
            Err(LoadError::MissingConfig { n, fields }) => {
                assert_eq!(n, 2);
                assert!(fields.contains("qwen35.ssm.conv_kernel"), "fields: {fields}");
                assert!(fields.contains("qwen35.rope.freq_base"), "fields: {fields}");
            }
            other => panic!("esperaba MissingConfig, obtuve {other:?}"),
        }
    }

    #[test]
    fn wrong_type_and_wrong_arch_refuse() {
        // block_count como f32 (todas las demás claves presentes y bien tipadas)
        // → corrupt nombrando la clave. OJO: las ausencias tienen precedencia
        // sobre los tipos (una corrida, un error); por eso aquí no falta ninguna.
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0)
            .u64(19)
            .str("general.architecture")
            .u32(8)
            .str("qwen35")
            .kv_f32("qwen35.block_count", 32.0) // tipo erróneo
            .kv_u32("qwen35.context_length", 262144)
            .kv_u32("qwen35.embedding_length", 4096)
            .kv_u32("qwen35.feed_forward_length", 12288)
            .kv_u32("qwen35.attention.head_count", 16)
            .kv_u32("qwen35.attention.head_count_kv", 4)
            .kv_u32("qwen35.attention.key_length", 256)
            .kv_u32("qwen35.attention.value_length", 256)
            .kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
            .kv_u32("qwen35.rope.dimension_count", 64)
            .kv_i32_arr("qwen35.rope.dimension_sections", &[11, 11, 10, 0])
            .kv_f32("qwen35.rope.freq_base", 1e7)
            .kv_u32("qwen35.ssm.conv_kernel", 4)
            .kv_u32("qwen35.ssm.state_size", 128)
            .kv_u32("qwen35.ssm.group_count", 16)
            .kv_u32("qwen35.ssm.time_step_rank", 32)
            .kv_u32("qwen35.ssm.inner_size", 4096)
            .kv_u32("qwen35.full_attention_interval", 4)
            .finish();
        match parse_cfg(&b) {
            Err(LoadError::BadFile(msg)) => assert!(msg.contains("qwen35.block_count"), "msg: {msg}"),
            other => panic!("esperaba BadFile, obtuve {other:?}"),
        }

        // arquitectura ajena → corrupt nombrando qwen35
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0)
            .u64(1)
            .str("general.architecture")
            .u32(8)
            .str("llama")
            .finish();
        match parse_cfg(&b) {
            Err(LoadError::BadFile(msg)) => assert!(msg.contains("qwen35"), "msg: {msg}"),
            other => panic!("esperaba BadFile, obtuve {other:?}"),
        }
    }

    #[test]
    fn structural_checks_refuse() {
        // wq: head_count × key_length debe cubrir n_embd exacto
        match parse_cfg(&fixture_with(Fx { n_embd: 4097, ..Default::default() })) {
            Err(LoadError::StructCheck(msg)) => assert!(msg.contains("wq"), "msg: {msg}"),
            other => panic!("esperaba StructCheck, obtuve {other:?}"),
        }

        // state_size debe ser el head dim lineal
        match parse_cfg(&fixture_with(Fx { state: 64, ..Default::default() })) {
            Err(LoadError::StructCheck(msg)) => assert!(msg.contains("state_size"), "msg: {msg}"),
            other => panic!("esperaba StructCheck, obtuve {other:?}"),
        }

        // n_rot debe ser par
        match parse_cfg(&fixture_with(Fx { n_rot: 63, ..Default::default() })) {
            Err(LoadError::StructCheck(msg)) => assert!(msg.contains("n_rot"), "msg: {msg}"),
            other => panic!("esperaba StructCheck, obtuve {other:?}"),
        }

        // key_length != value_length
        match parse_cfg(&fixture_with(Fx { val_len: 128, ..Default::default() })) {
            Err(LoadError::StructCheck(msg)) => assert!(msg.contains("value_length"), "msg: {msg}"),
            other => panic!("esperaba StructCheck, obtuve {other:?}"),
        }

        // d_inner no divisible por 2×group_count
        match parse_cfg(&fixture_with(Fx { d_inner: 4097, ..Default::default() })) {
            Err(LoadError::StructCheck(msg)) => assert!(msg.contains("d_inner"), "msg: {msg}"),
            other => panic!("esperaba StructCheck, obtuve {other:?}"),
        }
    }
}

/// Nombre corto del tipo de un valor, para mensajes de error y para el check de
/// tipo esperado. Los arrays incluyen su largo (`i32[4]`): el check de tipo
/// valida también el tamaño.
fn type_name(v: &GgufValue) -> String {
    let scalar = |s: &'static str| s.to_string();
    let arr = |t: &'static str, n: usize| format!("{t}[{n}]");
    match v {
        GgufValue::U8(_) => scalar("u8"),
        GgufValue::I8(_) => scalar("i8"),
        GgufValue::U16(_) => scalar("u16"),
        GgufValue::I16(_) => scalar("i16"),
        GgufValue::U32(_) => scalar("u32"),
        GgufValue::I32(_) => scalar("i32"),
        GgufValue::F32(_) => scalar("f32"),
        GgufValue::F64(_) => scalar("f64"),
        GgufValue::U64(_) => scalar("u64"),
        GgufValue::I64(_) => scalar("i64"),
        GgufValue::Bool(_) => scalar("bool"),
        GgufValue::Str(_) => scalar("string"),
        GgufValue::Arr(a) => match a {
            GgufArray::U8(v) => arr("u8", v.len()),
            GgufArray::I8(v) => arr("i8", v.len()),
            GgufArray::U16(v) => arr("u16", v.len()),
            GgufArray::I16(v) => arr("i16", v.len()),
            GgufArray::U32(v) => arr("u32", v.len()),
            GgufArray::I32(v) => arr("i32", v.len()),
            GgufArray::F32(v) => arr("f32", v.len()),
            GgufArray::F64(v) => arr("f64", v.len()),
            GgufArray::U64(v) => arr("u64", v.len()),
            GgufArray::I64(v) => arr("i64", v.len()),
            GgufArray::Bool(v) => arr("bool", v.len()),
            GgufArray::Str(v) => arr("string", v.len()),
        },
    }
}

