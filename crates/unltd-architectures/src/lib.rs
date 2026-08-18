//! La IR de arquitecturas: lo único que el motor de ejecución conoce.
//! Ver `docs/ARCHITECTURE.md` §4.
//!
//! Los adapters **describen, no ejecutan**. Cada adapter (llama, qwen2, qwen3,
//! qwen3-moe, mixtral, deepseek2-mla, …) valida la config con refusal, construye el
//! mapa nombre-de-tensor → rol con chequeo de element count ANTES de leer bytes
//! (el patrón `plan_layer` de kimi-k3-in-c), y emite un `ModelSpec`.
//!
//! Regla de diseño: el motor hace `match` sobre estos enums. Agregar una arquitectura =
//! agregar un adapter; nunca tocar el motor.

use unltd_core::WeightId;

/// Tipos de atención. La MLA tiene DOS semánticas en el ecosistema y la IR las distingue
/// explícitamente en vez de fusionarlas falsamente:
///
/// - `MlaDeepSeek`: comprime k/v a un latente (`kv_lora`), rota la parte rope (64 dims)
///   y cachea el latente comprimido (DeepSeek-V2/V3).
/// - `MlaK3NoPe`: las dims rope existen y se cachean pero NO se rotan (NoPE), k/v
///   expandidos en cache (Kimi K3). No en v1; el enum existe para no mentir después.
#[derive(Debug, Clone, PartialEq)]
pub enum AttnKind {
    Mha,
    Gqa { kv_groups: u32 },
    MlaDeepSeek { kv_lora: u32, qk_rope: u32, qk_nope: u32, v_head: u32 },
    MlaK3NoPe { qk_nope: u32, qk_rope: u32, v_head: u32 },
}

/// RoPE: variantes y escalado. Las frecuencias se precalculan una vez por modelo.
#[derive(Debug, Clone, PartialEq)]
pub enum RoPeKind {
    None,
    Llama { theta: f32, dims: u32 },
    /// YaRN (DeepSeek V2/V3): factor de escala + mscale por head-dim.
    LlamaYaRn { theta: f32, dims: u32, factor: f32, original_max: u32, mscale: f32 },
    NeoX { theta: f32, dims: u32 },
}

/// FFN: densa con su activación, o MoE con su ruteo.
#[derive(Debug, Clone, PartialEq)]
pub enum FfnKind {
    SwiGlu,
    GeGluTanh,
    /// ReLU^2 (PhiMoE). OJO con el mito: Qwen3 usa silu en TODOS sus configs
    /// verificados — nunca ReLU^2. Un adapter qwen3 con ReLU^2 decodifica texto
    /// fluido desde la arquitectura equivocada.
    Relu2,
    Moe {
        n_experts: u32,
        topk: u32,
        n_shared: u32,
        inter: u32,
        /// Renormalizar los pesos del top-k antes de escalar (DeepSeek-V3 y
        /// Qwen3-MoE: norm_topk_prob = true). Trap de corrección #1.
        norm_topk: bool,
        /// Gate: sigmoid sin bias + selección (DeepSeek-V3 noaux_tc; Qwen3-MoE
        /// sigmoid + renorm) vs softmax + bias (Qwen2-MoE, Mixtral, V2-Lite).
        /// Trap de corrección #2.
        sigmoid_gate: bool,
        routed_scaling: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormKind {
    RmsNorm { eps: f64 },
    LayerNorm { eps: f64 },
    /// RMSNorm extra sobre q/k antes de RoPE. Presente en Llama-4 (use_qk_norm),
    /// OLMo-2 (QK-norm) y Gemma (query_pre_attn_scalar). OJO con el mito: NI Qwen2.5
    /// NI Qwen3 la usan (configs verificados — campos ausentes). Atribuirla a Qwen
    /// produce un modelo que corre, fluido y equivocado.
    QkRmsNorm { eps: f64 },
}

/// Una capa del decoder. `attn: None` describe la capa densa 0 del linaje K3/DeepSeek
/// (`first_k_dense_replace`); en Llama/Qwen/Mistral todas las capas tienen atención.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub input_norm: NormKind,
    pub attn: Option<AttnSpec>,
    pub post_norm: NormKind,
    pub ffn: FfnKind,
    pub has_attn_bias: bool,
}

#[derive(Debug, Clone)]
pub struct AttnSpec {
    pub kind: AttnKind,
    pub n_heads: u32,
    pub head_dim: u32,
    /// Qwen2.5 tiene bias en q/k/v; Qwen3 no. Cambiar esto es cambiar el modelo.
    pub qk_norm: Option<NormKind>,
    pub rope: RoPeKind,
    /// Pesos: q/k/v/o (proyecciones de atención), referidos por WeightId.
    pub w_q: WeightId,
    pub w_k: WeightId,
    pub w_v: WeightId,
    pub w_o: WeightId,
}

/// La IR completa de un modelo. El motor ejecuta esto y nada más.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub vocab: u32,
    pub hidden: u32,
    pub layers: Vec<LayerSpec>,
    pub embed: WeightId,
    pub final_norm: WeightId,
    pub lm_head: WeightId,
    /// Qwen/Mistral/Gemma: true; el lm_head comparte (o no se carga) el embedding.
    pub tie_embeddings: bool,
    pub eos_ids: Vec<u32>,
    pub rope_global: RoPeKind,
    /// Norma del embedding: presente en algunos modelos, ausente en otros
    /// (Llama no la tiene; Qwen no; Gemma no; MiniCPM sí). Ausente ≠ default:
    /// el adapter lo decide desde la config del checkpoint.
    pub embed_norm: Option<NormKind>,
}

/// Lo que un adapter produce: la IR + el mapa de pesos que el MemoryManager consume.
/// Un adapter NUNCA abre archivos: resuelve nombres y formas contra el índice de
/// tensores que se le pasa.
pub trait Adapter {
    fn name(&self) -> &'static str;

    /// Valida la config del checkpoint y emite la IR. `config` es el JSON crudo:
    /// la política "campo ausente = error, nunca default" vive en esta llamada
    /// (acumulando TODAS las claves faltantes en un solo `LoadError::MissingConfig`).
    fn build_spec(&self, config: &serde_json::Value) -> Result<ModelSpec, unltd_core::LoadError>;
}
