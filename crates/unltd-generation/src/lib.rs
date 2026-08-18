//! Motor de ejecución: forward sobre la IR, KV cache, decode loop, sampler.
//! Ver `docs/ARCHITECTURE.md` §5.
//!
//! El forward es: `for layer in spec.layers { layer.execute(&mut state, &memory) }`.
//! El motor hace `match` sobre los enums de la IR y NADA más: no conoce arquitecturas
//! concretas, no abre archivos (los pesos llegan por el MemoryManager), y no conoce
//! el formato de los pesos (multiplica vistas empaquetadas).
//!
//! Decisiones heredadas de la auditoría:
//!
//! - **Incremental es el camino principal.** Full-recompute (O(T²)) queda como test de
//!   equivalencia: el oráculo exige tokens idénticos entre ambos.
//! - **La KV cache se dimensiona y se chequea ANTES de correr** (bytes/pos × posiciones
//!   vs disponible, negarse con ambos números). Es el único término que crece con el
//!   contexto.
//! - **Greedy es el sampler por defecto.** La salida determinista entre presupuestos de
//!   memoria es una propiedad de la que dependen los tests; sampling (temperatura,
//!   top-p) es opt-in y off por defecto (lección del ROADMAP de K3).
//! - **Un experto que no se pudo leer es un fallo de corrida** (exit code distinto),
//!   nunca un drop silencioso: tokens plausibles con parte del ruteo faltante son
//!   corrupción numérica que parece una corrida buena.

pub mod greedy;
pub mod min_forward;
pub mod qwen35_forward;

pub use greedy::{argmax, GreedyLoop};
pub use min_forward::MinForward;
pub use qwen35_forward::{LayerDump, NodeCapture, Qwen35Forward, Qwen35Session};

use unltd_architectures::ModelSpec;
use unltd_core::{LoadError, WeightId};
use unltd_memory::{MemoryManager, ResidencyClass};
use unltd_tokenizer::{TokenError, Tokenizer};

/// Estado que se lleva entre tokens: KV cache por capa + posición + buffers de scratch.
/// El scratch se pide al motor (`ScratchPlanner`), nunca se calcula a mano (lección
/// K3: el off-by-one silencioso).
pub struct DecodeState {
    pub kv_cache: KvCache,
    pub position: u64,
    pub scratch: ScratchArena,
}

/// KV cache en fp32, pre-dimensionada. GQA cachea por KV head (expandido) para
/// simplicidad; la optimización de cachear por grupo queda anotada como TODO medible.
pub struct KvCache {
    _private: (),
}

impl KvCache {
    /// `bytes_per_pos × positions` contra la memoria disponible; `DoesNotFit` con
    /// ambos números si no entra.
    pub fn plan(_spec: &ModelSpec, _positions: u64) -> Result<u64, LoadError> {
        todo!("Stage 0, ver docs/ROADMAP.md")
    }
}

pub struct ScratchArena {
    _private: (),
}

/// Una corrida completa: carga → prefill → decode.
pub struct Session<'m> {
    pub spec: ModelSpec,
    pub memory: &'m MemoryManager,
    pub tokenizer: Box<dyn Tokenizer>,
    pub state: DecodeState,
}

/// Resuelve la clase de residencia de un peso (por ahora: los expertos van al
/// `ExpertCache`, todo lo demás al tronco o a residente según tamaño).
pub fn residency_of(_spec: &ModelSpec, _id: WeightId) -> ResidencyClass {
    todo!("Stage 0, ver docs/ROADMAP.md")
}

/// Prefill: alimenta los tokens del prompt de una vez (chunked en etapas posteriores).
pub fn prefill(_session: &mut Session<'_>, _ids: &[u32]) -> Result<(), LoadError> {
    todo!("Stage 0, ver docs/ROADMAP.md")
}

/// Un paso de decode: forward de un token, sampler, append al KV.
pub fn decode_step(_session: &mut Session<'_>) -> Result<u32, DecodeError> {
    todo!("Stage 0, ver docs/ROADMAP.md")
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Token(#[from] TokenError),
    #[error("expert load failed: {0} experts dropped; output would be corrupt")]
    ExpertDrop(usize),
    #[error("eos token hit")]
    Eos,
}
