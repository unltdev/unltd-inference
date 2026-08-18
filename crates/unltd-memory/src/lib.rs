//! Gestión de memoria: presupuestos, políticas de residencia, lectores de disco y
//! cache de expertos. El diseño completo está en `docs/MEMORY-DESIGN.md`.
//!
//! Principios (heredados de kimi-k3-in-c, ver `docs/AUDIT.md`):
//!
//! - **La RAM es un dial, no un piso**: el mismo modelo corre con presupuestos
//!   distintos y produce salida idéntica.
//! - **El tráfico por token decide la asignación**: tronco antes que cache de expertos
//!   (1,69× medido a presupuesto total idéntico).
//! - **El plan de memoria se suma entero ANTES de asignar**; si no entra, negarse con
//!   ambos números (`LoadError::DoesNotFit`), nunca un OOM-kill a mitad de corrida.
//! - **Nunca desquantizar para cachear**: los slots guardan bloques empaquetados.
//! - **Hit rate VERDADERO**: `(hits - prefetch_reads) / requests` — el contador `hits`
//!   solo cuenta expertos que el prefetcher trajo del disco microsegundos antes, y
//!   reportarlo crudo es mentir.

use std::path::Path;

use unltd_core::LoadError;

pub mod budget;

pub use budget::{parse_size, MemoryAccounting, ParseSizeError};

/// Presupuesto del motor (pesos + KV), no del proceso. `None` = "lleno" para esa clase.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub total: u64,
    pub trunk: Option<u64>,
    pub expert_cache: Option<u64>,
    pub kv_limit: Option<u64>,
}

/// Clase de residencia de un peso. Decidida por el adapter de arquitectura, no por el
/// usuario (ver MEMORY-DESIGN.md §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyClass {
    /// Embeddings, lm_head, normas chicas: cargados al inicio, nunca evictados.
    AlwaysResident,
    /// Capas densas: recorrido fijo por token → pin prefix + ring de 2 slots.
    Trunk,
    /// Expertos ruteados: acceso dependiente de datos → LRU + prefetch batch.
    RoutedExpert,
}

/// El plan es un pronóstico. El PEAK RSS es el veredicto. Ambos se imprimen y el motor
/// declara cuál citar (lección de k3_run.c: el plan subestima; el RSS se mide).
pub struct MemoryManager {
    _private: (),
}

impl MemoryManager {
    /// Suma TODO (trunk, modelo siempre-residente, cache, estado, buffers, KV) contra
    /// la memoria disponible × 0.95 y se niega con números si no entra.
    pub fn plan(&self, _budget: &Budget) -> Result<MemoryPlan, LoadError> {
        todo!("Stage 0, ver docs/ROADMAP.md")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryPlan {
    pub trunk_gb: f64,
    pub expert_cache_gb: f64,
    pub kv_bytes: u64,
}

/// Fuente de lectura de pesos. Dos impls por plataforma (ver MEMORY-DESIGN.md §6):
///
/// - `Buffered`: pread/seek_read normal; el page cache decide; `fadvise`/hints.
/// - `Direct`: O_DIRECT en Linux (ventana ensanchada a 4 KB, fallback a buffered),
///   FILE_FLAG_NO_BUFFERING en Windows (alineación a sector).
pub trait DiskReader {
    /// Lee `nbytes` en `offset` hacia `buf`. `buf` DEBE tener capacidad para el
    /// ensanchado a alineación en el camino Direct.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), LoadError>;

    /// El alineamiento que este lector exige (4096 en Direct/Linux; sector en Windows).
    fn alignment(&self) -> u64;
}

/// Abre el lector correcto para la plataforma y el modo pedido.
pub fn open_reader(_path: &Path, _direct: bool) -> Result<Box<dyn DiskReader>, LoadError> {
    todo!("Stage 0, ver docs/ROADMAP.md")
}

/// Cache de expertos ruteados. Hereda el diseño de `k3_cache.c` completo:
///
/// - slots de 3 estados (EMPTY / INFLIGHT / ocupado): pick_victim nunca entrega un slot
///   cuya lectura no aterrizó;
/// - prefetch batch en 3 fases: reserva serial + dedup intra-batch, lecturas paralelas
///   ordenadas por offset de disco, publicación SOLO de lo que llegó;
/// - trace de accesos (8 bytes por request) para replay offline LRU/Belady/pinned;
/// - el presupuesto se redondea hacia abajo a slots enteros y exige ≥ topk+1 slots.
pub struct ExpertCache {
    _private: (),
}

impl ExpertCache {
    pub fn new(_budget_bytes: u64) -> Result<Self, LoadError> {
        todo!("Stage 4, ver docs/ROADMAP.md")
    }

    pub fn getmany(&self, _layer: u32, _experts: &[u32]) -> Result<(), LoadError> {
        todo!("Stage 4, ver docs/ROADMAP.md")
    }
}
