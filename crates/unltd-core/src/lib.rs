//! Config del modelo, errores y la política "refuse rather than guess".
//! Ver `docs/ARCHITECTURE.md` §3 y §8.
//!
//! **La regla** (heredada de `k3_cfg.h`, ver `docs/AUDIT.md` §3.3): un campo ausente es
//! un ERROR, nunca un default. El peor fallo posible de un motor de inferencia es un
//! modelo que carga, streamea y decodifica texto fluido desde la arquitectura
//! equivocada. Por eso:
//!
//! - prohibido `#[serde(default)]` salvo campos con default REAL y documentado;
//! - todas las claves ausentes se acumulan y se reportan juntas (una corrida, un error);
//! - los checks estructurales (capas > 0, topk ≤ n_experts, rango del mapa de capas)
//!   corren después del parseo, antes de tocar un solo byte de pesos.

use std::fmt;

/// Error de carga: el motor se NIEGA a correr. Distinto de un error de I/O: esto es una
/// decisión, no un accidente.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Config con claves requeridas ausentes. `fields` las lista TODAS.
    #[error("config is missing {n} required field(s): {fields}")]
    MissingConfig { n: usize, fields: String },

    /// Config que parsea pero no puede describir una arquitectura válida.
    #[error("config fails a structural check: {0}")]
    StructCheck(String),

    /// Tensor con element count distinto al que la config implica.
    /// Un peso con forma equivocada es un modelo distinto, no un error menor.
    #[error("tensor '{name}' has {got} elements, expected {want}")]
    ElementCount { name: String, got: usize, want: usize },

    /// El plan de memoria no cabe. `need` y `have` siempre se imprimen juntos.
    #[error("this run needs {need}, the machine has {have} available")]
    DoesNotFit { need: String, have: String },
}

/// Tamaños legibles por humanos (estilo K3: "8.24 GB", "1.45 TB").
pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut b = bytes as f64;
    let mut i = 0;
    while b >= 1000.0 && i < 4 {
        b /= 1000.0;
        i += 1;
    }
    format!("{b:.2} {}", UNITS[i])
}

/// Identificador de peso dentro del `ModelSpec`: índice simbólico que el MemoryManager
/// resuelve contra su fuente (arena, mmap, ring, cache de expertos).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightId(pub u64);

impl fmt::Display for WeightId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w{}", self.0)
    }
}
