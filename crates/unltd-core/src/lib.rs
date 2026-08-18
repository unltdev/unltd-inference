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
    ElementCount {
        name: String,
        got: usize,
        want: usize,
    },

    /// Tensor requerido que no está en el índice del archivo. Distinto de un
    /// archivo corrupto: el archivo es válido, le falta un peso.
    #[error("tensor '{name}' not found in the file index")]
    MissingTensor { name: String },

    /// El plan de memoria no cabe. `need` y `have` siempre se imprimen juntos.
    #[error("this run needs {need}, the machine has {have} available")]
    DoesNotFit { need: String, have: String },

    // ---- Errores de formato de pesos (GGUF / safetensors) ----
    // Un archivo corrupto/truncado NUNCA se degrada a "leer con ceros": eso produce
    // un modelo que corre, fluido y equivocado (lección k3_st.c, docs/AUDIT.md §3.2).
    /// Archivo corrupto con contexto (offsets y tamaños incluidos en el mensaje).
    #[error("corrupt file: {0}")]
    BadFile(String),

    /// Los primeros 4 bytes no son "GGUF".
    #[error("not a GGUF file: bad magic {magic:02X?} (expected 'GGUF')")]
    BadMagic { magic: [u8; 4] },

    /// Versión de GGUF fuera de {2, 3}.
    #[error("unsupported GGUF version {0} (supported: 2 and 3)")]
    UnsupportedVersion(u32),

    /// Un tensor declara bytes más allá del final del archivo.
    #[error(
        "tensor '{name}' claims bytes [{offset}, {offset}+{nbytes}) beyond file size {file_size}"
    )]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        nbytes: u64,
        file_size: u64,
    },

    /// Dos rangos de tensores se pisan: un archivo así no puede ser leído con
    /// confianza (el segundo tensor contendría bytes del primero).
    #[error("tensor data overlap: '{a}' ends at {a_end}, '{b}' starts at {b_start}")]
    TensorOverlap {
        a: String,
        b: String,
        a_end: u64,
        b_start: u64,
    },

    /// Offset de tensor no alineado a 32 (viola la spec GGUF).
    #[error("tensor '{name}' at offset {offset} is not aligned to {align} bytes")]
    MisalignedTensor {
        name: String,
        offset: u64,
        align: u64,
    },

    /// Id de tipo ggml fuera de la tabla conocida. Se reporta como error solo si
    /// el llamador necesita el tamaño; el reader lo tolera con `n_bytes: None`.
    #[error("unknown ggml type id {0}")]
    UnknownGgmlType(u32),

    /// Error de I/O del sistema.
    #[error("i/o error: {0}")]
    Io(#[source] std::io::Error),
}

impl LoadError {
    /// Archivo corrupto con contexto. Preferido a un panic: es una negativa, no un accidente.
    pub fn corrupt(msg: impl Into<String>) -> Self {
        Self::BadFile(msg.into())
    }

    pub fn io(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
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
