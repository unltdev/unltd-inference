//! Lectores de formatos de pesos: GGUF (primario) y safetensors (secundario).
//! Ver `docs/ARCHITECTURE.md` §3 y §7.
//!
//! GGUF es el estándar de facto de CPU (quants maduros, metadata); safetensors se usa
//! para experimentos de precisión completa y verificación contra referencias.
//!
//! Filosofía heredada de `k3_st.c` (ver `docs/AUDIT.md` §3.2): el lector valida antes de
//! servir un solo byte — offsets dentro del archivo, dtype conocido, nbytes coherentes
//! con la forma — y se NIEGA ante un archivo truncado o un tensor que no existe.
//! Un peso ausente leído como ceros produce un modelo que corre, fluido y equivocado.

use std::path::Path;

use unltd_core::LoadError;

/// Metadata de un tensor indexado: dónde están sus bytes y con qué dtype.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub offset: u64,
    pub nbytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

/// Índice de tensores unificado sobre ambos formatos.
pub trait WeightIndex {
    /// O(1) por nombre. `None` = ausente, y el llamador DEBE tratarlo como fatal.
    fn find(&self, name: &str) -> Option<&TensorMeta>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Lector GGUF. El header trae pares clave-valor de metadata + info de tensores
/// (offset/nbytes/quant) ya contiguos por tensor — la capa es la unidad natural de
/// stream. Validación de rango contra el tamaño real del archivo antes de aceptar.
pub struct GgufReader {
    _private: (),
}

impl GgufReader {
    pub fn open(_path: &Path) -> Result<Self, LoadError> {
        todo!("Stage 0, ver docs/ROADMAP.md")
    }
}

/// Lector safetensors: 8 bytes little-endian de largo de header + JSON + data contigua.
/// La lista de checks de `k3_st.c` completa (dtype/offsets presentes, nbytes ==
/// shape × elemsize, fin-pasado-EOF, cola con bytes extra reportada).
pub struct SafetensorsReader {
    _private: (),
}

impl SafetensorsReader {
    pub fn open(_path: &Path) -> Result<Self, LoadError> {
        todo!("Stage 1, ver docs/ROADMAP.md")
    }
}
