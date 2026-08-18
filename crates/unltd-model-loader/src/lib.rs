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

pub mod gguf;
pub mod weights;

pub use gguf::{
    ggml_type_blocksize, ggml_type_name, GgufArray, GgufReader, GgufValue, GgufValueType,
    GGUF_ALIGN,
};
pub use weights::MappedWeights;

/// Metadata de un tensor indexado: dónde están sus bytes y con qué dtype.
/// Tipo unificado para GGUF y safetensors.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    /// Offset ABSOLUTO en el archivo (GGUF: data_start + offset relativo del header).
    pub offset: u64,
    /// Bytes reales en el archivo (con padding de bloque). `None` = tipo ggml
    /// desconocido: el lector no adivina, el consumidor decide si lo necesita.
    pub nbytes: Option<u64>,
    /// Nombre legible del tipo ("F32", "Q4_K", "unknown(39)").
    pub dtype: String,
    /// Id numérico del tipo ggml (para dispatch de kernels).
    pub ggml_type_id: u32,
    /// Shape en orden GGUF (dims[0] es la dimensión contigua).
    pub shape: Vec<u64>,
    pub n_elements: u64,
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

/// Lector safetensors: 8 bytes little-endian de largo de header + JSON + data contigua.
/// La lista de checks de `k3_st.c` completa (dtype/offsets presentes, nbytes ==
/// shape × elemsize, fin-pasado-EOF, cola con bytes extra reportada).
pub struct SafetensorsReader {
    _private: (),
}

impl SafetensorsReader {
    #[allow(dead_code)] // Stage 1: verificación bf16 contra referencia
    pub fn open(_path: &std::path::Path) -> Result<Self, unltd_core::LoadError> {
        todo!("Stage 1, ver docs/ROADMAP.md")
    }
}
