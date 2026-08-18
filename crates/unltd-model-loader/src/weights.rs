//! Pesos mapeados en memoria: acceso SIN COPIA a los bytes de cada tensor.
//!
//! El archivo GGUF completo se mapea de solo lectura; cada tensor se sirve como
//! `&[u8]` sobre el mapa. El sistema operativo paga las páginas con demanda
//! desde disco — esto es la base del streaming de capas (Fases 9-10): "tocar"
//! un tensor solo lo trae a RAM cuando un kernel lo lee.
//!
//! Reglas heredadas de `k3_st.c` (ver `docs/AUDIT.md` §3.2):
//! - nunca se adivina: `tensor(name)` devuelve `None` si el tensor no existe o
//!   su tipo es desconocido (sin tamaño conocido) — el llamador decide si la
//!   ausencia es fatal para ese peso;
//! - `tensor_checked` convierte la ausencia en un error con contexto y exige el
//!   element count exacto que la config implica (un peso con forma equivocada es
//!   un modelo distinto, no un error menor).

use std::path::Path;

use unltd_core::LoadError;

use crate::gguf::GgufReader;
use crate::WeightIndex;

/// GGUF validado + mapeado. El índice vive en memoria (KB); los pesos, en el mapa.
pub struct MappedWeights {
    map: memmap2::Mmap,
    reader: GgufReader,
}

impl MappedWeights {
    /// Abre, valida y mapea. El parseo va PRIMERO: un archivo corrupto nunca se
    /// mapea (y un archivo válido se mapea una sola vez).
    pub fn open(path: &Path) -> Result<Self, LoadError> {
        let reader = GgufReader::open(path)?;
        let file = std::fs::File::open(path)?;
        // SAFETY: el mapa es de solo lectura y el archivo no cambia durante su
        // vida (nadie en este proceso escribe sobre un modelo). El len se toma
        // del metadata del archivo abierto.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(LoadError::io)?;
        Ok(Self { map, reader })
    }

    /// El índice parseado (metadata GGUF + tabla de tensores).
    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    pub fn file_size(&self) -> u64 {
        self.map.len() as u64
    }

    /// Bytes de un tensor por nombre, como slice del mapa. `None` = ausente o
    /// tipo ggml desconocido. Para pesos REQUERIDOS el `None` DEBE tratarse como
    /// fatal (regla k3_st.c: un peso ausente leído como ceros produce un modelo
    /// que corre, fluido y equivocado).
    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        let meta = self.reader.find(name)?;
        let nbytes = meta.nbytes?;
        let start = usize::try_from(meta.offset).ok()?;
        let end = start.checked_add(usize::try_from(nbytes).ok()?)?;
        self.map.get(start..end)
    }

    /// Como [`Self::tensor`], pero con la política de negativa completa:
    /// - tensor ausente → [`LoadError::MissingTensor`];
    /// - element count distinto al esperado → [`LoadError::ElementCount`];
    /// - tipo desconocido → [`LoadError::UnknownGgmlType`].
    pub fn tensor_checked(&self, name: &str, n_elements: usize) -> Result<&[u8], LoadError> {
        let meta = self
            .reader
            .find(name)
            .ok_or_else(|| LoadError::MissingTensor {
                name: name.to_string(),
            })?;
        let got = meta.n_elements as usize;
        if got != n_elements {
            return Err(LoadError::ElementCount {
                name: name.to_string(),
                got,
                want: n_elements,
            });
        }
        let nbytes = meta
            .nbytes
            .ok_or(LoadError::UnknownGgmlType(meta.ggml_type_id))?;
        let start = meta.offset as usize;
        let end = start.checked_add(nbytes as usize).ok_or_else(|| {
            LoadError::corrupt(format!("tensor '{name}': offset+nbytes overflow"))
        })?;
        self.map
            .get(start..end)
            .ok_or_else(|| LoadError::corrupt(format!("tensor '{name}': bytes fuera del mapa")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::fixture_valid;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Escribe la fixture en un archivo temporal único y lo borra al final.
    struct TempGguf(std::path::PathBuf);

    impl TempGguf {
        fn new(bytes: &[u8]) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "unltd-mapped-weights-{}-{n}.gguf",
                std::process::id()
            ));
            std::fs::write(&path, bytes).unwrap();
            TempGguf(path)
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn tensor_slices_are_the_file_bytes() {
        // fixture: tok_embd.weight = 48 B F32 en [224, 272); norm.weight = 68 B
        // Q8_0 en [288, 356). Marcamos bytes conocidos dentro de cada rango.
        let mut b = fixture_valid();
        b[224] = 0xAB;
        b[230] = 0xCD;
        b[300] = 0xEF;
        let tmp = TempGguf::new(&b);
        let mw = MappedWeights::open(&tmp.0).unwrap();
        assert_eq!(mw.file_size(), b.len() as u64);

        let t0 = mw.tensor("tok_embd.weight").unwrap();
        assert_eq!(t0.len(), 48);
        assert_eq!(t0[0], 0xAB);
        assert_eq!(t0[6], 0xCD);
        let t1 = mw.tensor("norm.weight").unwrap();
        assert_eq!(t1.len(), 68);
        assert_eq!(t1[12], 0xEF);
    }

    #[test]
    fn tensor_none_for_absent() {
        let tmp = TempGguf::new(&fixture_valid());
        let mw = MappedWeights::open(&tmp.0).unwrap();
        assert!(mw.tensor("no.existe").is_none());
    }

    #[test]
    fn tensor_checked_refusals() {
        let tmp = TempGguf::new(&fixture_valid());
        let mw = MappedWeights::open(&tmp.0).unwrap();

        // count correcto → Ok con los bytes exactos
        let t0 = mw.tensor_checked("tok_embd.weight", 12).unwrap();
        assert_eq!(t0.len(), 48);

        // count equivocado → ElementCount (un peso con forma distinta es fatal)
        match mw.tensor_checked("tok_embd.weight", 13) {
            Err(LoadError::ElementCount { name, got, want }) => {
                assert_eq!(name, "tok_embd.weight");
                assert_eq!((got, want), (12, 13));
            }
            other => panic!("esperaba ElementCount, obtuve {other:?}"),
        }

        // ausente → MissingTensor
        match mw.tensor_checked("no.existe", 1) {
            Err(LoadError::MissingTensor { name }) => assert_eq!(name, "no.existe"),
            other => panic!("esperaba MissingTensor, obtuve {other:?}"),
        }
    }

    #[test]
    fn open_refuses_truncated_file() {
        let b = fixture_valid();
        let tmp = TempGguf::new(&b[..b.len() - 40]);
        assert!(MappedWeights::open(&tmp.0).is_err());
    }
}
