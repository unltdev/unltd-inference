//! Forward MÍNIMO qwen3_5 (Fase 5): la primera aritmética real contra el modelo.
//!
//! NO es el motor (llega en Fase 6, sobre la IR): es el cableado de datos — config
//! con refusal, pesos por mmap sin copia, lookup de embeddings cuantizados, la
//! primera rmsnorm y el GEMV de la cabeza de salida — validado contra el oráculo
//! de llama-eval-callback (`benchmarks/reference/ornith-final/*.f32.bin`).
//!
//! Cada pieza atómica ya está validada en su crate (dequantize Q4_K bit-idéntico
//! a ggml, rmsnorm con contrato pairwise, dot Q6_K exacto en f64); aquí se valida
//! el CABLEADO: offsets de fila, formas reales, dtypes reales, y el contrato
//! "refuse rather than guess" de punta a punta.
//!
//! Límites declarados (y negativas explícitas, no defaults):
//! - embeddings soportados: F32 (copia) y Q4_K (dequantize, como GET_ROWS del
//!   oráculo); cualquier otro dtype de embedding es una negativa en `open`;
//! - la cabeza de salida acepta F32/Q4_K/Q6_K (los dtypes con dot implementado);
//! - el GEMV de salida se paraleliza por filas de salida con rayon (contrato §5:
//!   jamás dentro de una reducción — cada fila mantiene su árbol pairwise).

use unltd_architectures::qwen35::Qwen35Config;
use unltd_core::LoadError;
use unltd_model_loader::{MappedWeights, TensorMeta, WeightIndex};
use unltd_tensor::{dequantize_q4_k, gemv_quant, gemv_quant_q8k, rmsnorm, DType};

/// Modelo qwen3.5 abierto, validado y listo para el forward mínimo.
pub struct MinForward {
    pub cfg: Qwen35Config,
    weights: MappedWeights,
    /// Vocab REAL leído de la forma de `token_embd.weight` (dims[1]), no de
    /// metadata (el GGUF de ornith no trae clave de vocab).
    vocab: usize,
    /// Bytes por fila del embedding empaquetado.
    emb_row_bytes: usize,
    emb_dtype: DType,
}

/// dtype de motor para un tensor del índice. Negativa ante cualquier id ggml
/// que el motor no sabe interpretar — nunca se adivina.
pub(crate) fn dtype_of(meta: &TensorMeta) -> Result<DType, LoadError> {
    match meta.ggml_type_id {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        8 => Ok(DType::Q8_0),
        12 => Ok(DType::Q4K),
        14 => Ok(DType::Q6K),
        other => Err(LoadError::UnknownGgmlType(other)),
    }
}

/// Bytes de una fila empaquetada de dimensión `n` (dims múltiplo del bloque;
/// GGUF exige padding de bloque para los quants — el check es una negativa).
pub(crate) fn row_bytes(dtype: DType, n: usize) -> Result<usize, LoadError> {
    match dtype {
        DType::F32 => Ok(n * 4),
        DType::F16 => Ok(n * 2),
        DType::Q8_0 => {
            if n % 32 != 0 {
                return Err(LoadError::corrupt(format!(
                    "dimensión {n} no es múltiplo de 32 para Q8_0"
                )));
            }
            Ok(n / 32 * 34)
        }
        DType::Q4K | DType::Q6K => {
            if n % 256 != 0 {
                return Err(LoadError::corrupt(format!(
                    "dimensión {n} no es múltiplo de 256 para {dtype:?}"
                )));
            }
            Ok(n / 256
                * match dtype {
                    DType::Q4K => 144,
                    _ => 210,
                })
        }
        _ => Err(LoadError::corrupt(format!(
            "dtype {dtype:?} sin tamaño de fila"
        ))),
    }
}

impl MinForward {
    /// Abre y valida TODO antes de servir un byte (k3_cfg/k3_st: config,
    /// forma del embedding, dtype, tamaño del tensor). Devuelve el modelo o
    /// una negativa con contexto.
    pub fn open(weights: MappedWeights) -> Result<Self, LoadError> {
        let cfg = Qwen35Config::from_gguf(weights.reader())?;

        let emb_meta =
            weights
                .reader()
                .find("token_embd.weight")
                .ok_or_else(|| LoadError::MissingTensor {
                    name: "token_embd.weight".to_string(),
                })?;
        if emb_meta.shape.len() != 2 {
            return Err(LoadError::corrupt(format!(
                "token_embd.weight: se esperaban 2 dims, hay {}",
                emb_meta.shape.len()
            )));
        }
        let n_embd = emb_meta.shape[0] as usize;
        let vocab = emb_meta.shape[1] as usize;
        if n_embd != cfg.n_embd {
            return Err(LoadError::ElementCount {
                name: "token_embd.weight".to_string(),
                got: n_embd,
                want: cfg.n_embd,
            });
        }
        if vocab == 0 {
            return Err(LoadError::corrupt(
                "token_embd.weight: vocab = 0".to_string(),
            ));
        }

        let emb_dtype = dtype_of(emb_meta)?;
        match emb_dtype {
            // los caminos implementados; el resto es una negativa explícita
            DType::F32 | DType::Q4K => {}
            other => {
                return Err(LoadError::corrupt(format!(
                    "token_embd.weight es {other:?}: el lookup de embeddings solo \
                     soporta F32 y Q4_K (Fase 5)"
                )));
            }
        }
        let emb_row_bytes = row_bytes(emb_dtype, n_embd)?;
        let expect = vocab as u64 * emb_row_bytes as u64;
        if emb_meta.nbytes != Some(expect) {
            return Err(LoadError::ElementCount {
                name: "token_embd.weight".to_string(),
                got: emb_meta.nbytes.unwrap_or(0) as usize,
                want: expect as usize,
            });
        }

        Ok(Self {
            cfg,
            weights,
            vocab,
            emb_row_bytes,
            emb_dtype,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Los pesos mapeados (índice + bytes por mmap). El forward completo
    /// (Fase 6) lee las capas a través de este acceso, sin re-mapear el archivo.
    pub(crate) fn weights(&self) -> &MappedWeights {
        &self.weights
    }

    /// Lookup de embeddings: dequantiza (Q4_K) o copia (F32) las filas de
    /// `token_embd.weight` para cada token. Un token fuera del vocab es un
    /// ERROR, no un clamp.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, LoadError> {
        let n = self.cfg.n_embd;
        let all = self
            .weights
            .tensor("token_embd.weight")
            .expect("validado en open");
        let mut out = vec![0.0f32; tokens.len() * n];
        for (r, &tok) in tokens.iter().enumerate() {
            let id = tok as usize;
            if id >= self.vocab {
                return Err(LoadError::corrupt(format!(
                    "token {tok} fuera del vocab (0..{})",
                    self.vocab
                )));
            }
            let row = &all[id * self.emb_row_bytes..(id + 1) * self.emb_row_bytes];
            let dst = &mut out[r * n..(r + 1) * n];
            match self.emb_dtype {
                DType::F32 => dst.copy_from_slice(bytemuck::cast_slice(row)),
                DType::Q4K => dequantize_q4_k(dst, row),
                _ => unreachable!("negado en open"),
            }
        }
        Ok(out)
    }

    /// RMSNorm de `blk.0.attn_norm.weight` sobre cada fila de `x`
    /// (`x` = n_tokens × n_embd). Usa un scratch de fila: `rmsnorm` no declara
    /// soporte de aliasing in-place y el contrato no lo asume.
    pub fn attn_norm_rows(&self, x: &[f32]) -> Result<Vec<f32>, LoadError> {
        let n = self.cfg.n_embd;
        assert_eq!(
            x.len() % n,
            0,
            "attn_norm_rows: len(x) no es múltiplo de n_embd"
        );
        let w = self.weights.tensor_checked("blk.0.attn_norm.weight", n)?;
        let w: &[f32] = bytemuck::cast_slice(w);
        let mut out = vec![0.0f32; x.len()];
        let mut scratch = vec![0.0f32; n];
        for (src, dst) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            rmsnorm(&mut scratch, src, w, self.cfg.rms_eps);
            dst.copy_from_slice(&scratch);
        }
        Ok(out)
    }

    /// Cabeza de salida: `logits = output.weight · result_norm` (GEMV por fila,
    /// 248 320 filas de 4096 en ornith — paralelizado por filas con rayon).
    /// El input debe ser el `result_norm` del oráculo (n_embd floats).
    pub fn output_logits(&self, result_norm: &[f32]) -> Result<Vec<f32>, LoadError> {
        let n = self.cfg.n_embd;
        assert_eq!(
            result_norm.len(),
            n,
            "output_logits: len(result_norm) != n_embd"
        );
        let meta = self.weights.reader().find("output.weight").ok_or_else(|| {
            LoadError::MissingTensor {
                name: "output.weight".to_string(),
            }
        })?;
        if meta.shape[0] as usize != n || meta.shape[1] as usize != self.vocab {
            return Err(LoadError::ElementCount {
                name: "output.weight".to_string(),
                got: (meta.shape[0] * meta.shape[1]) as usize,
                want: n * self.vocab,
            });
        }
        let dtype = dtype_of(meta)?;
        let row_bytes = row_bytes(dtype, n)?;
        let expect = self.vocab as u64 * row_bytes as u64;
        if meta.nbytes != Some(expect) {
            return Err(LoadError::ElementCount {
                name: "output.weight".to_string(),
                got: meta.nbytes.unwrap_or(0) as usize,
                want: expect as usize,
            });
        }
        let w = self
            .weights
            .tensor("output.weight")
            .expect("validado arriba");

        let mut logits = vec![0.0f32; self.vocab];
        use rayon::prelude::*;
        // Filas de salida independientes → paralelismo permitido por el contrato.
        // Cada fila mantiene su árbol pairwise; el resultado es determinista.
        const CHUNK_ROWS: usize = 8192;
        logits
            .par_chunks_mut(CHUNK_ROWS)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let j0 = ci * CHUNK_ROWS;
                let c = chunk.len();
                let rows = &w[j0 * row_bytes..(j0 + c) * row_bytes];
                match dtype {
                    // el oráculo dota los pesos K-quant contra x cuantizado a Q8_K
                    DType::Q4K | DType::Q6K => {
                        gemv_quant_q8k(chunk, result_norm, rows, n, c, dtype)
                    }
                    // F32/Q8_0: camino existente (Q8_0 usa vec_dot Q8_0, no Q8_K)
                    _ => gemv_quant(chunk, result_norm, rows, n, c, dtype),
                }
            });
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bytes_sizes_are_exact() {
        assert_eq!(row_bytes(DType::F32, 4096).unwrap(), 16384);
        assert_eq!(row_bytes(DType::Q4K, 4096).unwrap(), 16 * 144);
        assert_eq!(row_bytes(DType::Q6K, 4096).unwrap(), 16 * 210);
        assert_eq!(row_bytes(DType::Q8_0, 64).unwrap(), 2 * 34);
        // negativas: dimensión no múltiplo del bloque
        assert!(row_bytes(DType::Q4K, 4097).is_err());
        assert!(row_bytes(DType::Q8_0, 33).is_err());
    }

    #[test]
    fn dtype_of_maps_and_refuses() {
        let meta = |id: u32| TensorMeta {
            name: "t".into(),
            offset: 0,
            nbytes: Some(4),
            dtype: "?".into(),
            ggml_type_id: id,
            shape: vec![1],
            n_elements: 1,
        };
        assert_eq!(dtype_of(&meta(0)).unwrap(), DType::F32);
        assert_eq!(dtype_of(&meta(12)).unwrap(), DType::Q4K);
        assert_eq!(dtype_of(&meta(14)).unwrap(), DType::Q6K);
        match dtype_of(&meta(39)) {
            Err(LoadError::UnknownGgmlType(39)) => {}
            other => panic!("esperaba UnknownGgmlType, obtuve {other:?}"),
        }
    }
}
