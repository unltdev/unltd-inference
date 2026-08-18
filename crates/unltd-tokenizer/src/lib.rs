//! Tokenizador BPE byte-level. Ver `docs/ARCHITECTURE.md` §6.
//!
//! Cobertura v1: `tokenizer.json` (Qwen2.5/3, Mistral, DeepSeek, Phi, SmolLM, Llama-3.1+
//! — y Llama-3.2, que también usa tiktoken BPE vocab 128,256, NO SentencePiece) y
//! `tiktoken.model` (los modelos que lo publiquen). Fuera de v1: SentencePiece unigram
//! (Gemma 256k/262k, Mixtral 32k, TinyLlama) — etapa posterior documentada.
//!
//! Invariantes heredadas de `k3_tok.h` (ver `docs/AUDIT.md` §3.1): cada una produce un
//! tokenizador que corre, emite ids, y es incorrecto — sin crash ni diagnóstico.
//!
//! 1. El vocabulario se clavea por la cadena BYTE-LEVEL (bytes → codepoints imprimibles),
//!    no por bytes crudos. Sin la conversión, cada pieza degrada a bytes sueltos y el
//!    modelo corre sobre ids basura.
//! 2. El regex de pre-tokenización se lee del archivo del checkpoint — nunca se
//!    re-implementa de memoria (lección K3 con `tokenization_kimi.py`).
//! 3. Los tokens especiales se resuelven antes del BPE y por coincidencia más larga
//!    primero.
//! 4. El decode NO emite secuencias UTF-8 parciales: los bytes se acumulan hasta el
//!    límite de codepoint (un multi-byte cortado no es mojibake si no se imprime).

use unltd_core::LoadError;

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("tokenizer has no vocab")]
    NoVocab,
    #[error("input token id {0} out of range")]
    IdOutOfRange(u32),
}

/// Interfaz del tokenizador. Implementaciones: `BpeTokenizer` (tokenizer.json) y
/// `TiktokenTokenizer` (tiktoken.model + tokenizer_config.json).
pub trait Tokenizer {
    fn encode(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokenError>;
    fn decode(&self, ids: &[u32], out: &mut Vec<u8>) -> Result<(), TokenError>;
    fn eos(&self) -> Option<u32>;
    fn vocab_size(&self) -> u32;
}

/// Carga el tokenizador correcto mirando QUÉ archivos existen en el directorio del
/// modelo — nunca adivinando por nombre de modelo. Si hay `tokenizer.json`, es BPE;
/// si hay `tiktoken.model`, es el loader tiktoken.
pub fn load(_dir: &std::path::Path) -> Result<Box<dyn Tokenizer>, LoadError> {
    todo!("Stage 0, ver docs/ROADMAP.md")
}
