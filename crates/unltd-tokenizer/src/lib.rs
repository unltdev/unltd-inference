//! Tokenizador BPE byte-level (gpt2) construido DESDE el GGUF del modelo.
//!
//! Fuente de verdad del comportamiento: llama.cpp (`src/llama-vocab.cpp` +
//! `src/unicode.cpp`, `D:\AI\runtimes\llama.cpp`). La puerta de la Fase 7 exige
//! IDs idénticos al oráculo, así que encode es una réplica EXACTA del flujo:
//!
//! 1. **Split del texto CRUDO** con el pre-tokenizador: llama.cpp no corre un
//!    engine regex genérico — para los patrones gpt2/qwen2/qwen35 hay splitters
//!    custom (`unicode_regex_split_custom_*`, unicode.cpp) que aquí se replican
//!    rama a rama (misma lista de checks, mismo orden, mismos límites). El
//!    patrón se selecciona por `tokenizer.ggml.pre` — igual que el dispatch de
//!    llama.cpp por string exacto.
//! 2. **Byte-encoding POR PALABRA**: cada palabra del split se traduce byte a
//!    byte con la tabla gpt2 (`unicode_byte_encoding_process`): un espacio real
//!    del texto crudo se vuelve Ġ recién aquí, después del split — por eso el
//!    gluing de " capital" se produce en el dominio crudo (espacio + letras) y
//!    no en el traducido.
//! 3. Por palabra traducida: símbolos = chars; BPE merge con heap por
//!    (rank, left) — réplica del priority queue de llama.cpp (min rank, empate
//!    por posición izquierda), un par por paso.
//! 4. Símbolo final → id por lookup en el vocab; si no está, fallback por CHAR
//!    suelto (piezas de 1 char). Si un char tampoco está, NOS NEGAMOS:
//!    llama.cpp lo silencia, un id faltante es corrupción del vocab.
//!
//! Decode (mismo contrato que `llama_vocab::token_to_piece`):
//! - CONTROL (3), UNKNOWN (2) y UNUSED (5) → suprimidos;
//! - USER_DEFINED (4) → texto crudo;
//! - NORMAL (1) y BYTE (6) → inversa de byte-to-unicode, char a char.
//!   Un char fuera de la tabla es corrupción → negativa, no "[UNK_BYTE_…]".
//!
//! Invariantes heredadas de `k3_tok.h` (ver `docs/AUDIT.md` §3.1): cada una
//! produce un tokenizador que corre, emite ids, y es incorrecto — sin crash ni
//! diagnóstico.
//! 1. El vocabulario se clavea por la cadena BYTE-LEVEL, no por bytes crudos.
//! 2. El pre-tokenizador se lee de la metadata del checkpoint y se replica del
//!    splitter custom de llama.cpp — nunca se re-implementa de memoria.
//! 3. Encode es RAW: sin BOS, sin parseo de tokens especiales (el modo del
//!    oráculo de la Fase 7 es `-no-cnv` sin add_bos).
//! 4. El decode NO emite secuencias UTF-8 parciales: emite bytes crudos
//!    completos; el texto decodificado se imprime como bloque.

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use unltd_core::LoadError;
use unltd_model_loader::{GgufArray, GgufReader, GgufValue};

/// Tipos de token del GGUF (`tokenizer.ggml.token_type`, `llama_token_type`).
pub mod token_type {
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("tokenizer has no vocab")]
    NoVocab,
    #[error("input token id {0} out of range")]
    IdOutOfRange(u32),
    /// Símbolo (o char del fallback) que no está en el vocab: llama.cpp lo
    /// silencia, el motor se niega — es corrupción del vocab, no texto exótico.
    #[error("piece {0:?} not in vocab (char-level byte fallback failed too)")]
    PieceMissing(String),
    /// Char de una pieza fuera de la tabla byte-to-unicode: el decode no puede
    /// invertirlo. llama.cpp emitiría "[UNK_BYTE_0x…]", el motor se niega.
    #[error("piece {0:?} contains a char outside the byte-to-unicode table")]
    BadPieceChar(String),
}

/// Interfaz del tokenizador. Implementación actual: `Gpt2Tokenizer` (GGUF gpt2 BPE).
pub trait Tokenizer {
    fn encode(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokenError>;
    fn decode(&self, ids: &[u32], out: &mut Vec<u8>) -> Result<(), TokenError>;
    fn bos(&self) -> Option<u32>;
    fn eos(&self) -> Option<u32>;
    fn vocab_size(&self) -> u32;
    /// Tipo del tokenizador para logs/checkpoints ("gpt2").
    fn kind(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// byte-to-unicode (gpt2) — tabla estándar: printables conservan su byte, los
// demás bytes (ascendentes) reciben 256, 257, …
// ---------------------------------------------------------------------------

fn byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut b2u = ['\0'; 256];
    let mut u2b = HashMap::new();
    let mut n = 0u32;
    for b in 0u32..256 {
        let c = if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
        b2u[b as usize] = c;
        u2b.insert(c, b as u8);
    }
    (b2u, u2b)
}

// ---------------------------------------------------------------------------
// Pre-tokenizadores — los strings EXACTOS que llama.cpp matchea para despachar
// a los splitters custom (`unicode_regex_split_custom`, unicode.cpp ~1053).
// Acá cumplen rol de documentación y de clave de dispatch; la implementación
// de cada uno es la transcripción rama a rama de su splitter custom.
// ---------------------------------------------------------------------------

/// QWEN35 (`LLAMA_VOCAB_PRE_TYPE_QWEN35`, llama-vocab.cpp ~386).
#[cfg_attr(not(test), allow(dead_code))] // documento del patrón oráculo; lo ancla el test regex_strings_pin
const RE_QWEN35: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// QWEN2 / STABLELM2 / HUNYUAN / SOLAR_OPEN (llama-vocab.cpp ~379).
#[cfg_attr(not(test), allow(dead_code))] // documento del patrón oráculo; lo ancla el test regex_strings_pin
const RE_QWEN2: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// GPT2 / MPT / OLMO / JAIS / TRILLION / GRANITE_DOCLING (llama-vocab.cpp ~369).
#[cfg_attr(not(test), allow(dead_code))] // documento del patrón oráculo; lo ancla el test regex_strings_pin
const RE_GPT2: &str =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreKind {
    Gpt2,
    Qwen2,
    Qwen35,
}

impl PreKind {
    fn from_pre(pre: &str) -> Result<Self, LoadError> {
        match pre {
            "qwen35" => Ok(Self::Qwen35),
            "qwen2" => Ok(Self::Qwen2),
            "default" | "" => Ok(Self::Gpt2),
            other => Err(LoadError::corrupt(format!(
                "tokenizer.ggml.pre = {other:?} — pre-tokenizador no implementado \
                 (qwen35, qwen2, default)"
            ))),
        }
    }
}

/// Clasificación Unicode por char — réplica de `unicode_cpt_flags_from_cpt` de
/// llama.cpp (mismas propiedades: \p{L}, \p{N}, \p{M}, White_Space). El engine
/// regex del crate se usa SOLO como clasificador por propiedad, nunca para el
/// split (los splitters custom se replican a mano abajo).
#[derive(Clone, Copy, Default, PartialEq)]
struct Flags {
    letter: bool,
    number: bool,
    mark: bool,
    ws: bool,
}

static RE_PROP_LETTER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\p{L}").unwrap());
static RE_PROP_NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\p{N}").unwrap());
static RE_PROP_MARK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\p{M}").unwrap());
static RE_PROP_WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s").unwrap());

fn classify(c: char) -> Flags {
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    Flags {
        letter: RE_PROP_LETTER.is_match(s),
        number: RE_PROP_NUMBER.is_match(s),
        mark: RE_PROP_MARK.is_match(s),
        ws: RE_PROP_WS.is_match(s),
    }
}

/// `unicode_tolower` de llama.cpp: lowercase simple de UN char. Rust da el
/// mapping completo (multi-char); si baja a más de un char, no es comparable
/// (en la práctica los chars de los sets de contracciones bajan a 1).
fn lower(c: char) -> Option<char> {
    let mut it = c.to_lowercase();
    let first = it.next()?;
    if it.next().is_some() {
        None
    } else {
        Some(first)
    }
}

// ---------------------------------------------------------------------------
// Splitters custom — transcripciones rama a rama de
// `unicode_regex_split_custom_qwen35` / `_qwen2` / `_gpt2` (unicode.cpp).
// Trabajan sobre el texto CRUDO (cpts); el byte-encoding ocurre por palabra
// después del split, como en `unicode_regex_split` → `unicode_byte_encoding_process`.
//
// Divergencia documentada (única): llama.cpp chequea `flags.as_uint()` — un
// codepoint no asignado/private-use tiene flags cero y NO matchea el patrón de
// símbolos. Acá el chequeo es `pos < n`; los codepoints sin asignar caen al
// patrón de símbolos. No afecta texto real en ningún idioma.
// ---------------------------------------------------------------------------

struct SplitCtx<'a> {
    chars: &'a [char],
    flags: &'a [Flags],
}

impl<'a> SplitCtx<'a> {
    fn cpt(&self, pos: usize) -> Option<char> {
        self.chars.get(pos).copied()
    }
    fn flags(&self, pos: usize) -> Flags {
        self.flags.get(pos).copied().unwrap_or_default()
    }
}

fn split_words(text: &str, pre: PreKind) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let flags: Vec<Flags> = chars.iter().map(|&c| classify(c)).collect();
    let ctx = SplitCtx {
        chars: &chars,
        flags: &flags,
    };
    let n = chars.len();

    let mut words: Vec<String> = Vec::new();
    let mut prev_end = 0usize;
    // _add_token de llama.cpp: token contiguo [prev_end, end), solo si len > 0.
    let mut add_token = |end: usize| {
        if end > prev_end {
            words.push(chars[prev_end..end].iter().collect());
        }
        prev_end = end;
    };

    let mut pos = 0usize;
    while pos < n {
        let cpt = ctx.cpt(pos).unwrap();
        let flags = ctx.flags(pos);

        // Contractions: gpt2 case-sensitive; qwen2/qwen35 con unicode_tolower.
        let case_insensitive = pre != PreKind::Gpt2;
        if cpt == '\'' && pos + 1 < n {
            let c1 = ctx.cpt(pos + 1).unwrap();
            let n1 = if case_insensitive {
                lower(c1)
            } else {
                Some(c1)
            };
            if matches!(n1, Some('s' | 't' | 'm' | 'd')) {
                pos += 2;
                add_token(pos);
                continue;
            }
            if pos + 2 < n {
                let c2 = ctx.cpt(pos + 2).unwrap();
                let n2 = if case_insensitive {
                    lower(c2)
                } else {
                    Some(c2)
                };
                if (n1 == Some('r') && n2 == Some('e'))
                    || (n1 == Some('v') && n2 == Some('e'))
                    || (n1 == Some('l') && n2 == Some('l'))
                {
                    pos += 3;
                    add_token(pos);
                    continue;
                }
            }
        }

        match pre {
            // ------------------------- gpt2 -------------------------
            PreKind::Gpt2 => {
                let flags2 = if cpt == ' ' {
                    ctx.flags(pos + 1)
                } else {
                    flags
                };
                // <space>?\p{L}+
                if flags2.letter {
                    pos += (cpt == ' ') as usize;
                    while ctx.flags(pos).letter {
                        pos += 1;
                    }
                    add_token(pos);
                    continue;
                }
                // <space>?\p{N}+
                if flags2.number {
                    pos += (cpt == ' ') as usize;
                    while ctx.flags(pos).number {
                        pos += 1;
                    }
                    add_token(pos);
                    continue;
                }
                // <space>?[^\s\p{L}\p{N}]+  (sin [\r\n]* en gpt2)
                if !(flags2.ws || flags2.letter || flags2.number) && pos < n {
                    pos += (cpt == ' ') as usize;
                    while !(ctx.flags(pos).ws || ctx.flags(pos).letter || ctx.flags(pos).number)
                        && pos < n
                    {
                        pos += 1;
                    }
                    add_token(pos);
                    continue;
                }
                // \s+(?!\S) y \s+ (gpt2 no tiene \s*[\r\n]+: el \r\n es parte de \s+)
                let mut num_ws = 0;
                while ctx.flags(pos + num_ws).ws {
                    num_ws += 1;
                }
                if num_ws > 1 && pos + num_ws < n {
                    pos += num_ws - 1;
                    add_token(pos);
                    continue;
                }
                if num_ws > 0 {
                    pos += num_ws;
                    add_token(pos);
                    continue;
                }
            }
            // ----------------------- qwen2/qwen35 -----------------------
            _ => {
                // [^\r\n\p{L}\p{N}]?[\p{L}(\p{M})]+ — qwen35 también consume marcas.
                let with_marks = pre == PreKind::Qwen35;
                if !(cpt == '\r' || cpt == '\n' || flags.number) {
                    let nf = ctx.flags(pos + 1);
                    let run_ok = if with_marks {
                        flags.letter || flags.mark || nf.letter || nf.mark
                    } else {
                        flags.letter || nf.letter
                    };
                    if run_ok {
                        pos += 1;
                        while {
                            let f = ctx.flags(pos);
                            if with_marks {
                                f.letter || f.mark
                            } else {
                                f.letter
                            }
                        } {
                            pos += 1;
                        }
                        add_token(pos);
                        continue;
                    }
                }
                // \p{N} — UN char por token (el patrón qwen es \p{N}, no {1,3}).
                if flags.number {
                    pos += 1;
                    add_token(pos);
                    continue;
                }
                // <space>?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*
                let flags2 = if cpt == ' ' {
                    ctx.flags(pos + 1)
                } else {
                    flags
                };
                let excluded = if with_marks {
                    flags2.ws || flags2.letter || flags2.mark || flags2.number
                } else {
                    flags2.ws || flags2.letter || flags2.number
                };
                if !excluded && pos < n {
                    pos += (cpt == ' ') as usize;
                    loop {
                        let f = ctx.flags(pos);
                        let ex = if with_marks {
                            f.ws || f.letter || f.mark || f.number
                        } else {
                            f.ws || f.letter || f.number
                        };
                        if ex || pos >= n {
                            break;
                        }
                        pos += 1;
                    }
                    while matches!(ctx.cpt(pos), Some('\r' | '\n')) {
                        pos += 1;
                    }
                    add_token(pos);
                    continue;
                }
                // \s*[\r\n]+ — el run de whitespace HASTA el último \r/\n incluido.
                let mut num_ws = 0;
                let mut last_rn = None;
                while ctx.flags(pos + num_ws).ws {
                    if matches!(ctx.cpt(pos + num_ws), Some('\r' | '\n')) {
                        last_rn = Some(pos + num_ws + 1);
                    }
                    num_ws += 1;
                }
                if let Some(end) = last_rn {
                    pos = end;
                    add_token(pos);
                    continue;
                }
                // \s+(?!\S) — run de ≥2 ws no-final: todo menos el último char.
                if num_ws > 1 && pos + num_ws < n {
                    pos += num_ws - 1;
                    add_token(pos);
                    continue;
                }
                // \s+
                if num_ws > 0 {
                    pos += num_ws;
                    add_token(pos);
                    continue;
                }
            }
        }

        // no matches: char suelto
        pos += 1;
        add_token(pos);
    }
    words
}

/// Tokenizador BPE byte-level gpt2 construido desde la metadata del GGUF.
/// El modelo debe declarar `tokenizer.ggml.model = "gpt2"`; el pre-tokenizador
/// se elige por `tokenizer.ggml.pre` — cualquier otro valor es una negativa.
#[derive(Debug)]
pub struct Gpt2Tokenizer {
    b2u: [char; 256],
    u2b: HashMap<char, u8>,
    pre: PreKind,
    pre_str: String,
    /// (izquierda, derecha) → rank (índice en `tokenizer.ggml.merges`, primera
    /// aparición si hay duplicados — emplace de llama.cpp).
    merges: HashMap<(String, String), u32>,
    /// pieza → id (última aparición si la pieza está duplicada — asignación de
    /// llama.cpp).
    vocab: HashMap<String, u32>,
    /// Piezas de 1 CHAR (fallback por byte de llama.cpp, tokenize ~696-711).
    char_vocab: HashMap<char, u32>,
    tokens: Vec<String>,
    token_types: Vec<i32>,
    eos: Option<u32>,
    bos: Option<u32>,
}

impl Gpt2Tokenizer {
    /// Construye el tokenizador desde la metadata de un GGUF. Se NIEGA con
    /// `LoadError` si el modelo no es gpt2, el pre-tokenizador es desconocido,
    /// falta el vocab/merges/token_type, o alguna pieza está corrupta.
    pub fn from_gguf(r: &GgufReader) -> Result<Self, LoadError> {
        let model = r.get_str("tokenizer.ggml.model").ok_or_else(|| {
            LoadError::corrupt("GGUF sin tokenizer.ggml.model — no hay tokenizador BPE")
        })?;
        if model != "gpt2" {
            return Err(LoadError::corrupt(format!(
                "tokenizer.ggml.model = {model:?} — solo gpt2 (BPE byte-level) implementado"
            )));
        }
        let pre_str = r
            .get_str("tokenizer.ggml.pre")
            .unwrap_or("default")
            .to_string();
        let pre = PreKind::from_pre(&pre_str)?;

        let tokens = match r.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Arr(GgufArray::Str(v))) => v.clone(),
            _ => {
                return Err(LoadError::corrupt(
                    "tokenizer.ggml.tokens ausente o no es un array de strings",
                ))
            }
        };
        if tokens.is_empty() {
            return Err(LoadError::corrupt("tokenizer.ggml.tokens vacío"));
        }
        let token_types = match r.get("tokenizer.ggml.token_type") {
            Some(GgufValue::Arr(GgufArray::I32(v))) => v.clone(),
            Some(GgufValue::Arr(GgufArray::U32(v))) => v.iter().map(|&x| x as i32).collect(),
            _ => {
                return Err(LoadError::corrupt(
                    "tokenizer.ggml.token_type ausente o no es un array i32/u32 — \
                     sin tipos el decode no puede suprimir tokens de control",
                ))
            }
        };
        if token_types.len() != tokens.len() {
            return Err(LoadError::corrupt(format!(
                "tokenizer.ggml.token_type tiene {} entradas, tokens {}",
                token_types.len(),
                tokens.len()
            )));
        }
        let merges = match r.get("tokenizer.ggml.merges") {
            Some(GgufValue::Arr(GgufArray::Str(v))) => v.clone(),
            _ => {
                return Err(LoadError::corrupt(
                    "tokenizer.ggml.merges ausente o no es un array de strings",
                ))
            }
        };

        let eos = scalar_u32(r.get("tokenizer.ggml.eos_token_id"));
        let bos = scalar_u32(r.get("tokenizer.ggml.bos_token_id"));

        // Las piezas viven en el espacio traducido: TODOS sus chars deben estar en
        // la tabla byte-to-unicode. U+FFFD (from_utf8_lossy del loader sobre una
        // pieza no-UTF-8) no está en la imagen de la tabla → detectado aquí.
        let (b2u, u2b) = byte_to_unicode();
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        let mut char_vocab: HashMap<char, u32> = HashMap::new();
        for (id, piece) in tokens.iter().enumerate() {
            if let Some(bad) = piece.chars().find(|c| !u2b.contains_key(c)) {
                return Err(LoadError::corrupt(format!(
                    "pieza {id} {piece:?}: char {bad:?} fuera de la tabla byte-to-unicode \
                     (vocab corrupto, no texto exótico)"
                )));
            }
            vocab.insert(piece.clone(), id as u32); // última aparición gana (llama.cpp)
            if piece.chars().count() == 1 {
                char_vocab.insert(piece.chars().next().unwrap(), id as u32);
            }
        }
        let mut merge_ranks: HashMap<(String, String), u32> = HashMap::new();
        for (rank, m) in merges.iter().enumerate() {
            let Some((l, r)) = m.split_once(' ') else {
                return Err(LoadError::corrupt(format!(
                    "merge {rank} {m:?} sin separador de espacio"
                )));
            };
            if l.is_empty() || r.is_empty() {
                return Err(LoadError::corrupt(format!(
                    "merge {rank} {m:?} con lado vacío"
                )));
            }
            merge_ranks
                .entry((l.to_string(), r.to_string()))
                .or_insert(rank as u32);
        }

        Ok(Self {
            b2u,
            u2b,
            pre,
            pre_str,
            merges: merge_ranks,
            vocab,
            char_vocab,
            tokens,
            token_types,
            eos,
            bos,
        })
    }

    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokenError> {
        if self.vocab.is_empty() {
            return Err(TokenError::NoVocab);
        }
        // 1) split del texto CRUDO; 2) byte-encode POR PALABRA; 3) BPE; 4) lookup.
        for word in split_words(text, self.pre) {
            let translated: String = word
                .as_bytes()
                .iter()
                .map(|&b| self.b2u[b as usize])
                .collect();
            let mut symbols: Vec<String> = translated.chars().map(|c| c.to_string()).collect();
            bpe_merge(&mut symbols, &self.merges);
            for sym in &symbols {
                if let Some(&id) = self.vocab.get(sym.as_str()) {
                    out.push(id);
                    continue;
                }
                let mut ok = true;
                for ch in sym.chars() {
                    match self.char_vocab.get(&ch) {
                        Some(&id) => out.push(id),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    return Err(TokenError::PieceMissing(sym.clone()));
                }
            }
        }
        Ok(())
    }

    pub fn decode_into(&self, ids: &[u32], out: &mut Vec<u8>) -> Result<(), TokenError> {
        for &id in ids {
            let Some(piece) = self.tokens.get(id as usize) else {
                return Err(TokenError::IdOutOfRange(id));
            };
            match self.token_types[id as usize] {
                // Suprimidos como el detokenizador de llama.cpp (token_to_piece,
                // attr UNKNOWN|CONTROL devuelve pieza vacía; UNUSED cae en el
                // mismo grupo).
                token_type::CONTROL | token_type::UNKNOWN | token_type::UNUSED => {}
                // Texto crudo (attr USER_DEFINED en el switch BPE).
                token_type::USER_DEFINED => out.extend_from_slice(piece.as_bytes()),
                // NORMAL y BYTE: inversa de byte-to-unicode, char a char.
                _ => {
                    for ch in piece.chars() {
                        match self.u2b.get(&ch) {
                            Some(&b) => out.push(b),
                            None => return Err(TokenError::BadPieceChar(piece.clone())),
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Pieza (texto traducido) del id, para reportes del CLI.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Pre-tokenizador seleccionado (para el checkpoint: "qwen35" | "qwen2" | "default").
    pub fn pre(&self) -> &str {
        &self.pre_str
    }

    /// Estimación CONSERVADORA del heap que ocupa el tokenizador en bytes,
    /// para la contabilidad del presupuesto de memoria (Fase 9): bytes de los
    /// Strings del vocab (más sus claves clonadas en el HashMap) + entries de
    /// los tres mapas. Las constantes por entry son el orden del layout real
    /// de std HashMap (String ≈ 24 B + heap; entry ≈ 1-2 palabras de
    /// control); el RSS medido al final de la corrida es la cifra autoritativa.
    pub fn heap_bytes(&self) -> u64 {
        let mut b = 0u64;
        for t in &self.tokens {
            b += (t.len() as u64).saturating_add(48); // String 24 + slot Vec 8 + heap
        }
        for (l, r) in self.merges.keys() {
            b += (l.len() as u64)
                .saturating_add(r.len() as u64)
                .saturating_add(88); // (String,String) 48 + u32 + entry
        }
        for k in self.vocab.keys() {
            b += (k.len() as u64).saturating_add(48); // String 24 + u32 + entry
        }
        b += (self.char_vocab.len() as u64).saturating_mul(48); // char 4 + u32 + entry
        b += (self.u2b.len() as u64).saturating_mul(32); // char 4 + u8 + entry
        b = b.saturating_add(256 * 4); // b2u: [char; 256]
        b = b.saturating_add((self.token_types.len() as u64).saturating_mul(4));
        b.saturating_add(self.pre_str.len() as u64)
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokenError> {
        self.encode_into(text, out)
    }

    fn decode(&self, ids: &[u32], out: &mut Vec<u8>) -> Result<(), TokenError> {
        self.decode_into(ids, out)
    }

    fn bos(&self) -> Option<u32> {
        self.bos
    }

    fn eos(&self) -> Option<u32> {
        self.eos
    }

    fn vocab_size(&self) -> u32 {
        self.tokens.len() as u32
    }

    fn kind(&self) -> &'static str {
        "gpt2"
    }
}

fn scalar_u32(v: Option<&GgufValue>) -> Option<u32> {
    match v {
        Some(GgufValue::U32(x)) => Some(*x),
        Some(GgufValue::I32(x)) => Some(*x as u32),
        _ => None,
    }
}

/// BPE merge, réplica del priority queue de llama.cpp (`llm_tokenizer_bpe::tokenize`):
/// en cada paso se mergea UNA ocurrencia del par con (rank, posición izquierda)
/// mínimo, y se re-escanea. Con ranks únicos (rank = índice en merges, el caso de
/// los GGUF) esto coincide con "merge de todas las ocurrencias del par mínimo".
fn bpe_merge(symbols: &mut Vec<String>, ranks: &HashMap<(String, String), u32>) {
    loop {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..symbols.len().saturating_sub(1) {
            if let Some(&r) = ranks.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                if best.map_or(true, |(br, bi)| (r, i) < (br, bi)) {
                    best = Some((r, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let merged = symbols[i].clone() + &symbols[i + 1];
        symbols[i] = merged;
        symbols.remove(i + 1);
    }
}

/// Carga el tokenizador desde un GGUF. Si `path` es un directorio, toma el primer
/// `*.gguf` ordenado por nombre; si es un archivo, lo usa directo.
pub fn load(path: &Path) -> Result<Box<dyn Tokenizer>, LoadError> {
    let file = if path.is_file() {
        path.to_path_buf()
    } else if path.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(path)
            .map_err(LoadError::io)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
            .collect();
        files.sort();
        match files.into_iter().next() {
            Some(f) => f,
            None => {
                return Err(LoadError::corrupt(format!(
                    "{} no contiene ningún .gguf",
                    path.display()
                )))
            }
        }
    } else {
        return Err(LoadError::corrupt(format!(
            "{} no es un archivo .gguf ni un directorio",
            path.display()
        )));
    };
    let reader = GgufReader::open(&file)?;
    Ok(Box::new(Gpt2Tokenizer::from_gguf(&reader)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Fixture de vocab sintético (piezas en espacio traducido, id = índice).
    // Diseñado para que "The capital of France is" produzca 5 piezas, como el
    // prompt del oráculo de ornith [760, 6511, 314, 9338, 369].
    // -----------------------------------------------------------------------

    const TOKENS: &[&str] = &[
        "Ġ",             // 0
        "The",           // 1
        "Ġcapital",      // 2
        "Ġof",           // 3
        "ĠFrance",       // 4
        "Ġis",           // 5
        "ĠParis",        // 6
        ".",             // 7
        "Ċ",             // 8
        "h",             // 9
        "e",             // 10
        "l",             // 11
        "o",             // 12
        "w",             // 13
        "a",             // 14
        "b",             // 15
        "c",             // 16
        "n",             // 17
        "'",             // 18
        "t",             // 19
        "r",             // 20
        "d",             // 21
        "y",             // 22
        "?",             // 23
        "<|endoftext|>", // 24 USER_DEFINED
        "<|im_end|>",    // 25 CONTROL
        "'t",            // 26
        "can",           // 27
    ];

    const MERGES: &[&str] = &[
        "T h",
        "Th e", // 0,1 → The
        "Ġ c",
        "Ġc a",
        "Ġca p",
        "Ġcap i",
        "Ġcapi t",
        "Ġcapit a",
        "Ġcapita l", // 2..9 → Ġcapital
        "Ġ o",
        "Ġo f", // 10,11 → Ġof
        "Ġ F",
        "ĠF r",
        "ĠFr a",
        "ĠFra n",
        "ĠFran c",
        "ĠFranc e", // 12..17 → ĠFrance
        "Ġ i",
        "Ġi s", // 18,19 → Ġis
        "Ġ P",
        "ĠP a",
        "ĠPa r",
        "ĠPar i",
        "ĠPari s", // 20..24 → ĠParis
        "c a",
        "ca n", // 25,26 → can
        "' t",  // 27 → 't
    ];

    fn fixture(pre: PreKind) -> Gpt2Tokenizer {
        let mut types = vec![token_type::NORMAL; TOKENS.len()];
        types[24] = token_type::USER_DEFINED;
        types[25] = token_type::CONTROL;
        Gpt2Tokenizer::from_parts(
            TOKENS.iter().map(|s| s.to_string()).collect(),
            types,
            MERGES.iter().map(|s| s.to_string()).collect(),
            pre,
        )
        .expect("fixture válido")
    }

    impl Gpt2Tokenizer {
        /// Constructor directo para tests unitarios (sin GGUF): el camino
        /// `from_gguf` está cubierto por `from_gguf_*` más abajo.
        fn from_parts(
            tokens: Vec<String>,
            token_types: Vec<i32>,
            merges: Vec<String>,
            pre: PreKind,
        ) -> Result<Self, LoadError> {
            assert_eq!(tokens.len(), token_types.len());
            let (b2u, u2b) = byte_to_unicode();
            let mut vocab = HashMap::new();
            let mut char_vocab = HashMap::new();
            for (id, piece) in tokens.iter().enumerate() {
                assert!(
                    piece.chars().all(|c| u2b.contains_key(&c)),
                    "pieza {piece:?} fuera de tabla"
                );
                vocab.insert(piece.clone(), id as u32);
                if piece.chars().count() == 1 {
                    char_vocab.insert(piece.chars().next().unwrap(), id as u32);
                }
            }
            let mut merge_ranks = HashMap::new();
            for (rank, m) in merges.iter().enumerate() {
                let (l, r) = m.split_once(' ').unwrap();
                merge_ranks
                    .entry((l.to_string(), r.to_string()))
                    .or_insert(rank as u32);
            }
            Ok(Self {
                b2u,
                u2b,
                pre,
                pre_str: match pre {
                    PreKind::Qwen35 => "qwen35",
                    PreKind::Qwen2 => "qwen2",
                    PreKind::Gpt2 => "default",
                }
                .to_string(),
                merges: merge_ranks,
                vocab,
                char_vocab,
                tokens,
                token_types,
                eos: Some(25),
                bos: None,
            })
        }
    }

    fn encode(t: &Gpt2Tokenizer, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        t.encode(text, &mut out).expect("encode ok");
        out
    }

    fn decode(t: &Gpt2Tokenizer, ids: &[u32]) -> String {
        let mut out = Vec::new();
        t.decode(ids, &mut out).expect("decode ok");
        String::from_utf8(out).expect("decode emite UTF-8 válido")
    }

    /// El caso del oráculo: 5 piezas, sin espacios colgando.
    #[test]
    fn encode_oracle_prompt() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(encode(&t, "The capital of France is"), vec![1, 2, 3, 4, 5]);
    }

    /// La puerta de Fase 7 es ida y vuelta: encode + decode = texto original.
    #[test]
    fn encode_decode_roundtrip() {
        let t = fixture(PreKind::Qwen35);
        for text in [
            "The capital of France is",
            "The capital of France is Paris.\nThe",
            "hello world",
        ] {
            let ids = encode(&t, text);
            assert_eq!(decode(&t, &ids), text, "roundtrip de {text:?}");
        }
    }

    /// Fallback por byte (char suelto): símbolo sin merge y fuera del vocab.
    #[test]
    fn byte_fallback_chars() {
        let t = fixture(PreKind::Qwen35);
        // "hello": sin pares con rank → símbolo "hello" ausente → chars sueltos.
        assert_eq!(encode(&t, "hello"), vec![9, 10, 11, 11, 12]);
    }

    /// Contracciones: el splitter parte "can't" en "can" + "'t" (rama 1 del
    /// patrón qwen35). Case-insensitive: el split de "CAN'T" es el mismo
    /// (el vocab del fixture no tiene piezas mayúsculas, así que el encode
    /// fallaría — acá se valida el SPLIT, que es el contrato de la rama).
    #[test]
    fn contraction_split() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(encode(&t, "can't"), vec![27, 26]);
        assert_eq!(split_words("CAN'T", PreKind::Qwen35), vec!["CAN", "'T"]);
    }

    /// gpt2 es case-SENSITIVE en contracciones: "'T" no matchea → char fallback
    /// de ' y T... 'T tampoco tiene pieza → PieceMissing.
    #[test]
    fn contraction_case_sensitive_gpt2() {
        let t = fixture(PreKind::Gpt2);
        assert_eq!(encode(&t, "can't"), vec![27, 26]);
        let e = t.encode("CAN'T", &mut Vec::new()).unwrap_err();
        assert!(matches!(e, TokenError::PieceMissing(_)), "{e:?}");
    }

    /// Espacios: la palabra siguiente empieza con espacio real, que el
    /// byte-encode traduce a Ġ; el decode lo devuelve a ' '.
    #[test]
    fn leading_space_word() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(encode(&t, " Paris"), vec![6]);
        assert_eq!(decode(&t, &[6]), " Paris");
    }

    /// Newline: en el dominio crudo el "\n" es un run de whitespace con \r\n
    /// (rama 5) → palabra "\n" → byte-encode → "Ċ" (pieza id 8).
    #[test]
    fn newline_piece() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(encode(&t, "a\nb"), vec![14, 8, 15]);
        assert_eq!(decode(&t, &[14, 8, 15]), "a\nb");
    }

    /// Run de whitespace de ≥2 en medio del texto: la rama \s+(?!\S) emite el
    /// run menos el último char; el último char sale por \s+. "a  b" → [" ",
    /// " "] → cada una "Ġ" (id 0).
    #[test]
    fn mid_string_double_space() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(encode(&t, "a  b"), vec![14, 0, 0, 15]);
    }

    /// Run de whitespace final: \s+(?!\S) NO aplica (no hay char después) y
    /// \s+ emite el run entero como UNA palabra → "ĠĠ" → fallback de chars
    /// (no hay merge "Ġ Ġ" en el fixture).
    #[test]
    fn trailing_whitespace_run() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(encode(&t, "a  "), vec![14, 0, 0]);
        assert_eq!(encode(&t, "a "), vec![14, 0]);
    }

    /// Decode suprime CONTROL y UNKNOWN/UNUSED como el detokenizador de llama.cpp.
    #[test]
    fn decode_skips_control() {
        let t = fixture(PreKind::Qwen35);
        // CONTROL <|im_end|> en el medio no deja rastro.
        assert_eq!(decode(&t, &[1, 25, 7]), "The.");
    }

    /// USER_DEFINED se copia crudo.
    #[test]
    fn decode_user_defined_raw() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(decode(&t, &[24]), "<|endoftext|>");
    }

    #[test]
    fn decode_id_out_of_range() {
        let t = fixture(PreKind::Qwen35);
        let mut out = Vec::new();
        let e = t.decode(&[999], &mut out).unwrap_err();
        assert!(matches!(e, TokenError::IdOutOfRange(999)));
    }

    #[test]
    fn eos_bos_from_parts() {
        let t = fixture(PreKind::Qwen35);
        assert_eq!(t.eos(), Some(25));
        assert_eq!(t.bos(), None);
        assert_eq!(t.vocab_size(), TOKENS.len() as u32);
        assert_eq!(t.kind(), "gpt2");
        assert_eq!(t.piece(1), Some("The"));
        assert_eq!(t.piece(999), None);
        assert_eq!(t.pre(), "qwen35");
    }

    /// Greedy determinista: dos encodes del mismo texto son idénticos.
    #[test]
    fn encode_deterministic() {
        let t = fixture(PreKind::Qwen35);
        let text = "The capital of France is Paris.\nThe";
        assert_eq!(encode(&t, text), encode(&t, text));
    }

    /// Los tres pre-tokenizadores dan el mismo resultado para el caso del
    /// oráculo (texto sin marcas ni diferencias de patrón entre familias).
    #[test]
    fn pre_kinds_agree_on_oracle_prompt() {
        for pre in [PreKind::Qwen35, PreKind::Qwen2, PreKind::Gpt2] {
            let t = fixture(pre);
            assert_eq!(
                encode(&t, "The capital of France is"),
                vec![1, 2, 3, 4, 5],
                "{pre:?}"
            );
        }
    }

    /// Splitter de símbolos: "!" y "?" son piezas propias; los números salen
    /// de a un char (rama \p{N} del patrón qwen35).
    #[test]
    fn symbols_and_digits() {
        let _t = fixture(PreKind::Qwen35);
        // "12" → "1" y "2" como palabras separadas → char fallback (no hay
        // piezas "1"/"2" en el fixture → PieceMissing). Verificamos el split
        // directamente con split_words para no depender de piezas.
        assert_eq!(
            split_words("ab 12!", PreKind::Qwen35),
            vec!["ab", " ", "1", "2", "!"]
        );
    }

    /// El splitter qwen35 consume la marca de combinación junto con la letra
    /// ([\p{L}\p{M}]+) — la diferencia con qwen2.
    #[test]
    fn qwen35_marks_glue() {
        assert_eq!(
            split_words("a\u{0301}b", PreKind::Qwen35),
            vec!["a\u{0301}b"]
        );
        assert_eq!(
            split_words("a\u{0301}b", PreKind::Qwen2),
            vec!["a", "\u{0301}b"]
        );
    }

    /// El símbolo seguido de newline los lleva en UNA palabra (rama
    /// [\r\n]* del patrón de símbolos) — réplica exacta del splitter custom.
    #[test]
    fn symbol_eats_trailing_newline() {
        assert_eq!(split_words("a.\nb", PreKind::Qwen35), vec!["a", ".\n", "b"]);
        // gpt2 NO tiene [\r\n]*: "." y "\n" separados.
        assert_eq!(
            split_words("a.\nb", PreKind::Gpt2),
            vec!["a", ".", "\n", "b"]
        );
    }

    /// La contracción requiere una letra del set justo después del apóstrofo:
    /// en "''s" el char siguiente a la primera comilla es otra comilla → NO
    /// hay contracción y el run de símbolos se lleva "''" completo (mismo
    /// resultado que llama.cpp: el P1 mira cpt == '\'' && next, y next es '\'').
    #[test]
    fn contraction_only_at_start() {
        assert_eq!(split_words("''s", PreKind::Qwen35), vec!["''", "s"]);
    }

    // -----------------------------------------------------------------------
    // from_gguf con un GGUF sintético en disco (camino completo del parser).
    // -----------------------------------------------------------------------

    /// Escritor de GGUF mínimo para fixtures (header + KV, 0 tensores).
    struct W(Vec<u8>);

    impl W {
        fn new() -> Self {
            W(Vec::new())
        }
        fn u32(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(mut self, v: u64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn raw(mut self, b: &[u8]) -> Self {
            self.0.extend_from_slice(b);
            self
        }
        fn str(mut self, s: &str) -> Self {
            self = self.u64(s.len() as u64);
            self.raw(s.as_bytes())
        }
        fn kv_str(self, k: &str, v: &str) -> Self {
            self.str(k).u32(8).str(v)
        }
        fn kv_arr_str(self, k: &str, items: &[&str]) -> Self {
            // elem type 8 = GGUF_TYPE_STRING (12 es F64: un array de strings con
            // elem 12 se lee como 28×f64 y descuadra todo lo que sigue).
            let mut w = self.str(k).u32(9).u32(8).u64(items.len() as u64);
            for s in items {
                w = w.str(s);
            }
            w
        }
        fn kv_arr_i32(self, k: &str, items: &[i32]) -> Self {
            let mut w = self.str(k).u32(9).u32(5).u64(items.len() as u64);
            for v in items {
                w = w.u32(*v as u32);
            }
            w
        }
        fn finish(self) -> Vec<u8> {
            self.0
        }
    }

    /// GGUF v3 con la metadata del tokenizador fixture y 0 tensores.
    fn fixture_gguf(
        model: &str,
        pre: &str,
        tokens: &[&str],
        types: &[i32],
        _merges: &[&str],
    ) -> Vec<u8> {
        W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0) // n_tensors
            .u64(5) // n_kv
            .kv_str("general.architecture", "ornith")
            .kv_str("tokenizer.ggml.model", model)
            .kv_str("tokenizer.ggml.pre", pre)
            .kv_arr_str("tokenizer.ggml.tokens", tokens)
            .kv_arr_i32("tokenizer.ggml.token_type", types)
            .finish()
    }

    fn fixture_gguf_with_merges(
        model: &str,
        pre: &str,
        tokens: &[&str],
        types: &[i32],
        merges: &[&str],
    ) -> Vec<u8> {
        W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0) // n_tensors
            .u64(6) // n_kv
            .kv_str("general.architecture", "ornith")
            .kv_str("tokenizer.ggml.model", model)
            .kv_str("tokenizer.ggml.pre", pre)
            .kv_arr_str("tokenizer.ggml.tokens", tokens)
            .kv_arr_i32("tokenizer.ggml.token_type", types)
            .kv_arr_str("tokenizer.ggml.merges", merges)
            .finish()
    }

    /// Nombre único por test: los tests corren en paralelo sobre el mismo PID.
    fn with_file(bytes: Vec<u8>, f: impl FnOnce(&Path)) {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "unltd-tokenizer-test-{}-{n}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("escribir fixture");
        f(&path);
        let _ = std::fs::remove_file(&path);
    }

    fn fixture_types() -> Vec<i32> {
        TOKENS
            .iter()
            .enumerate()
            .map(|(i, _)| match i {
                24 => token_type::USER_DEFINED,
                25 => token_type::CONTROL,
                _ => token_type::NORMAL,
            })
            .collect()
    }

    #[test]
    fn from_gguf_parses_fixture() {
        let types = fixture_types();
        let bytes = fixture_gguf_with_merges("gpt2", "qwen35", TOKENS, &types, MERGES);
        with_file(bytes, |p| {
            let r = GgufReader::open(p).expect("parse");
            let t = Gpt2Tokenizer::from_gguf(&r).expect("from_gguf");
            assert_eq!(encode(&t, "The capital of France is"), vec![1, 2, 3, 4, 5]);
            assert_eq!(decode(&t, &[1, 25, 7]), "The.");
        });
    }

    /// pre = "default" selecciona el splitter gpt2 (misma puerta, distinto selector).
    #[test]
    fn from_gguf_default_pre() {
        let types = fixture_types();
        let bytes = fixture_gguf_with_merges("gpt2", "default", TOKENS, &types, MERGES);
        with_file(bytes, |p| {
            let r = GgufReader::open(p).expect("parse");
            let t = Gpt2Tokenizer::from_gguf(&r).expect("from_gguf");
            assert_eq!(t.pre(), "default");
            assert_eq!(encode(&t, "The capital of France is"), vec![1, 2, 3, 4, 5]);
        });
    }

    #[test]
    fn from_gguf_refuses_unknown_model() {
        let types = fixture_types();
        let bytes = fixture_gguf_with_merges("llama", "qwen35", TOKENS, &types, MERGES);
        with_file(bytes, |p| {
            let r = GgufReader::open(p).expect("parse");
            let e = Gpt2Tokenizer::from_gguf(&r).unwrap_err();
            assert!(e.to_string().contains("tokenizer.ggml.model"), "{e}");
        });
    }

    #[test]
    fn from_gguf_refuses_unknown_pre() {
        let types = fixture_types();
        let bytes = fixture_gguf_with_merges("gpt2", "deepseek3", TOKENS, &types, MERGES);
        with_file(bytes, |p| {
            let r = GgufReader::open(p).expect("parse");
            let e = Gpt2Tokenizer::from_gguf(&r).unwrap_err();
            assert!(e.to_string().contains("tokenizer.ggml.pre"), "{e}");
        });
    }

    #[test]
    fn from_gguf_refuses_corrupt_piece() {
        let mut tokens: Vec<&str> = TOKENS.to_vec();
        tokens[0] = "\u{FFFD}"; // U+FFFD no está en la imagen de byte_to_unicode
        let types = fixture_types();
        let bytes = fixture_gguf_with_merges("gpt2", "qwen35", &tokens, &types, MERGES);
        with_file(bytes, |p| {
            let r = GgufReader::open(p).expect("parse");
            let e = Gpt2Tokenizer::from_gguf(&r).unwrap_err();
            assert!(e.to_string().contains("byte-to-unicode"), "{e}");
        });
    }

    #[test]
    fn from_gguf_refuses_missing_token_type() {
        let bytes = fixture_gguf("gpt2", "qwen35", TOKENS, &[], &[]);
        with_file(bytes, |p| {
            let r = GgufReader::open(p).expect("parse");
            let e = Gpt2Tokenizer::from_gguf(&r).unwrap_err();
            assert!(e.to_string().contains("token_type"), "{e}");
        });
    }

    /// Pin contra la fuente del oráculo: estos strings son EXACTAMENTE los de
    /// llama-vocab.cpp (el dispatch de llama.cpp matchea por igualdad de
    /// string — si alguien los edita "arreglando" algo, este test frena).
    #[test]
    fn regex_strings_pin() {
        assert_eq!(RE_QWEN35, "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
        assert_eq!(RE_QWEN2, "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
        assert_eq!(
            RE_GPT2,
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
        );
    }

    /// La tabla gpt2: printables → byte, resto ascendente → 256+.
    #[test]
    fn byte_to_unicode_table() {
        let (b2u, u2b) = byte_to_unicode();
        assert_eq!(b2u[65], 'A');
        assert_eq!(b2u[32] as u32, 0x120); // space → Ġ
        assert_eq!(b2u[10] as u32, 0x10A); // \n → Ċ
        assert_eq!(b2u[33], '!');
        assert_eq!(b2u[255], 'ÿ');
        assert_eq!(u2b[&'Ġ'], 32);
        assert_eq!(u2b[&'Ċ'], 10);
        assert_eq!(u2b[&'A'], 65);
    }
}
