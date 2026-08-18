//! Presupuesto de memoria controlada (Fase 9) y contabilidad de memoria.
//!
//! Qué ES el presupuesto: el tope de la memoria que UNLTD mantiene residente
//! DELIBERADAMENTE — pesos materializados en heap, KV cache, scratch de
//! activaciones, estados recurrentes y overhead de runtime.
//!
//! Qué NO es (y por qué `mapped > budget` es VÁLIDO):
//! - **NO limita el archivo mapeado**: el modelo entero vive en mmap virtual;
//!   `mapped bytes` (espacio virtual) != `resident bytes` (RAM).
//! - **NO limita el page cache del SO**: las páginas del archivo que el SO
//!   cachea tras tocarlas aparecen en el RSS del proceso y son exactamente el
//!   mecanismo disk-first (demand paging). No son asignación nuestra.
//! - El PEAK RSS medido puede por eso superar el presupuesto sin violarlo; la
//!   cifra autoritativa del presupuesto es `used_controlled_bytes()`.

use std::fmt;

/// Error de parseo de un tamaño (`parse_size`). Distinto por directiva:
/// vacío, cero, sufijo desconocido, overflow y no-número son rechazos
/// separados — el CLI imprime cuál.
#[derive(Debug, PartialEq, Eq)]
pub enum ParseSizeError {
    Empty,
    Zero,
    UnknownSuffix(String),
    Overflow,
    NotANumber,
}

impl fmt::Display for ParseSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseSizeError::Empty => write!(f, "vacío"),
            ParseSizeError::Zero => write!(f, "cero — el presupuesto debe ser > 0"),
            ParseSizeError::UnknownSuffix(s) => write!(
                f,
                "sufijo {s:?} desconocido (válidos: K/KB, M/MB, G/GB, T/TB, o bytes sin sufijo)"
            ),
            ParseSizeError::Overflow => write!(f, "desborda u64"),
            ParseSizeError::NotANumber => write!(f, "no es un número"),
        }
    }
}

impl std::error::Error for ParseSizeError {}

/// Sufijo de memoria (K/M/G/T, opcional B) contra una cadena, case-insensitive.
/// `strip_suffix_ci("4GB", "GB") == Some("4")`.
fn strip_suffix_ci<'a>(s: &'a str, suf: &str) -> Option<&'a str> {
    if s.len() >= suf.len() && s[s.len() - suf.len()..].eq_ignore_ascii_case(suf) {
        Some(&s[..s.len() - suf.len()])
    } else {
        None
    }
}

/// Parsea un tamaño de memoria a bytes (u64). Acepta como mínimo:
/// `512M`, `1G`, `2G`, `4G`, `8G`, `512MB`, `4GB` y bytes crudos (`536870912`).
/// Prefijos BINARIOS (convención de memoria: 1G = 1024³), case-insensitive.
/// Rechaza: vacío, cero, sufijo desconocido, overflow u64 y no-números.
pub fn parse_size(s: &str) -> Result<u64, ParseSizeError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ParseSizeError::Empty);
    }
    // Sufijos largos ("KB") antes que cortos ("K").
    const SUFFIXES: [(&str, u64); 8] = [
        ("KB", 1024),
        ("MB", 1024 * 1024),
        ("GB", 1024 * 1024 * 1024),
        ("TB", 1024 * 1024 * 1024 * 1024),
        ("K", 1024),
        ("M", 1024 * 1024),
        ("G", 1024 * 1024 * 1024),
        ("T", 1024 * 1024 * 1024 * 1024),
    ];
    let (num_part, mult) = SUFFIXES
        .iter()
        .find_map(|&(suf, m)| strip_suffix_ci(s, suf).map(|n| (n, m)))
        .unwrap_or_else(|| {
            // Sin sufijo conocido: bytes crudos si es todo dígitos; si no,
            // `mult = 0` marca el error y se tipifica abajo.
            if s.bytes().all(|b| b.is_ascii_digit()) {
                (s, 1)
            } else {
                (s, 0)
            }
        });
    if mult == 0 {
        // Sufijo desconocido (termina en letras que no son un sufijo válido)
        // vs no-número (dígitos mezclados con otra cosa).
        let alpha_tail = s.trim_start_matches(|c: char| c.is_ascii_digit());
        if !alpha_tail.is_empty() && alpha_tail.bytes().all(|b| b.is_ascii_alphabetic()) {
            return Err(ParseSizeError::UnknownSuffix(alpha_tail.to_string()));
        }
        return Err(ParseSizeError::NotANumber);
    }
    let num: u64 = num_part.parse().map_err(|_| {
        // Todo-dígitos que no caben en u64 = Overflow (el usuario dio un
        // número válido en formato, demasiado grande); basura mezclada =
        // NotANumber.
        if !num_part.is_empty() && num_part.bytes().all(|b| b.is_ascii_digit()) {
            ParseSizeError::Overflow
        } else {
            ParseSizeError::NotANumber
        }
    })?;
    if num == 0 {
        return Err(ParseSizeError::Zero);
    }
    num.checked_mul(mult).ok_or(ParseSizeError::Overflow)
}

/// Contabilidad de la memoria controlada (Fase 9).
///
/// Las 7 cifras que el componente DEBE conocer:
/// `configured_budget` (tope), `mandatory_bytes` (mínimo para correr),
/// `weight_buffer_bytes` (pesos materializados en heap — 0 con mmap puro),
/// `weight_cache_bytes` (cache de bloques empaquetados — 0 sin cache),
/// `kv_cache_bytes`, `scratch_bytes` (pico de activaciones), `runtime_bytes`
/// (tokenizer + índice + overhead).
#[derive(Debug, Clone, Copy)]
pub struct MemoryAccounting {
    pub configured_budget: u64,
    pub mandatory_bytes: u64,
    pub weight_buffer_bytes: u64,
    pub weight_cache_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub runtime_bytes: u64,
}

impl MemoryAccounting {
    pub fn new(configured_budget: u64, mandatory_bytes: u64) -> Self {
        Self {
            configured_budget,
            mandatory_bytes,
            weight_buffer_bytes: 0,
            weight_cache_bytes: 0,
            kv_cache_bytes: 0,
            scratch_bytes: 0,
            runtime_bytes: 0,
        }
    }

    /// Bytes controlados en uso: las 5 categorías deliberadamente residentes.
    /// Suma saturante (un plan corrupto no desborda en silencio).
    pub fn used_controlled_bytes(&self) -> u64 {
        self.weight_buffer_bytes
            .saturating_add(self.weight_cache_bytes)
            .saturating_add(self.kv_cache_bytes)
            .saturating_add(self.scratch_bytes)
            .saturating_add(self.runtime_bytes)
    }

    /// Presupuesto disponible: `configured - used`, nunca negativo.
    pub fn available_budget(&self) -> u64 {
        self.configured_budget
            .saturating_sub(self.used_controlled_bytes())
    }

    /// true sii el presupuesto respeta AMBOS contratos: el mínimo mandatorio
    /// para correr entra, y lo usado no excede lo configurado. Un plan que
    /// reserva más que el presupuesto se NIEGA antes de asignar (contrato del
    /// CLI: "REFUSING TO RUN" con números, nunca un OOM a mitad de corrida).
    pub fn budget_respected(&self) -> bool {
        self.mandatory_bytes <= self.configured_budget
            && self.used_controlled_bytes() <= self.configured_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1024 * 1024 * 1024;
    const M: u64 = 1024 * 1024;

    #[test]
    fn parses_short_suffixes() {
        assert_eq!(parse_size("4G"), Ok(4 * G));
        assert_eq!(parse_size("2G"), Ok(2 * G));
        assert_eq!(parse_size("1G"), Ok(G));
        assert_eq!(parse_size("8G"), Ok(8 * G));
        assert_eq!(parse_size("512M"), Ok(512 * M));
    }

    #[test]
    fn parses_long_suffixes() {
        assert_eq!(parse_size("4GB"), Ok(4 * G));
        assert_eq!(parse_size("512MB"), Ok(512 * M));
        assert_eq!(parse_size("1KB"), Ok(1024));
        assert_eq!(parse_size("1K"), Ok(1024));
    }

    #[test]
    fn parses_case_insensitive_and_trim() {
        assert_eq!(parse_size("4g"), Ok(4 * G));
        assert_eq!(parse_size("512mb"), Ok(512 * M));
        assert_eq!(parse_size(" 4G "), Ok(4 * G));
    }

    #[test]
    fn parses_raw_bytes() {
        assert_eq!(parse_size("536870912"), Ok(512 * M));
        assert_eq!(parse_size("1"), Ok(1));
        assert_eq!(parse_size("18446744073709551615"), Ok(u64::MAX));
    }

    #[test]
    fn rejects_empty_and_zero() {
        assert_eq!(parse_size(""), Err(ParseSizeError::Empty));
        assert_eq!(parse_size("   "), Err(ParseSizeError::Empty));
        assert_eq!(parse_size("0"), Err(ParseSizeError::Zero));
        assert_eq!(parse_size("0G"), Err(ParseSizeError::Zero));
        assert_eq!(parse_size("000"), Err(ParseSizeError::Zero));
    }

    #[test]
    fn rejects_unknown_suffix() {
        assert_eq!(
            parse_size("4X"),
            Err(ParseSizeError::UnknownSuffix("X".into()))
        );
        assert_eq!(
            parse_size("512BLAH"),
            Err(ParseSizeError::UnknownSuffix("BLAH".into()))
        );
        assert_eq!(
            parse_size("foo"),
            Err(ParseSizeError::UnknownSuffix("foo".into()))
        );
    }

    #[test]
    fn rejects_not_a_number() {
        assert_eq!(parse_size("1.5G"), Err(ParseSizeError::NotANumber));
        assert_eq!(parse_size("-4G"), Err(ParseSizeError::NotANumber));
        assert_eq!(parse_size("5GB9"), Err(ParseSizeError::NotANumber));
        assert_eq!(parse_size("4G5M"), Err(ParseSizeError::NotANumber));
    }

    #[test]
    fn rejects_overflow() {
        // u64::MAX × 1024 no cabe; u64::MAX sin sufijo sí (bytes crudos).
        assert_eq!(
            parse_size("18446744073709551615G"),
            Err(ParseSizeError::Overflow)
        );
        assert_eq!(
            parse_size("99999999999999999999999G"),
            Err(ParseSizeError::Overflow)
        );
    }

    #[test]
    fn accounting_sums_and_releases() {
        let mut a = MemoryAccounting::new(4 * G, 0);
        assert_eq!(a.used_controlled_bytes(), 0);
        assert_eq!(a.available_budget(), 4 * G);
        assert!(a.budget_respected());

        a.kv_cache_bytes = 1 * M;
        a.scratch_bytes = 5 * M;
        a.runtime_bytes = 30 * M;
        assert_eq!(a.used_controlled_bytes(), 36 * M);
        assert_eq!(a.available_budget(), 4 * G - 36 * M);
        assert!(a.budget_respected());

        // Release: volver una categoría a cero baja el uso.
        a.scratch_bytes = 0;
        assert_eq!(a.used_controlled_bytes(), 31 * M);
    }

    #[test]
    fn accounting_exact_limit_is_respected() {
        let mut a = MemoryAccounting::new(10 * M, 10 * M);
        a.kv_cache_bytes = 10 * M;
        assert_eq!(a.used_controlled_bytes(), 10 * M);
        assert_eq!(a.available_budget(), 0);
        assert!(a.budget_respected());
    }

    #[test]
    fn accounting_exceeded_is_not_respected() {
        let mut a = MemoryAccounting::new(10 * M, 0);
        a.kv_cache_bytes = 11 * M;
        assert_eq!(a.available_budget(), 0); // satura, nunca negativo
        assert!(!a.budget_respected());
    }

    #[test]
    fn accounting_mandatory_over_budget_is_not_respected() {
        // El mínimo mandatorio no entra aunque todavía no se haya asignado nada.
        let a = MemoryAccounting::new(512 * M, 734 * M);
        assert_eq!(a.used_controlled_bytes(), 0);
        assert!(!a.budget_respected());
    }

    #[test]
    fn accounting_overflow_saturates() {
        let mut a = MemoryAccounting::new(u64::MAX, 0);
        a.kv_cache_bytes = u64::MAX;
        a.scratch_bytes = u64::MAX;
        assert_eq!(a.used_controlled_bytes(), u64::MAX);
        assert_eq!(a.available_budget(), 0);
    }
}
