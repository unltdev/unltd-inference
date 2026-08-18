//! Lector GGUF hecho a mano, std-only (sin serde para parsear).
//! Formato: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//!
//! Diseño heredado de `k3_st.c` (ver `docs/AUDIT.md` §3.2): el lector valida ANTES de
//! servir un byte — offsets dentro del archivo, tipo conocido, nbytes coherente con la
//! forma — y se NIEGA ante un archivo truncado o un tensor que no existe. Un peso
//! ausente leído como ceros produce un modelo que corre, fluido y equivocado.
//!
//! Checks duros (orden de ejecución):
//! 1. magic "GGUF"; versión 2 o 3 (el resto se rechaza);
//! 2. caps anti-OOM sobre n_kv, n_tensors, largos de string y arrays (nunca confiar
//!    en un largo leído del archivo para dimensionar una alocación);
//! 3. n_dims ≤ 64, n_elements con mul checked, tensores zero-size = error;
//! 4. offsets de tensor RELATIVOS al inicio de la sección de datos (así los escribe
//!    llama.cpp — verificado en gguf.cpp y byte a byte contra el archivo real):
//!    absoluto = data_start + relativo; absoluto % 32 == 0, absoluto+nbytes ≤ file_size,
//!    sin overlap entre rangos (orden por offset). NO hay padding de 32 entre pares
//!    KV ni entre infos de tensores: el único padding del header precede a los datos;
//! 5. tipo ggml desconocido → `n_bytes: None` + warning en el resumen, NUNCA adivinar.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Seek};
use std::path::Path;

use unltd_core::LoadError;

use crate::TensorMeta;

/// Alineación de todas las secciones del header y de los datos de tensores.
pub const GGUF_ALIGN: u64 = 32;

// ---------------------------------------------------------------------------
// Tipos de valor de metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufValueType {
    /// Id de tipo → tipo. `None` = id fuera del rango 0..=12: se rechaza, no se adivina.
    fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            _ => None,
        }
    }

    /// Bytes por elemento para dimensionar el cap anti-OOM de arrays.
    fn elem_bytes(self) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => 1, // strings de largo variable: el cap real lo impone el largo total
            Self::Array => 8,  // no ocurre: los elementos de array no son arrays
        }
    }

    /// Nombre legible para mensajes de error.
    fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }
}

/// Un valor de metadata GGUF. Los arrays son vectores tipados (no `Vec<Value>` anidado):
/// el consumidor real (p. ej. `tokenizer.ggml.tokens`) quiere un `Vec<String>` directo.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    U64(u64),
    I64(i64),
    Bool(bool),
    Str(String),
    Arr(GgufArray),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

impl GgufValue {
    /// Descripción compacta para el `inspect` (arrays resumidos, strings literales).
    pub fn describe(&self) -> String {
        match self {
            Self::U8(v) => format!("u8 {v}"),
            Self::I8(v) => format!("i8 {v}"),
            Self::U16(v) => format!("u16 {v}"),
            Self::I16(v) => format!("i16 {v}"),
            Self::U32(v) => format!("u32 {v}"),
            Self::I32(v) => format!("i32 {v}"),
            Self::F32(v) => format!("f32 {v}"),
            Self::F64(v) => format!("f64 {v}"),
            Self::U64(v) => format!("u64 {v}"),
            Self::I64(v) => format!("i64 {v}"),
            Self::Bool(v) => format!("bool {v}"),
            Self::Str(s) => format!("{s:?}"),
            Self::Arr(a) => a.describe(),
        }
    }
}

impl GgufArray {
    fn describe(&self) -> String {
        let (t, n) = match self {
            Self::U8(v) => ("u8", v.len()),
            Self::I8(v) => ("i8", v.len()),
            Self::U16(v) => ("u16", v.len()),
            Self::I16(v) => ("i16", v.len()),
            Self::U32(v) => ("u32", v.len()),
            Self::I32(v) => ("i32", v.len()),
            Self::F32(v) => ("f32", v.len()),
            Self::F64(v) => ("f64", v.len()),
            Self::U64(v) => ("u64", v.len()),
            Self::I64(v) => ("i64", v.len()),
            Self::Bool(v) => ("bool", v.len()),
            Self::Str(v) => ("string", v.len()),
        };
        format!("{t}[{n}]")
    }
}

// ---------------------------------------------------------------------------
// Tabla de tipos ggml
// ---------------------------------------------------------------------------

/// (bytes por bloque, elementos por bloque). Verificada contra
/// `ggml/src/ggml-common.h` de llama.cpp (fdb1db8) — cada entrada sale de un
/// `static_assert(sizeof(block_*))` de la fuente, no de memoria.
///
/// `None` = tipo desconocido o retirado: el lector lo reporta y deja `n_bytes: None`;
/// NUNCA adivina un tamaño.
pub fn ggml_type_blocksize(id: u32) -> Option<(u64, u64)> {
    match id {
        0 => Some((4, 1)),      // F32
        1 => Some((2, 1)),      // F16
        2 => Some((18, 32)),    // Q4_0
        3 => Some((20, 32)),    // Q4_1
        4 | 5 => None,          // Q4_2 / Q4_3: retirados del formato
        6 => Some((22, 32)),    // Q5_0
        7 => Some((24, 32)),    // Q5_1
        8 => Some((34, 32)),    // Q8_0
        9 => Some((36, 32)),    // Q8_1
        10 => Some((84, 256)),  // Q2_K
        11 => Some((110, 256)), // Q3_K
        12 => Some((144, 256)), // Q4_K
        13 => Some((176, 256)), // Q5_K
        14 => Some((210, 256)), // Q6_K
        15 => Some((292, 256)), // Q8_K
        16 => Some((66, 256)),  // IQ2_XXS
        17 => Some((74, 256)),  // IQ2_XS
        18 => Some((98, 256)),  // IQ3_XXS
        19 => Some((50, 256)),  // IQ1_S
        20 => Some((18, 32)),   // IQ4_NL
        21 => Some((110, 256)), // IQ3_S
        22 => Some((82, 256)),  // IQ2_S
        23 => Some((136, 256)), // IQ4_XS
        24 => Some((1, 1)),     // I8
        25 => Some((2, 1)),     // I16
        26 => Some((4, 1)),     // I32
        27 => Some((8, 1)),     // I64
        28 => Some((8, 1)),     // F64
        29 => Some((56, 256)),  // IQ1_M
        30 => Some((2, 1)),     // BF16
        31..=33 => None,        // Q4_0_4_4/4_8/8_8: retirados de gguf
        34 => Some((54, 256)),  // TQ1_0
        35 => Some((66, 256)),  // TQ2_0
        36..=38 => None,        // IQ4_NL_4_4/4_8/8_8: retirados
        39 => Some((17, 32)),   // MXFP4
        40 => Some((36, 64)),   // NVFP4
        41 => Some((18, 128)),  // Q1_0
        _ => None,
    }
}

/// Nombre legible del tipo ggml para reportes.
pub fn ggml_type_name(id: u32) -> String {
    match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        4 | 5 => "retired(Q4_2/Q4_3)",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31..=33 | 36..=38 => "retired",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        other => return format!("unknown({other})"),
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// El lector
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GgufReader {
    pub file_size: u64,
    pub version: u32,
    pub metadata: Vec<(String, GgufValue)>,
    tensors: Vec<TensorMeta>,
    by_name: HashMap<String, u32>,
    /// Offset absoluto donde empieza la sección de datos (primera posición alineada a
    /// 32 tras la última info de tensor). Los offsets de `TensorMeta` ya son absolutos.
    pub data_start: u64,
}

impl GgufReader {
    /// Abre un archivo GGUF y valida el header completo. Los bytes de pesos NO se leen.
    pub fn open(path: &Path) -> Result<Self, LoadError> {
        let f = File::open(path).map_err(LoadError::io)?;
        let file_size = f.metadata().map_err(LoadError::io)?.len();
        let mut br = BufReader::new(f);
        parse(&mut br, file_size)
    }

    /// Metadata por clave. Primera aparición si la clave está duplicada.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Metadata string por clave.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(GgufValue::Str(s)) => Some(s),
            _ => None,
        }
    }

    pub fn tensors(&self) -> &[TensorMeta] {
        &self.tensors
    }

    /// (suma de bytes de tensores conocidos, nº de tensores con tipo desconocido).
    /// Si hay desconocidos, la suma es un piso, nunca un invento.
    pub fn tensor_bytes_summary(&self) -> (u64, usize) {
        let mut sum = 0u64;
        let mut unknown = 0usize;
        for t in &self.tensors {
            match t.nbytes {
                Some(b) => sum += b,
                None => unknown += 1,
            }
        }
        (sum, unknown)
    }

    pub fn unknown_type_tensors(&self) -> impl Iterator<Item = &TensorMeta> {
        self.tensors.iter().filter(|t| t.nbytes.is_none())
    }
}

impl crate::WeightIndex for GgufReader {
    fn find(&self, name: &str) -> Option<&TensorMeta> {
        self.by_name.get(name).map(|&i| &self.tensors[i as usize])
    }

    fn len(&self) -> usize {
        self.tensors.len()
    }
}

/// Parser puro sobre cualquier `Read + Seek` (archivo o buffer de tests).
/// `file_size` es el tamaño REAL del archivo; todos los checks de rango lo usan.
pub fn parse<R: Read + Seek>(r: &mut R, file_size: u64) -> Result<GgufReader, LoadError> {
    // 1. magic + versión
    let mut magic = [0u8; 4];
    read_checked(r, &mut magic, file_size)?;
    if &magic != b"GGUF" {
        return Err(LoadError::BadMagic { magic });
    }
    let version = rd_u32(r, file_size)?;
    if version != 2 && version != 3 {
        return Err(LoadError::UnsupportedVersion(version));
    }

    // 2. conteos con caps anti-OOM (una alocación jamás se dimensiona por un número
    //    leído del archivo sin un cap contra el tamaño real)
    let n_tensors = rd_u64(r, file_size)?;
    let n_kv = rd_u64(r, file_size)?;
    if n_kv > file_size / 8 {
        return Err(LoadError::corrupt(format!(
            "n_kv {n_kv} is implausible for a file of {file_size} bytes"
        )));
    }
    if n_tensors > file_size / 16 {
        return Err(LoadError::corrupt(format!(
            "n_tensors {n_tensors} is implausible for a file of {file_size} bytes"
        )));
    }

    // 3. metadata. Los pares KV son contiguos: NO hay padding entre ellos
    //    (verificado byte a byte contra el archivo real y en gguf.cpp).
    let mut metadata = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = rd_string(r, file_size)?;
        let value = rd_value(r, file_size)?;
        metadata.push((key, value));
    }
    // `general.alignment` (llama.cpp lo honra como potencia de 2). Un valor distinto
    // haría que este parser leyera los offsets mal: negarse, no adivinar.
    if let Some(GgufValue::U32(a)) = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .map(|(_, v)| v)
    {
        if *a != GGUF_ALIGN as u32 {
            return Err(LoadError::corrupt(format!(
                "general.alignment = {a}: only {GGUF_ALIGN} is supported"
            )));
        }
    }

    // 4. infos de tensores, contiguas (sin padding entre ellas; el nombre del
    //    siguiente tensor empieza justo tras el offset del anterior)
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = rd_string(r, file_size)?;
        let n_dims = rd_u32(r, file_size)?;
        if n_dims == 0 {
            return Err(LoadError::corrupt(format!(
                "tensor '{name}' has n_dims = 0"
            )));
        }
        if n_dims > 64 {
            return Err(LoadError::corrupt(format!(
                "tensor '{name}' has n_dims {n_dims} (sane limit: 64)"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(rd_u64(r, file_size)?);
        }
        let ggml_type = rd_u32(r, file_size)?;
        let offset = rd_u64(r, file_size)?;

        let n_elements = dims
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d))
            .ok_or_else(|| {
                LoadError::corrupt(format!(
                    "tensor '{name}': element count overflows u64"
                ))
            })?;
        if n_elements == 0 {
            return Err(LoadError::corrupt(format!(
                "tensor '{name}' has zero elements"
            )));
        }
        // nbytes: ceil(n_elements / elems_por_bloque) * bytes_por_bloque (el último
        // bloque se rellena). Tipo desconocido → None, nunca adivinar.
        let nbytes = ggml_type_blocksize(ggml_type).and_then(|(bb, be)| {
            n_elements
                .checked_add(be - 1)
                .map(|padded| (padded / be) * bb)
        });

        tensors.push(TensorMeta {
            name,
            offset,
            nbytes,
            dtype: ggml_type_name(ggml_type),
            ggml_type_id: ggml_type,
            shape: dims,
            n_elements,
        });
    }

    // La sección de datos empieza en la primera posición alineada a 32 tras la
    // última info (llama.cpp: GGML_PAD(infos_end, alignment)). Los offsets leídos
    // son RELATIVOS a ese punto: absoluto = data_start + relativo.
    let infos_end = r.stream_position().map_err(LoadError::io)?;
    let data_start = (infos_end + GGUF_ALIGN - 1) / GGUF_ALIGN * GGUF_ALIGN;
    for t in &mut tensors {
        t.offset = data_start.checked_add(t.offset).ok_or_else(|| {
            LoadError::corrupt(format!(
                "tensor '{}': absolute offset overflows u64",
                t.name
            ))
        })?;
    }

    // 5. checks duros sobre los rangos de datos (orden por offset absoluto)
    let mut sorted: Vec<&TensorMeta> = tensors.iter().collect();
    sorted.sort_by_key(|t| t.offset);
    let mut prev: Option<&TensorMeta> = None;
    for t in &sorted {
        if t.offset % GGUF_ALIGN != 0 {
            return Err(LoadError::MisalignedTensor {
                name: t.name.clone(),
                offset: t.offset,
                align: GGUF_ALIGN,
            });
        }
        if let Some(nb) = t.nbytes {
            if t.offset.checked_add(nb).map_or(true, |end| end > file_size) {
                return Err(LoadError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.offset,
                    nbytes: nb,
                    file_size,
                });
            }
            if let Some(pr) = prev {
                if let Some(pr_end) = pr.nbytes.map(|nb| pr.offset + nb) {
                    if pr_end > t.offset {
                        return Err(LoadError::TensorOverlap {
                            a: pr.name.clone(),
                            b: t.name.clone(),
                            a_end: pr_end,
                            b_start: t.offset,
                        });
                    }
                }
            }
        }
        prev = Some(t);
    }

    // 6. índice por nombre
    let mut by_name = HashMap::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        by_name.insert(t.name.clone(), i as u32);
    }

    Ok(GgufReader {
        file_size,
        version,
        metadata,
        tensors,
        by_name,
        data_start,
    })
}

// ---------------------------------------------------------------------------
// Lectores primitivos con posición (todos los errores llevan contexto de offset)
// ---------------------------------------------------------------------------

fn pos<R: Read + Seek>(r: &mut R) -> Result<u64, LoadError> {
    r.stream_position().map_err(LoadError::io)
}

/// `read_exact` con checks previos: nunca se pide un buffer más allá del EOF declarado,
/// y un EOF inesperado se reporta como archivo truncado (con offset), no como I/O genérico.
fn read_checked<R: Read + Seek>(r: &mut R, buf: &mut [u8], file_size: u64) -> Result<(), LoadError> {
    let at = pos(r)?;
    if at.checked_add(buf.len() as u64).map_or(true, |end| end > file_size) {
        return Err(LoadError::corrupt(format!(
            "read of {} bytes at offset {at} exceeds file size {file_size}",
            buf.len()
        )));
    }
    r.read_exact(buf).map_err(|e| match e.kind() {
        ErrorKind::UnexpectedEof => LoadError::corrupt(format!(
            "truncated file: unexpected EOF at offset {at} (file size {file_size})"
        )),
        _ => LoadError::io(e),
    })
}

fn rd_u8<R: Read + Seek>(r: &mut R, fs: u64) -> Result<u8, LoadError> {
    let mut b = [0u8; 1];
    read_checked(r, &mut b, fs)?;
    Ok(b[0])
}

fn rd_u32<R: Read + Seek>(r: &mut R, fs: u64) -> Result<u32, LoadError> {
    let mut b = [0u8; 4];
    read_checked(r, &mut b, fs)?;
    Ok(u32::from_le_bytes(b))
}

fn rd_i32<R: Read + Seek>(r: &mut R, fs: u64) -> Result<i32, LoadError> {
    Ok(rd_u32(r, fs)? as i32)
}

fn rd_u64<R: Read + Seek>(r: &mut R, fs: u64) -> Result<u64, LoadError> {
    let mut b = [0u8; 8];
    read_checked(r, &mut b, fs)?;
    Ok(u64::from_le_bytes(b))
}

fn rd_i64<R: Read + Seek>(r: &mut R, fs: u64) -> Result<i64, LoadError> {
    Ok(rd_u64(r, fs)? as i64)
}

fn rd_f32<R: Read + Seek>(r: &mut R, fs: u64) -> Result<f32, LoadError> {
    Ok(f32::from_bits(rd_u32(r, fs)?))
}

fn rd_f64<R: Read + Seek>(r: &mut R, fs: u64) -> Result<f64, LoadError> {
    Ok(f64::from_bits(rd_u64(r, fs)?))
}

/// String: u64 de largo + bytes UTF-8 (lossy: la metadata admite bytes no-UTF-8,
/// p. ej. `tokenizer.ggml.tokens`).
fn rd_string<R: Read + Seek>(r: &mut R, fs: u64) -> Result<String, LoadError> {
    let at = pos(r)?;
    let len = rd_u64(r, fs)?;
    if len > fs {
        return Err(LoadError::corrupt(format!(
            "string length {len} at offset {at} exceeds file size {fs}"
        )));
    }
    let mut b = vec![0u8; len as usize];
    read_checked(r, &mut b, fs)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn rd_value<R: Read + Seek>(r: &mut R, fs: u64) -> Result<GgufValue, LoadError> {
    let at = pos(r)?;
    let vt_id = rd_u32(r, fs)?;
    let vt = GgufValueType::from_id(vt_id).ok_or_else(|| {
        LoadError::corrupt(format!(
            "unknown metadata value type id {vt_id} at offset {at}"
        ))
    })?;
    let v = match vt {
        GgufValueType::U8 => GgufValue::U8(rd_u8(r, fs)?),
        GgufValueType::I8 => GgufValue::I8(rd_u8(r, fs)? as i8),
        GgufValueType::U16 => {
            let mut b = [0u8; 2];
            read_checked(r, &mut b, fs)?;
            GgufValue::U16(u16::from_le_bytes(b))
        }
        GgufValueType::I16 => {
            let mut b = [0u8; 2];
            read_checked(r, &mut b, fs)?;
            GgufValue::I16(i16::from_le_bytes(b))
        }
        GgufValueType::U32 => GgufValue::U32(rd_u32(r, fs)?),
        GgufValueType::I32 => GgufValue::I32(rd_i32(r, fs)?),
        GgufValueType::F32 => GgufValue::F32(rd_f32(r, fs)?),
        GgufValueType::F64 => GgufValue::F64(rd_f64(r, fs)?),
        GgufValueType::U64 => GgufValue::U64(rd_u64(r, fs)?),
        GgufValueType::I64 => GgufValue::I64(rd_i64(r, fs)?),
        GgufValueType::Bool => GgufValue::Bool(rd_u8(r, fs)? != 0),
        GgufValueType::String => GgufValue::Str(rd_string(r, fs)?),
        GgufValueType::Array => {
            let elem_id = rd_u32(r, fs)?;
            let elem = GgufValueType::from_id(elem_id).ok_or_else(|| {
                LoadError::corrupt(format!(
                    "unknown array element type id {elem_id} at offset {at}"
                ))
            })?;
            let count = rd_u64(r, fs)?;
            // cap anti-OOM: el array completo no puede exceder el archivo entero
            let elem_bytes = elem.elem_bytes();
            if count
                .checked_mul(elem_bytes)
                .map_or(true, |total| total > fs)
            {
                return Err(LoadError::corrupt(format!(
                    "array of {count} {}(s) at offset {at} exceeds file size {fs}",
                    elem.name()
                )));
            }
            GgufValue::Arr(rd_array(r, fs, elem, count)?)
        }
    };
    Ok(v)
}

fn rd_array<R: Read + Seek>(
    r: &mut R,
    fs: u64,
    elem: GgufValueType,
    count: u64,
) -> Result<GgufArray, LoadError> {
    let n = count as usize;
    macro_rules! vec_of {
        ($rd:ident, $variant:ident, $t:ty) => {{
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push($rd(r, fs)? as $t);
            }
            GgufArray::$variant(v)
        }};
    }
    Ok(match elem {
        GgufValueType::U8 => vec_of!(rd_u8, U8, u8),
        GgufValueType::I8 => vec_of!(rd_u8, I8, i8),
        GgufValueType::U16 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let mut b = [0u8; 2];
                read_checked(r, &mut b, fs)?;
                v.push(u16::from_le_bytes(b));
            }
            GgufArray::U16(v)
        }
        GgufValueType::I16 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let mut b = [0u8; 2];
                read_checked(r, &mut b, fs)?;
                v.push(i16::from_le_bytes(b));
            }
            GgufArray::I16(v)
        }
        GgufValueType::U32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_u32(r, fs)?);
            }
            GgufArray::U32(v)
        }
        GgufValueType::I32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_i32(r, fs)?);
            }
            GgufArray::I32(v)
        }
        GgufValueType::F32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_f32(r, fs)?);
            }
            GgufArray::F32(v)
        }
        GgufValueType::F64 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_f64(r, fs)?);
            }
            GgufArray::F64(v)
        }
        GgufValueType::U64 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_u64(r, fs)?);
            }
            GgufArray::U64(v)
        }
        GgufValueType::I64 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_i64(r, fs)?);
            }
            GgufArray::I64(v)
        }
        GgufValueType::Bool => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_u8(r, fs)? != 0);
            }
            GgufArray::Bool(v)
        }
        GgufValueType::String => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rd_string(r, fs)?);
            }
            GgufArray::Str(v)
        }
        GgufValueType::Array => {
            return Err(LoadError::corrupt("nested array in GGUF metadata"));
        }
    })
}

// ---------------------------------------------------------------------------
// Tests: fixtures sintéticos construidos byte a byte (escritor de GGUF mínimo)
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    /// Escritor de GGUF sintético para fixtures (little-endian; header contiguo
    /// como lo escribe llama.cpp: el único alineado son los datos de tensores).
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
        /// Rellena con ceros hasta `target` bytes exactos (para simular datos de tensores).
        fn pad_to(mut self, target: usize) -> Self {
            assert!(self.0.len() <= target, "pad_to: already past {target}");
            self.0.resize(target, 0);
            self
        }
        fn finish(self) -> Vec<u8> {
            self.0
        }
    }

    fn parse_bytes(b: &[u8]) -> Result<GgufReader, LoadError> {
        parse(&mut Cursor::new(b), b.len() as u64)
    }

    /// GGUF v3 válido: 1 kv string + 1 kv array u32, 2 tensores (F32 y Q8_0).
    /// Layout REAL (verificado contra llama.cpp): pares KV e infos de tensor
    /// contiguos, sin padding; las infos terminan en 217 → data_start = 224.
    /// Offsets relativos: t0 @0 (abs 224, 48 B de F32), t1 @64 (abs 288, 68 B de
    /// Q8_0); el archivo se rellena hasta 356.
    pub(crate) fn fixture_valid() -> Vec<u8> {
        let header = W::new()
            .raw(b"GGUF")
            .u32(3) // version
            .u64(2) // n_tensors
            .u64(2) // n_kv
            // kv 1: general.architecture = "llama"
            .str("general.architecture")
            .u32(8) // STRING
            .str("llama")
            // kv 2: test.counts = [u32; 3]
            .str("test.counts")
            .u32(9) // ARRAY
            .u32(4) // elem U32
            .u64(3) // count
            .u32(10)
            .u32(20)
            .u32(30)
            // tensor 1: tok_embd.weight, 2 dims [4, 3], F32(0), offset relativo 0
            .str("tok_embd.weight")
            .u32(2)
            .u64(4)
            .u64(3)
            .u32(0)
            .u64(0)
            // tensor 2: norm.weight, 1 dim [33], Q8_0(8), offset relativo 64
            // (48 B de datos de t0 + 16 de padding a 32 → abs 288)
            .str("norm.weight")
            .u32(1)
            .u64(33)
            .u32(8)
            .u64(64);
        let data_start = (header.0.len() as u64 + 31) / 32 * 32;
        assert_eq!(data_start, 224);
        header
            .pad_to((data_start + 64 + 68) as usize) // datos: t0 [224,272) + t1 [288,356)
            .finish()
    }

    #[test]
    fn parses_valid_v3() {
        let b = fixture_valid();
        let g = parse_bytes(&b).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.file_size, b.len() as u64);
        assert_eq!(g.data_start, 224);
        assert_eq!(g.metadata.len(), 2);
        assert_eq!(g.get_str("general.architecture"), Some("llama"));
        match g.get("test.counts") {
            Some(GgufValue::Arr(GgufArray::U32(v))) => assert_eq!(v, &[10, 20, 30]),
            other => panic!("unexpected value: {other:?}"),
        }
        assert_eq!(g.tensors.len(), 2);
        let t0 = &g.tensors[0];
        assert_eq!(t0.name, "tok_embd.weight");
        assert_eq!(t0.shape, vec![4, 3]);
        assert_eq!(t0.n_elements, 12);
        assert_eq!(t0.nbytes, Some(48)); // 12 × 4B
        assert_eq!(t0.dtype, "F32");
        assert_eq!(t0.offset, 224); // data_start + 0
        let t1 = &g.tensors[1];
        assert_eq!(t1.shape, vec![33]);
        assert_eq!(t1.nbytes, Some(68)); // ceil(33/32) × 34 = 2 bloques
        assert_eq!(t1.dtype, "Q8_0");
        assert_eq!(t1.offset, 288); // data_start + 64
        // WeightIndex
        let idx = &g;
        assert_eq!(crate::WeightIndex::len(idx), 2);
        assert!(crate::WeightIndex::find(idx, "tok_embd.weight").is_some());
        assert!(crate::WeightIndex::find(idx, "ausente").is_none());
    }

    #[test]
    fn rejects_bad_magic() {
        let b = W::new().raw(b"NOPE").u32(3).u64(0).u64(0).finish();
        match parse_bytes(&b) {
            Err(LoadError::BadMagic { magic }) => assert_eq!(&magic, b"NOPE"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rejects_unsupported_version() {
        for v in [1u32, 4u32] {
            let b = W::new().raw(b"GGUF").u32(v).u64(0).u64(0).finish();
            match parse_bytes(&b) {
                Err(LoadError::UnsupportedVersion(got)) => assert_eq!(got, v),
                other => panic!("unexpected: {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_implausible_counts() {
        // n_kv enorme en un archivo chico → cap anti-OOM
        let b = W::new().raw(b"GGUF").u32(3).u64(0).u64(1 << 40).finish();
        assert!(matches!(parse_bytes(&b), Err(LoadError::BadFile(_))));
        // n_tensors enorme
        let b = W::new().raw(b"GGUF").u32(3).u64(1 << 40).u64(0).finish();
        assert!(matches!(parse_bytes(&b), Err(LoadError::BadFile(_))));
    }

    #[test]
    fn rejects_truncated_file() {
        let full = fixture_valid();
        // cortar a la mitad: el parseo muere con "truncated", nunca con un OOM
        let b = &full[..full.len() / 2];
        assert!(matches!(parse_bytes(b), Err(LoadError::BadFile(_))));
    }

    #[test]
    fn rejects_huge_string_len() {
        // string que declara más bytes que el archivo entero → error antes de alocar
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0)
            .u64(1)
            .u64(1 << 50)
            .finish();
        assert!(matches!(parse_bytes(&b), Err(LoadError::BadFile(_))));
    }

    /// Header de un tensor: magic+versión+conteos+info contiguos, sin padding.
    /// Para name "w", n_dims 1 y un dim: las infos terminan en 57 → data_start = 64.
    /// `raw_offset` es el offset RELATIVO a data_start (absoluto = data_start + raw).
    fn header_one_tensor(name: &str, n_dims: u32, dims: &[u64], ggml_type: u32, raw_offset: u64) -> Vec<u8> {
        let mut w = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(1)
            .u64(0)
            .str(name)
            .u32(n_dims);
        for &d in dims {
            w = w.u64(d);
        }
        w.u32(ggml_type).u64(raw_offset).finish()
    }

    #[test]
    fn rejects_zero_dims_and_zero_elements() {
        // n_dims = 0
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(1)
            .u64(0)
            .str("w")
            .u32(0) // n_dims 0
            .finish();
        assert!(matches!(parse_bytes(&b), Err(LoadError::BadFile(_))));

        // un dim = 0 → zero elements
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(1)
            .u64(0)
            .str("w")
            .u32(1)
            .u64(0) // dim 0
            .u32(0)
            .u64(0)
            .finish();
        assert!(matches!(parse_bytes(&b), Err(LoadError::BadFile(_))));
    }

    #[test]
    fn rejects_tensor_out_of_bounds() {
        // data_start = 64 (infos terminan en 57), tensor F32 de 4 elems @abs 64
        // → [64,80) > file_size 57: el archivo declara datos que no existen.
        let b = header_one_tensor("w", 1, &[4], 0, 0);
        match parse_bytes(&b) {
            Err(LoadError::TensorOutOfBounds { name, offset, file_size, .. }) => {
                assert_eq!(name, "w");
                assert_eq!(offset, 64); // data_start + 0
                assert_eq!(file_size, 57);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rejects_misaligned_tensor() {
        // offset relativo 100 → absoluto 164, no múltiplo de 32
        let b = header_one_tensor("w", 1, &[4], 0, 100);
        match parse_bytes(&b) {
            Err(LoadError::MisalignedTensor { name, offset, align }) => {
                assert_eq!(name, "w");
                assert_eq!(offset, 164); // 64 + 100
                assert_eq!(align, 32);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rejects_overlapping_tensors() {
        // Dos infos contiguas terminan en 90 → data_start = 96. Ambos tensores con
        // offset relativo 0 (abs 96), file 128: a @[96,112) en rango, b pisa a → overlap.
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(2)
            .u64(0)
            .str("a")
            .u32(1)
            .u64(4)
            .u32(0)
            .u64(0)
            .str("b")
            .u32(1)
            .u64(4)
            .u32(0)
            .u64(0) // mismo offset relativo que a → overlap
            .pad_to(128)
            .finish();
        match parse_bytes(&b) {
            Err(LoadError::TensorOverlap { a, b: b2, .. }) => {
                assert_eq!(a, "a");
                assert_eq!(b2, "b");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn tolerates_unknown_type_without_guessing() {
        // tipo 999: nbytes None, sin checks de rango, dtype "unknown(999)"
        let b = header_one_tensor("w", 1, &[4], 999, 0);
        let g = parse_bytes(&b).unwrap();
        assert_eq!(g.tensors[0].nbytes, None);
        assert_eq!(g.tensors[0].dtype, "unknown(999)");
        assert_eq!(g.tensor_bytes_summary(), (0, 1));
        assert_eq!(g.unknown_type_tensors().count(), 1);
    }

    #[test]
    fn rejects_unknown_value_type() {
        let b = W::new()
            .raw(b"GGUF")
            .u32(3)
            .u64(0)
            .u64(1)
            .str("k")
            .u32(77) // tipo de valor desconocido
            .finish();
        assert!(matches!(parse_bytes(&b), Err(LoadError::BadFile(_))));
    }

    #[test]
    fn table_sizes_match_llama_cpp() {
        // Los valores que el PoC necesita, clavados contra ggml-common.h (fdb1db8)
        assert_eq!(ggml_type_blocksize(0), Some((4, 1))); // F32
        assert_eq!(ggml_type_blocksize(1), Some((2, 1))); // F16
        assert_eq!(ggml_type_blocksize(30), Some((2, 1))); // BF16
        assert_eq!(ggml_type_blocksize(8), Some((34, 32))); // Q8_0
        assert_eq!(ggml_type_blocksize(12), Some((144, 256))); // Q4_K
        assert_eq!(ggml_type_blocksize(14), Some((210, 256))); // Q6_K
        assert_eq!(ggml_type_blocksize(15), Some((292, 256))); // Q8_K
        assert_eq!(ggml_type_blocksize(31), None); // retirados
        assert_eq!(ggml_type_blocksize(34), Some((54, 256))); // TQ1_0
        assert_eq!(ggml_type_blocksize(999), None);
    }
}
