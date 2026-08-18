//! Tensores, dtypes y kernels del motor. Ver `docs/ARCHITECTURE.md` §3.
//!
//! **Contrato numérico** (heredado de kimi-k3-in-c, ver `docs/AUDIT.md` §3.3):
//!
//! 1. acumuladores `f64` en reducciones largas; `f32` solo donde el modelo lo define;
//! 2. orden de reducción fijo y documentado por kernel (partición `((a0+a1)+(a2+a3))...`);
//! 3. `mul_add` explícito donde se quiere FMA — nunca FMA del autovectorizador;
//! 4. el backend scalar es la referencia; AVX2 debe ser **bit-idéntico** (tests dedicados);
//! 5. el paralelismo (rayon) ocurre solo sobre filas de salida independientes, jamás
//!    dentro de una reducción.
//!
//! **Principio de formato empaquetado**: los bloques cuantizados (Q4_K, Q6_K, Q8_0, IQ…)
//! se multiplican directamente desde sus vistas, nunca se desquantizan a una copia.

use std::alloc::{alloc, dealloc, Layout};

/// Dtypes que el motor conoce. Los formatos empaquetados de GGUF son vistas, no copias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    /// GGUF Q4_K: bloques de 256 pesos, 2 escalas f16 + 12 bytes de submínimos.
    Q4K,
    /// GGUF Q6_K: bloques de 256 pesos, 1 escala f16 + q8 en 128 bytes.
    Q6K,
    /// GGUF Q8_0: bloques de 32 pesos, escala f32 por bloque + int8.
    Q8_0,
}

impl DType {
    /// Bytes por elemento lógico para dtypes no empaquetados; `None` para empaquetados.
    pub fn elem_bytes(self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::I8 | DType::U8 => Some(1),
            DType::Q4K | DType::Q6K | DType::Q8_0 => None,
        }
    }
}

/// Vista prestada sobre bytes que NO posee. Los datos pueden venir de una arena propia,
/// de un mmap, o de un slot del ring de streaming (nunca se copian).
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: DType,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Buffer alineado (`posix_memalign` en K3; `std::alloc::Layout` aquí).
///
/// Requisito del camino Direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING): el buffer, el
/// offset y la longitud deben ser múltiplos del alineamiento. Ver `docs/MEMORY-DESIGN.md`
/// §2 y §6.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    align: usize,
}

impl AlignedBuf {
    /// Asigna `len` bytes alineados a `align` (potencia de dos). `None` si el layout es
    /// inválido o la asignación falla.
    pub fn new(len: usize, align: usize) -> Option<Self> {
        assert!(align.is_power_of_two(), "align must be a power of two");
        let layout = Layout::from_size_align(len.max(1), align).ok()?;
        // SAFETY: layout válido, verificamos el puntero nulo.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, len, align })
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: el puntero viene de `alloc` con `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: idem, acceso exclusivo por &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn align(&self) -> usize {
        self.align
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: mismo layout con el que se asignó; len/align invariantes.
        let layout = unsafe { Layout::from_size_align_unchecked(self.len.max(1), self.align) };
        unsafe { dealloc(self.ptr, layout) };
    }
}

// El buffer no comparte nada; moverlo entre hilos es seguro.
unsafe impl Send for AlignedBuf {}

// ---------------------------------------------------------------------------
// Kernels: por ahora solo las firmas del backend scalar de referencia.
// Cada kernel documenta su orden de reducción ANTES de tener cuerpo.
// ---------------------------------------------------------------------------

/// RMSNorm: `out = x / sqrt(mean(x^2) + eps) * w`, acumulador f64, eps DENTRO de la raíz.
/// Orden de reducción: pares consecutivos `((a0+a1)+(a2+a3))...` en f64.
pub fn rmsnorm(_out: &mut [f32], _x: &[f32], _w: &[f32], _eps: f64) {
    todo!("Stage 0, ver docs/ROADMAP.md");
}

/// MatMul f32×f32 acumulando en f64, salida `acc += a @ b` (b en row-major [k, n]).
/// Partición de reducción: 16 acumuladores f64 como en `k3_matmul`.
pub fn matmul_f32_acc(_acc: &mut [f32], _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) {
    todo!("Stage 0, ver docs/ROADMAP.md");
}

/// RoPE intercalado estilo Llama (pares (2i, 2i+1)); variante NeoX en otro kernel.
/// Frecuencias precalculadas en f32; la multiplicación compleja usa `mul_add` explícito.
pub fn rope_apply_llama(_x: &mut [f32], _freqs: &[f32], _pos: usize) {
    todo!("Stage 0, ver docs/ROADMAP.md");
}
