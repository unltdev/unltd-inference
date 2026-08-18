//! Dot products sobre formatos empaquetados GGUF (Fase 4): F32, Q4_K y Q6_K.
//! Los layouts de bloque están verificados contra `ggml-common.h` del oráculo
//! local (D:\AI\runtimes\llama.cpp) y, en tests, contra la dequantización de
//! gguf-py sobre bytes REALES de ornith-1.0-9b-Q4_K_M.gguf (ver `ornith_decode_pin`).
//!
//! Principio (ARCHITECTURE §3): los bloques se multiplican directamente desde
//! sus bytes, NUNCA se desquantizan a una copia de pesos — salvo el lookup de
//! embeddings ([`dequantize_q4_k`]): GET_ROWS es por definición una copia, y el
//! oráculo materializa exactamente eso (dequantize_row_q4_K).
//!
//! Contrato numérico por elemento (AUDIT §3.3, igual que los kernels f32):
//! 1. el peso cuantizado se reconstruye en f64 y es EXACTO — cada fórmula de
//!    decode es un producto de enteros chicos por un f16/f32, con mantisa
//!    siempre < 2^53 (demostrado por cotas en cada función);
//! 2. el producto `x_i * w_i` se computa en f64 (una sola multiplicación, un
//!    solo redondeo);
//! 3. la reducción de la fila completa usa [`crate::kernels::pairwise_sum_f64`]:
//!    el MISMO árbol `((p0+p1)+(p2+p3))...` que los kernels f32, así que un dot
//!    F32 y un dot Q4_K de la misma dimensión difieren solo en los valores de
//!    los productos, nunca en el orden;
//! 4. el resultado es un parcial f64; el llamador lo acumula en f64 (ver
//!    `matmul_f32_acc` para el patrón de acumulación).
//!
//! **Excepción — los dots Q8_K** ([`dot_q4_k_q8_k`] / [`dot_q6_k_q8_k`]): el
//! oráculo los ejecuta con `ggml_vec_dot_q4_K_q8_K` / `q6_K`, que en su build
//! MSVC cae a los kernels GENERIC (GGML_AVX2=OFF en build-msvc/CMakeCache.txt).
//! Ahí NO rige el contrato f64: estos dos dots son réplicas escalares EXACTAS de
//! esos generics (ggml-cpu/quants.c:645 y :800) — carriles i32, d f32,
//! mul/add f32 separados sin FMA — bit-idénticos a lo que midió el oráculo.

use crate::kernels::pairwise_sum_f64;

/// Elementos por superbloque K-quant.
pub const QK_K: usize = 256;
/// Bytes por bloque Q4_K: d(2) + dmin(2) + scales(12) + qs(128).
pub const BLOCK_Q4_K_BYTES: usize = 144;
/// Bytes por bloque Q6_K: ql(128) + qh(64) + scales(16) + d(2). ¡El struct de
/// ggml va en ese orden: d está al FINAL del bloque!
pub const BLOCK_Q6_K_BYTES: usize = 210;

/// Conversión IEEE 754 half → f32, bit a bit (incluye subnormales, ±0, ±inf, NaN).
/// No usa la instrucción hardware (que redondea según el modo actual): es la
/// conversión canónica round-to-nearest de la especificación.
pub fn f16_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h >> 15) << 31;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    let bits = if exp == 0 {
        if frac == 0 {
            sign // ±0
        } else {
            // subnormal: m * 2^-24, renormalizar (exacto en f32)
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        if frac == 0 {
            sign | 0x7F80_0000 // ±inf
        } else {
            sign | 0x7FC0_0000 // NaN (quiet, payload 0)
        }
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// f16 → f64. Exacta: f16 → f32 es exacto por construcción y f32 → f64 también.
#[inline]
fn f16_to_f64(h: u16) -> f64 {
    f16_to_f32(h) as f64
}

#[inline]
fn f16_at(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Desempaqueta escala y mínimo del grupo `g` (0..8) de un bloque Q4_K.
/// Replica exacta de `get_scale_min_k4` (ggml-quants.c:822):
/// - g < 4:   `sc = s[g] & 63`,              `m = s[g+4] & 63`
/// - g >= 4:  `sc = (s[g+4] & 0xF) | ((s[g-4] >> 6) << 4)`,
///            `m  = (s[g+4] >> 4)  | ((s[g]   >> 6) << 4)`
/// Los bits 6-7 de cada byte de `s[0..8]` son los bits altos del grupo g+4.
#[inline]
fn scale_min_k4(s: &[u8], g: usize) -> (u8, u8) {
    debug_assert!(g < 8, "scale_min_k4: grupo {g} fuera de 0..8");
    if g < 4 {
        (s[g] & 63, s[g + 4] & 63)
    } else {
        (
            (s[g + 4] & 0xF) | ((s[g - 4] >> 6) << 4),
            (s[g + 4] >> 4) | ((s[g] >> 6) << 4),
        )
    }
}

/// Dot F32×F32 sobre una fila de dimensión `n`: Σ `x_i * w_i` en f64 con el
/// árbol pairwise del contrato. Peso f32 → f64 exacto.
pub fn dot_f32(x: &[f32], w: &[f32]) -> f64 {
    assert_eq!(x.len(), w.len(), "dot_f32: len(x) != len(w)");
    let prods: Vec<f64> = x
        .iter()
        .zip(w)
        .map(|(&xv, &wv)| (xv as f64) * (wv as f64))
        .collect();
    pairwise_sum_f64(&prods)
}

/// Dot F32×Q4_K. Layout del bloque (ggml-common.h `block_q4_K`, 144 B):
/// ```text
/// off  campo
/// 0    d (f16)          escala del superbloque
/// 2    dmin (f16)       escala del mínimo del superbloque
/// 4    scales[12]       sc y m de 8 grupos de 32, 6 bits cada uno
/// 16   qs[128]          nibbles NO intercalados: los 32 bytes de q[0..32]
///                       son el nibble BAJO de los grupos 0 y ALTO del 1
/// ```
/// Mapeo elemento → nibble (grupo `g = e/32`, posición `l = e%32`):
/// byte `qs[32*(g/2) + l]`, nibble bajo si `g` par, alto si `g` impar.
/// Peso: `w = (d * sc[g]) * nib - (dmin * m[g])`
///
/// Exactitud de la reconstrucción en f64: `d * sc` tiene mantisa ≤ 2^11·2^6 = 2^17,
/// `(d*sc) * nib` ≤ 2^17·2^4 = 2^21, `dmin * m` ≤ 2^17 — todo exacto; la resta
/// es el redondeo correcto de dos operandos exactos. Un solo redondeo posible.
pub fn dot_q4_k(x: &[f32], w: &[u8]) -> f64 {
    let n = x.len();
    assert_eq!(n % QK_K, 0, "dot_q4_k: n ({n}) no es múltiplo de {QK_K}");
    assert_eq!(
        w.len(),
        n / QK_K * BLOCK_Q4_K_BYTES,
        "dot_q4_k: len(w) != bloques exactos"
    );
    let mut prods = vec![0.0f64; n];
    for (b, block) in w.chunks_exact(BLOCK_Q4_K_BYTES).enumerate() {
        let d = f16_to_f64(f16_at(&block[0..2]));
        let dmin = f16_to_f64(f16_at(&block[2..4]));
        let scales = &block[4..16];
        let qs = &block[16..];
        for e in 0..QK_K {
            let g = e / 32;
            let (sc, m) = scale_min_k4(scales, g);
            let byte = qs[32 * (g / 2) + (e % 32)];
            let nib = if g % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let wv = (d * f64::from(sc)) * f64::from(nib) - (dmin * f64::from(m));
            prods[b * QK_K + e] = f64::from(x[b * QK_K + e]) * wv;
        }
    }
    pairwise_sum_f64(&prods)
}

/// Dot F32×Q6_K. Layout del bloque (ggml-common.h `block_q6_K`, 210 B):
/// ```text
/// off  campo
/// 0    ql[128]   nibbles bajos
/// 128  qh[64]    2 bits altos, 4 elementos por byte
/// 192  scales[16] escalas i8, una por cada 16 elementos
/// 208  d (f16)   escala del superbloque
/// ```
/// Elemento `e`: byte ql = `64*(e/128) + (e%64)`, nibble alto si `(e/64)%2 == 1`;
/// hi2 = `(qh[(e%32) + 32*(e/128)] >> (2*((e%128)/32))) & 3`;
/// `q = (nib | hi2<<4) - 32` (rango -32..31); `w = d * (scales[e/16] * q)`.
///
/// Exactitud en f64: `d * sc * q` tiene mantisa ≤ 2^11·2^7·2^5 = 2^23 — exacto.
pub fn dot_q6_k(x: &[f32], w: &[u8]) -> f64 {
    let n = x.len();
    assert_eq!(n % QK_K, 0, "dot_q6_k: n ({n}) no es múltiplo de {QK_K}");
    assert_eq!(
        w.len(),
        n / QK_K * BLOCK_Q6_K_BYTES,
        "dot_q6_k: len(w) != bloques exactos"
    );
    let mut prods = vec![0.0f64; n];
    for (b, block) in w.chunks_exact(BLOCK_Q6_K_BYTES).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f64(f16_at(&block[208..210]));
        for e in 0..QK_K {
            let byte = ql[64 * (e / 128) + (e % 64)];
            let nib = if (e / 64) % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let hi2 = (qh[(e % 32) + 32 * (e / 128)] >> (2 * ((e % 128) / 32))) & 3;
            let q = (i32::from(nib) | (i32::from(hi2) << 4)) - 32;
            let sc = scales[e / 16] as i8;
            let wv = d * (f64::from(sc) * f64::from(q));
            prods[b * QK_K + e] = f64::from(x[b * QK_K + e]) * wv;
        }
    }
    pairwise_sum_f64(&prods)
}

// ---------------------------------------------------------------------------
// Camino del ORÁCULO: MUL_MAT con x cuantizado a Q8_K.
//
// ggml-cpu no dota el vector f32 contra los pesos cuantizados: para Q4_K/Q6_K
// su `vec_dot_type` es GGML_TYPE_Q8_K (ggml-cpu.c:304-313), así que ANTES del
// dot cuantiza `x` a block_q8_K con `quantize_row_q8_K_ref` (ggml-quants.c:2696)
// — una cuantización CON pérdida (~0.4% por elemento). El dot resultante NO es
// el dot matemáticamente exacto del contrato §3.3: es el dot que ejecuta el
// oráculo, y el gate exige replicarlo bit a bit. Los kernels de aquí replican
// en escalar las versiones AVX2 de ggml_vec_dot_q4_K_q8_K (arch/x86/quants.c:2038)
// y ggml_vec_dot_q6_K_q8_K (arch/x86/quants.c:2426), incluida la aritmética:
//
// - sumas de enteros EXACTAS en i32 por carril (8 carriles idénticos a AVX2);
// - cadenas FMA f32 por carril: `acc[k] = fma(d, sumi[k] as f32, acc[k])`;
// - suma horizontal de los 8 carriles en el orden exacto de hsum_float_8
//   (arch/x86/quants.c:43): ((a4+a0)+(a6+a2)) + ((a5+a1)+(a7+a3));
// - término dmin en 4 carriles (q4_K): ((m0+m2)+(m1+m3)) al final.
//
// Los dots exactos ([`dot_q4_k`] / [`dot_q6_k`]) siguen existiendo como
// referencia matemática y para los tests; el forward usa gemv_quant_q8k.
// ---------------------------------------------------------------------------

/// Redondeo a entero de ggml (`nearest_int`, ggml-quants.c:563): el truco del
/// número mágico. `val = f + 12582912.f` redondea en f32 a entero (ties-to-even,
/// el rango de trabajo es [2^23, 2^24) donde ulp = 1) y los 23 bits de mantisa
/// contienen el entero desplazado por 0x400000. Replica EXACTA, incluido el
/// desempate a par (2.5 → 2, 3.5 → 4).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    ((val.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Bloque Q8_K: `{float d; int8 qs[256]; int16 bsums[16]}` (ggml-common.h).
/// `d` es f32 (verificado por static_assert en ggml-common.h:364-366).
#[derive(Debug, Clone, Copy)]
pub struct Q8KBlock {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i32; 16],
}

/// Replica exacta de `quantize_row_q8_K_ref` (ggml-quants.c:2696):
/// por bloque de 256: max con SIGNO (primer argmax estricto de |x|); si amax = 0
/// el bloque queda d = 0 y qs = 0 (ggml deja bsums sin tocar — irrelevante porque
/// d = 0 anula todo uso de bsums; aquí se ponen a 0 por higiene);
/// `iscale = -127.f/max` (f32); `qs[j] = MIN(127, nearest_int(iscale*x[j]))`;
/// `bsums[j] = Σ16 qs`; `d = 1/iscale`.
pub fn quantize_q8_k(x: &[f32]) -> Vec<Q8KBlock> {
    assert_eq!(x.len() % QK_K, 0, "quantize_q8_k: len no es múltiplo de {QK_K}");
    let mut out = Vec::with_capacity(x.len() / QK_K);
    for chunk in x.chunks_exact(QK_K) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in chunk {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        let mut b = Q8KBlock {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; 16],
        };
        if amax == 0.0 {
            out.push(b);
            continue;
        }
        let iscale = -127.0f32 / max;
        for j in 0..QK_K {
            b.qs[j] = nearest_int(iscale * chunk[j]).min(127) as i8;
        }
        for j in 0..16 {
            b.bsums[j] = b.qs[16 * j..16 * j + 16]
                .iter()
                .map(|&q| i32::from(q))
                .sum();
        }
        b.d = 1.0f32 / iscale;
        out.push(b);
    }
    out
}

/// Dot Q8_K×Q4_K, réplica escalar EXACTA de `ggml_vec_dot_q4_K_q8_K_generic`
/// (ggml-cpu/quants.c:645). El build MSVC del oráculo está configurado con
/// GGML_AVX=OFF y GGML_AVX2=OFF (build-msvc/CMakeCache.txt), así que la rama
/// SIMD de arch/x86/quants.c NO se compiló y `ggml_vec_dot_q4_K_q8_K` cae al
/// generic — medido contra el volcado: la suma del tensor z-0 solo cierra con
/// la aritmética del generic (el AVX2 replica diverge ~3e-6/elem).
/// Por bloque (nb = 16 en ornith):
/// - nibbles de `qs` expandidos en orden de elemento: `a[e]` = nibble bajo si
///   `e%64 < 32`, alto si no;
/// - dance `utmp` (== [`scale_min_k4`]): sc[0..7] (uno por grupo de 32),
///   m[0..7] (uno por par de bsums);
/// - 8 carriles i32: carril k acumula los elementos e ≡ k (mod 8):
///   `aux32[k] += sc[e/32] · (a[e] · q8[e])` — productos i16/i32 EXACTOS;
/// - `sums[k] += d * aux32[k]` en f32 con mul y add SEPARADOS (sin FMA — MSVC
///   sin /arch:AVX2 no emite FMA), `d = f16(x.d) * y.d`;
/// - `sumi = Σ_{g=0..15} bsums[g]·m[g/2]` (i32 exacto, orden ascendente);
///   `sumf -= dmin * sumi` con `dmin = f16(x.dmin) * y.d` POSITIVO (el signo
///   menos va en la resta — NO es el `-y.d*f16(x.dmin)` del AVX2);
/// - `s = ((sumf + sums[0]) + sums[1]) + ... + sums[7]` (secuencial, DESPUÉS
///   de todas las restas dmin).
pub fn dot_q4_k_q8_k(x: &[Q8KBlock], w: &[u8]) -> f32 {
    assert_eq!(
        w.len() % BLOCK_Q4_K_BYTES,
        0,
        "dot_q4_k_q8_k: len(w) != bloques exactos"
    );
    assert_eq!(
        x.len(),
        w.len() / BLOCK_Q4_K_BYTES,
        "dot_q4_k_q8_k: bloques de x ({}) != bloques de w ({})",
        x.len(),
        w.len() / BLOCK_Q4_K_BYTES
    );
    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    for (q8, block) in x.iter().zip(w.chunks_exact(BLOCK_Q4_K_BYTES)) {
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for g in 0..8 {
            let (s, mn) = scale_min_k4(&block[4..16], g);
            sc[g] = s;
            m[g] = mn;
        }
        let qs = &block[16..];
        let mut sumi = [0i32; 8];
        for e in 0..QK_K {
            let pos = e % 64;
            let byte = qs[32 * (e / 64) + pos % 32];
            let a = if pos < 32 { byte & 0xF } else { byte >> 4 };
            sumi[e % 8] += i32::from(sc[e / 32]) * i32::from(a) * i32::from(q8.qs[e]);
        }
        let d = q8.d * f16_to_f32(f16_at(&block[0..2]));
        for k in 0..8 {
            // mul y add f32 SEPARADOS: el generic no usa FMA
            sums[k] += d * sumi[k] as f32;
        }
        let mut minsum = 0i32;
        for g in 0..16 {
            minsum += q8.bsums[g] * i32::from(m[g / 2]);
        }
        let dmin = f16_to_f32(f16_at(&block[2..4])) * q8.d;
        sumf -= dmin * minsum as f32;
    }
    let mut s = sumf;
    for k in 0..8 {
        s += sums[k];
    }
    s
}

/// Dot Q8_K×Q6_K, réplica escalar EXACTA de `ggml_vec_dot_q6_K_q8_K_generic`
/// (ggml-cpu/quants.c:800) — mismo motivo que en [`dot_q4_k_q8_k`]: el build
/// del oráculo compiló solo el generic. Por bloque:
/// - `a[e]` = valor 6-bit crudo del elemento e (nib ql | 2 bits qh) MENOS 32
///   (el −32 va DENTRO de a — no hay término q8sclsub como en el AVX2);
/// - 8 carriles i32: `aux32[k] += sc[e/16]·(a[e]·q8[e])` con sc i8 CON signo;
/// - `sums[k] += d * aux32[k]` (f32, mul+add separados), `d = f16(x.d) * y.d`;
/// - `s = sums[0] + ... + sums[7]` secuencial (sumf arranca en 0 — Q6_K no
///   tiene término de mínimos).
pub fn dot_q6_k_q8_k(x: &[Q8KBlock], w: &[u8]) -> f32 {
    assert_eq!(
        w.len() % BLOCK_Q6_K_BYTES,
        0,
        "dot_q6_k_q8_k: len(w) != bloques exactos"
    );
    assert_eq!(
        x.len(),
        w.len() / BLOCK_Q6_K_BYTES,
        "dot_q6_k_q8_k: bloques de x ({}) != bloques de w ({})",
        x.len(),
        w.len() / BLOCK_Q6_K_BYTES
    );
    let mut sums = [0.0f32; 8];
    for (q8, block) in x.iter().zip(w.chunks_exact(BLOCK_Q6_K_BYTES)) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let mut sumi = [0i32; 8];
        for e in 0..QK_K {
            // mapeo de dequantize_row_q6_K_reference: por chunk de 128, los
            // primeros 64 elementos usan los nibbles BAJOS de ql[pos%64] y los
            // últimos 64 los ALTOS (q1/q2: ql[l]&0xF y ql[l+32]&0xF; q3/q4:
            // ql[l]>>4 y ql[l+32]>>4).
            let pos = e % 128;
            let byte = ql[64 * (e / 128) + pos % 64];
            let nib = if pos < 64 { byte & 0xF } else { byte >> 4 };
            let hi2 = (qh[32 * (e / 128) + pos % 32] >> (2 * (pos / 32))) & 3;
            let a = i32::from(nib | (hi2 << 4)) - 32; // crudo − 32
            sumi[e % 8] += i32::from(scales[e / 16] as i8) * a * i32::from(q8.qs[e]);
        }
        let d = q8.d * f16_to_f32(f16_at(&block[208..210]));
        for k in 0..8 {
            sums[k] += d * sumi[k] as f32;
        }
    }
    let mut s = 0.0f32;
    for k in 0..8 {
        s += sums[k];
    }
    s
}

/// GEMV sobre una matriz de pesos empaquetados: `out[j] = Σ_i x[i] * w[i, j]`.
/// La matriz `w` es `[dim_in, dim_out]` con filas de `dim_in` contiguas (como las
/// escribe GGUF); cada fila es un vector de bloques del dtype.
///
/// Contrato: el dot de cada fila sigue su kernel (F32/Q4_K/Q6_K — mismo árbol
/// pairwise, mismas cotas de exactitud); las filas son independientes y el
/// llamador puede paralelizar por rangos de `out` (contrato §5). El dtype del
/// archivo se toma de [`crate::DType`], NUNCA se adivina desde los bytes.
pub fn gemv_quant(out: &mut [f32], x: &[f32], w: &[u8], dim_in: usize, dim_out: usize, dtype: crate::DType) {
    assert_eq!(out.len(), dim_out, "gemv_quant: len(out) != dim_out");
    assert!(x.len() >= dim_in, "gemv_quant: len(x) < dim_in");
    let row_bytes = match dtype {
        crate::DType::F32 => dim_in * 4,
        crate::DType::Q4K | crate::DType::Q6K => {
            assert_eq!(dim_in % QK_K, 0, "gemv_quant: dim_in no es múltiplo de {QK_K}");
            dim_in / QK_K
                * match dtype {
                    crate::DType::Q4K => BLOCK_Q4_K_BYTES,
                    _ => BLOCK_Q6_K_BYTES,
                }
        }
        other => panic!("gemv_quant: dtype {other:?} sin dot implementado"),
    };
    assert_eq!(w.len(), row_bytes * dim_out, "gemv_quant: len(w) != filas exactas");
    let x = &x[..dim_in];
    for j in 0..dim_out {
        let row = &w[j * row_bytes..(j + 1) * row_bytes];
        let sum = match dtype {
            crate::DType::F32 => {
                let row: &[f32] = bytemuck::cast_slice(row);
                dot_f32(x, row)
            }
            crate::DType::Q4K => dot_q4_k(x, row),
            _ => dot_q6_k(x, row),
        };
        out[j] = sum as f32;
    }
}

/// GEMV como lo ejecuta el ORÁCULO (ggml_compute_forward_mul_mat): cuantiza `x`
/// a Q8_K UNA vez (como hace ggml con src1 para vec_dot_type = Q8_K) y dota cada
/// fila con el kernel q8_K del dtype de la fila. Es CON pérdida respecto del dot
/// exacto (~0.4%/elemento de x); es lo que corre llama.cpp y lo que el gate
/// compara. F32 no tiene sentido aquí (el oráculo usa vec_dot f32 directo, sin
/// cuantizar): se rechaza explícito.
pub fn gemv_quant_q8k(
    out: &mut [f32],
    x: &[f32],
    w: &[u8],
    dim_in: usize,
    dim_out: usize,
    dtype: crate::DType,
) {
    assert_eq!(out.len(), dim_out, "gemv_quant_q8k: len(out) != dim_out");
    assert!(x.len() >= dim_in, "gemv_quant_q8k: len(x) < dim_in");
    let row_bytes = match dtype {
        crate::DType::Q4K | crate::DType::Q6K => {
            assert_eq!(dim_in % QK_K, 0, "gemv_quant_q8k: dim_in no es múltiplo de {QK_K}");
            dim_in / QK_K
                * match dtype {
                    crate::DType::Q4K => BLOCK_Q4_K_BYTES,
                    _ => BLOCK_Q6_K_BYTES,
                }
        }
        other => panic!(
            "gemv_quant_q8k: dtype {other:?} no usa vec_dot Q8_K en el oráculo (usa gemv_quant)"
        ),
    };
    assert_eq!(w.len(), row_bytes * dim_out, "gemv_quant_q8k: len(w) != filas exactas");
    let x = &x[..dim_in];
    let xq = quantize_q8_k(x);
    for j in 0..dim_out {
        let row = &w[j * row_bytes..(j + 1) * row_bytes];
        out[j] = match dtype {
            crate::DType::Q4K => dot_q4_k_q8_k(&xq, row),
            _ => dot_q6_k_q8_k(&xq, row),
        };
    }
}

/// Dequantiza una fila Q4_K a f32 con la fórmula EXACTA de la referencia
/// (`dequantize_row_q4_K`, ggml-quants.c:1471 — la usa GET_ROWS para los
/// embeddings cuantizados, que es el ÚNICO camino que materializa pesos a f32;
/// los dots nunca dequantizan, ver doc del módulo):
///
/// ```text
/// por bloque: d = f16→f32(d), min = f16→f32(dmin)
/// por par de grupos (is = 0, 2, 4, 6):
///   d1 = d * sc(is);   m1 = min * m(is)      (f32)
///   d2 = d * sc(is+1); m2 = min * m(is+1)    (f32)
///   y[32·(is)]   = d1 * (q[l] & 0xF) - m1    (f32, l = 0..32)
///   y[32·(is+1)] = d2 * (q[l] >> 4)  - m2    (f32, l = 0..32)
/// ```
///
/// Bit-idéntico al oráculo: mismas operaciones f32 en el mismo orden (los u8 se
/// ensanchan a f32 exactamente). NOTA: el nibble NO lleva shift -16 — el offset
/// vive en `min` (misma convención que [`dot_q4_k`]).
pub fn dequantize_q4_k(out: &mut [f32], w: &[u8]) {
    assert_eq!(
        w.len(),
        out.len() / QK_K * BLOCK_Q4_K_BYTES,
        "dequantize_q4_k: len(w) != bloques exactos para len(out)"
    );
    let mut e = 0usize;
    for block in w.chunks_exact(BLOCK_Q4_K_BYTES) {
        let d = f16_to_f32(f16_at(&block[0..2]));
        let min = f16_to_f32(f16_at(&block[2..4]));
        let scales = &block[4..16];
        let mut qs = &block[16..];
        let mut is = 0usize;
        while is < 8 {
            let (sc0, m0) = scale_min_k4(scales, is);
            let d1 = d * sc0 as f32;
            let m1 = min * m0 as f32;
            let (sc1, m1s) = scale_min_k4(scales, is + 1);
            let d2 = d * sc1 as f32;
            let m2 = min * m1s as f32;
            for l in 0..32 {
                out[e] = d1 * (qs[l] & 0xF) as f32 - m1;
                e += 1;
            }
            for l in 0..32 {
                out[e] = d2 * (qs[l] >> 4) as f32 - m2;
                e += 1;
            }
            qs = &qs[32..];
            is += 2;
        }
    }
    assert_eq!(e, out.len());
}

// ---------------------------------------------------------------------------
// Tests: casos a mano (valores exactos), pines de orden del árbol, y el pin
// de bytes reales de ornith contra gguf-py (generado por
// `benchmarks/reference/ornith-decode-crosscheck.py`).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const F16_ONE: u16 = 0x3C00;
    const F16_TWO: u16 = 0x4000;

    #[test]
    fn f16_to_f32_known_patterns() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000); // -0.0
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // subnormal mínimo
        assert_eq!(f16_to_f32(0x03FF), 2f32.powi(-14) - 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // f16 máximo
        assert_eq!(f16_to_f32(0x3555), 0.333_251_953_125); // 1/3 en f16
    }

    #[test]
    fn dot_f32_small_exact() {
        // enteros chicos: árbol de n=4 = (p0+p1)+(p2+p3), todo exacto
        assert_eq!(dot_f32(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]), 70.0);
        // cola impar n=3: (p0+p1)+p2
        assert_eq!(dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn dequantize_q4_k_hand_values_and_multi_block() {
        // bloque A: d=1, dmin=2; g0: sc=1 (scales[0]&63) / m=3 (scales[4]&63);
        // g1: sc=2 / m=1; g2..7: 0. qs[0] = 0x12: nib bajo 2 (elem 0, g0),
        // nib alto 1 (elem 32, g1).
        let mut scales = [0u8; 12];
        scales[0] = 1;
        scales[4] = 3;
        scales[1] = 2;
        scales[5] = 1;
        let mut qs = [0u8; 128];
        qs[0] = 0x12;
        let block_a = q4_block(F16_ONE, F16_TWO, scales, qs);
        // bloque B: d = 0 → todos los valores 0 (prueba el offset del 2º bloque)
        let block_b = q4_block(0x0000, 0x0000, [0; 12], [0xAB; 128]);
        let mut w = block_a.clone();
        w.extend_from_slice(&block_b);

        let mut out = vec![0.0f32; 512];
        dequantize_q4_k(&mut out, &w);

        assert_eq!(out[0], -4.0); // 1·(0x12 & 0xF) − 2·3 = 2 − 6
        assert_eq!(out[1], -6.0); // nib 0: 0 − 6
        assert_eq!(out[31], -6.0);
        assert_eq!(out[32], 0.0); // 2·(0x12 >> 4) − 2·1 = 2 − 2
        assert_eq!(out[33], -2.0); // nib 0: 0 − 2
        for i in 64..256 {
            assert_eq!(out[i], 0.0, "elem {i} (grupos 2..7)");
        }
        for i in 256..512 {
            assert_eq!(out[i], 0.0, "elem {i} (bloque B)");
        }

        // consistencia con el dot: dot_q4_k reconstruye en f64 exacto, la
        // dequantizada está redondeada a f32 — mismas fórmulas, ≤ 1 ulp/elem
        let x = [0.5f32; 256];
        let exact = dot_q4_k(&x, &block_a);
        let via_deq = dot_f32(&x, &out[..256]);
        assert!(
            (via_deq - exact).abs() < 1e-4,
            "dot exacto {exact} vs via dequantize {via_deq}"
        );
    }

    #[test]
    fn dequantize_q4_k_matches_reference_formula_bitwise() {
        // Bytes arbitrarios (scales con bits altos para grupos 4..7) + réplica
        // independiente en f32 de dequantize_row_q4_K (ggml-quants.c:1471).
        let scales: [u8; 12] = [0x1F, 0x2E, 0x3D, 0x4C, 0x5B, 0x6A, 0x79, 0x03, 0x92, 0x81, 0x70, 0x6F];
        let qs: [u8; 128] = core::array::from_fn(|i| (i * 7 + 3) as u8);
        let d_bits = 0x3B21u16;
        let dmin_bits = 0x2C10u16;
        let block = q4_block(d_bits, dmin_bits, scales, qs);

        let mut out = vec![0.0f32; 256];
        dequantize_q4_k(&mut out, &block);

        let d = f16_to_f32(d_bits);
        let min = f16_to_f32(dmin_bits);
        let mut e = 0usize;
        let mut is = 0usize;
        while is < 8 {
            // el kernel avanza 32 bytes de qs por par de grupos (is += 2)
            let qs_pair = &qs[32 * (is / 2)..32 * (is / 2) + 32];
            for (g, shift) in [(is, 0u32), (is + 1, 4u32)] {
                let (sc, m) = scale_min_k4(&scales, g);
                let dg = d * sc as f32;
                let mg = min * m as f32;
                for l in 0..32 {
                    let expected = dg * ((qs_pair[l] >> shift) & 0xF) as f32 - mg;
                    assert_eq!(out[e].to_bits(), expected.to_bits(), "elem {e}");
                    e += 1;
                }
            }
            is += 2;
        }
    }

    fn q4_block(d: u16, dmin: u16, scales: [u8; 12], qs: [u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(BLOCK_Q4_K_BYTES);
        b.extend_from_slice(&d.to_le_bytes());
        b.extend_from_slice(&dmin.to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&qs);
        assert_eq!(b.len(), BLOCK_Q4_K_BYTES);
        b
    }

    fn q6_block(d: u16, ql: [u8; 128], qh: [u8; 64], scales: [i8; 16]) -> Vec<u8> {
        // orden de struct ggml: ql, qh, scales, d (¡d al final!)
        let mut b = Vec::with_capacity(BLOCK_Q6_K_BYTES);
        b.extend_from_slice(&ql);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&scales.map(|s| s as u8));
        b.extend_from_slice(&d.to_le_bytes());
        assert_eq!(b.len(), BLOCK_Q6_K_BYTES);
        b
    }

    #[test]
    fn q4_k_zero_block_and_hand_values() {
        // todo a cero (sc=0, m=0): pesos 0 para cualquier nibble
        let block = q4_block(F16_ONE, 0x0000, [0; 12], [0xFF; 128]);
        let x = [0.5f32; 256];
        assert_eq!(dot_q4_k(&x, &block), 0.0);

        // grupo 0: sc=1 (scales[0]&63), m=3 (scales[4]&63); d=1, dmin=2
        // elemento 0: nib=1 → w = (1*1)*1 - (2*3) = -5
        // elemento 1: nib=0 → w = 0 - 6 = -6
        let mut scales = [0u8; 12];
        scales[0] = 1;
        scales[4] = 3;
        let mut qs = [0u8; 128];
        qs[0] = 0x01; // elem0 = 1, elem1 = 0
        let block = q4_block(F16_ONE, F16_TWO, scales, qs);
        let mut x = [0.0f32; 256];
        x[0] = 1.0;
        assert_eq!(dot_q4_k(&x, &block), -5.0);
        x[0] = 0.0;
        x[1] = 1.0;
        assert_eq!(dot_q4_k(&x, &block), -6.0);

        // grupo 4 (elemento 128): sc = (s[8]&0xF)|((s[0]>>6)<<4),
        // m = (s[8]>>4)|((s[4]>>6)<<4) → sc=49, m=19; d=1, dmin=1
        // nib(128)=1 → w = 49*1 - 19 = 30
        let mut scales = [0u8; 12];
        scales[0] = 0xC0; // bits altos de sc[4]
        scales[4] = 0x43; // bits altos de m[4]
        scales[8] = 0x31; // nibbles bajos de sc[4] y m[4]
        let mut qs = [0u8; 128];
        qs[64] = 0x01; // elem 128 = nibble bajo del byte 64
        let block = q4_block(F16_ONE, F16_ONE, scales, qs);
        let mut x = [0.0f32; 256];
        x[128] = 1.0;
        assert_eq!(dot_q4_k(&x, &block), 30.0);
    }

    #[test]
    fn q4_k_pairwise_tree_is_the_contract() {
        // pesos [10, 1, 10, 1] (grupo 0: sc=1, m=0, d=1). Cada elemento del
        // grupo 0 usa el nibble BAJO de su propio byte: elem0..3 → qs[0..4].
        let mut scales = [0u8; 12];
        scales[0] = 1;
        let mut qs = [0u8; 128];
        qs[0] = 0x0A; // elem0 = 10
        qs[1] = 0x01; // elem1 = 1
        qs[2] = 0x0A; // elem2 = 10
        qs[3] = 0x01; // elem3 = 1
        let block = q4_block(F16_ONE, 0x0000, scales, qs);
        // productos [H, 1, -H, 1] con H = 1e19f32 * 10 ≈ 1e20 (exacto en f64,
        // H+1 redondea a H porque ulp ≈ 2^15). Serial: H-H+1 = 1. Pairwise:
        // (H+1)+(-H+1) = 0. El contrato exige 0.
        let mut x = [0.0f32; 256];
        x[0] = 1e19;
        x[1] = 1.0;
        x[2] = -1e19;
        x[3] = 1.0;
        assert_eq!(dot_q4_k(&x, &block), 0.0);
    }

    #[test]
    fn q4_k_full_block_matches_layout_reference() {
        // sc=1, m=0 en los 8 grupos (todos los bytes de escala = 1), d=1:
        // w_e = nibble(e). x_e = e+1. Productos enteros < 2^53: el orden del
        // árbol no influye, el test clava el MAPEO elemento→nibble completo.
        let mut qs = [0u8; 128];
        for (i, b) in qs.iter_mut().enumerate() {
            *b = i as u8; // nibbles distintos (0x00, 0x10, 0x21, ...)
        }
        let block = q4_block(F16_ONE, 0x0000, [1; 12], qs);
        let x: Vec<f32> = (0..256).map(|e| (e + 1) as f32).collect();
        // referencia independiente: formulación directa del layout del bloque
        let mut expected = 0.0f64;
        for e in 0..QK_K {
            let g = e / 32;
            let byte = qs[32 * (g / 2) + (e % 32)];
            let nib = if g % 2 == 0 { byte & 0xF } else { byte >> 4 };
            expected += (e + 1) as f64 * f64::from(nib);
        }
        assert_eq!(dot_q4_k(&x, &block), expected);
    }

    #[test]
    fn q4_k_multiblock_stride() {
        // dos bloques con d distinto: valida el stride de 144 bytes
        let mut qs = [0u8; 128];
        qs[0] = 0x01; // elem0 = 1 (nibble bajo del byte 0), elem1 = qs[1]&0xF = 0
        let b1 = q4_block(F16_ONE, 0x0000, [1; 12], qs);
        let b2 = q4_block(F16_TWO, 0x0000, [1; 12], qs);
        let mut w = b1;
        w.extend_from_slice(&b2);
        let x = [1.0f32; 512];
        // bloque 1: Σ w = 1; bloque 2: Σ w = 2
        assert_eq!(dot_q4_k(&x, &w), 1.0 + 2.0);
    }

    #[test]
    fn q6_k_zero_block_and_hand_values() {
        // escalas 0 → q irrelevante → w = 0
        let block = q6_block(F16_TWO, [0xFF; 128], [0xFF; 64], [0; 16]);
        let x = [0.5f32; 256];
        assert_eq!(dot_q6_k(&x, &block), 0.0);

        // elemento 0: ql[0]&0xF=1, hi2=0 → q = -31; sc[0]=1, d=2 → w = -62
        let mut ql = [0u8; 128];
        ql[0] = 0x01;
        let mut scales = [1i8; 16];
        scales[0] = 1;
        let block = q6_block(F16_TWO, ql, [0; 64], scales);
        let mut x = [0.0f32; 256];
        x[0] = 1.0;
        assert_eq!(dot_q6_k(&x, &block), -62.0);

        // elemento 32 (camino q2): ql[32]&0xF=3, qh[0] bits 2-3 = 1 → q = 19-32 = -13;
        // sc[2] = -1 (escala NEGATIVA), d=2 → w = 2*(-1)*(-13) = 26
        let mut ql = [0u8; 128];
        ql[32] = 0x03;
        let mut qh = [0u8; 64];
        qh[0] = 0x04;
        let mut scales = [0i8; 16];
        scales[2] = -1;
        let block = q6_block(F16_TWO, ql, qh, scales);
        let mut x = [0.0f32; 256];
        x[32] = 1.0;
        assert_eq!(dot_q6_k(&x, &block), 26.0);

        // elemento 64 (camino q3, nibble ALTO): ql[0]>>4=5, qh[0] bits 4-5 = 1 →
        // q = 21-32 = -11; sc[4]=3, d=2 → w = -66
        let mut ql = [0u8; 128];
        ql[0] = 0x50;
        let mut qh = [0u8; 64];
        qh[0] = 0x10;
        let mut scales = [0i8; 16];
        scales[4] = 3;
        let block = q6_block(F16_TWO, ql, qh, scales);
        let mut x = [0.0f32; 256];
        x[64] = 1.0;
        assert_eq!(dot_q6_k(&x, &block), -66.0);

        // elemento 96 (camino q4): ql[32]>>4=7, qh[0] bits 6-7 = 1 → q = 23-32 = -9;
        // sc[6]=1, d=2 → w = -18
        let mut ql = [0u8; 128];
        ql[32] = 0x70;
        let mut qh = [0u8; 64];
        qh[0] = 0x40;
        let mut scales = [1i8; 16];
        scales[6] = 1;
        let block = q6_block(F16_TWO, ql, qh, scales);
        let mut x = [0.0f32; 256];
        x[96] = 1.0;
        assert_eq!(dot_q6_k(&x, &block), -18.0);

        // elemento 128 (segunda mitad): ql[64]&0xF=2, qh[32] bits 0-1 = 2 →
        // q = 34-32 = 2; sc[8]=5, d=2 → w = 20
        let mut ql = [0u8; 128];
        ql[64] = 0x02;
        let mut qh = [0u8; 64];
        qh[32] = 0x02;
        let mut scales = [0i8; 16];
        scales[8] = 5;
        let block = q6_block(F16_TWO, ql, qh, scales);
        let mut x = [0.0f32; 256];
        x[128] = 1.0;
        assert_eq!(dot_q6_k(&x, &block), 20.0);
    }

    #[test]
    fn q6_k_pairwise_tree_is_the_contract() {
        // pesos [10, 1, 10, 1]: q = nib | hi2<<4 - 32 con d=1, sc[0]=1.
        // OJO con qh: el byte e%32 sirve a 4 elementos (e, e+32, e+64, e+96)
        // en los bits 0-1, 2-3, 4-5, 6-7. Aquí cada elemento usa su PROPIO
        // byte qh[e] en bits 0-1: elem0..3 → qh[0..4] = 2.
        let mut ql = [0u8; 128];
        ql[0] = 0x0A; // elem0: q = 42-32 = 10
        ql[1] = 0x01; // elem1: q = 1
        ql[2] = 0x0A; // elem2: q = 10
        ql[3] = 0x01; // elem3: q = 1
        let mut qh = [0u8; 64];
        qh[0] = 0x02;
        qh[1] = 0x02;
        qh[2] = 0x02;
        qh[3] = 0x02;
        let mut scales = [0i8; 16];
        scales[0] = 1;
        let block = q6_block(F16_ONE, ql, qh, scales);
        // productos [H, 1, -H, 1] con H = 1e19f32 * 10 ≈ 1e20 (exacto; H+1 → H).
        // Serial = 1; pairwise = 0. El contrato exige 0.
        let mut x = [0.0f32; 256];
        x[0] = 1e19;
        x[1] = 1.0;
        x[2] = -1e19;
        x[3] = 1.0;
        assert_eq!(dot_q6_k(&x, &block), 0.0);
    }

    #[test]
    fn q6_k_full_block_matches_layout_reference() {
        // escalas i8 [-8..7] por grupos de 16, d=1, ql[i]=i, qh[i]=i:
        // w_e = sc[e/16] * q(e). Productos enteros < 2^53: clava el mapeo.
        let mut ql = [0u8; 128];
        for (i, b) in ql.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut qh = [0u8; 64];
        for (i, b) in qh.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut scales = [0i8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = i as i8 - 8;
        }
        let block = q6_block(F16_ONE, ql, qh, scales);
        let x: Vec<f32> = (0..256).map(|e| (e + 1) as f32).collect();
        let mut expected = 0.0f64;
        for e in 0..QK_K {
            let byte = ql[64 * (e / 128) + (e % 64)];
            let nib = if (e / 64) % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let hi2 = (qh[(e % 32) + 32 * (e / 128)] >> (2 * ((e % 128) / 32))) & 3;
            let q = i32::from(nib) | (i32::from(hi2) << 4);
            expected += (e + 1) as f64 * f64::from(scales[e / 16]) * f64::from(q - 32);
        }
        assert_eq!(dot_q6_k(&x, &block), expected);
    }

    #[test]
    fn q6_k_multiblock_stride() {
        // dos bloques: valida el stride de 210 bytes (d al final de cada bloque)
        let mut ql = [0u8; 128];
        ql[0] = 0x01; // elem0: q = 1-32 = -31
        let mut scales = [0i8; 16];
        scales[0] = 1;
        let b1 = q6_block(F16_ONE, ql, [0; 64], scales);
        let b2 = q6_block(F16_TWO, ql, [0; 64], scales);
        let mut w = b1;
        w.extend_from_slice(&b2);
        let x = [1.0f32; 512];
        // elementos 0..16 del grupo sc[0]=1: elem0 → q=-31; elems 1..16 → q=-32
        // (nib 0 NO es peso 0 en Q6_K: el cero está en q=32).
        // bloque 1: -31 + 15·(-32) = -511; bloque 2 (d=2): -62 + 15·(-64) = -1022
        assert_eq!(dot_q6_k(&x, &w), -511.0 + -1022.0);
    }

    #[test]
    fn nearest_int_replicates_ggml_magic() {
        // ties-to-even del truco del número mágico (2.5 → 2, 3.5 → 4)
        assert_eq!(nearest_int(0.0), 0);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(3.5), 4);
        assert_eq!(nearest_int(-3.5), -4);
        assert_eq!(nearest_int(126.7), 127);
        assert_eq!(nearest_int(-126.7), -127);
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
    }

    #[test]
    fn quantize_q8_k_hand_block_and_zero_block() {
        // x = [1, 0, ...]: max = 1 → iscale = -127 → qs[0] = -127, d = -1/127
        let mut x = [0.0f32; 256];
        x[0] = 1.0;
        let q = quantize_q8_k(&x);
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].qs[0], -127);
        assert!(q[0].qs[1..].iter().all(|&v| v == 0));
        assert_eq!(q[0].bsums[0], -127);
        assert!(q[0].bsums[1..].iter().all(|&v| v == 0));
        assert_eq!(q[0].d, 1.0f32 / -127.0f32);

        // max negativo: x[0] = -1 → iscale = 127 → qs[0] = -127, d = 1/127
        let mut x = [0.0f32; 256];
        x[0] = -1.0;
        let q = quantize_q8_k(&x);
        assert_eq!(q[0].qs[0], -127);
        assert_eq!(q[0].d, 1.0f32 / 127.0f32);

        // bloque todo cero: d = 0, qs = 0 (ggml no toca bsums; d=0 lo anula todo)
        let z = [0.0f32; 256];
        let q = quantize_q8_k(&z);
        assert_eq!(q[0].d, 0.0);
        assert!(q[0].qs.iter().all(|&v| v == 0));

        // dos bloques: stride de 256
        let mut two = [0.0f32; 512];
        two[256] = 2.0;
        let q = quantize_q8_k(&two);
        assert_eq!(q.len(), 2);
        assert_eq!(q[0].d, 0.0);
        assert_eq!(q[1].qs[0], -127); // iscale = -127/2 → 2·-63.5 = -127
    }

    #[test]
    fn dot_q4_k_q8_k_hand_values() {
        // Semántica del kernel GENERIC (quants.c:645). pesos: d = 1 (f16),
        // dmin = 2 (f16), g0: sc = 1 / m = 3, elem0 nib = 1.
        // x = one-hot(0) = 127: xq d = -1, qs[0] = -127, bs[0] = -127.
        // sumi[0] = sc[0]·a[0]·q8[0] = 1·1·(-127) = -127; d = q8.d·f16(d) = -1
        // → sums[0] += -1·(-127) = 127 (mul + add separados).
        // minsum = bs[0]·m[0] = -127·3 = -381; dmin = f16(dmin)·q8.d = 2·(-1) = -2
        // → sumf -= (-2)·(-381) = -762. Total = -762 + 127 = -635.
        let mut scales = [0u8; 12];
        scales[0] = 1;
        scales[4] = 3;
        let mut qs = [0u8; 128];
        qs[0] = 0x01; // elem0 = 1, elem1 = 0
        let block = q4_block(F16_ONE, F16_TWO, scales, qs);
        let mut x = [0.0f32; 256];
        x[0] = 127.0;
        let got = dot_q4_k_q8_k(&quantize_q8_k(&x), &block);
        assert_eq!(got, -635.0); // exacto: d, dmin, sumi y minsum son enteros·potencias
    }

    #[test]
    fn dot_q6_k_q8_k_hand_values() {
        // Semántica del kernel GENERIC (quants.c:800). pesos: d = 1 (f16),
        // sc[0] = 1, elem0 raw = 1 (ql[0] = 0x01). x = one-hot(0) = 127:
        // xq d = -1, qs[0] = -127.
        // a[0] = raw − 32 = -31; sumi[0] = sc[0]·a[0]·q8[0] = 1·(-31)·(-127) = 3937
        // → sums[0] += -1·3937 = -3937. Comprobación exacta:
        // w0 = d·sc·(raw-32) = 1·(1-32) = -31; x0·w0 = 127·(-31) = -3937.
        let mut ql = [0u8; 128];
        ql[0] = 0x01;
        let mut scales = [0i8; 16];
        scales[0] = 1;
        let block = q6_block(F16_ONE, ql, [0; 64], scales);
        let mut x = [0.0f32; 256];
        x[0] = 127.0;
        let got = dot_q6_k_q8_k(&quantize_q8_k(&x), &block);
        assert_eq!(got, -3937.0);

        // Escala negativa: sc[2] = -1, elem 32 raw = 1 (ql[32] = 0x01 → nibble
        // BAJO, chunk de 128: elem 32 usa ql[32]&0xF), x one-hot(32) = 127.
        // El −32 va dentro de a[32] = 1−32 = -31, MISMO carril 0 que su escala
        // sc[2] (no hay corrección por carril en el generic):
        // sumi[0] += -1·(-31)·(-127) = -3937 → sums[0] += -1·(-3937) = 3937.
        // Verificación real: w32 = d·sc·(raw-32) = 1·(-1)·(-31) = 31;
        // x·w = 127·31 = 3937.
        let mut ql = [0u8; 128];
        ql[32] = 0x01;
        let mut scales = [0i8; 16];
        scales[2] = -1;
        let block = q6_block(F16_ONE, ql, [0; 64], scales);
        let mut x = [0.0f32; 256];
        x[32] = 127.0;
        let got = dot_q6_k_q8_k(&quantize_q8_k(&x), &block);
        assert_eq!(got, 3937.0);
    }

    #[test]
    fn q8k_dots_close_to_exact_dots() {
        // Sanidad sobre los bytes REALES clavados de ornith: el dot q8_K pierde
        // ~0.4%/elemento de x (cuantización de x), el exacto no. Cota holgada.
        let xp: Vec<f32> = (0..256)
            .map(|i| ((i * 7919) % 1009) as f32 / 1009.0 - 0.5)
            .collect();
        let q = quantize_q8_k(&xp);
        for raw in [Q4K_B0.as_slice(), Q4K_B15.as_slice()] {
            let exact = dot_q4_k(&xp, raw);
            let via_q8k = f64::from(dot_q4_k_q8_k(&q, raw));
            assert!(
                (exact - via_q8k).abs() < 0.5,
                "q4_k: exact {exact} vs q8k {via_q8k}"
            );
        }
        for raw in [Q6K_B0.as_slice(), Q6K_B15.as_slice()] {
            let exact = dot_q6_k(&xp, raw);
            let via_q8k = f64::from(dot_q6_k_q8_k(&q, raw));
            assert!(
                (exact - via_q8k).abs() < 0.5,
                "q6_k: exact {exact} vs q8k {via_q8k}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Pin de bytes REALES de ornith-1.0-9b-Q4_K_M.gguf (5.63 GB,
    // D:/AI/models/Ornith-1.0-9B-GGUF/) contra la dequantización INDEPENDIENTE
    // de gguf-py (árbol fuente de llama.cpp). Generado por
    // benchmarks/reference/ornith-decode-crosscheck.py el 2026-08-18.
    // Tolerancia 1e-4: absorbe la diferencia de redondeo f32-vs-f64 del decode
    // (~1e-7/elem) pero NO un mapeo nibble/escala equivocado (error ~O(1)).
    // ------------------------------------------------------------------

    const Q6K_B0: [u8; 210] = [
        0x0d, 0x04, 0x06, 0x61, 0xef, 0xb2, 0x70, 0x0e, 0xed, 0x6f, 0x5a, 0x3d, 0x0b, 0x04, 0x71, 0xe4,
        0x75, 0x2e, 0x86, 0x56, 0xea, 0x5f, 0x6f, 0x40, 0x53, 0xfa, 0xcd, 0x47, 0xbd, 0x0f, 0x70, 0xe2,
        0xfa, 0x3e, 0x63, 0x1e, 0x45, 0xcd, 0xc2, 0x63, 0x44, 0xf0, 0x99, 0x5b, 0xe5, 0x08, 0xc8, 0xb0,
        0x02, 0xed, 0x28, 0xb8, 0x11, 0xf1, 0xb2, 0xfa, 0xc6, 0x3e, 0x62, 0x6b, 0x03, 0x0a, 0xc2, 0xa0,
        0x10, 0xec, 0xbe, 0x83, 0x66, 0x0c, 0x07, 0x94, 0xa1, 0x6c, 0x00, 0xa6, 0x0e, 0x1c, 0x96, 0xdc,
        0x17, 0xc7, 0x01, 0xd0, 0x59, 0x05, 0x53, 0xcd, 0x2a, 0xc9, 0x16, 0xb4, 0xc4, 0xd5, 0x20, 0xcb,
        0x86, 0x34, 0xef, 0xa4, 0xf9, 0x7e, 0xe1, 0x40, 0x03, 0x9c, 0xad, 0x05, 0xbb, 0x69, 0x95, 0xa8,
        0xa1, 0x46, 0xf6, 0x48, 0x2e, 0xa1, 0xdb, 0x1b, 0x3f, 0xc2, 0xdc, 0xad, 0x19, 0xd5, 0x7e, 0x32,
        0x61, 0xa7, 0xad, 0xea, 0xca, 0x65, 0x98, 0x59, 0xb8, 0x62, 0x10, 0xa7, 0x8f, 0x29, 0xa8, 0xff,
        0x08, 0x16, 0x1a, 0xf6, 0xaf, 0x11, 0x56, 0x24, 0x46, 0x60, 0xa5, 0x54, 0xaa, 0x86, 0x56, 0xea,
        0x6a, 0x9a, 0x92, 0xae, 0xa5, 0x47, 0xa1, 0xaa, 0x25, 0xa6, 0x60, 0x36, 0x56, 0x5a, 0xaa, 0x99,
        0xe5, 0x5c, 0x48, 0x6b, 0x69, 0xb1, 0xa5, 0x16, 0x64, 0x22, 0x26, 0x53, 0x99, 0x18, 0xa9, 0xe2,
        0x56, 0xa5, 0x93, 0x95, 0x52, 0x4a, 0x66, 0x54, 0x69, 0x37, 0x6d, 0x46, 0x6c, 0x80, 0xaf, 0x45,
        0xec, 0x00,
    ];

    const Q6K_B15: [u8; 210] = [
        0x7e, 0x81, 0x09, 0x7f, 0x1c, 0xd0, 0x1b, 0xe4, 0xde, 0x79, 0xfc, 0x3e, 0xd1, 0xa0, 0x91, 0xac,
        0x5e, 0x6e, 0x5c, 0xdb, 0x04, 0x11, 0xbc, 0x94, 0xb0, 0xc7, 0x70, 0x27, 0xd6, 0x7a, 0xa5, 0x4a,
        0xaa, 0x9c, 0xfe, 0x22, 0xc0, 0xec, 0x43, 0x40, 0x09, 0xe2, 0xd4, 0x7c, 0x2a, 0x4f, 0x42, 0x16,
        0x07, 0xda, 0x8b, 0xd6, 0x63, 0xf5, 0xd3, 0x31, 0xdd, 0x99, 0x18, 0x5d, 0x92, 0x05, 0xf4, 0x3c,
        0x15, 0x94, 0x66, 0x3d, 0x08, 0x89, 0xf4, 0xe4, 0xf8, 0x61, 0x37, 0x86, 0x8e, 0xa8, 0xea, 0xed,
        0x4c, 0x5d, 0xe6, 0x25, 0x84, 0xca, 0xa0, 0x39, 0xb7, 0x09, 0x21, 0xa3, 0xa9, 0x13, 0xe6, 0x53,
        0x26, 0x0c, 0x03, 0x61, 0x34, 0x68, 0xc2, 0x03, 0x92, 0xec, 0x98, 0x1b, 0x6e, 0x3c, 0x94, 0x19,
        0x3b, 0x79, 0xaa, 0x06, 0x9b, 0x82, 0xc5, 0x9b, 0xb4, 0x98, 0x1c, 0x39, 0x8f, 0x6e, 0x50, 0x9b,
        0x69, 0x96, 0x91, 0x9c, 0x41, 0x92, 0x65, 0x9a, 0x31, 0x2a, 0xb6, 0xe1, 0xe5, 0x66, 0x58, 0x61,
        0xb5, 0x6b, 0xa9, 0x4f, 0xb9, 0x04, 0x56, 0xa1, 0x9a, 0x9a, 0xea, 0x6b, 0xea, 0x28, 0x0e, 0x39,
        0xc9, 0x9a, 0xd9, 0x61, 0x2a, 0x55, 0x5a, 0x2b, 0xd9, 0x20, 0xa2, 0x51, 0x96, 0xa9, 0x89, 0x95,
        0xda, 0x9a, 0xa5, 0x14, 0x92, 0x9b, 0x8c, 0xa9, 0x49, 0x02, 0xda, 0x56, 0xd5, 0x8a, 0xa3, 0xb5,
        0x68, 0xc4, 0xb7, 0x5c, 0xa8, 0x45, 0x3e, 0x4c, 0xa9, 0x57, 0x73, 0x50, 0x80, 0x41, 0xb1, 0xaf,
        0x05, 0x01,
    ];

    const Q4K_B0: [u8; 144] = [
        0x70, 0x06, 0x08, 0x12, 0xf5, 0xee, 0xa2, 0xbf, 0xf5, 0xeb, 0xa0, 0xbf, 0x84, 0x0b, 0xbd, 0x6c,
        0xa5, 0x69, 0x7f, 0x74, 0x59, 0xc7, 0x0a, 0xfb, 0x46, 0x99, 0x86, 0x87, 0x79, 0x48, 0x35, 0xb4,
        0x5a, 0x93, 0x84, 0x68, 0x05, 0x47, 0x20, 0x97, 0xb8, 0x83, 0xab, 0x65, 0x1a, 0x7a, 0x08, 0xd8,
        0x5f, 0x6b, 0xa5, 0x5b, 0x68, 0xc6, 0x58, 0xd2, 0x84, 0x09, 0x6e, 0x72, 0x72, 0x87, 0xb9, 0x59,
        0x6c, 0x97, 0xc6, 0xe7, 0x62, 0x6c, 0x97, 0x8f, 0x04, 0x6e, 0x7d, 0x91, 0x40, 0x48, 0x8f, 0x93,
        0x8e, 0xe6, 0x93, 0x07, 0x68, 0x17, 0x67, 0x51, 0x88, 0x97, 0x48, 0x1b, 0x69, 0x38, 0x52, 0x59,
        0x6a, 0x6c, 0x67, 0x58, 0x69, 0xb9, 0xaf, 0x8a, 0x77, 0x87, 0x40, 0x55, 0x77, 0x36, 0x4b, 0xe5,
        0x50, 0x68, 0x0a, 0x97, 0x00, 0x67, 0xa2, 0x63, 0x67, 0x86, 0x09, 0x89, 0x58, 0x55, 0x62, 0x7f,
        0xf6, 0x1c, 0x27, 0x55, 0x71, 0x74, 0x42, 0x45, 0x87, 0xab, 0xa7, 0xbb, 0x52, 0x5d, 0xf7, 0x64,
    ];

    const Q4K_B15: [u8; 144] = [
        0xfe, 0x05, 0xc8, 0x10, 0xb7, 0xa9, 0xf4, 0xb2, 0xb4, 0xb6, 0xf0, 0x79, 0xa5, 0xad, 0xff, 0xd0,
        0x66, 0x72, 0xf5, 0x65, 0x65, 0x57, 0xeb, 0x97, 0x53, 0xd8, 0x00, 0x47, 0x85, 0x24, 0x85, 0x94,
        0x84, 0x43, 0x75, 0x4b, 0xc0, 0x56, 0xaa, 0x45, 0x77, 0xb5, 0x72, 0xff, 0x70, 0x65, 0x67, 0x5a,
        0x38, 0x66, 0x92, 0x66, 0x94, 0x92, 0xae, 0xfb, 0xb0, 0xc5, 0x79, 0xa9, 0x86, 0xf2, 0x53, 0x01,
        0x3a, 0x91, 0x72, 0xa4, 0x59, 0x8d, 0xc0, 0xd6, 0x95, 0xb2, 0x44, 0x61, 0x94, 0x82, 0xac, 0x85,
        0xa6, 0x57, 0x48, 0x74, 0x73, 0x07, 0xab, 0x66, 0xbd, 0x6b, 0xb7, 0xa3, 0x03, 0xd0, 0xfc, 0x87,
        0x87, 0x75, 0x12, 0xaf, 0xbb, 0xa8, 0x74, 0x8e, 0x65, 0x98, 0xc8, 0x96, 0x98, 0xad, 0x85, 0xd9,
        0x66, 0x3e, 0x93, 0x96, 0x68, 0x4a, 0x66, 0xb8, 0xc2, 0x60, 0x97, 0xc6, 0xf7, 0xd4, 0x87, 0x44,
        0x28, 0xd5, 0xc6, 0x0a, 0x75, 0xaa, 0x7a, 0xe7, 0x56, 0x94, 0xd9, 0x38, 0x94, 0x42, 0x75, 0xca,
    ];

    #[test]
    fn ornith_decode_pin() {
        // x deterministas (sin RNG): misma fórmula que el script generador
        let xp: Vec<f32> = (0..256)
            .map(|i| ((i * 7919) % 1009) as f32 / 1009.0 - 0.5)
            .collect();
        let xa: Vec<f32> = (0..256)
            .map(|i| ((i * 31 + 17) % 257) as f32 / 257.0 - 0.5)
            .collect();
        let mut hot7 = vec![0.0f32; 256];
        hot7[7] = 1.0;
        let mut hot200 = vec![0.0f32; 256];
        hot200[200] = 1.0;

        // (bytes crudos, dot_x_pattern, dot_x_alt, w[7], w[200]) — valores gguf-py
        let cases: [(&[u8], f64, f64, f64, f64); 4] = [
            (
                &Q6K_B0,
                0.009389617280043283,
                -0.06928441728599341,
                -0.0024194717407226562,
                0.015192031860351562,
            ),
            (
                &Q6K_B15,
                0.051974415601894794,
                -0.04944705545670326,
                0.0064716339111328125,
                0.00199127197265625,
            ),
            (
                &Q4K_B0,
                -0.059268411725198325,
                0.015680094173446263,
                0.01824665069580078,
                -0.0007162094116210938,
            ),
            (
                &Q4K_B15,
                -0.07894214695343532,
                0.00035575484487332734,
                0.004852175712585449,
                -0.025249242782592773,
            ),
        ];
        for (raw, dp, da, w7, w200) in cases {
            let dot = |x: &[f32]| {
                if raw.len() == BLOCK_Q6_K_BYTES {
                    dot_q6_k(x, raw)
                } else {
                    dot_q4_k(x, raw)
                }
            };
            let got_dp = dot(&xp);
            let got_da = dot(&xa);
            let got_w7 = dot(&hot7);
            let got_w200 = dot(&hot200);
            assert!((got_dp - dp).abs() < 1e-4, "dot_x_pattern: {got_dp} vs {dp}");
            assert!((got_da - da).abs() < 1e-4, "dot_x_alt: {got_da} vs {da}");
            assert!((got_w7 - w7).abs() < 1e-4, "w[7]: {got_w7} vs {w7}");
            assert!((got_w200 - w200).abs() < 1e-4, "w[200]: {got_w200} vs {w200}");
        }
    }
}
