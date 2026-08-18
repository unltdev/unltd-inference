"""Valida la transcripción del gemm REPACKED AVX2 (ggml_gemm_q4_K_8x8_q8_K,
arch/x86/repack.cpp) contra el volcado prec9 (ornith-prec9-ffn0.txt).

Hipótesis: el oráculo (build con __AVX2__, REPACK=1) computa MUL_MAT Q4_K con
  - repack de src0 a block_q4_Kx8 (make_block_q4_Kx8, interleave 8)
  - cuantización de src1 (filas 0..3) con ggml_quantize_mat_q8_K_4x8 (AVX2)
  - gemm ggml_gemm_q4_K_8x8_q8_K (camino AVX2, 4 filas × 8 cols por pasada)
Aritmética por elemento de salida (m=fila src1, c=col src0):
  acc = 0; acc_min = 0
  for b in 0..nb-1:                       # bloques de 256 de la contracción
    for sb in 0..3:                       # pares de sub-bloques (2sb, 2sb+1)
      iacc   = Σ_{e∈64sb..64sb+63} v(e,c)*s_{e//32}(c)*q8(e,m)     (i32 exacto)
      prod   = f32mul(d_col[c], d_row[m])
      acc    = f32fma(iacc, prod, acc)          # round32(iacc*prod + acc)
      minacc = bs_{2sb}(m)*m_{2sb}(c) + bs_{2sb+1}(m)*m_{2sb+1}(c) (i32 exacto)
      prodm  = f32mul(dmin_col[c], d_row[m])
      acc_min = f32fma(minacc, prodm, acc_min)
  out = f32sub(acc, acc_min)

Uso: python gemm-avx2-validate.py --parse   (parsea el dump de 11.9 GB y cachea)
     python gemm-avx2-validate.py           (valida desde cache)
"""
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- reutilizar diag
_D = importlib.util.spec_from_file_location(
    "diag", r"C:\Users\gpsan\.claude\jobs\b4d2a9ae\tmp\diag-ffn8322.py")
diag = importlib.util.module_from_spec(_D)
_D.loader.exec_module(diag)

MODEL = diag.MODEL
DUMP = diag.DUMP
CACHE = r"C:\Users\gpsan\.claude\jobs\b4d2a9ae\tmp\gemm-dump-cache.npz"
f32 = diag.f32
f32bits = diag.f32bits
f16 = diag.f16
scale_min_k4 = diag.scale_min_k4

# ---------------------------------------------------------------- aritmética f32 exacta

def f32_to_sigexp(v):
    """f32 → (significando entero con signo, exp) con v == sig * 2^exp EXACTO."""
    u = f32bits(v)
    s = -1 if u >> 31 else 1
    e = (u >> 23) & 0xFF
    f = u & 0x7FFFFF
    if e == 0:
        return s * f, -149  # cero o subnormal
    return s * (f | 0x800000), e - 150


def int2_to_f32(T, e):
    """round32(T * 2^e) con redondeo ties-to-even. T entero de Python exacto."""
    if T == 0:
        return f32(0.0)
    neg = T < 0
    T = abs(T)
    bl = T.bit_length()
    lo_e = e + bl - 1   # valor >= 2^lo_e
    hi_e = e + bl       # valor <  2^hi_e
    if lo_e >= 128:     # valor >= 2^128 > f32 max + media ulp
        return f32(float("-inf") if neg else float("inf"))
    if hi_e <= -150:    # valor < 2^-150 → 0 (el empate exacto cae en la rama subnormal)
        return f32(-0.0) if neg else f32(0.0)
    if hi_e <= -126:    # valor < 2^-126 → subnormal: UNA sola pasada de redondeo
        sh = -(e + 149)  # f = round_half_even(T * 2^(e+149))
        if sh <= 0:
            f = T << (-sh)
        else:
            f, rem = divmod(T, 1 << sh)
            half = 1 << (sh - 1)
            if rem > half or (rem == half and (f & 1)):
                f += 1
        if f >= (1 << 23):  # redondeó hasta el normal mínimo 2^-126
            return f32(struct.unpack("<f", struct.pack("<I",
                                                       (0x80000000 if neg else 0) | (1 << 23)))[0])
        return f32(struct.unpack("<f", struct.pack("<I",
                                                   (0x80000000 if neg else 0) | f))[0])
    # normal: normalizar a 24 bits y codificar
    s = bl - 24
    if s > 0:
        q, rem = divmod(T, 1 << s)
        half = 1 << (s - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
        if q >= (1 << 24):
            q >>= 1
            s += 1
    else:
        q = T << (-s)
    E = e + s  # valor = q * 2^E, q ∈ [2^23, 2^24) (significando normalizado 24 bits)
    be = E + 150  # campo de exponente: field = E + 127 + 23 (q ya incluye el bit implícito)
    if be >= 255:
        return f32(float("-inf") if neg else float("inf"))
    return f32(struct.unpack("<f", struct.pack("<I",
                                               (0x80000000 if neg else 0) | (be << 23) | (q & 0x7FFFFF)))[0])


def fma32_int(iacc, b, c):
    """round32(iacc * b + c) EXACTO: iacc entero (≤2^24 → cvtepi32_ps exacto),
    b y c np.float32. Emula _mm256_fmadd_ps(cvtepi32_ps(iacc), prod, acc)."""
    if iacc == 0 or b == 0.0:
        return f32(c)
    if c == 0.0 and not np.signbit(c):  # +0 exacto
        T = iacc * f32_to_sigexp(b)[0]
        return int2_to_f32(T, f32_to_sigexp(b)[1])
    sb, eb = f32_to_sigexp(b)
    sc, ec = f32_to_sigexp(c)
    if eb <= ec:
        T = iacc * sb + (sc << (ec - eb))  # el término con exponente MENOR se alinea
        E = eb
    else:
        T = iacc * sb * (1 << (eb - ec)) + sc
        E = ec
    return int2_to_f32(T, E)


def fma32_vec(iacc, prod, acc):
    """fma32 vectorizado: camino rápido con f64 + chequeo de doble redondeo;
    los casos en el borde caen al camino exacto (fma32_int)."""
    s64 = iacc.astype(np.float64) * prod.astype(np.float64) + acc.astype(np.float64)
    sl = np.nextafter(s64, -np.inf)
    sh = np.nextafter(s64, np.inf)
    r = s64.astype(np.float32)
    rl = sl.astype(np.float32)
    rh = sh.astype(np.float32)
    mask = (rl == r) & (rh == r)
    bad = np.nonzero(~mask)
    if len(bad[0]):
        out = r.copy()
        for mi, ci in zip(*bad):
            out[mi, ci] = fma32_int(int(iacc[mi, ci]), prod[mi, ci], acc[mi, ci])
        return out, len(bad[0])
    return r, 0


# ---------------------------------------------------------------- cuantización 4x8 (AVX2)

def quantize_4x8(x4):
    """ggml_quantize_mat_q8_K_4x8 (AVX2): por bloque de 256, por fila:
    maxScalar = max|x|; iscale = ±127/maxScalar (regla AVX2: negativo si hay
    positivo en el max, positivo si no); d = 1/iscale; qs = RNE(x*iscale).
    Devuelve por bloque: d (4,), qs (256,4) int32, bs (8,4) int32 (bsums de 32)."""
    n = x4.shape[1]
    assert n % 256 == 0
    nb = n // 256
    ds = np.empty((nb, 4), np.float32)
    qss = np.empty((nb, 256, 4), np.int32)
    bss = np.empty((nb, 8, 4), np.int32)
    for b in range(nb):
        chunk = x4[:, b * 256:(b + 1) * 256]
        for m in range(4):
            row = chunk[m]
            maxsc = np.max(np.abs(row.astype(np.float32)))
            if maxsc == 0.0:
                ds[b, m] = f32(0.0)
                qss[b, :, m] = 0
                bss[b, :, m] = 0
                continue
            if np.any(row == f32(maxsc)):
                iscale = f32(f32(-127.0) / f32(maxsc))
            else:
                iscale = f32(f32(127.0) / f32(maxsc))
            val = np.float32(row.astype(np.float32) * iscale)
            q = ((np.float32(val + f32(12582912.0)).view(np.uint32) & 0x7FFFFF)
                 - 0x400000).astype(np.int32)
            qss[b, :, m] = q
            bss[b, :, m] = q.reshape(8, 32).sum(axis=1)
            ds[b, m] = f32(f32(1.0) / iscale)
    return ds, qss, bss


# ---------------------------------------------------------------- pesos Q4_K (semántico)

def group_arrays(w8):
    """w8: (8, nb*144) uint8 (8 filas consecutivas de src0). Devuelve:
    dcol (nb,8) f32, dmincol (nb,8) f32, s (nb,8,8) int32, m (nb,8,8) int32,
    vs (nb,256,8) int32 = v(e,c)*s_{e//32}(c)."""
    nb = w8.shape[1] // 144
    dcol = np.empty((nb, 8), np.float32)
    dmincol = np.empty((nb, 8), np.float32)
    ss = np.empty((nb, 8, 8), np.int32)
    mm_ = np.empty((nb, 8, 8), np.int32)
    vs = np.empty((nb, 256, 8), np.int32)
    for b in range(nb):
        block = w8[:, b * 144:(b + 1) * 144]
        d16 = np.frombuffer(np.ascontiguousarray(block[:, 0:2]).tobytes(), dtype="<f2")
        dm16 = np.frombuffer(np.ascontiguousarray(block[:, 2:4]).tobytes(), dtype="<f2")
        dcol[b] = d16.astype(np.float32)
        dmincol[b] = dm16.astype(np.float32)
        scales = block[:, 4:16]
        for c in range(8):
            for g in range(8):
                sg, mg = scale_min_k4(scales[c], g)
                ss[b, g, c] = sg
                mm_[b, g, c] = mg
        qs = block[:, 16:144]
        a = np.empty((256, 8), np.int32)
        for c in range(8):
            q = qs[c]
            aa = np.empty(256, np.int32)
            for h in range(4):
                aa[h * 64:h * 64 + 32] = q[h * 32:h * 32 + 32] & 0xF
                aa[h * 64 + 32:h * 64 + 64] = q[h * 32:h * 32 + 32] >> 4
            a[:, c] = aa
        # v(e,c)*s_{e//32}(c): escala por grupo de 32 dentro del bloque
        gidx = np.repeat(np.arange(8), 32)
        vs[b] = a * ss[b, gidx]  # ss[b, gidx] ya es (256, 8), fila e = gidx[e]
    return dcol, dmincol, ss, mm_, vs


# ---------------------------------------------------------------- gemm

def gemm_q4k_8x8_avx2(dcol, dmincol, ss, mm_, vs, q8ds, q8qs, q8bs, nr=4, nc=8):
    """Réplica de ggml_gemm_q4_K_8x8_q8_K (AVX2) para un grupo de 8 columnas.
    dcol/dmincol/ss/mm_/vs: (nb, ...) de group_arrays.
    q8ds: (nb,4) f32; q8qs: (nb,256,4) int32; q8bs: (nb,8,4) int32.
    Devuelve (nr, nc) f32: out[m][j]."""
    nb = dcol.shape[0]
    acc = np.zeros((nr, 8), np.float32)
    acc_min = np.zeros((nr, 8), np.float32)
    nfallback = 0
    for b in range(nb):
        drow = q8ds[b]          # (4,)
        for sb in range(4):
            e0 = 64 * sb
            vsel = vs[b, e0:e0 + 64]        # (64, 8) int32
            qsel = q8qs[b, e0:e0 + 64]      # (64, 4) int32
            iacc = np.matmul(qsel.T, vsel).astype(np.int64)   # (4, 8) exacto
            prod = np.float32(dcol[b][None, :] * drow[:, None])
            acc, nb_fb = fma32_vec(iacc, prod, acc)
            nfallback += nb_fb
            bs_ = q8bs[b]                    # (8, 4)
            mns = mm_[b]                     # (8, 8)
            minacc = np.matmul(bs_[[2 * sb, 2 * sb + 1]].T,
                               mns[[2 * sb, 2 * sb + 1]]).astype(np.int64)  # (4, 8)
            prodm = np.float32(dmincol[b][None, :] * drow[:, None])
            acc_min, nb_fb = fma32_vec(minacc, prodm, acc_min)
            nfallback += nb_fb
    out = np.float32(acc - acc_min)
    return out, nfallback


def gemm_one_col_generic(dcol, dmincol, ss, mm_, vs, q8ds, q8qs, q8bs, m, j,
                         exact=False):
    """Gemm GENERIC (repack.cpp:1905) para (m, j) — para CONTRASTE: por k
    (8 productos) sumf += ((sumi * d_col) * d_row) (3 redondeos por término),
    min por sb: sum_minf += ((mins*(bs0+bs1)) * dmin) * d; final sub.
    Si exact=True usa fma32_int para el camino rápido de validación."""
    nb = dcol.shape[0]
    sumf = f32(0.0)
    sum_minf = f32(0.0)
    for b in range(nb):
        dc = f32(dcol[b, j])
        dminc = f32(dmincol[b, j])
        dr = f32(q8ds[b, m])
        for k in range(16):
            # v0 = nibble bajo del byte 8k+i (col j), v1 = alto
            # a0 = q8 elem 64*(k//4)+8*(k%4)+i ; a1 = a0+32
            sumi = 0
            for i in range(8):
                e0 = 64 * (k // 4) + 8 * (k % 4) + i
                v0 = vs[b, e0, j]  # ojo: vs ya incluye la escala... NO para generic
                # vs incluye s — para el generic necesito v crudo y s por grupo.
                # Reconstruir: v = vs / s  (entero exacto)
                s0 = int(ss[b, (e0 // 32), j])
                s1 = int(ss[b, (e0 // 32) + 1, j]) if False else None
                a0 = int(q8qs[b, e0, m])
                a1 = int(q8qs[b, e0 + 32, m])
                sumi += (int(v0) // s0 if s0 else 0) * a0 * s0
                v1 = int(vs[b, e0 + 32, j])
                sumi += (v1 // s0 if s0 else 0) * a1 * s0
            # generic: scales_0 = grupo k//4 (s del sub-bloque 2*(k//4)??)
            # -> NO: scales_0 = utmp + (k//4)*32 → grupos 2*(k//4) y 2*(k//4)+1
            # corregir abajo; aquí placeholder de estructura
            raise NotImplementedError("revisar mapeo de escalas del generic")


# ---------------------------------------------------------------- main

def main():
    tensors, data_start = diag.gguf_tensors(MODEL)

    if "--parse" in sys.argv:
        want = {"attn_norm-0", "z-0", "attn_post_norm-0", "ffn_gate-0"}
        found = diag.parse_dump(DUMP, want)
        np.savez_compressed(CACHE, **{k: np.stack(v) for k, v in found.items()})
        print("cache guardado:", CACHE, {k: v.shape for k, v in found.items()})
        return

    cache = np.load(CACHE)
    attn_norm = cache["attn_norm-0"]       # (5, 4096)
    z0 = cache["z-0"]                      # (5, 4096)
    attn_post = cache["attn_post_norm-0"]  # (5, 4096)
    ffn_gate = cache["ffn_gate-0"]         # (5, 12288)
    print("cache:", {k: cache[k].shape for k in ["attn_norm-0", "z-0", "attn_post_norm-0", "ffn_gate-0"]})

    def load_rows(name, rows, row_bytes):
        d, tt, off = tensors[name]
        mm = np.memmap(MODEL, mode="r", dtype=np.uint8)
        out = np.empty((len(rows), row_bytes), np.uint8)
        for i, r in enumerate(rows):
            base = data_start + off + r * row_bytes
            out[i] = mm[base:base + row_bytes]
        return out

    # ================= FASE 2: z-0 completo (attn_gate 4096×4096, tokens 0..3)
    print("\n== FASE 2: z-0 (gemm AVX2) contra oráculo ==")
    ag = tensors["blk.0.attn_gate.weight"]
    ag_dims, ag_tt, _ = ag
    dim_in = int(ag_dims[0])
    nrows = int(ag_dims[1])
    row_bytes = diag.BLOCK[ag_tt] * (dim_in // 256)
    nb = dim_in // 256
    x4 = attn_norm[:4]                     # (4, 4096)
    q8ds, q8qs, q8bs = quantize_4x8(x4)
    assert q8ds.shape[0] == nb

    nbad_total = 0
    nfb_total = 0
    nelem = 0
    for g in range(nrows // 8):
        w8 = load_rows("blk.0.attn_gate.weight", range(g * 8, g * 8 + 8), row_bytes)
        dcol, dmincol, ss, mm_, vs = group_arrays(w8)
        out, nfb = gemm_q4k_8x8_avx2(dcol, dmincol, ss, mm_, vs, q8ds, q8qs, q8bs)
        nfb_total += nfb
        ref = z0[:4, g * 8:g * 8 + 8]
        neq = np.count_nonzero(np.not_equal(out, ref))
        nbad_total += neq
        nelem += out.size
        if neq:
            print(f"  grupo {g}: {neq} de 32 difieren")
            mi, ci = np.nonzero(np.not_equal(out, ref))[0][0], \
                np.nonzero(np.not_equal(out, ref))[1][0]
            print(f"    primer diff [{mi},{ci}]: mine={out[mi,ci]!r} ora={ref[mi,ci]!r}")
        del w8
    print(f"z-0: {nbad_total}/{nelem} elementos difieren "
          f"(fallbacks exactos totales: {nfb_total})")

    # ============ FASE 3: ffn_gate-0 col 8322 exacto (token 1) + contraste generic
    print("\n== FASE 3: ffn_gate-0[token 1] cols 8320..8327 (exacto) ==")
    fg = tensors["blk.0.ffn_gate.weight"]
    fg_dims, fg_tt, _ = fg
    dim_in_fg = int(fg_dims[0])
    row_bytes_fg = diag.BLOCK[fg_tt] * (dim_in_fg // 256)
    nb_fg = dim_in_fg // 256
    xcol = 1040  # grupo de 8 columnas que contiene la 8322
    w8 = load_rows("blk.0.ffn_gate.weight", range(xcol * 8, xcol * 8 + 8), row_bytes_fg)
    dcol, dmincol, ss, mm_, vs = group_arrays(w8)

    x4 = attn_post[:4]
    q8ds, q8qs, q8bs = quantize_4x8(x4)
    assert q8ds.shape[0] == nb_fg

    # camino EXACTO elemento a elemento (sin f64): todos los (m, j)
    acc = np.zeros((4, 8), np.float32)
    acc_min = np.zeros((4, 8), np.float32)
    for b in range(nb_fg):
        drow = q8ds[b]
        for sb in range(4):
            e0 = 64 * sb
            vsel = vs[b, e0:e0 + 64]
            qsel = q8qs[b, e0:e0 + 64]
            iacc = np.matmul(qsel.T, vsel).astype(np.int64)
            for m in range(4):
                for j in range(8):
                    prod = f32(f32(dcol[b, j]) * f32(drow[m]))
                    acc[m, j] = fma32_int(int(iacc[m, j]), prod, acc[m, j])
            bs_ = q8bs[b]
            mns = mm_[b]
            minacc = np.matmul(bs_[[2 * sb, 2 * sb + 1]].T,
                               mns[[2 * sb, 2 * sb + 1]]).astype(np.int64)
            for m in range(4):
                for j in range(8):
                    prodm = f32(f32(dmincol[b, j]) * f32(drow[m]))
                    acc_min[m, j] = fma32_int(int(minacc[m, j]), prodm, acc_min[m, j])
    out = np.float32(acc - acc_min)
    ref = ffn_gate[:4, xcol * 8:xcol * 8 + 8]
    for m in range(4):
        for j in range(8):
            c = xcol * 8 + j
            ok = f32bits(out[m, j]) == f32bits(ref[m, j])
            print(f"    col {c} token {m}: mine={out[m,j]!r} ora={ref[m,j]!r} "
                  f"{'BIT-EXACTO' if ok else '*** DIFIERE ***'}")
    neq = np.count_nonzero(np.not_equal(out, ref))
    print(f"  total: {neq}/32 difieren")

    # contraste: aritmética del gemm GENERIC sobre la misma col 8322, token 1
    print("\n== contraste: gemm GENERIC (3 redondeos) col 8322 token 1 ==")
    m, j = 1, 2
    sumf = f32(0.0)
    sum_minf = f32(0.0)
    for b in range(nb_fg):
        dc = f32(dcol[b, j])
        dminc = f32(dmincol[b, j])
        dr = f32(q8ds[b, m])
        for k in range(16):
            # generic: v0/v1 nibbles del byte 8k+i (x8 qs = natural byte 8k+i)
            # a0 = q8 elem 64*(k//4)+8*(k%4)+i ; a1 = +32
            # s para v0: grupo 2*(k//4)?? NO: scales_0 = utmp+(k//4)*32 → grupo
            # 2*(k//4) para v0 y 2*(k//4)+1 para v1 (ver derivación).
            sumi = 0
            for i in range(8):
                e0 = 64 * (k // 4) + 8 * (k % 4) + i
                # v crudo: vs[b,e0,j] = v * s_{e0//32}; e0//32 == 2*(k//4) ✓
                s0 = int(ss[b, 2 * (k // 4), j])
                v0 = int(vs[b, e0, j]) // s0
                a0 = int(q8qs[b, e0, m])
                sumi += v0 * a0 * s0
                s1 = int(ss[b, 2 * (k // 4) + 1, j])
                v1 = int(vs[b, e0 + 32, j]) // s1
                a1 = int(q8qs[b, e0 + 32, m])
                sumi += v1 * a1 * s1
            t = f32(f32(f32(sumi) * dc) * dr)   # 2 muls redondeados
            sumf = f32(f32(sumf) + t)
        for sb in range(8):
            bs_ = int(q8bs[b, sb, m])
            mn = int(mm_[b, sb, j])
            t = f32(f32(f32(f32(mn) * f32(bs_)) * dminc) * dr)
            sum_minf = f32(f32(sum_minf) + t)
    gen = f32(f32(sumf) - f32(sum_minf))
    tgt = ref[m, j]
    print(f"    generic: {gen!r}  vs oráculo {tgt!r}  "
          f"{'BIT-EXACTO' if f32bits(gen) == f32bits(tgt) else 'difiere (como esperado si el oráculo corrió AVX2)'}")


if __name__ == "__main__":
    main()
