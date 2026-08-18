"""Hipótesis 'qs flip': el motor cuantiza SU p (que difiere del oráculo en
~1e-7/elemento) y algunos qs caen del otro lado de un límite de redondeo.

Para el token 1: computa qs_eng = q8(p_eng) y qs_ora = q8(p_ora) (cada uno con
su propio max de bloque), cuenta flips, y verifica que
W·(xq_eng − xq_ora) ≈ eng − ora en las filas clavadas de ffn_gate/ffn_up.
"""
import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AI\runtimes\llama.cpp\gguf-py")
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

REF = Path(r"D:\AI\projects\unltd-inference\benchmarks\reference")
spec = importlib.util.spec_from_file_location("cno", REF / "compare-nodes-oracle.py")
cno = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cno)

MODEL = "D:/AI/models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf"
PREC9 = REF / "ornith-prec9-ffn0.txt"
ENGINE = REF / "engine-nodes.f32.bin"
QK = 256


def qs_block(x: np.ndarray):
    """qs por bloque de 256 (ref): iscale = −127/max_abs, rint ties-to-even."""
    m = float(np.max(np.abs(x)))
    if m == 0.0:
        m = 1.0  # el ref nunca recibe bloques nulos aquí
    return np.rint((-127.0 / m) * x), m


def qs_all(p: np.ndarray):
    qs = np.empty_like(p)
    mx = np.empty(p.shape[0])
    for t in range(p.shape[0]):
        for b in range(p.shape[1] // QK):
            seg = p[t, b * QK : (b + 1) * QK]
            q, m = qs_block(seg)
            qs[t, b * QK : (b + 1) * QK] = q
            mx[t] = 0.0  # no se usa el max global; d sale del bloque
    return qs


def main():
    with open(PREC9, encoding="utf-8", errors="replace") as f:
        dump = cno.parse_dump(f)
    n_tok, _names, engine = cno.load_engine(ENGINE)
    p_ora = np.array(dump["attn_post_norm-0"]["rows"], dtype=np.float64)  # (5,4096)
    p_eng = np.array(engine["attn_post_norm-0"], dtype=np.float64)  # (5,4096)
    r = GGUFReader(MODEL, "r")

    qs_ora = qs_all(p_ora)
    qs_eng = qs_all(p_eng)

    for t in range(min(n_tok, 5)):
        n_flips = int(np.sum(qs_ora[t] != qs_eng[t]))
        d = p_eng[t] - p_ora[t]
        print(
            f"token {t}: flips={n_flips:3d}  |d| mean={np.mean(np.abs(d)):.2e} "
            f"max={np.max(np.abs(d)):.2e} @k{int(np.argmax(np.abs(d)))}"
        )

    # reproducir eng − ora con los flips, en las filas clavadas
    t = 1
    xq_ora = qs_ora[t] / (-127.0)  # reconstrucción hasta el factor max_bloque
    xq_eng = qs_eng[t] / (-127.0)
    # (d = −max/127 por bloque; para la diferencia usamos d REAL por bloque)
    xq_ora = np.empty_like(p_ora[t])
    xq_eng = np.empty_like(p_eng[t])
    for b in range(p_ora.shape[1] // QK):
        sl = slice(b * QK, (b + 1) * QK)
        m_o = float(np.max(np.abs(p_ora[t, sl])))
        m_e = float(np.max(np.abs(p_eng[t, sl])))
        xq_ora[sl] = qs_ora[t, sl] * (-m_o / 127.0)
        xq_eng[sl] = qs_eng[t, sl] * (-m_e / 127.0)

    for tname, node, rows_pin, bs in [
        ("blk.0.ffn_gate.weight", "ffn_gate-0", [0, 8322, 12286], 144),
        ("blk.0.ffn_up.weight", "ffn_up-0", [0, 3934, 12286], 144),
    ]:
        tinfo = next(t for t in r.tensors if t.name == tname)
        qtype = tinfo.tensor_type
        n_blocks = tinfo.data.shape[1] // bs
        ora = dump[node]["rows"][1]
        eng = engine[node][1]
        print(f"=== {tname} token 1: W·(xq_eng−xq_ora) vs eng−ora")
        for j in rows_pin:
            deq = dequantize(
                tinfo.data[j].astype(np.uint8).reshape(n_blocks, bs), qtype
            ).astype(np.float64).ravel()
            e_flips = float(np.dot(xq_eng - xq_ora, deq))
            e_meas = eng[j] - ora[j]
            print(
                f"  row {j:5d}: e_flips={e_flips:+.3e}  e_meas={e_meas:+.3e}  "
                f"match={'OK' if abs(e_flips - e_meas) < 1e-5 else 'NO'}"
            )


if __name__ == "__main__":
    main()
