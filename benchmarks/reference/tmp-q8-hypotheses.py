"""Prueba hipótesis de cuantización Q8_K de src1 para explicar la anomalía del
token 1 en ffn_gate/ffn_up del oráculo.

Para cada candidato x' (distinta cuantización de p_t1), computa con gguf-py
(dequantize exacto f64) e_pred = W·(x' − p_t1) en las filas clavadas y lo
compara con e_medido = oráculo − W·p_t1 (que debe ser ≈ oráculo − motor, ya
medido). Si e_pred ≈ e_medido → esa es la cuantización que corrió el oráculo.

Candidatos:
  - per-token:    q8 por bloque de 256 con max del bloque (referencia)
  - shared-all:   max compartido con los 5 tokens (mismo bloque de cada token)
  - shared-pair:  max compartido token1+token0
  - f16:          p_t1 redondeado a f16 antes de cuantizar (q8 de f16(p))
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AI\runtimes\llama.cpp\gguf-py")
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

import importlib.util

REF = Path(r"D:\AI\projects\unltd-inference\benchmarks\reference")
spec = importlib.util.spec_from_file_location("cno", REF / "compare-nodes-oracle.py")
cno = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cno)

MODEL = "D:/AI/models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf"
PREC9 = REF / "ornith-prec9-ffn0.txt"
QK = 256


def q8_block(x: np.ndarray, max_abs: float) -> np.ndarray:
    """Réplica de quantize_row_q8_K_ref para UN bloque: devuelve la
    RECONSTRUCCIÓN qs·d (f64), no los bytes."""
    iscale = -127.0 / max_abs
    qs = np.clip(np.rint(iscale * x), -128, 127)  # rint = ties-to-even
    d = 1.0 / iscale
    return qs * d


def q8_per_token(p: np.ndarray) -> np.ndarray:
    """p: (5, 4096) → reconstrucción q8 con max por bloque DE SU token."""
    out = np.empty_like(p)
    for t in range(p.shape[0]):
        for b in range(p.shape[1] // QK):
            seg = p[t, b * QK : (b + 1) * QK]
            out[t, b * QK : (b + 1) * QK] = q8_block(seg, float(np.max(np.abs(seg))))
    return out


def q8_shared(p: np.ndarray, partners) -> np.ndarray:
    """max por bloque compartido con los tokens en `partners`."""
    out = np.empty_like(p)
    for t in range(p.shape[0]):
        for b in range(p.shape[1] // QK):
            segs = [p[u, b * QK : (b + 1) * QK] for u in partners if u < p.shape[0]]
            m = max(float(np.max(np.abs(s))) for s in segs)
            out[t, b * QK : (b + 1) * QK] = q8_block(p[t, b * QK : (b + 1) * QK], m)
    return out


def main():
    with open(PREC9, encoding="utf-8", errors="replace") as f:
        dump = cno.parse_dump(f)
    p = np.array(dump["attn_post_norm-0"]["rows"], dtype=np.float64)  # (5, 4096)
    r = GGUFReader(MODEL, "r")

    q8_pt = q8_per_token(p)
    q8_all = q8_shared(p, [0, 1, 2, 3, 4])
    q8_p01 = q8_shared(p, [0, 1])
    q8_p12 = q8_shared(p, [1, 2])
    p16 = p.astype(np.float16).astype(np.float64)
    q8_f16 = q8_per_token(p16)

    cands = {
        "per-token": q8_pt,  # baseline: cuantización correcta (ref)
        "shared-all5": q8_all,
        "shared-t0t1": q8_p01,
        "shared-t1t2": q8_p12,
        "f16-then-q8": q8_f16,
        "q8-of-t0": q8_pt[0],  # ¿leyó la columna del token 0?
        "q8-of-t2": q8_pt[2],  # ¿leyó la columna del token 2?
    }

    for tname, node, rows_pin in [
        ("blk.0.ffn_gate.weight", "ffn_gate-0", [0, 8322, 12286]),
        ("blk.0.ffn_up.weight", "ffn_up-0", [0, 3934, 12286]),
    ]:
        t = next(t for t in r.tensors if t.name == tname)
        qtype = t.tensor_type
        n_blocks = t.data.shape[1] // 144
        ora = dump[node]["rows"][1]
        print(f"=== {tname} token 1")
        for j in rows_pin:
            deq = dequantize(
                t.data[j].astype(np.uint8).reshape(n_blocks, 144), qtype
            ).astype(np.float64).ravel()
            exact = float(np.dot(p[1], deq))
            e_target = ora[j] - exact
            parts = [f"e_target={e_target:+.3e}"]
            for name, q8x in cands.items():
                x_cand = q8x if q8x.ndim == 1 else q8x[1]
                e_pred = float(np.dot(x_cand - p[1], deq))
                parts.append(f"{name}={e_pred:+.3e}")
            print(f"  row {j:5d}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
