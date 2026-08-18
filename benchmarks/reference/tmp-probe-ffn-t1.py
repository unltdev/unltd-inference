"""Probe: en token 1, ¿quién se desvía de W·p_t1 exacto — el oráculo o el motor?

Para las filas con error máximo (ffn_gate fila 8322, ffn_up fila 3934) y unas de
control, computa el dot EXACTO (f64, dequantize gguf-py) con x = attn_post_norm
del ORÁCULO (prec9), y compara contra (a) el oráculo y (b) el motor.
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


def main():
    with open(PREC9, encoding="utf-8", errors="replace") as f:
        dump = cno.parse_dump(f)
    n_tok, _names, engine = cno.load_engine(ENGINE)
    p_ora = dump["attn_post_norm-0"]["rows"]  # 5 x 4096
    r = GGUFReader(MODEL, "r")

    for tname, node, rows_pin, bs in [
        ("blk.0.ffn_gate.weight", "ffn_gate-0", [0, 8322, 12286], 144),
        ("blk.0.ffn_up.weight", "ffn_up-0", [0, 3934, 12286], 144),
        ("blk.0.ffn_down.weight", "ffn_out-0", [0, 751, 4094], 210),
    ]:
        t = next(t for t in r.tensors if t.name == tname)
        qtype = t.tensor_type
        row_bytes = t.data.shape[1]
        n_blocks = row_bytes // bs
        print(f"=== {tname} [{qtype.name}] token 1 (x = attn_post_norm oráculo)")
        x = p_ora[1]
        for j in rows_pin:
            row = t.data[j].astype(np.uint8).reshape(n_blocks, bs)
            deq = dequantize(row, qtype).astype(np.float64).ravel()
            exact = float(np.dot(x, deq))
            eng = engine[node][1][j]
            ora = dump[node]["rows"][1][j]
            print(
                f"  row {j:5d}: exact={exact: .9f}  engine={eng: .9f}  ora={ora: .9f} "
                f"| d(eng-exact)={eng-exact:+.3e} d(ora-exact)={ora-exact:+.3e}"
            )


if __name__ == "__main__":
    main()
