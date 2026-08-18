"""Compara (prec9) los nodos del oráculo contra engine-nodes.f32.bin, elemento a
elemento: media, std, max de |d|, índice del max, y energía de error por bloque
de 256 (para detectar estructura periódica de bloques Q8_K).

Uso: python tmp-compare-prec9.py dump-prec9.txt engine-nodes.f32.bin
"""
import importlib.util
import sys
from pathlib import Path

REF = Path(r"D:\AI\projects\unltd-inference\benchmarks\reference")
spec = importlib.util.spec_from_file_location("compare_nodes_oracle", REF / "compare-nodes-oracle.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NODES = ["z-0", "attn_post_norm-0", "ffn_gate-0", "ffn_up-0", "ffn_out-0", "l_out-0"]


def main():
    dump_path, engine_path = sys.argv[1], sys.argv[2]
    with open(dump_path, encoding="utf-8", errors="replace") as f:
        dump = mod.parse_dump(f)
    n_tok, _names, engine = mod.load_engine(Path(engine_path))
    print(f"dump: {len(dump)} tensores; engine: {n_tok} tokens")
    for name in NODES:
        blk = dump.get(name)
        toks = engine.get(name)
        if blk is None or toks is None:
            print(f"{name}: faltan datos (dump={blk is not None} engine={toks is not None})")
            continue
        rows = blk["rows"]
        print(f"=== {name}: shape={blk['shape']} {len(toks)} tokens x {len(toks[0])}")
        for t in range(min(n_tok, len(toks))):
            if len(rows) <= t:
                print(f"  token {t}: falta en dump (solo {len(rows)} filas)")
                break
            ora = rows[t]
            eng = toks[t]
            n = min(len(ora), len(eng))
            if n == 0:
                continue
            d = [eng[i] - ora[i] for i in range(n)]
            absd = [abs(x) for x in d]
            mean = sum(d) / n
            std = (sum(x * x for x in d) / n) ** 0.5
            mx = max(absd)
            mi = absd.index(mx)
            nb = (n + 255) // 256
            blk_e = [0.0] * nb
            for i, x in enumerate(absd):
                blk_e[i // 256] += x * x
            topb = sorted(range(nb), key=lambda b: -blk_e[b])[:4]
            topb_s = ", ".join(f"{b}({blk_e[b] ** 0.5:.2e})" for b in topb)
            print(
                f"  t{t}: mean={mean:+.3e} std={std:.3e} max|d|={mx:.3e} "
                f"@i{mi} (eng={eng[mi]:.9g} ora={ora[mi]:.9g}) blk256 top: {topb_s}"
            )


if __name__ == "__main__":
    main()
