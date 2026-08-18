"""Validación cruzada del decode de quants (Fase 4).

Decode de bloques REALES de ornith-1.0-9b-Q4_K_M.gguf con la implementación
INDEPENDIENTE de gguf-py (árbol fuente de llama.cpp) y emite los bytes crudos
+ dots de referencia para el test Rust `ornith_decode_pin` de
crates/unltd-tensor/src/quants.rs.

Uso:  python benchmarks/reference/ornith-decode-crosscheck.py
Sin argumentos: rutas fijas de este repo. Requiere numpy (no el paquete gguf:
se importa desde el árbol fuente con sys.path).
"""
import sys

sys.path.insert(0, r"D:\AI\runtimes\llama.cpp\gguf-py")
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = "D:/AI/models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf"


def x_pattern(n=256):
    """Vector x determinista (sin RNG): el test Rust lo reproduce literalmente."""
    return np.array([((i * 7919) % 1009) / 1009.0 - 0.5 for i in range(n)], dtype=np.float64)


def x_alt(n=256):
    return np.array([((i * 31 + 17) % 257) / 257.0 - 0.5 for i in range(n)], dtype=np.float64)


def main():
    r = GGUFReader(MODEL, "r")
    # (nombre, índice de bloque, n_elems por bloque, nota)
    # fila = 4096 elementos = 16 bloques de 256
    targets = [
        ("blk.0.attn_qkv.weight", 0, "Q6_K first block of row 0"),
        ("blk.0.attn_qkv.weight", 15, "Q6_K last block of row 0"),
        ("blk.0.attn_gate.weight", 0, "Q4_K first block of row 0"),
        ("blk.0.attn_gate.weight", 15, "Q4_K last block of row 0"),
    ]
    xa = x_pattern()
    xb = x_alt()
    print(f"model: {MODEL}")
    print(f"x_pattern = [((i * 7919) % 1009) / 1009.0 - 0.5]")
    print(f"x_alt    = [((i * 31 + 17) % 257) / 257.0 - 0.5]")
    print()
    for name, blk, note in targets:
        t = next(t for t in r.tensors if t.name == name)
        qtype = t.tensor_type
        # gguf-py guarda los cuantizados como (col, bytes_de_fila):
        # una fila = n_bloques × bytes_por_bloque contiguos.
        bs = 210 if qtype.name == "Q6_K" else 144
        row = t.data[0].astype(np.uint8)
        n_blocks = row.shape[0] // bs
        blocks = row.reshape(n_blocks, bs)
        raw = blocks[blk]  # bloque crudo (uint8)
        deq = dequantize(blocks, qtype).astype(np.float64)  # (n_blocks, 256)
        vals = deq[blk]  # los 256 pesos del bloque
        dots = {
            "dot_x_pattern": float(np.dot(xa, vals)),
            "dot_x_alt": float(np.dot(xb, vals)),
            "w[7]": float(vals[7]),
            "w[200]": float(vals[200]),
        }
        print(f"=== {name} [{qtype.name}] {note} (shape raw {t.data.shape})")
        print("raw_hex = [" + ", ".join(f"0x{v:02x}" for v in raw) + "]")
        for k, v in dots.items():
            print(f"{k} = {v!r}")
        print()


if __name__ == "__main__":
    main()
