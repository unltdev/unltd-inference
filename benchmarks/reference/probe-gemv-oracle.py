"""Probe (Fase 6, debugging): ¿dónde diverge el GEMV del motor?

Computa dots de referencia con gguf-py sobre los pesos REALES de ornith usando
como x el attn_norm del MOTOR (que ya pasa la puerta contra el volcado), y los
compara con (a) el volcado del oráculo y (b) los valores del motor
(engine-nodes.f32.bin). Tres fuentes, un veredicto por fila.

Uso: python probe-gemv-oracle.py
"""
import struct
import sys

sys.path.insert(0, r"D:\AI\runtimes\llama.cpp\gguf-py")
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = "D:/AI/models/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf"
DUMP = "D:/AI/projects/unltd-inference/benchmarks/reference/ornith-evalcallback-prompt5.txt"
ENGINE = "D:/AI/projects/unltd-inference/benchmarks/reference/engine-nodes.f32.bin"

import re

HDR_RE = re.compile(
    r"^(?:[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+)?"
    r"common_debug_cb_eval:\s+(\S+)\s+=\s+\(f32\)"
    r".*=\s+\{(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\}\s*$"
)
PREFIX_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+")
SUM_RE = re.compile(r"^sum\s*=\s*([-+0-9.eE]+)\s*$")


def parse_dump_blocks(names):
    """Devuelve {nombre: [filas de valores]} para los nombres pedidos."""
    out = {n: [] for n in names}
    state = None
    with open(DUMP, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = HDR_RE.match(line)
            if m:
                name = m.group(1)
                state = name if name in out else None
                pending = 2
                continue
            if state is None:
                continue
            if pending > 0:
                pending -= 1
                continue
            s = PREFIX_RE.sub("", line).strip()
            if SUM_RE.match(s):
                state = None
                continue
            if s.startswith("["):
                s = s[1:]
            s = s.rstrip()
            if s.endswith("],"):
                s = s[:-2]
            elif s.endswith("]"):
                s = s[:-1]
            s = s.strip()
            if s in ("...", "...,"):
                continue
            toks = [t.strip() for t in s.split(", ") if t.strip()]
            if not toks:
                continue
            if len(toks) == 7 and toks[3] == "...":
                out[state].append([float(t) for t in toks[:3] + toks[4:]])
            else:
                out[state].append([float(t) for t in toks])
    return out


def load_engine():
    b = open(ENGINE, "rb").read()
    (n_tokens,) = struct.unpack_from("<I", b, 0)
    off = 4
    nodes = {}
    for _ in range(n_tokens):
        (n_nodes,) = struct.unpack_from("<I", b, off)
        off += 4
        for _ in range(n_nodes):
            (name_len,) = struct.unpack_from("<I", b, off)
            off += 4
            name = b[off : off + name_len].decode()
            off += name_len
            (ne,) = struct.unpack_from("<I", b, off)
            off += 4
            vals = struct.unpack_from(f"<{ne}f", b, off)
            off += ne * 4
            nodes.setdefault(name, []).append(vals)
    return nodes


def main():
    nodes = load_engine()
    attn_norm = np.array(nodes["attn_norm-0"], dtype=np.float64)  # (5, 4096)
    dump = parse_dump_blocks({"linear_attn_qkv_mixed-0", "z-0", "beta-0", "alpha-0"})

    r = GGUFReader(MODEL, "r")

    def probe(tname, node, tok, pairs, bs, d3=False):
        """pairs = [(fila del peso, posición en la fila del volcado)]"""
        t = next(t for t in r.tensors if t.name == tname)
        qtype = t.tensor_type
        n_rows, row_bytes = t.data.shape
        n_blocks = row_bytes // bs
        x = attn_norm[tok]
        rows = dump[node]
        eng = nodes[node][tok]
        print(f"=== {tname} [{qtype.name}] token {tok} (x = attn_norm motor)")
        for j, dp in pairs:
            row = t.data[j].astype(np.uint8).reshape(n_blocks, bs)
            deq = dequantize(row, qtype).astype(np.float64).ravel()
            dot = float(np.dot(x, deq))
            ora = rows[tok * 6 + dp][0] if d3 else rows[tok][dp]
            print(
                f"  row {j:5d}: gguf-py={dot: .6f}  engine={eng[j]: .6f}  "
                f"dump={ora: .6f}"
            )

    # qkv: Q6_K, filas clavadas 0,1,2,8189,8190,8191 → posiciones 0..5 del volcado
    probe("blk.0.attn_qkv.weight", "linear_attn_qkv_mixed-0", 0,
          [(0, 0), (1, 1), (2, 2), (8189, 3), (8190, 4), (8191, 5)], 210)
    # z: Q4_K 4096x4096
    probe("blk.0.attn_gate.weight", "z-0", 2,
          [(0, 0), (1, 1), (2, 2), (4093, 3), (4094, 4), (4095, 5)], 144)
    # beta: Q4_K 4096x32 (3D: 6 filas clavadas por token, 1 valor por fila)
    probe("blk.0.ssm_beta.weight", "beta-0", 0,
          [(0, 0), (1, 1), (2, 2), (29, 3), (30, 4), (31, 5)], 144, d3=True)


if __name__ == "__main__":
    main()
