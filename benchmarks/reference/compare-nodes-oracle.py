#!/usr/bin/env python3
"""Compara los nodos intermedios del camino recurrente contra el volcado del oráculo.

Uso:
  python compare-nodes-oracle.py ornith-evalcallback-prompt5.txt \\
      engine-nodes.f32.bin [--layer 0] [--all] [--tol 1e-4] [--tol-sum 1e-2]

El motor (`unltd forward-oracle --debug-nodes`) escribe:
  [u32 n_tokens] y por token [u32 n_nodes], por nodo
  [u32 name_len][name][u32 ne][f32×ne].
Los nombres llevan sufijo de capa (`attn_norm-0`, `gate-0`, ...), igual que
los cbs del volcado (`common_debug_cb_eval: attn_norm-0 = ... = {ne0, ne1,
ne2, ne3}`).

El volcado imprime bloques 3D por token (ne2 = tokens), con filas (ne1) y
valores (ne0) ELIDIDOS a 6 cuando la dimensión supera 6:
  - valores clavados por fila: posiciones 0,1,2, ne0-3, ne0-2, ne0-1;
  - filas clavadas por token:  0,1,2, ne1-3, ne1-2, ne1-1 (p. ej. beta-0
    {1,32,5} imprime 6 filas de 32 por token, una por head);
  - tensores planos {4096, 5} imprimen las 5 filas completas elididas a 6.
El motor captura un vector plano por nodo y por token (orden fila-mayor por
head igual que el bloque del volcado), así que la comparación es directa:
fila (token t, head h) del volcado ↔ vals[h*ne0 .. (h+1)*ne0] del token t.
Además se gata la suma del tensor completo (el volcado imprime una suma
global por bloque).

Exit 0 = todos los nodos comparados en verde; exit 1 = primer nodo divergente
con detalle (RUN INVALID).
"""

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

HDR_RE = re.compile(
    r"^(?:[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+)?"
    r"common_debug_cb_eval:\s+(\S+)\s+=\s+\(f32\)"
    r".*=\s+\{(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\}\s*$"
)
PREFIX_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+")
SUM_RE = re.compile(r"^sum\s*=\s*([-+0-9.eE]+)\s*$")


def parse_values_line(line: str):
    """Devuelve (valores, fila_elidida). Maneja [a,b,c,...,x,y,z] (6 valores),
    filas de 1 valor, y la línea `...,` de filas omitidas ([] = saltar)."""
    s = PREFIX_RE.sub("", line).strip()
    if s.startswith("["):
        s = s[1:]
    s = s.rstrip()
    if s.endswith("],"):
        s = s[:-2]
    elif s.endswith("]"):
        s = s[:-1]
    s = s.strip()
    if s in ("...", "...,"):
        return [], True
    toks = [t.strip() for t in s.split(", ") if t.strip()]
    if not toks:
        return [], False  # aperturas/cierres de sub-bloque: `[`, `]`
    if len(toks) == 7 and toks[3] == "...":
        return [float(t) for t in toks[:3] + toks[4:]], True
    return [float(t) for t in toks], False


def parse_dump(stream):
    """Devuelve {nombre: {"shape": (ne0, ne1, ne2), "rows": [[v...]], "sum": float}}
    con la ÚLTIMA aparición. Solo entran bloques que matchean HDR_RE
    (las variantes `(reshaped)` no matchean y se ignoran)."""
    found = {}
    state = None
    for raw in stream:
        line = raw.rstrip("\n")
        m = HDR_RE.match(line)
        if m:
            name = m.group(1)
            shape = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            state = {"name": name, "pending": 2, "rows": [], "sum": None, "shape": shape}
            continue
        if state is None:
            continue
        if state["pending"] > 0:
            state["pending"] -= 1
            continue
        s = PREFIX_RE.sub("", line).strip()
        sm = SUM_RE.match(s)
        if sm is not None:
            found[state["name"]] = {
                "shape": state["shape"],
                "rows": state["rows"],
                "sum": float(sm.group(1)),
            }
            state = None
            continue
        vals, _ = parse_values_line(s)
        if vals:
            state["rows"].append(vals)
    return found


def load_engine(path: Path):
    """Devuelve {nombre: [vals_por_token]} en orden de captura (el orden del
    grafo por token), más la lista de nombres en ese orden."""
    b = path.read_bytes()
    (n_tokens,) = struct.unpack_from("<I", b, 0)
    off = 4
    names = []
    nodes = {}
    for _ in range(n_tokens):
        (n_nodes,) = struct.unpack_from("<I", b, off)
        off += 4
        for _ in range(n_nodes):
            (name_len,) = struct.unpack_from("<I", b, off)
            off += 4
            name = b[off : off + name_len].decode("utf-8")
            off += name_len
            (ne,) = struct.unpack_from("<I", b, off)
            off += 4
            vals = struct.unpack_from(f"<{ne}f", b, off)
            off += ne * 4
            if name not in nodes:
                nodes[name] = []
                names.append(name)
            nodes[name].append(list(vals))
    if off != len(b):
        print(
            f"ERROR: {path}: {len(b) - off} bytes sin leer al final", file=sys.stderr
        )
        sys.exit(1)
    return n_tokens, names, nodes


def pinned_cols(ne0: int, n_shown: int):
    if ne0 > 6 and n_shown == 6:
        return [0, 1, 2, ne0 - 3, ne0 - 2, ne0 - 1]
    if n_shown == ne0:
        return list(range(ne0))
    return None  # no comparable


def pinned_rows(ne1: int, n_shown: int):
    if ne1 > 6 and n_shown == 6:
        return [0, 1, 2, ne1 - 3, ne1 - 2, ne1 - 1]
    if n_shown == ne1:
        return list(range(ne1))
    return None


def compare_node(name, tokens, block, args):
    """Compara un nodo: engine tokens (lista de vectores) contra el bloque del
    volcado. Devuelve (ok, worst_diff, detail).

    Dos layoutes según la forma del bloque:
    - 2D {ne0, ne1, 1, 1}: una fila por token (ne1 = tokens); cada fila es el
      vector completo del token (elidida a 6 valores si ne0 > 6).
    - 3D {ne0, ne1, ne2, 1}: ne2 = tokens, ne1 filas por token (p. ej.
      beta {1, 32, 5}: 32 heads por token), filas elididas a 6 si ne1 > 6."""
    ne0, ne1, ne2 = block["shape"]
    rows = block["rows"]
    if ne2 == 1:
        n_tok = ne1
        if n_tok != len(tokens):
            return False, 0.0, f"tokens: dump ne1={n_tok} != engine {len(tokens)}"
        if any(len(v) != ne0 for v in tokens):
            return False, 0.0, f"len vals {[len(v) for v in tokens]} != ne0={ne0}"
        if len(rows) != n_tok:
            return False, 0.0, f"filas {len(rows)} != tokens {n_tok}"
        cmap = pinned_cols(ne0, len(rows[0]))
        if cmap is None:
            return False, 0.0, f"valores por fila {len(rows[0])} (ne0={ne0}) no comparable"
        worst = 0.0
        where = ""
        for t, vals in enumerate(tokens):
            dump_row = rows[t]
            if len(dump_row) != len(cmap):
                return False, 0.0, (
                    f"fila t{t}: dump {len(dump_row)} valores, se esperaban {len(cmap)}"
                )
            for ci, c in enumerate(cmap):
                d = abs(vals[c] - dump_row[ci])
                if d > worst:
                    worst = d
                    where = f"token {t} col {c}: mine={vals[c]:.6f} ora={dump_row[ci]:.6f}"
    else:
        n_tok = ne2
        if n_tok != len(tokens):
            return False, 0.0, f"tokens: dump ne2={n_tok} != engine {len(tokens)}"
        if any(len(v) != ne0 * ne1 for v in tokens):
            return False, 0.0, f"len vals {[len(v) for v in tokens]} != ne0*ne1={ne0*ne1}"
        if len(rows) % n_tok != 0:
            return False, 0.0, f"filas {len(rows)} no múltiplo de tokens {n_tok}"
        rpt = len(rows) // n_tok
        rmap = pinned_rows(ne1, rpt)
        if rmap is None:
            return False, 0.0, f"filas por token {rpt} (ne1={ne1}) no comparable"
        cmap = pinned_cols(ne0, len(rows[0]))
        if cmap is None:
            return False, 0.0, f"valores por fila {len(rows[0])} (ne0={ne0}) no comparable"
        worst = 0.0
        where = ""
        for t, vals in enumerate(tokens):
            for hi, h in enumerate(rmap):
                dump_row = rows[t * rpt + hi]
                if len(dump_row) != len(cmap):
                    return False, 0.0, (
                        f"fila t{t} h{h}: dump {len(dump_row)} valores, se esperaban {len(cmap)}"
                    )
                for ci, c in enumerate(cmap):
                    d = abs(vals[h * ne0 + c] - dump_row[ci])
                    if d > worst:
                        worst = d
                        where = (
                            f"token {t} head {h} col {c}: "
                            f"mine={vals[h*ne0+c]:.6f} ora={dump_row[ci]:.6f}"
                        )
    # suma global del tensor: el oráculo acumula `float sum = 0; sum += v;`
    # (f32 SECUENCIAL, common/debug.cpp) sobre TODO el tensor en orden
    # row-major (i0 más rápido → tokens concatenados, mismo orden que el
    # volcado del motor). Replicar ese orden y tipo EXACTOS — una suma f64
    # aquí produciría ~0.1 de ruido fantasma en tensores con cancelación.
    acc = np.float32(0.0)
    for v in tokens:
        for x in v:
            acc = np.float32(acc + np.float32(x))
    mine_sum = float(acc)
    dsum = abs(mine_sum - block["sum"])
    ok = worst <= args.tol and dsum <= args.tol_sum
    return ok, max(worst, 0.0), (
        f"pin={worst:.2e} ({where}) sum_d={dsum:.2e} (mine={mine_sum:.4f} ora={block['sum']:.4f})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="volcado llama-eval-callback (prompt5)")
    ap.add_argument("engine", help="engine-nodes.f32.bin del motor")
    ap.add_argument("--layer", type=int, default=0, help="capa a comparar (default 0)")
    ap.add_argument("--all", action="store_true", help="comparar las 24 capas recurrentes")
    ap.add_argument("--tol", type=float, default=1e-4, help="tol. valores clavados (default 1e-4)")
    ap.add_argument("--tol-sum", type=float, default=1e-2, help="tol. suma del tensor (default 1e-2)")
    args = ap.parse_args()

    with open(args.dump, "r", encoding="utf-8") as f:
        dump = parse_dump(f)
    n_tokens, names, engine = load_engine(Path(args.engine))

    layers = range(0, 32) if args.all else [args.layer]
    n_fail = 0
    n_pass = 0
    for L in layers:
        if L % 4 == 3:
            continue  # capas de atención: sin captura de nodos
        suffix = f"-{L}"
        for name in names:
            if not name.endswith(suffix):
                continue
            block = dump.get(name)
            if block is None:
                print(f"MISS  {name}: no está en el volcado")
                n_fail += 1
                continue
            ok, _, detail = compare_node(name, engine[name], block, args)
            if ok:
                n_pass += 1
                print(f"PASS  {name:<26} {detail}")
            else:
                n_fail += 1
                print(f"FAIL  {name:<26} {detail}")

    print(f"\n{n_pass} PASS, {n_fail} FAIL (pin tol {args.tol:.1e}, sum tol {args.tol_sum:.1e})")
    if n_fail:
        print("RUN INVALID — un nodo diverge del oráculo.")
        return 1
    print("NODES-ORACLE PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
