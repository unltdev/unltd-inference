#!/usr/bin/env python3
"""Compara las salidas por capa del motor contra el volcado del oráculo (Fase 6).

Uso:
  python compare-layers-oracle.py ornith-evalcallback-prompt5.txt engine-layers.f32.bin [--tol 1e-4]

El motor (`unltd forward-oracle`) escribe engine-layers.f32.bin con:
  [u32 n_layer][u32 n_tokens][u32 n_embd]
  por capa il, por token t (capa-major): l_out[n], attn_residual[n],
  linear_attn_out[n] en f32 LE.

El volcado imprime cada fila de 4096 ELIDIDA (primeros 3 valores, "...",
últimos 3) con %12.4f → piso 5e-5, y cierra cada bloque con `sum = <suma>`.
Dos puertas por (capa, token, tensor):
  1. 6 valores clavados por fila (posiciones 0,1,2,4093,4094,4095) ≤ tol;
  2. suma del tensor completo: |Δ| ≤ tol_sum (1e-2 — la suma difiere por
     acumulación de ~1e-7 por valor, cualquier divergencia real la rompe).
Las capas de atención no tienen `linear_attn_out` en el volcado: se compara
solo donde existe.

Exit 0 = todas las capas en verde; exit 1 = tabla de fallos (RUN INVALID).
"""

import argparse
import re
import struct
import sys
from pathlib import Path

N_LAYER = 32
N_TOKENS = 5
N_EMBD = 4096
PINNED = [0, 1, 2, N_EMBD - 3, N_EMBD - 2, N_EMBD - 1]

# capas recurrentes: todas menos (il+1) % 4 == 0
RECURRENT = [il for il in range(N_LAYER) if (il + 1) % 4 != 0]

TARGETS = {f"l_out-{il}": (N_EMBD, N_TOKENS) for il in range(N_LAYER)}
TARGETS.update({f"attn_residual-{il}": (N_EMBD, N_TOKENS) for il in range(N_LAYER)})
TARGETS.update({f"linear_attn_out-{il}": (N_EMBD, N_TOKENS) for il in RECURRENT})

HDR_RE = re.compile(
    r"^(?:[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+)?"
    r"common_debug_cb_eval:\s+(\S+)\s+=\s+\(f32\)"
    r".*=\s+\{(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\}\s*$"
)
PREFIX_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+")
SUM_RE = re.compile(r"^sum\s*=\s*([-+0-9.eE]+)\s*$")


def parse_values_line(line: str):
    """Devuelve (valores, elidida). Formato elidido: [a, b, c, ..., x, y, z]."""
    s = PREFIX_RE.sub("", line).strip()
    if s.startswith("["):
        s = s[1:]
    if s.rstrip().endswith("],"):
        s = s[: s.rindex("],")]
    toks = [t.strip() for t in s.split(", ") if t.strip()]
    if "..." in toks:
        if not (len(toks) == 7 and toks[3] == "..."):
            raise ValueError(f"formato elidido inesperado: {toks}")
        return [float(t) for t in toks[:3] + toks[4:]], True
    return [float(t) for t in toks], False


def parse_dump(stream) -> dict:
    """Devuelve {nombre: {"rows": [[6 valores] x ne1], "sum": float}} con la
    ÚLTIMA aparición. Bloque completo = filas + línea `sum = <valor>`."""
    found = {}
    state = None  # dict(name=, pending=2, rows=[], sum=None, want_sum=False)
    for raw in stream:
        line = raw.rstrip("\n")
        m = HDR_RE.match(line)
        if m:
            name, ne0, ne1 = m.group(1), int(m.group(2)), int(m.group(3))
            if name in TARGETS:
                w0, w1 = TARGETS[name]
                if (ne0, ne1) != (w0, w1):
                    print(
                        f"WARNING: {name} con forma {{{ne0}, {ne1}}} != "
                        f"{{{w0}, {w1}}} — bloque saltado",
                        file=sys.stderr,
                    )
                    state = None
                    continue
                state = {"name": name, "pending": 2, "rows": [], "sum": None}
                continue
        if state is not None:
            if state["pending"] > 0:
                state["pending"] -= 1
                continue
            s = PREFIX_RE.sub("", line).strip()
            sm = SUM_RE.match(s)
            if sm is not None:
                name = state["name"]
                w0, w1 = TARGETS[name]
                if len(state["rows"]) != w1:
                    print(
                        f"ERROR: {name}: {len(state['rows'])} filas != {w1}",
                        file=sys.stderr,
                    )
                    return {}
                found[name] = {"rows": state["rows"], "sum": float(sm.group(1))}
                state = None
                continue
            if s in ("],", "]", ")", "}"):
                continue  # cierres intermedios; el cierre real es la línea sum
            vals, _ = parse_values_line(s)
            if len(vals) != 6:
                print(f"ERROR: {state['name']}: fila con {len(vals)} valores", file=sys.stderr)
                return {}
            state["rows"].append(vals)
    return found


def load_engine(path: Path):
    b = path.read_bytes()
    n_layer, n_tokens, n = struct.unpack_from("<3I", b, 0)
    expect = 12 + n_layer * n_tokens * 3 * n * 4
    if len(b) != expect:
        print(
            f"ERROR: {path}: {len(b)} bytes, se esperaban {expect} "
            f"(n_layer={n_layer}, n_tokens={n_tokens}, n={n})",
            file=sys.stderr,
        )
        sys.exit(1)
    return n_layer, n_tokens, n, b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="volcado llama-eval-callback (prompt5)")
    ap.add_argument("engine", help="engine-layers.f32.bin del motor")
    ap.add_argument("--tol", type=float, default=1e-4, help="tol. valores clavados (default 1e-4)")
    ap.add_argument("--tol-sum", type=float, default=1e-2, help="tol. suma del tensor (default 1e-2)")
    ap.add_argument("--detail", type=int, default=None, help="detalle por tensor de UNA capa")
    args = ap.parse_args()

    with open(args.dump, "r", encoding="utf-8") as f:
        found = parse_dump(f)

    missing = set(TARGETS) - set(found)
    if missing:
        print(
            f"ERROR: cbs no encontrados en el volcado: {sorted(missing)[:10]}"
            f"{'...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )
        return 1

    n_layer, n_tokens, n, eng = load_engine(Path(args.engine))
    if (n_layer, n_tokens, n) != (N_LAYER, N_TOKENS, N_EMBD):
        print(
            f"ERROR: header del motor ({n_layer}, {n_tokens}, {n}) != "
            f"esperado ({N_LAYER}, {N_TOKENS}, {N_EMBD})",
            file=sys.stderr,
        )
        return 1

    names = ("l_out", "attn_residual", "linear_attn_out")
    stride = n_tokens * 3 * n
    failures = []
    n_pass = 0
    for il in range(n_layer):
        off = il * stride
        worst_pin = (0.0, "")
        worst_sum = (0.0, "")
        for k, name in enumerate(names):
            key = f"{name}-{il}"
            if key not in TARGETS:
                continue  # linear_attn_out no existe en capas de atención
            # suma sobre TODO el tensor (el volcado imprime una suma global)
            tot = 0.0
            for t in range(n_tokens):
                mine = struct.unpack_from(f"<{n}f", eng, off + t * 3 * n * 4 + k * n * 4)
                tot += sum(mine)
                ora_rows = found[key]["rows"]
                pin = max(abs(mine[PINNED[p]] - ora_rows[t][p]) for p in range(6))
                if args.detail == il:
                    print(
                        f"  {key} t{t}: pin={pin:.2e} "
                        f"(mine[{PINNED[0]}]={mine[PINNED[0]]:.4f} "
                        f"ora={ora_rows[t][0]:.4f})"
                    )
                if pin > worst_pin[0]:
                    worst_pin = (pin, f"{key}[token {t}][max pin]")
            ds = abs(tot - found[key]["sum"])
            if args.detail == il:
                print(f"  {key} total: sum_d={ds:.2e} (mine={tot:.4f} ora={found[key]['sum']:.4f})")
            if ds > worst_sum[0]:
                worst_sum = (ds, f"{key}[total]")
        ok = worst_pin[0] <= args.tol and worst_sum[0] <= args.tol_sum
        if ok:
            n_pass += 1
            print(
                f"PASS capa {il:2d}  pin={worst_pin[0]:.2e} ({worst_pin[1]})  "
                f"sum={worst_sum[0]:.2e} ({worst_sum[1]})"
            )
        else:
            failures.append((il, worst_pin, worst_sum))
            print(
                f"FAIL capa {il:2d}  pin={worst_pin[0]:.2e} ({worst_pin[1]})  "
                f"sum={worst_sum[0]:.2e} ({worst_sum[1]})"
            )

    print(f"\n{n_pass}/{n_layer} capas PASS (pin tol {args.tol:.1e}, sum tol {args.tol_sum:.1e})")
    if failures:
        print("FAIL:", ", ".join(f"capa {il}" for il, _, _ in failures))
        print("RUN INVALID — el forward diverge del oráculo en estas capas.")
        return 1
    print("FORWARD-ORACLE PASS: las 32 capas dentro de tolerancia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
