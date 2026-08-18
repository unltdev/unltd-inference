#!/usr/bin/env python3
"""Extrae tensores COMPLETOS de un volcado de llama-eval-callback.

El volcado se genera con LLAMA_DEBUG_N grande (n > ne[0], p. ej. 1000000) y,
para precisión de round-trip f32, LLAMA_DEBUG_PREC=9 (%.9g — 9 dígitos
significativos == FLT_DECIMAL_DIG, ida y vuelta exacta). Con el formato por
defecto (%12.4f) la extracción sigue funcionando pero con ~5e-5 de error.

Uso:
  python extract-logits-oracle.py dump.txt -o benchmarks/reference/ornith-final/
  llama-eval-callback.exe ... 2>&1 | python extract-logits-oracle.py - -o out/

Tensores capturados (nombres exactos del grafo qwen3.5, ver ornith-tensor-table.txt):
  model.input_embed  {4096, 5}       — embeddings de los 5 tokens del prompt
  attn_norm-0        {4096, 5}       — primera norma (Fase 5: emb + rmsnorm)
  norm               {4096, 5}       — norma final del modelo
  result_norm        {4096, 1}       — estado final (último token)
  result_output      {248320, 1}     — logits del último token

Salidas (en -o): ornith-<name>.f32.bin (floats LE, row-major con ne[0] contiguo)
y ornith-final-summary.txt (argmax, top-5, sumas, recuentos).
"""

import argparse
import re
import struct
import sys
from pathlib import Path

# nombre → (ne0, ne1)
TARGETS = {
    "model.input_embed": (4096, 5),
    "attn_norm-0": (4096, 5),
    "norm": (4096, 5),
    "result_norm": (4096, 1),
    "result_output": (248320, 1),
}

# Línea de cabecera:  common_debug_cb_eval:  <nombre> = (f32)  OP(...) = {ne0, ne1, ne2, ne3}
# Opcionalmente prefijada por el timestamp del log de llama.cpp ("0.05.579.108 I ").
HDR_RE = re.compile(
    r"^(?:[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+)?"
    r"common_debug_cb_eval:\s+(\S+)\s+=\s+\(f32\)"
    r".*=\s+\{(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\}\s*$"
)
# Prefijo de log opcional en CUALQUIER línea del bloque.
PREFIX_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\s+[IWED]\s+")


def parse_values_line(line: str) -> list[float]:
    """Parsea una línea de valores. Cada fila (i1) empieza con '[' pegado al
    primer valor; la línea de cierre es '],' (i1) o ']' (i2/i3)."""
    s = PREFIX_RE.sub("", line).strip()
    if s.startswith("["):
        s = s[1:]
    if s.rstrip().endswith("],"):
        s = s[: s.rindex("],")]
    vals = [float(tok) for tok in s.split(", ") if tok.strip()]
    return vals


def process_stream(stream, out_dir: Path) -> dict:
    """Devuelve {nombre: [floats]} con la ÚLTIMA aparición de cada tensor
    (el grafo se evalúa en varios splits y los nodos compartidos repiten; los
    valores son idénticos, la última aparición es la definitiva)."""
    found = {}
    state = None  # dict(nombre=, espera=2) cuando estamos dentro de un bloque objetivo

    for raw in stream:
        line = raw.rstrip("\n")
        m = HDR_RE.match(line)
        if m:
            name, ne0, ne1 = m.group(1), int(m.group(2)), int(m.group(3))
            if name in TARGETS:
                w0, w1 = TARGETS[name]
                if (ne0, ne1) != (w0, w1):
                    # El grafo repite nodos con distinta forma (p. ej. `norm`:
                    # RMS_NORM {4096,5} en la capa final y GET_ROWS {4096,1}
                    # en la selección de resultado; input_embed {4096,1} en
                    # pasos de decode). No aborta: se salta el bloque y el
                    # check final de `missing` informa si nunca apareció la
                    # forma buscada.
                    print(
                        f"WARNING: {name} con forma {{{ne0}, {ne1}}} != "
                        f"{{{w0}, {w1}}} — bloque saltado",
                        file=sys.stderr,
                    )
                    state = None
                    continue
                state = {"name": name, "pending": 2, "vals": []}
                continue
        if state is not None:
            if state["pending"] > 0:
                state["pending"] -= 1
                continue
            s = PREFIX_RE.sub("", line).strip()
            if s in ("],", "]", ")", "}") or s.startswith("sum"):
                # bloque completo
                name = state["name"]
                w0, w1 = TARGETS[name]
                expect = w0 * w1
                assert len(state["vals"]) == expect, (
                    f"{name}: {len(state['vals'])} != {expect}"
                )
                found[name] = state["vals"]
                state = None
                continue
            state["vals"].extend(parse_values_line(line))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="ruta del volcado o '-' para stdin")
    ap.add_argument("-o", "--out-dir", default=str(Path(__file__).parent / "ornith-final"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dump == "-":
        found = process_stream(sys.stdin, out_dir)
    else:
        with open(args.dump, "r", encoding="utf-8") as f:
            found = process_stream(f, out_dir)

    missing = set(TARGETS) - set(found)
    if missing:
        print(f"ERROR: tensores no encontrados en el volcado: {sorted(missing)}", file=sys.stderr)
        return 1

    lines = []
    for name in ("model.input_embed", "attn_norm-0", "norm", "result_norm", "result_output"):
        vals = found[name]
        s = sum(vals)
        packed = struct.pack(f"<{len(vals)}f", *[float(v) for v in vals])
        bin_path = out_dir / f"ornith-{name}.f32.bin"
        bin_path.write_bytes(packed)
        lines.append(f"{name}: n={len(vals)} sum={s:.9f} -> {bin_path.name}")

    logits = found["result_output"]
    top5 = sorted(enumerate(logits), key=lambda kv: kv[1], reverse=True)[:5]
    lines.append("result_output top-5 (idx, valor):")
    for i, (idx, v) in enumerate(top5):
        lines.append(f"  #{i+1} idx={idx} v={v:.9f}")
    lines.append(f"argmax={top5[0][0]}")

    summary = out_dir / "ornith-final-summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
