# UNLTD Inference (resumen en español)

Runtime de inferencia LLM **disk-first y CPU-first** escrito en Rust, desarrollado por UNLTD como proyecto experimental/de investigación. Versión: **1.0.0** · Plataforma validada: Windows x86-64 · Formato: GGUF.

> El README principal es `README.md` (inglés). Este documento es un resumen.

## La idea central

```
model_size > configured_memory
```

El límite de memoria es una **característica explícita** del runtime: los pesos viven en un mmap del archivo (paginado por demanda por el SO) y el runtime solo controla una cantidad acotada de memoria propia. Con un modelo de 5.63 GB se genera texto con presupuesto de 3-4 GB — y hasta con el mínimo exacto de ~113 MB.

## Capacidades v1.0 (honestas)

- Parsing GGUF + acceso a pesos vía mmap (vistas, sin copias a heap)
- Forward Qwen3.5-compatible (Ornith 1.0 9B, Q4_K_M) — híbrido recurrente (GatedDeltaNet) + atención completa, M-RoPE
- Kernels de cuantización Q4_K/Q6_K/Q8_0 (camino scalar de referencia)
- Tokenizador BPE byte-level (gpt2) construido desde el GGUF — bit-exacto contra llama.cpp para el prompt validado
- Generación greedy determinista (temperatura 0) con KV cache incremental
- Presupuesto de memoria (`--memory-budget`) con plan antes de asignar, contabilidad y negativa `REFUSING TO RUN` (exit 2)
- Medición del working set / I/O del proceso (Windows, vía PowerShell)

**Importante:** el modo es **mmap demand paging del SO** — NO es "explicit layer streaming" (eso no existe todavía en v1.0).

## Comandos

```powershell
cargo build --release -p unltd-cli

.\target\release\unltd.exe inspect  "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf"
.\target\release\unltd.exe tokenize "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" --text "The capital of France is"

.\target\release\unltd.exe run `
  "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" `
  --prompt "The capital of France is" `
  --max-tokens 3 `
  --temperature 0 `
  --memory-budget 4G
```

El modelo es argumento **POSICIONAL** (no existe `--model`). `--memory-budget` acepta `512M`/`1G`/`2G`/`4G`/`8G`/`512MB`/`4GB`/bytes crudos (prefijos binarios) y controla SOLO la memoria controlada (KV + scratch + estado de runtime): el mapped del modelo y el page cache del SO no cuentan contra el presupuesto.

## Resultados reales (campaña Fases 9-10, 2026-08-18)

| Métrica | Resultado |
|---|---|
| Modelo | Ornith 1.0 9B Q4_K_M |
| Tamaño | 5.63 GB |
| Presupuestos probados | 4G, 3G, mínimo exacto 113.06 MB |
| Memoria controlada | ~113.19 MB (ctx 8) |
| Peak working set | ~5.24 GB (incluye page cache del SO) |
| Tests | 107 verdes |
| Generación | PASS (1/3/5 tokens == oráculo en el tramo testeado) |
| Modelo > presupuesto | PASS |

## Validación contra llama.cpp (oráculo)

- Tokenizer **bit-exacto** para el prompt validado.
- Los primeros **11 pasos** greedy coinciden con el oráculo.
- Divergencia posterior conocida y medida (primer mismatch en el token generado 12): diferencia numérica scalar vs AVX2/repacked amplificada por el contexto. Documentada en `docs/PHASE-7-8-CHECKPOINT.md`. No hay garantía de bit-exactitud de tensores internos; el contrato es corrección + determinismo del runtime.

## Limitaciones

CPU-only (scalar, ~14 s/token de decode vs ~230 ms de llama.cpp), sin sampling, sin GPU/SIMD/hilos, sin MoE, sin streaming explícito de capas, solo Windows validado end-to-end, divergencia numérica documentada contra el oráculo optimizado en contextos largos.

## Licencia

**No especificada todavía**: el repositorio no tiene archivo LICENSE (el manifest lleva metadata `license = "Apache-2.0"` a la espera de la decisión final del dueño del proyecto).

Documentación completa: `README.md` y `docs/` (AUDIT, ARCHITECTURE, MEMORY-DESIGN, checkpoints por fase, etc.).
