# Fases 9-10 — Checkpoint: presupuesto de memoria + ejecución disk-first

Fecha: 2026-08-18. Modelo: `ornith-1.0-9b-Q4_K_M.gguf` (Ornith-1.0-9B, 5.63 GB, no movido ni modificado). Prompt: "The capital of France is".

## Arquitectura (qué significa `--memory-budget`)

- **Flag nueva en `unltd run`:** `--memory-budget <SIZE>`. El modelo sigue siendo POSICIONAL (sintaxis real verificada con `run --help`): `unltd.exe run [OPTIONS] --prompt <PROMPT> <MODEL>`. Sin la flag: comportamiento Fase 8 sin cambios.
- **Parser** (`unltd_memory::parse_size`, crates/unltd-memory/src/budget.rs): acepta como mínimo `512M`, `1G`, `2G`, `4G`, `8G`, `512MB`, `4GB` y bytes crudos; prefijos BINARIOS (1G = 1024³, convención de memoria), case-insensitive. Rechaza con error tipificado: vacío, cero, sufijo desconocido (`4X`), overflow u64 y no-número (`1.5G`, `-4G`). 13 tests.
- **Componente de contabilidad** (`unltd_memory::MemoryAccounting`): conoce las 7 cifras del contrato — `configured_budget`, `mandatory_bytes`, `weight_buffer_bytes`, `weight_cache_bytes`, `kv_cache_bytes`, `scratch_bytes`, `runtime_bytes` — y responde `used_controlled_bytes()`, `available_budget()`, `budget_respected()`. Suma saturante; `used > budget` o `mandatory > budget` ⇒ no respetado.
- **Qué controla el presupuesto (memoria CONTROLADA):** KV cache (fórmula exacta ctx × n_head_kv × head_dim_v × 2 × 4 × 8 capas full-attn), estado recurrente (conv ring + GDN, residente deliberado de `new_session` — contabilizado en `runtime_bytes`), scratch (pico del paso: activaciones + logits del loop + índice del top-5) y runtime (heap real del tokenizer vía `Gpt2Tokenizer::heap_bytes()` + índice GGUF). Los pesos aportan **0**: son vistas sobre el mmap, sin copias heap ni cache de dequant.
- **Plan ANTES de asignar:** el plan se suma entero antes de `new_session`; si `mandatory_bytes > configured_budget` → `REFUSING TO RUN` con la tabla de números reales y **exit 2** (1 = error de carga/uso, 2 = presupuesto, 3 = RUN INVALID — contrato de la CLI). Reporte de memoria al inicio (plan) y al final (con cifras MEDIDAS del proceso).

## mmap vs RAM (mapped bytes != resident bytes)

- **File size = mapped virtual = 5.63 GB** (5629108704 bytes): `MappedWeights` mapea el archivo COMPLETO con `memmap2::Mmap` — eso es espacio VIRTUAL, no RAM.
- **El presupuesto NO limita el mapped ni el espacio virtual**: `mapped bytes > memory budget` es VÁLIDO y es el punto disk-first — el Test D corre con modelo de 5.63 GB y presupuesto de 3G (3.22 GB).
- **El PEAK RSS puede superar el presupuesto sin violarlo**: medido 5.24 GB — incluye las páginas del archivo que el SO cachea tras tocarlas (demand paging). Eso es page cache del sistema, no asignación nuestra; el presupuesto gobierna SOLO `used_controlled_bytes` (113.19 MB en la corrida canónica).
- **Métricas del proceso** vía PowerShell (API de Windows existente, sin dependencias nuevas): `PeakWorkingSet64`/`WorkingSet64` de `Get-Process` y `GetProcessIoCounters` (bytes leídos) de kernel32 vía `Add-Type`. Best-effort: si no está disponible, el reporte dice "Unavailable" — no se inventan números.

## Streaming (el mecanismo REALMENTE implementado)

Inspección previa a implementar (directiva §7), verificada en el código:

- `MappedWeights::open` = mmap del archivo completo (crates/unltd-model-loader/src/weights.rs); los tensores se leen como **VISTAS** `&[u8]` sobre el mapa (`tensor_checked` → `bytemuck::cast_slice`); los kernels de dot leen bloques empaquetados directamente de la vista.
- **No hay** copias heap de pesos, ni buffers de capa vivos entre capas, ni lectura completa del GGUF a RAM (el reader parsea solo header + índice + metadata). Los únicos residentes controlados son: KV, estado recurrente (conv + GDN) y el scratch transitorio del paso (~4.7 MB de pico).

Por lo tanto, el mecanismo implementado es **mmap demand paging del SO, NO explicit layer streaming** — no hay lectura por capa explícita ni prefetch, y el documento no llama "streaming explícito" al page cache del SO. La cadena real es: GGUF → mmap archivo completo → vista por tensor/capa → buffers acotados por paso (activaciones) → reuso → capa siguiente. La evidencia de que la memoria controlada es independiente del tamaño del modelo: la corrida con presupuesto EXACTO de 113.06 MB (mínimo viable) genera con un modelo de 5.63 GB. Doble buffering / prefetch no son requeridos para PASS y quedan fuera.

## Test (comando exacto)

```
.\target\release\unltd.exe run "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" --prompt "The capital of France is" --max-tokens 3 --temperature 0 --memory-budget 4G
```

Corridas del gate (todas deterministas, mismas secuencias que Fase 8 en su tramo):

| Test | budget | max-tokens | Resultado |
|------|--------|-----------|-----------|
| A | 4G | 1 | **1/1 MATCH** (11751 " Paris"), exit 0 |
| B (canónica) | 4G | 3 | **3/3 MATCH** (11751, 13, 198), exit 0 |
| C (opcional) | 4G | 5 | **5/5 MATCH**, exit 0 |
| D (restrictivo) | 3G | 3 | **3/3 MATCH**, exit 0 — modelo 5.63 GB > presupuesto 3.22 GB |
| Mínimo viable | 113063271 (bytes crudos) | 1 | **1/1 MATCH**, exit 0 — available_budget = 0 B, budget_respected = true |
| Refusal | 32M | 1 | **REFUSING TO RUN**, exit 2 |

## Memory Report (valores reales, corrida canónica B)

| Categoría | Valor |
|-----------|-------|
| file size (GGUF) | 5.63 GB (5629108704 bytes) |
| mapped (virtual) | 5.63 GB (mmap archivo completo — NO cuenta contra el presupuesto) |
| configured budget | 4.29 GB (4294967296 bytes) |
| weight_buffer_bytes | 0 B (pesos = vistas sobre mmap) |
| weight_cache_bytes | 0 B (sin cache de dequant) |
| kv_cache_bytes | 524.29 KB (ctx 8) |
| scratch_bytes | 4.69 MB (pico estimado del paso) |
| runtime_bytes | 107.98 MB (tokenizer 55.29 MB + índice GGUF + estado recurrente 52.69 MB) |
| used_controlled_bytes | 113.19 MB |
| available_budget | 4.18 GB |
| mandatory_bytes | 113.19 MB |
| budget_respected | true |
| peak RSS (medido) | 5.24 GB (PeakWorkingSet64 — incluye páginas del modelo cacheadas por el SO) |
| working set (medido) | 5.24 GB |
| bytes read (medido) | 21.94 MB (GetProcessIoCounters — lecturas FÍSICAS del proceso; las páginas del modelo salen del cache del SO, caliente por las corridas previas del día) |

**Mínimo viable controlado: 113.06 MB** (113063271 bytes, ctx 6) — corrido con ese presupuesto exacto: PASS, `used = budget`, `available = 0 B`. Rechazo verificado: 32M → `REFUSING TO RUN` con la tabla completa (exit 2).

## Tests y validación

- **Workspace:** `cargo fmt --check` verde (rustfmt 1.9.0-stable 2026-07-14; se aplicó reformateo mecánico del workspace por drift de versión de rustfmt preexistente — solo whitespace), `cargo check --workspace` sin warnings, `cargo test --workspace` **107 verdes** (94 previos + 13 nuevos del módulo budget), `cargo build --release -p unltd-cli` OK.
- **Nuevos (Fase 9):** parser (sufijos cortos/largos, case-insensitive, trim, bytes crudos, vacío, cero, sufijo desconocido, no-número, overflow), contabilidad (suma, release, límite exacto, excedido, mandatory > budget, saturación de overflow). Fases 0-8 intactas (greedy 11, tokenizer 27, loader 17, tensor 35, architectures 4).
- **Performance (registrada):** prefill 73.0 s (14.59 s/token, cache OS tibio), TTFT ~= prefill, decode 14.29 s/token (2 forwards en 28.6 s), total 101.5 s (Test B). El presupuesto no toca el camino numérico: secuencias idénticas a Fase 8 en todo el tramo común.

## Resultado

**PHASE 9-10 RESULT: PASS** — presupuesto de memoria controlada con plan ante asignación, rechazo con números, reporte plan/final con RSS e IO medidos; ejecución disk-first sobre mmap (mapped 5.63 GB > presupuesto 3G / 113.06 MB mínimo viable, sin copias heap de pesos); gate real con Ornith verde en los 5 tests y la negativa de rechazo.
