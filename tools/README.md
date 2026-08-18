# tools/ — herramientas de análisis y fixtures

Análogos directos de las herramientas de `kimi-k3-in-c/tools/`, que probaron
ser tan valiosas como el runtime mismo (ver `docs/AUDIT.md`). La regla: cada
hallazgo del runtime debe poderse reproducir OFFLINE, sin re-correr el modelo.

## Por venir (en orden de aparición en el ROADMAP)

| Herramienta | Análogo K3 | Qué hace |
|---|---|---|
| `make_tiny.py` | `make_tiny_checkpoint.py` | Checkpoint sintético adversarial para Stage 0 (SiTU caps, routing reordenado, dims hostiles al SIMD) |
| `doctor.sh` | `k3-doctor.sh` | Probe de máquina (CPU/flags, RAM, SSD, OS) + presupuesto recomendado |
| `sim_cache.py` | `sim_cache.py` | Replay OFFLINE del trace de accesos: LRU vs Belady vs pinned_lru, techo de compulsory misses, hit rate verdadero |
| `budget.py` | `budget.py` | Clasifica tensores del checkpoint por bytes REALES desde headers (shared experts = residentes, la trampa) |
| `pack.py` | `pack_trunk.py` | (Si se adopta formato propio) empaquetado secuencial de tronco con alineación |

## Contrato del formato de trace (para `sim_cache.py`)

El trace que escribe `--dump-cache-trace` es un CSV mínimo, 8 bytes por request
en binario: `layer u16 | expert u32 | t u16` (mismo esquema que K3, que grababa
12 KB por token). `sim_cache.py` lo lee y reproduce los resultados del runtime
sin tocar el modelo — condición necesaria para confiar en el runtime.
