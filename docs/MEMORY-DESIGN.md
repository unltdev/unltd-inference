# MEMORY-DESIGN.md — Estrategias de memoria y disco para `unltd-inference`

> Premisa del proyecto: `model_size > available_RAM` es una **característica central**, no un caso excepcional.
> Hardware objetivo: 16 GB de RAM, CPU x86-64 (AVX2), Windows como sistema principal, WSL2/Linux disponible, ≤ 1 TB de disco, GPU no obligatoria.

---

## 1. Principios heredados de la auditoría

1. **La RAM es un dial, no un piso.** El mismo modelo debe correr con presupuestos distintos y producir salida idéntica (K3 lo garantiza byte a byte en 12 escalones).
2. **El tráfico por token decide la asignación.** Cada clase de pesos tiene un tráfico garantizado por token distinto; el presupuesto va donde el valor marginal es mayor (tronco antes que cache de expertos: 1,69× medido).
3. **Contar antes que cronometrar.** GB leídos, requests, evictions, retención — los contadores no se mueven con el ruido (piso medido: 33% en relojes).
4. **Refuse rather than guess.** El plan de memoria se suma entero ANTES de asignar; si no entra, el motor se niega con ambos números.
5. **Nunca desquantizar para cachear.** Se cachea y se multiplica en formato empaquetado (un slot de 17,55 MB vs 132 MB = 7,5× más capacidad).
6. **El orden de acceso es el activo.** Recorridos fijos (capas) → prefetch perfecto + pin determinístico; accesos dependientes de datos (expertos) → LRU + prefetch batch.

---

## 2. Taxonomía de estrategias: qué usa k3, qué adoptamos, qué rechazamos

| Estrategia | ¿K3 la usa? | Decisión para `unltd-inference` | Razón |
|---|---|---|---|
| **mmap + page cache** | NO (pread explícito; comentario en `k3_st.h`: mmap haría inmedible el RSS a 1,56 TB) | **Adoptar como política opcional** ("resident" / "mmap"), NO como mecanismo único | mmap es lo que llama.cpp usa y funciona muy bien para modelos que caben; para modelos >> RAM el page cache thrashing es incontrolable y el RSS deja de ser una medición. En un runtime de investigación, mmap es UNA política más del MemoryManager, elegible por presupuesto. |
| **Layer streaming (trunk)** | SÍ: pin prefix + ring de 2 slots + reader thread | **Adoptar completo** (núcleo del diseño) | Es LA técnica que convierte 108,81 GB en ~2,4 GB sin perder exactitud. Generaliza a todo decoder: orden de capas fijo por token. |
| **Chunked weight loading** | SÍ (implícito): la unidad de carga es la capa (corrida contigua en `trunk.bin`, una pread por capa); widen por tensores chicos | **Adoptar**: unidad = capa, con chunking interno de tensores grandes | Para modelos sin tronco empaquetado, el GGUF ya es contiguo por tensor; la capa sigue siendo la unidad natural de stream + widen. |
| **Expert streaming** | SÍ: 1.472 expertos/token leídos por demanda, nunca residentes | **Adoptar para MoE** | Cualquier MoE (DeepSeek V2/V3, Mixtral, Qwen3-MoE) tiene >90% del checkpoint en expertos ruteados; el patrón de 6 tensores contiguos → una pread coalescida se replica empaquetando per-experts en GGUF. |
| **Expert LRU cache + prefetch batch** | SÍ: 3 fases (reserva serial, lecturas ordenadas por offset, publicación parcial), estado INFLIGHT, dedup intra-batch | **Adoptar completo** | Genérico a cualquier MoE; la calidad depende del hot set del modelo (ver §5). |
| **Selective residency (pin)** | SÍ: capas 0..K-1 pineadas con asignación exacta por capa; hit rate = K/N determinístico | **Adoptar**: pin por CLASE de pesos (embeddings, capas iniciales, expertos calientes) | Los embeddings + lm_head (4,7 GB en K3) son siempre residentes; en modelos chicos caben; en grandes se pueden stream-ear igual que capas. |
| **Prefetch / double buffering** | SÍ: reader thread + ring de 2 slots (71,75 → 42,27 s/token); prefetch de capa L+1 mientras se computa L; fadvise en camino buffered | **Adoptar**: ring ≥2 slots con publicación explícita post-éxito; prefetch de la siguiente capa; fadvise/pareto en modo buffered | La lección crítica del código: con 1 slot + reader async hay corrupción silenciosa. El invariante de publicación es parte del diseño. |
| **Acceso secuencial al disco** | SÍ: `pack_trunk.py` empaqueta corridas por capa; expertos contiguos; orden de offsets en el prefetch batch | **Adoptar**: el GGUF ya es contiguo por tensor; para MoE, ordenar lecturas por offset dentro del batch | Ordenar accesos por offset físico es la optimización más barata que existe para NVMe/SSD. |
| **Page-cache awareness** | SÍ (medido): bajo cap de 32 GB el camino buffered colapsa (1.247 vs 6.400 MB/s) porque el page cache copia dos veces y expulsa lo útil; O_DIRECT lo recupera | **Adoptar como política**: modo `Buffered` (madvise/evict) vs `Direct` (O_DIRECT/NO_BUFFERING) seleccionable por plataforma y por clase de pesos | Los pesos leídos UNA vez (expertos fríos, capas streameadas) son basura para el page cache; los reutilizados (embeddings, capas pineadas) son su mejor uso. Ver §6. |
| **Direct I/O** | SÍ: O_DIRECT con ventana ensanchada a 4 KB, fallback buffered, 2 MB hugepages | **Adoptar en Linux/WSL2; emular en Windows** (FILE_FLAG_NO_BUFFERING, buffers alineados a sector) | En Windows el equivalente existe pero con alineación por sector y sin hugepages; el trait de storage abstrae la diferencia. |
| **Kernel page eviction control** | Parcial (madvise HUGEPAGE; nada más) | **Adoptar**: madvise(MADV_DONTNEED)/posix_madvise sobre slots consumidos; en Windows, VirtualUnlock/MEM_RESET | Evita que el OS conserve en RAM bytes ya consumidos y deja el presupuesto al runtime. |
| **Doble buffering de cómputo** | No explícito (el overlap es lector-vs-cómputo) | **Adoptar**: el ring de 2 slots ES el doble buffering de lectura; no hace falta más | El cómputo de capa L se solapa con la lectura de L+1; medido 1,70×. |
| **Quantización de pesos residentes** | NO (decisión documentada: int4 post-hoc = 17,4% error en tensores sensibles de K3) | **Adoptar la clasificación, no la decisión**: el GGUF ya trae quants; el runtime respeta la elección del usuario por clase de pesos, y el MEMORY-DESIGN documenta la sensibilidad como asunto del modelo, no del motor | En el ecosistema GGUF, Q4_K_M es el estándar de CPU; el motor no re-quantiza, consume lo empaquetado (principio "nunca desquantizar"). |

---

## 3. Clases de residencia

Todo peso del modelo pertenece a exactamente una clase, decidida por el adapter de arquitectura (no por el usuario):

| Clase | Ejemplos | Política por defecto | Notas |
|---|---|---|---|
| **Always-resident** | Config, tokenizer, embeddings, lm_head, normas y biases de tamaño menor a un umbral | Cargar al inicio, nunca evictar | En K3: 4,70 GB + widen 2,43 GB. El umbral de "chico" es configurable. |
| **Trunk (dense, por capa)** | Proyecciones de atención/FFN de capas densas, router, shared experts | Pin prefix + ring de 2 slots, prefetch L+1 | Orden de acceso fijo → hit rate determinístico = K/N. |
| **Routed experts (MoE)** | Expertos de capas MoE | Cache LRU con prefetch batch; presupuesto separado | Acceso dependiente de datos; retención depende del hot set del modelo. |
| **Activaciones / KV cache / scratch** | Buffers por token, KV | Arena propia con contabilidad estricta | Nunca compite con pesos por el mismo presupuesto. |

La contabilidad (plan de memoria) suma las cuatro clases ANTES de asignar y se niega con números si el total > MemAvailable×0,95 (el factor 0,95 y la lectura de MemAvailable son herencia directa de K3; en Windows: `GlobalMemoryStatusEx.ullAvailPhys`).

---

## 4. El MemoryManager

Crate `unltd-memory`. API conceptual:

```rust
trait MemoryPolicy {
    fn plan(&self, cfg: &ModelCfg, budget: &Budget) -> Result<MemoryPlan, Refusal>;
    fn allocate(&self, plan: &MemoryPlan) -> Result<Arenas, Refusal>;
}

struct Budget {
    total: u64,              // presupuesto total del motor (no del proceso)
    trunk: Option<u64>,      // None = full-resident / mmap
    expert_cache: Option<u64>,
    kv_limit: Option<u64>,   // techo duro de KV cache (tokens máximos)
}
```

Reglas:

1. **Un solo presupuesto, reparto declarado.** `--budget auto` reparte con la heurística de K3: reserva fija + embeddings + estado, luego tronco primero, cache de expertos con lo que sobre; con techo de RSS para evitar el hazard de pinning parcial (medido en K3: 51/109 GB pineados fue 14% más lento que 0).
2. **El plan es un pronóstico; el PEAK RSS es el veredicto.** El motor imprime ambos y declara cuál citar (getrusage en Linux/macOS; en Windows `GetProcessMemoryInfo.PeakWorkingSetSize`).
3. **Refusal con números.** "esto necesita X, la máquina tiene Y, faltan Z. Opciones: menos cache, menos capas pineadas, menos contexto." Nunca un OOM-kill a mitad de corrida.
4. **La KV cache se chequea antes que nada** (es el único término que crece con el contexto): bytes/posición × posiciones vs disponible.

---

## 5. Presupuesto para 16 GB (el caso del usuario)

Cálculo de referencia (a ajustar con el modelo concreto; el motor lo hace solo):

```
16,0 GB RAM total
 − 1,5 GB OS + shell + fondo (medible con k3-doctor-style probe)
 = ~14,5 GB disponibles para el motor
 − 0,5 GB tokenizer + config + index de tensores
 − 0,3 GB scratch/activaciones de un token
 = ~13,7 GB para pesos + KV
```

Reparto típico por régimen:

| Régimen | Trunk | Expert cache | KV (1024 tokens) | Cabe |
|---|---|---|---|---|
| Modelo denso 4-8B Q4_K_M (2,5-5 GB) | full-resident | — | 0,5-1 GB | Sí, holgado |
| Modelo denso 14B Q4_K_M (~9 GB) | full-resident ajustado | — | 0,5-1 GB | Sí, justo |
| MoE 30B-A3B Q4_K_M (~17 GB) | ring 2-4 GB + pin parcial | 2-4 GB | 0,5-1 GB | Sí, stream-eado; es el régimen estrella |
| Denso 32-70B Q4 (18-40 GB) | ring de capas, 0 pin | — | 0,5-1 GB | Corre, lento; clase C/D |
| MoE grande (>100 GB GGUF) | ring | LRU con trace | 0,5-1 GB | Corre; la retención manda el trace |

Regla de oro heredada: **en un MoE, llenar el tronco (ring + pin) antes de darle un solo GB a la cache de expertos** — salvo que el trace del modelo demuestre un hot set. El runtime nuevo graba el trace desde el día 1 (es barato: 8 bytes por request) y el replay offline (`sim_cache` reimplementado en Rust) decide la política por modelo.

---

## 6. Matriz de plataformas (storage)

| | Linux / WSL2 | Windows nativo |
|---|---|---|
| Lecturas posicionadas | `pread` (std::os::unix) | `ReadFile` con OVERLAPPED / `FileExt::seek_read` (síncrono) |
| mmap | `memmap2` | `memmap2` (funciona; más lento en page-faulting, y los archivos abiertos no se pueden truncar — irrelevante para lectura) |
| Direct I/O | O_DIRECT, ventana 4 KB, hugepages 2 MB | `FILE_FLAG_NO_BUFFERING`, buffers y offsets alineados a tamaño de sector (consultar `GetDiskFreeSpace`/`IOCTL_STORAGE_QUERY_PROPERTY`); sin hugepages |
| Evictar página | `madvise(MADV_DONTNEED)` | `VirtualUnlock` + `MEM_RESET` sobre la vista de mapa |
| Prefetch hint | `posix_fadvise(WILLNEED)` | `PrefetchVirtualMemory` / FILE_FLAG_SEQUENTIAL_SCAN |
| Memoria disponible | `/proc/meminfo MemAvailable` | `GlobalMemoryStatusEx` |
| Peak RSS | `getrusage` | `GetProcessMemoryInfo` |
| Cgroups (escalera de medición) | systemd-run MemoryMax+MemorySwapMax=0 | Job Objects (con `JOB_OBJECT_LIMIT_PROCESS_MEMORY`); si no, la escalera se corre en WSL2 |

Diseño: trait `DiskReader` con dos impls (`BufferedReader`, `DirectReader`) y un factory que detecta la plataforma; el modo Direct es **fallback a Buffered** ante cualquier incompatibilidad (igual que K3). WSL2 es el camino recomendado para correr experimentos de medición (cgroups), y el binario nativo Windows es el camino para uso diario.

Evidencia verificada del ecosistema (llama.cpp, 2026): Windows nativo con mmap ≈ Linux para MoE grandes (397B: 140/16 PP/TG vs 150/15.2), pero WSL2 se degrada para MoE grandes (~25% de pérdida en PP + ~15% de overhead I/O del filesystem 9P), y Windows con mmap pierde ~70% contra Linux en modelos clase 671B. Consecuencia práctica: **el régimen de streaming extremo (Stage 5) se mide en WSL2/Linux** (cgroups + O_DIRECT), mientras el binario nativo Windows es el objetivo de uso diario con mmap — la comparación A/B entre modos de lectura es parte del protocolo de benchmarks.

---

## 7. La escalera de medición del runtime nuevo

Herencia directa de `benchmarks/memory-ladder.sh`, adaptada:

1. Un solo binario, presupuesto por flags (`--memory-gb X --trunk-gb Y --cache-gb Z` o preset).
2. Escalones: 4, 6, 8, 12, 16 GB (+ el del usuario: 16) — siempre bajo techo real (cgroups / Job Objects, swap=0).
3. Por escalón: 3 repeticiones mínimo; se reporta media, sd y spread; se declara el piso de ruido de LA máquina del usuario (no heredar el 33%: medirlo).
4. La aserción central: **tokens idénticos en todos los escalones** (y, más fuerte, logits idénticos bit a bit en el primer paso con el backend de referencia).
5. Se reportan contadores (GB leídos por clase, requests/evictions, hit rates verdaderos) que no necesitan replicación.
6. `machine.txt` con CPU, RAM, disco, ancho de banda medido (probe estilo `k3-doctor.sh`), versión del binario, ids del prompt.

---

## 8. Qué NO hacemos (explícito)

- **Swap.** El motor no asume swap; la escalera corre con swap=0. Si el usuario tiene swap, el plan de memoria lo ignora (el presupuesto es RAM física).
- **Compresión transparente.** Los pesos se leen en su formato empaquetado (GGUF); comprimir para disco es asunto del ecosistema, no del motor.
- **Checkpointing de pesos a mitad de corrida.** El estado que se persiste es el de la conversación (KV + posición), nunca pesos.
- **Memoria compartida entre procesos.** Un motor por proceso; la cache de expertos no se comparte (nota de K3: la cache/trunk/safetensors index no son thread-safe; un runtime de investigación no necesita multi-proceso).
