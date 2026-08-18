# benchmarks/ — disciplina de medición

La medición hereda el protocolo de `kimi-k3-in-c/benchmarks/` (ver
`docs/AUDIT.md` §3.3 "disciplina de medición" y `docs/PERFORMANCE.md` del
proyecto de referencia). La lección central: en workloads de streaming el
**ruido de fondo es enorme** — en K3, la misma corrida variaba 14.78/14.67/20.14
s/token sin cambio alguno (piso de ruido del 33%). Cualquier número sin el
protocolo de abajo es anecdótico.

## Protocolo obligatorio para TODA medición publicable

1. **Machine file**: cada corrida escribe `machine.txt` (CPU exacta, modelo de
   RAM y canales, SSD modelo + firmware, kernel/OS, versión del binario, commit).
   Sin machine.txt el número no se puede comparar ni con la semana pasada.
2. **3 repeticiones mínimo**, misma corrida, y se reporta la MEDIANA con el
   rango. Una sola corrida no es un dato.
3. **Salida idéntica entre repeticiones**: aserción de token ids byte a byte.
   Una repetición que produjo otra salida no es una repetición: es un bug.
4. **Presupuesto forzado por el kernel**, no por el programa: cgroup
   `MemoryMax` + `MemorySwapMax=0` en Linux (Job Objects en Windows), para que
   el proceso NO pueda exceder el presupuesto ni siquiera por bug.
5. **Cache de disco fría o caliente declarada**: cada corrida dice si el page
   cache estaba poblado (repetir con `sync` + drop de caches donde se pueda).
6. **Métricas por token**: `s/token`, `MB leídos/token` (I/O share), hit rate
   VERDADERO del expert cache, PEAK RSS medido con `getrusage` (no el plan).

## La escalera de memoria (por venir, Stage 4+)

```
benchmarks/
├── memory-ladder.sh   # 8/12/16/24/32 GB → misma salida, curva s/token
├── noise-floor.sh     # N corridas idénticas → desvío del piso de ruido
└── machine.txt        # ejemplo del formato
```

El propósito no es "cuántos t/s": es la CURVA de degradación con el presupuesto
(y en qué punto la salida sigue siendo idéntica). Eso es lo que no existe en
ningún otro runtime (ver `docs/MODEL-SUPPORT.md` §comparación con llama.cpp).
