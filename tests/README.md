# tests/ — fixtures de integración y la escalera de pruebas

La estrategia de testing está definida en `docs/ARCHITECTURE.md` §8 y hereda la
filosofía de `kimi-k3-in-c/tests/fixtures/README.md`: fixtures ADVERSARIOS, no
"caminos felices". Un fixture adversario es uno que elige exactamente los inputs
que rompen la suposición ingenua (rank chico con poco espaciado, caps de escala
que se activan, routing que reordena el top-k, dimensión no divisible por el
ancho SIMD).

## Organización

```
tests/
├── fixtures/        # checkpoints sintéticos (generados por tools/make_tiny.py)
├── oracle/          # oráculo de referencia (torch, en Python; solo Stage 0/1)
├── reference/       # salidas de referencia capturadas (token ids, logits)
└── README.md        # este archivo
```

Cada crate además lleva sus tests unitarios en `src` (o `tests/` propio). El
comando de integración es `cargo test --workspace`.

## La escalera (resumen; detalle en ROADMAP.md)

| Stage | Modelo | Qué se verifica |
|---|---|---|
| 0 | sintético 1 capa | kernels vs naive, refusal de config, budget plan |
| 1 | 0.5–1.5B real | forward completo vs oráculo torch (max diff < 1e-5) |
| 2 | 3–4B | GQA, RoPE, rendimiento razonable |
| 3 | 7–8B | primer modelo que no cabe entero en arena de tests |
| 4 | MoE chico | router, expert cache, trace/replay |
| 5 | modelo > RAM | streaming extremo bajo presupuesto artificial |

## Contratos que los tests HACEN valer (no negocian)

1. **Determinismo bit a bit entre presupuestos de memoria**: misma corrida con
   8 GB y con 32 GB de presupuesto produce los mismos token ids. Si un test de
   streaming pasa sin esta aserción, no prueba nada.
2. **Backends bit-idénticos**: scalar vs AVX2, mismo resultado byte a byte.
   El backend scalar es la referencia; el AVX2 tiene tests dedicados por kernel.
3. **`--no-alloc`**: el plan de memoria se puede computar sin asignar nada.
   Los tests de refusal (no cabe, config inválida) corren sin tocar pesos.
4. **Incremental ≡ full-recompute**: el decode con KV cache debe dar los
   mismos logits que recomputar con atención completa (diferencia < 1e-5).
5. **Hit rate verdadero**: los tests del ExpertCache asertan sobre
   `(hits - prefetch_reads) / requests`, nunca sobre `hits / requests`.
6. **Experto caído = corrida inválida**: un fallo de lectura simulado debe
   producir exit code distinto + "RUN INVALID", nunca tokens plausibles.
7. **Campo ausente = refusal**: cada test de loader incluye el caso de quitar
   una clave del config y verificar el error que LISTA TODAS las ausentes.
