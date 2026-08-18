# Inventario REAL de modelos locales — D:\AI\models

Generado 2026-08-18 por inspección directa del filesystem (nada teórico: cada fila fue verificada con `Get-ChildItem` y, donde se cita, leyendo el `config.json` real).

## 1. Tabla maestra

| Modelo | Ruta | Formato | Arquitectura | Cuantización | Tamaño | Estado | Stage | Notas |
|---|---|---|---|---|---|---|---|---|
| **Ornith-1.0-9B (GGUF)** | `D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf` | GGUF v3 | **Qwen3.5-9B híbrido** (`qwen3_5_text`: 32 capas, patrón 3×linear+1×full, GatedDeltaNet, M-RoPE interleaved, partial rotary 0.25, attn_output_gate, vocab 248,320) | Q4_K_M | 5.37 GB | ✅ completo (un archivo + start-server.bat) | **1º modelo end-to-end** | Único GGUF de texto utilizable. Corrió en esta máquina con llama.cpp CUDA el 12/08 (logs de kilo-local). |
| Ornith-1.0-9B (safetensors) | `D:\AI\models\Ornith-1.0-9B` | safetensors BF16, 4 shards (~4.7 GB c/u) + config.json + tokenizer.json (19.1 MB) + vocab.json (6.4 MB) | igual que arriba + `vision_config` (tower 27 capas) | BF16 | 17.9 GB | ✅ completo | futuro (referencia bf16) | Útil para verificación peso a peso contra el GGUF; NO corre en 16 GB sin streaming extremo. |
| Gemma-4-31B-it | `D:\AI\models\gemma-4-31B-it` | safetensors BF16, 2 shards (47.5 + 12.2 GB) + config + tokenizer.json (30.7 MB) | `gemma4_text`: 60 capas, patrón 5×sliding+1×full, GeGLU-tanh, RoPE dual (θ=10k sliding / θ=1M full proportional), softcap 30, vocab 262,144, tie | BF16 | 59.7 GB | ✅ completo | futuro lejano | Sin GGUF local. Convertir exige ~62 GB de RAM (no en esta máquina). Ni siquiera streameado en BF16 es razonable ahora. |
| Bonsai-27B (1-bit) | `D:\AI\models\Bonsai-demo\models\gguf\27B\Bonsai-27B-Q1_0.gguf` | GGUF | ternaria/1-bit PrismML (fork `prism-b9596` de llama.cpp) | Q1_0 | 3.63 GB | ✅ presente | experimental | 27B params en 3.6 GB: fascinante para tests de RAM, PERO arquitectura no-estándar y requiere el fork CUDA — no es camino v1. |
| Bonsai-27B-dspark | `...\Bonsai-27B-dspark-Q4_1.gguf` | GGUF | ídem (empaque ternario ~1.7 bpw) | Q4_1 "dspark" | 1.70 GB | ✅ presente | experimental | Ídem anterior; el nombre Q4_1 es engañoso (1.7 GB / 27B ≈ 0.5 B/param — no es Q4_1 estándar). |
| Bonsai mmproj | `...\Bonsai-27B-mmproj-{BF16,Q8_0}.gguf` | GGUF | projector de visión | BF16 / Q8_0 | 888 / 600 MB | ✅ presente | no aplica | No son modelos de texto. |

**No existe localmente**: Qwen3-0.6B, SmolLM, Llama, Phi, Mistral, OLMoE, DeepSeek — el único camino GGUF de texto local es Ornith.

## 2. Máquina real

- **CPU**: Intel i7-8700 @ 3.20 GHz (6C/12T, AVX2/FMA/BMI2, sin AVX-512)
- **RAM**: 15.9 GB
- **GPU**: NVIDIA GTX 1060 6 GB (presente; NO es requisito — el runtime es CPU-first, pero explica el stack CUDA local)
- **Disco D:**: 0.21 TB libres de 0.91 TB
- **OS**: Windows 11; WSL2 disponible pero SIN distros instaladas

## 3. Toolchain real (verificado por comando)

| Herramienta | Versión | Ruta |
|---|---|---|
| rustc / cargo / rustup | instalado (falta chequear versión exacta) | `%USERPROFILE%\.cargo\bin\` (NO en PATH de shells viejos — usar `$env:Path += ";$env:USERPROFILE\.cargo\bin"`) |
| git | 2.53.0.windows.1 | C:\Program Files\Git |
| cmake | 4.3.3 | C:\Program Files\CMake |
| MSVC cl.exe | 14.44.35207 (Build Tools 2022) | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` |
| msys64 gcc | presente | C:\msys64\mingw64\bin\gcc.exe |
| python | 3.13.12 | %LOCALAPPDATA%\Programs\Python\Python313 |
| clang | ausente | — |

## 4. El oráculo: llama.cpp local

- **Binarios listos**: `D:\AI\models\Bonsai-demo\bin\cuda\` — fork **Prism** de llama.cpp (`prism-b9596-9fcaed7`, archivo `.llama_release`), compilado con CUDA 12 (ggml-cuda.dll 556 MB, cublasLt). Incluye `llama-cli.exe`, `llama-server.exe`, `llama-bench.exe`, `llama-gguf.exe` + DLLs de runtime en el mismo dir (requiere ese dir en PATH).
- **Evidencia de uso real**: `D:\AI\launchers\kilo-local\logs\llama-20260812-*.stderr.log` — llama-server corrió ornith Q4_K_M (n_ctx 4096, n_threads 6, CUDA0 GTX 1060, "model loaded" en ~10.3 s, chat template con thinking). Launcher: `kilo-local.ps1` → llama-server en puerto 8080, OpenAI-compatible.
- **Fuente mainline**: `D:\AI\runtimes\llama.cpp` (commit fdb1db8, 2026-07-02, tag b9860) con `build-msvc/` y `build-cuda/` parciales; los .exe esperados en `.cpp\build-msvc\bin\Release\` NO estaban presentes al momento de la auditoría (el primer listado recursivo los mostró, `Test-Path` directo falló — asumir build incompleto; el fork Prism bin es el oráculo operativo).

## 5. Decisión de primer modelo end-to-end (Fase 5)

**Elegido: `ornith-1.0-9b-Q4_K_M.gguf`** (Ornith 1.0 9B de DeepReinforce-AI — un Qwen3.5-9B).

Criterios del usuario y veredicto:

1. *Arquitectura simple* — ❌ NO. Es Qwen3.5 híbrido (24 capas GatedDeltaNet + 8 full attention + MTP). Es el punto en contra real.
2. *Tamaño pequeño* — ✅ 5.37 GB (Q4_K_M); cabe residente en 16 GB con KV (n_ctx 4096 como lo usa kilo) y streameable con presupuesto forzado.
3. *GGUF válido* — ✅ un archivo limpio.
4. *Tokenizer disponible* — ✅ tokenizer.json (19.1 MB) y vocab.json locales + todo dentro del GGUF.
5. *Soportado por llama.cpp* — ✅ probado EN ESTA MÁQUINA (logs del 12/08).
6. *Razonable para 16 GB* — ✅ ~5.4 GB de pesos + KV Q4 + overhead < 8 GB.

**Por qué a pesar del criterio 1**: es el ÚNICO GGUF de texto local con oráculo funcionando. Las alternativas locales son peores para empezar (Gemma-4-31B: 60 GB BF16 sin GGUF; Bonsai: fork exótico). El plan de fases absorbe la complejidad: las fases 2–4 (GGUF, kernels, quants) son agnósticas de arquitectura; la Fase 5 implementa el forward de qwen3_5_text por piezas con tests sintéticos (primero las 8 capas full-attention del fixture sintético, después GatedDeltaNet validado contra la fórmula y contra la fuente de llama.cpp local), y el MTP se omite para decode normal (no afecta los logits del modelo principal — solo especulación). **Plan de contingencia documentado**: si GatedDeltaNet bloquea la Fase 5 más allá de lo razonable, se descarga Qwen3-0.6B (~0.5 GB — no es "gigante", y cumple el criterio 1 del usuario) como camino simple de desbloqueo, y Ornith pasa a ser el objetivo de Fase 13. Decisión de contingencia explícita, no silenciosa.

## 6. Consumos para el roadmap (qué fase usa qué)

| Fase | Modelo local |
|---|---|
| 2 (GGUF inspect) | los 4 GGUFs reales (ornith Q4_K_M, Bonsai Q1_0/dspark/mmproj) — el inspector es agnóstico |
| 4 (cuantización) | Q4_K_M (ornith) + F32/F16/Q8_0 (KV y activaciones). Q1_0 de Bonsai: solo decodificar metadata, no kernels |
| 5–8 (end-to-end) | ornith Q4_K_M, oráculo = llama-cli del fork Prism con `-ngl 0` (CPU justo) |
| 9–10 (budget/streaming) | ornith con `--memory-budget 4G` forzado (pesos 5.37 GB > presupuesto → streaming real sin necesitar modelo gigante) |
| 13 (escalera) | ornith 5.4 GB + (opcional futuro) gemma-4-31B y Bonsai como tests de RAM |
| 14–16 (MoE) | NINGUNO local → tests sintéticos + diseño (documentado); no descargar MoE grande sin decisión explícita |
| 18 (bench vs llama.cpp) | ornith en llama-cli CPU (`-ngl 0 -t 6`) vs unltd; los logs de kilo dan el contexto GPU |
