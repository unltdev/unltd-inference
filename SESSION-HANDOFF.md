# HANDOFF ÔÇö unltd-inference (nueva sesi├│n)

## 1. Objetivo del proyecto

Runtime de inferencia en **Rust puro**, CPU-first, disk-first, con presupuesto expl├¡cito de RAM, para correr LLMs cuyo tama├▒o excede la RAM disponible (`model_size > available_RAM` como caracter├¡stica central, no caso excepcional). Referencia t├®cnica: auditor├¡a de `kimi-k3-in-c`. **No** es wrapper de llama.cpp: llama.cpp es solo or├ículo de validaci├│n, formato de entrada (GGUF) y benchmark.

**Constraints del usuario (verbatim, innegociables):** no borrar/mover modelos en `D:\AI\models`; no descargar modelos gigantes; no APIs pagas; no cloud; no wrapper de llama.cpp; correctness > performance; measurements > assumptions; una arquitectura bien implementada antes que 50 a medias.

**M├íquina real:** Windows 11 Pro, i7-8700 (6C/12T, AVX2), 15.9 GB RAM, GTX 1060 6GB (opcional, no requisito), D: 0.21/0.91 TB libres, WSL2 habilitado sin distros, modelos en `D:\AI\models`.

**Definici├│n de ├®xito:** `unltd-inference inspect <gguf>` + `unltd-inference tokenize --model ... --text "Hola"` + `unltd-inference run --model ... --prompt "Hola" --max-tokens 20 --temperature 0` generando texto real con NUESTRO runtime.

**Gran checkpoint (no avanzar a optimizaciones sin esto):** token de UNLTD == token de llama.cpp con el mismo modelo, mismo prompt, greedy, temperatura 0 ÔÇö o explicaci├│n matem├ítica precisa del mismatch.

## 2. Arquitectura decidida (docs ya escritos)

Workspace de 8 crates en `D:\AI\projects\unltd-inference`:

```
unltd-tensor (kernels, DType) ÔåÉ unltd-core (errores, refusal)
  ÔåÉ {unltd-model-loader, unltd-architectures, unltd-memory, unltd-tokenizer}
  ÔåÉ unltd-generation ÔåÉ unltd-cli
```

- **IR propia** (`unltd-architectures`): `AttnKind::{Mha,Gqa,MlaDeepSeek,MlaK3NoPe}`, `RoPeKind`, `FfnKind::{SwiGlu,GeGluTanh,Relu2,Moe}`, `NormKind`, `LayerSpec`, `ModelSpec`.
- **Refuse-rather-than-guess**: campo ausente = error, nunca default; prohibido `#[serde(default)]`. El refusal lista TODAS las claves faltantes.
- **Contrato num├®rico**: acumuladores f64, orden de reducci├│n fijo, scalar = referencia, AVX2 bit-id├®ntico.
- **Nunca desquantizar**: multiplicar nibbles empaquetados directo; cachear en formato empaquetado.
- **pread no mmap** como camino investigable (RSS medible), mmap como modo alternativo para comparaci├│n A/B (llama.cpp es mmap-first ÔÇö se respeta su evidencia).
- **Streaming**: pin prefix + ring de 2 slots (publicaci├│n expl├¡cita post-├®xito, el invariante anti-corrupci├│n de K3), prefetch L+1.
- **MoE**: expert LRU + prefetch batch 3 fases (reserva serial ÔåÆ lecturas paralelas ordenadas por offset ÔåÆ publicar solo lo llegado); hit rate verdadero `(hits ÔêÆ prefetch_reads)/requests`; trace 8 B/request + replay offline.
- **Determinismo entre presupuestos**: misma salida en 4 GB y en 32 GB (firma del proyecto).
- **GGUF primario** (parsers hand-written, sin serde para parsear), safetensors secundario. tokio rechazado; rayon solo sobre filas de salida independientes.
- Deps de workspace ya fijadas: half 2, bytemuck 1, serde 1, serde_json 1, thiserror 2, anyhow 1, clap 4, memmap2 0.9, rayon 1. Release: lto thin + codegen-units 1. `rust-toolchain.toml`: stable.

Docs completos (Fase de investigaci├│n terminada): `docs/AUDIT.md`, `MODEL-CANDIDATES.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `MEMORY-DESIGN.md`, `MODEL-SUPPORT.md`, `LOCAL-MODEL-INVENTORY.md`.

## 3. Fases completadas

| # | Fase | Estado |
|---|---|---|
| 5 | Fase 0: Auditor├¡a real (m├íquina, toolchain, D:\AI\models) | Ô£à **completada** ÔÇö entregable `docs/LOCAL-MODEL-INVENTORY.md` escrito |
| 6 | Fase 1: Bootstrap Rust + workspace verde | Ô£à **completada** ÔÇö cargo check y test verdes |
| 7 | Fase 2: GGUF reader + CLI `inspect` | ­ƒöä **en curso ÔÇö C├ôDIGO SIN EMPEZAR** |
| 8 | Fase 3: Tensor core scalar + tests | pendiente |
| 9 | Fase 4: Cuantizaci├│n (seg├║n quants reales de ornith: Q4_K_M) | pendiente |
| 10 | Fase 5ÔÇô8: End-to-end ornith (token == llama.cpp) | pendiente |
| 11 | Fase 9ÔÇô10: Memory budget + streaming de capas | pendiente |
| 12 | Fase 17: Refusal / safe config | pendiente |

## 4. Modelos analizados (inventario REAL de D:\AI\models)

| Modelo | Ruta | Formato | Tama├▒o | Veredicto |
|---|---|---|---|---|
| **Ornith-1.0-9B GGUF** | `D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf` | GGUF v3, Q4_K_M | 5.37 GB | **MODELO #1 end-to-end** (decisi├│n justificada abajo) |
| Ornith-1.0-9B safetensors | `D:\AI\models\Ornith-1.0-9B` | BF16, 4 shards ~4.7 GB c/u + config.json + tokenizer.json (19.1 MB) + vocab.json (6.4 MB) | 17.9 GB | referencia bf16 futura (peso a peso vs GGUF) |
| Gemma-4-31B-it | `D:\AI\models\gemma-4-31B-it` | BF16, 2 shards 47.5+12.2 GB | 59.7 GB | futuro lejano; sin GGUF local |
| Bonsai-27B | `D:\AI\models\Bonsai-demo\models\gguf\27B\` | Q1_0 (3.63 GB), dspark-Q4_1 (1.70 GB), mmproj BF16/Q8_0 | ÔÇö | experimental (ternaria PrismML, fork ex├│tico) |

**Detalles de ornith (del config.json real)**: `Qwen3_5ForConditionalGeneration` ÔÇö 32 capas patr├│n 3├ùlinear_attention + 1├ùfull_attention (`full_attention_interval: 4`); GatedDeltaNet (conv kernel 4, key heads 16├ù128, value heads 32├ù128, mamba_ssm_dtype f32); full attention: 16 Q / 4 KV heads, head_dim 256, partial_rotary_factor 0.25, M-RoPE interleaved (mrope_section [11,11,10]), rope_theta 1e7, attn_output_gate: true; SwiGLU 4096ÔåÆ12288; RMSNorm 1e-6; vocab 248,320 (Qwen2Tokenizer tiktoken-BPE); eos 248044/248046; max_pos 262144; untied; MTP 1 capa (**omisible para decode base** ÔÇö no afecta logits del modelo principal); vision tower 27 capas (no necesaria para texto). `use_cache: false`.

**Justificaci├│n de ornith como modelo #1** (a pesar de ser la arquitectura M├üS dif├¡cil de las 3 locales ÔÇö h├¡brido DeltaNet): es el **├║nico GGUF de texto local con or├ículo funcionando** (probado en esta m├íquina el 2026-08-12, logs de kilo-local). Las fases 2ÔÇô4 (GGUF, kernels, quants) son agn├│sticas de arquitectura. **Contingencia documentada**: si GatedDeltaNet bloquea la Fase 5, descargar Qwen3-0.6B (~0.5 GB, permitido ÔÇö no es gigante) y ornith pasa a Fase 13. Decisi├│n expl├¡cita, no silenciosa.

## 5. Toolchain verificado (resultados reales de comandos)

- **Rust 1.97.1** (rustc 8bab26f4f 2026-07-14), cargo 1.97.1, toolchain `stable-x86_64-pc-windows-msvc` (activa, default) ÔÇö en `C:\Users\gpsan\.cargo\bin\` (cargo.exe, rustc.exe, rustup.exe, rustfmt, clippy, rust-analyzer presentes). Ya est├í en el PATH de usuario del registro, PERO los shells del harness arrancaron antes ÔåÆ **usar ruta completa**: `& "$env:USERPROFILE\.cargo\bin\cargo.exe" ...`.
- git 2.53, cmake 4.3.3, MSVC Build Tools 14.44.35207, msys64 mingw64, python 3.13.12, clang ausente, WSL2 sin distros.
- **Or├ículo**: fork **Prism** de llama.cpp (`prism-b9596-9fcaed7`), build CUDA, en `D:\AI\models\Bonsai-demo\bin\cuda\` (llama-cli/server/bench/gguf + ggml-cuda.dll 556 MB + DLLs runtime en el mismo dir ÔåÆ ese dir debe ir al PATH al ejecutar). Evidencia de uso real: `D:\AI\launchers\kilo-local\logs\llama-20260812-*.stderr.log` (ornith carg├│ en ~10.3 s, CUDA0 GTX 1060, n_ctx 4096). Launcher kilo: llama-server puerto 8080. Fuente mainline en `D:\AI\runtimes\llama.cpp` (fdb1db8, 2026-07-02; builds msvc/cuda incompletos ÔÇö no confiar en sus .exe).

## 6. Comandos ejecutados y resultados (sesi├│n actual)

- `cargo check --workspace` ÔåÆ **VERDE a la primera** (20.72 s; baj├│ deps: clap 4.6.6, serde 1.0.229, thiserror 2.0.20, rayon 1.12.0, memmap2 0.9.11, half 2.7.1, bytemuck 1.25.2, anyhow 1.0.104, zerocopy 0.8.56ÔÇª). Los 8 crates compilan con los esqueletos `todo!()`.
- `cargo test --workspace` ÔåÆ **VERDE** con **0 tests en todos los crates** (esqueleto puro).
- Rust verificado: rustc/cargo 1.97.1; user PATH contiene `C:\Users\gpsan\.cargo\bin` Ô£ô.
- Or├ículo llama-cli (ornith Q4_K_M, `-ngl 0 -t 6 -c 512 --temp 0`, prompt "The capital of France is", -n 20) ÔåÆ **intentado 3 veces, SIEMPRE matado** (killed por crash de sesi├│n). **PENDIENTE: correrlo en FOREGROUND con timeout 600000 ms.**

## 7. Errores / bloqueos encontrados

1. **`cargo` no reconocido** en shells del harness ÔåÆ usar ruta completa `$env:USERPROFILE\.cargo\bin\cargo.exe` (el binario existe; es artefacto de PATH heredado del proceso padre, no de instalaci├│n).
2. **Tasks de background se matan al instante** (`[killed]`, sin output) ÔåÆ Claude Code se cierra solo (reportado por el usuario). **Regla de supervivencia: todo en foreground; checkpoint git tras cada hito.**
3. **git NUNCA inicializado**: el repositorio todav├¡a NO es git (`git init` pendiente; `Glob` confirm├│ que no existe `.git/HEAD`). Los intentos murieron con las sesiones.
4. **Permisos**: comandos compuestos `Set-Location ...; git ...` y hasta `git -C ...` simple ahora piden aprobaci├│n (guard anti "bare repository attacks"). `Set-Location` + cargo s├¡ pas├│. Si git se bloquea, pedir aprobaci├│n expl├¡cita o usar el Bash tool (Git Bash).
5. Sesi├│n inestable (crashs) ÔåÆ mantener este HANDOFF al d├¡a tras cada hito.

## 8. Estado actual EXACTO

- Workspace: check verde + test verde, **0 tests reales**, todos los kernels/readers a├║n `todo!()`.
- Fase 2 (GGUF reader + `inspect`): **sin una l├¡nea de c├│digo todav├¡a** ÔÇö es el punto exacto de continuaci├│n.
- Sin repo git, sin or├ículo capturado, sin implementaci├│n de kernels.
- Tasks: #7 in_progress; #8ÔÇô#12 pendientes.
- Transcript completo de la sesi├│n pre-compactaci├│n: `C:\Users\gpsan\.claude\projects\D--AI-projects-kimi-k3-in-c\e4f5f3b5-2de3-497a-97b6-00b3a4fd54b7.jsonl`.
- ÔÜá´©Å El working directory del harness es `D:\AI\projects\kimi-k3-in-c`; el proyecto vive en `D:\AI\projects\unltd-inference`.

## 9. Pr├│ximos pasos (orden exacto)

1. **Guardar este HANDOFF en `docs/HANDOFF.md`** (primer acto de la nueva sesi├│n).
2. **git init + primer commit** (checkpoint: workspace verde + docs). Mensaje con `Co-Authored-By: Claude <noreply@anthropic.com>`.
3. **Capturar el or├ículo en FOREGROUND** (timeout 600 s):
   ```powershell
   $env:Path = "D:\AI\models\Bonsai-demo\bin\cuda;$env:Path"
   & "D:\AI\models\Bonsai-demo\bin\cuda\llama-cli.exe" -m "D:\AI\models\Ornith-1.0-9B-GGUF\ornith-1.0-9b-Q4_K_M.gguf" -p "The capital of France is" -n 20 -t 6 -ngl 0 -c 512 --temp 0 2>&1 | Out-File -Append "D:\AI\projects\unltd-inference\benchmarks\reference\ornith-llama-cpu-greedy.txt"
   ```
   (guardar tambi├®n `llama-cli --version` al inicio del archivo). Verificar que salga exit (no interactive).
4. **Fase 2 ÔÇö GGUF reader** (`crates/unltd-model-loader/src/gguf.rs`, parser hand-written std-only):
   - Magic `"GGUF"`, version u32 (aceptar 2 y 3, rechazar resto), n_tensors/n_kv u64 con sanity caps anti-OOM (n_kv < file_size/8, n_tensors < file_size/16, len de string < file_size, n_dims Ôëñ 64).
   - Metadata: key string (u64 len + bytes), value_type u32, valores: 0 UINT8, 1 INT8, 2 UINT16, 3 INT16, 4 UINT32, 5 INT32, 6 FLOAT32, 7 BOOL, 8 STRING, 9 ARRAY (elem_type + count + valores), 10 UINT64, 11 INT64, 12 FLOAT64.
   - Padding a `GGUF_ALIGNMENT=32` tras metadata y tras cada tensor info. Tensor info: name, n_dims u32, dims u64├ùn_dims, ggml_type u32, offset u64.
   - Validaci├│n por tensor: n_elements con mul checked; n_bytes v├¡a tabla de tipos conocidos (F32 4B, F16/BF16 2B, Q4_0 18B/32, Q4_1 20/32, Q5_0 22/32, Q5_1 24/32, Q8_0 34/32, Q8_1 40/32, Q2_K 84/256, Q3_K 110/256, **Q4_K 144/256**, Q5_K 176/256, Q6_K 210/256, Q8_K 292/256, IQ2_XXS 66/256, IQ2_XS 74/256, IQ3_XXS 66/256, IQ1_S 66/256, IQ4_NL 34/32, IQ3_S 110/256, IQ2_S 84/256, IQ4_XS 136/256, IQ1_M 98/256, I8/I16/I32/I64/F64 1/2/4/8/8B); ids 31/32 = TQ1_0/TQ2_0 (nombres del merge bonsai upstream ÔÇö **verificar contra `llama-gguf.exe` antes de afirmar bytes**); tipo desconocido ÔåÆ `n_bytes: None` + warning, nunca adivinar.
   - Checks duros: offset ÔëÑ header_end; offset % 32 == 0 (v3); offset+nbytes Ôëñ file_size; sin overlap entre rangos (ordenar por offset); tensores zero-size = error.
   - Extender `LoadError` en unltd-core: `BadFile(String)`, `BadMagic([u8;4])`, `UnsupportedVersion(u32)`, `TensorOutOfBounds{name,offset,nbytes,file_size}`, `TensorOverlap{a,b,a_end,b_start}`, `MisalignedTensor{name,offset,align}`, `UnknownGgmlType(u32)` (+ helpers `LoadError::corrupt/io`). Conectar con `WeightIndex`/`TensorMeta`.
5. **CLI `inspect`** (`unltd-cli/src/main.rs`, clap derive): subcomandos `inspect <gguf> [--no-tensors]` (header: version, counts, tama├▒o con `human()`, header_end; metadata completa con nombres de tipo; tabla de tensores name|dims|type|offset|bytes|%file; resumen: total bytes, tensores top, warnings de tipos desconocidos; pista de arquitectura desde `general.architecture`), `tokenize` y `run` como stubs expl├¡citos "not implemented (Fase 5)".
6. `cargo check` + `cargo test` verde ÔåÆ **commit**.
7. Ejecutar `inspect` contra los 4 GGUFs reales (ornith + Bonsai Q1_0/dspark/mmproj) y comparar contra `llama-gguf.exe` (or├ículo del parser).
8. Fase 3: kernels scalar (dot/matmul f64-acc orden fijo, RMSNorm, SiLU, RoPE, softmax, attention, embedding) + unit tests con tolerancias expl├¡citas y fixtures precomputados.
9. Fase 4: decode/dot para F32/F16/Q8_0/Q4_0/Q4_K_M (los tipos reales de ornith) con tests vs referencia.
10. Fases 5ÔÇô8: forward qwen3_5_text por piezas (primero las 8 capas full-attention en fixture sint├®tico, luego GatedDeltaNet validado contra f├│rmula + fuente local de llama.cpp; MTP omitido) ÔåÆ token == llama.cpp greedy ÔåÆ generaci├│n multi-token con m├®tricas (t/s, TTFT, RSS, bytes le├¡dos).
11. Fases 9ÔÇô10: `--memory-budget` (ej. 4G forzado con ornith 5.37 GB ÔåÆ streaming real), informe de memoria, equivalencia full-recompute vs KV. Fase 17: refusal con n├║meros.

**Referencias r├ípidas de la investigaci├│n previa** (si se necesitan): llama.cpp tiene 146 archs GGUF incl. `LLM_ARCH_KIMI_K3` y Qwen3.5 (PR #16095); MTP spec PR #25589 mergeado 2026-05-16; `--lazy-experts` PR #26003 sin mergear (medici├│n: pinning neutral <24 GB, peor arriba ÔÇö r├®gimen distinto al de K3); Q4_K_M recomendado para CPU; IQ quants m├ís lentos en CPU; Windows nativo Ôëê Linux para MoE grande v├¡a mmap; WSL2 ~25% PP loss + overhead I/O 9P; Windows ~70% m├ís lento en clase 671B; oxillama = "llama.cpp en Rust" ya existe (~165k LOC) ÔÇö nuestro diferenciador es la instrumentaci├│n de memoria con determinismo entre presupuestos, no la amplitud de arquitecturas.
