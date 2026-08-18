//! CLI de unltd. Subcomandos por fase del ROADMAP:
//! - `inspect` (Fase 2): header, metadata y tabla de tensores de un GGUF;
//! - `min-forward` (Fase 5): embedding + attn_norm + cabeza de salida, validado
//!   contra los bins del oráculo llama-eval-callback;
//! - `forward-oracle` (Fase 6): prefill de 5 tokens por 32 capas contra el oráculo;
//! - `tokenize` (Fase 7): tokenizador real del GGUF (BPE byte-level gpt2);
//! - `run` (Fase 8): generación greedy multi-token.
//!
//! Contratos de la CLI (heredados de k3_run.c, ver docs/AUDIT.md):
//! - el banner de memoria es un PLAN que se imprime ANTES de asignar; el PEAK RSS se
//!   imprime al final y es la cifra autoritativa;
//! - negarse con números ("necesita X, hay Y") antes de un OOM-kill;
//! - un experto caído = exit code distinto + "RUN INVALID";
//! - la salida de texto se imprime como bloque, no streameada (un multi-byte cortado
//!   no es UTF-8 válido).

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use unltd_core::{human, LoadError};
use unltd_generation::{GreedyLoop, LayerDump, MinForward, NodeCapture, Qwen35Forward};
use unltd_memory::{parse_size, MemoryAccounting};
use unltd_model_loader::{GgufReader, MappedWeights};
use unltd_tokenizer::{Gpt2Tokenizer, TokenError, Tokenizer};

#[derive(Parser, Debug)]
#[command(name = "unltd", version, about = "Disk-first CPU inference runtime")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Inspecciona un GGUF: header, metadata y tabla de tensores (Fase 2).
    Inspect {
        /// Archivo .gguf a inspeccionar.
        gguf: PathBuf,

        /// No imprimir la tabla de tensores (solo header, metadata y resumen).
        #[arg(long)]
        no_tensors: bool,
    },

    /// Forward mínimo Fase 5 contra el oráculo (embeddings, attn_norm, cabeza
    /// de salida). Sin `--oracle-dir` solo ejercita el forward y reporta.
    MinForward {
        /// Modelo: archivo .gguf.
        model: PathBuf,

        /// Tokens (ids) a embeder, separados por coma.
        #[arg(long)]
        tokens: String,

        /// Directorio con los bins del oráculo (ornith-*.f32.bin).
        #[arg(long)]
        oracle_dir: Option<PathBuf>,

        /// Bins del volcado legacy %12.4f: relaja las puertas al límite del
        /// redondeo de impresión (embeddings ≤ 5e-5, attn_norm ≤ 1e-4) en
        /// lugar de bit-exacto / 1e-5 (solo válido con bins %.9g).
        #[arg(long)]
        legacy_prec: bool,
    },

    /// Prefill de los 5 tokens del oráculo por las 32 capas (Fase 6):
    /// escribe las salidas por capa y por token a un binario f32 para
    /// `compare-layers-oracle.py` y, con `--oracle-dir` (bins prec9),
    /// gata result_norm y el argmax de los logits contra el oráculo.
    ForwardOracle {
        /// Modelo: archivo .gguf.
        model: PathBuf,

        /// Directorio donde escribir `engine-layers.f32.bin` + summary.
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,

        /// Directorio con los bins prec9 del oráculo (puertas de cola).
        #[arg(long)]
        oracle_dir: Option<PathBuf>,

        /// Si se pasa, escribe los nodos intermedios del camino recurrente a
        /// este binario (para `compare-nodes-oracle.py`): [u32 n_tokens] y por
        /// token [u32 n_nodes], por nodo [u32 name_len][name][u32 ne0][f32×ne0].
        #[arg(long)]
        debug_nodes: Option<PathBuf>,
    },

    /// Tokeniza texto con el tokenizador del modelo (Fase 7).
    Tokenize {
        /// Modelo: archivo .gguf.
        model: PathBuf,

        #[arg(long)]
        text: String,
    },

    /// Genera texto con el motor (Fases 8-10): greedy determinista, temperatura
    /// 0, con presupuesto de memoria controlada opcional (Fase 9) y ejecución
    /// disk-first sobre mmap (Fase 10: el modelo nunca se copia a heap).
    Run {
        /// Modelo: archivo .gguf.
        model: PathBuf,

        #[arg(long)]
        prompt: String,

        /// Tokens a generar.
        #[arg(long, default_value_t = 20)]
        max_tokens: u32,

        /// Temperatura: solo 0 (greedy determinista) — sampling llega después.
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,

        /// Presupuesto de memoria CONTROLADA (Fase 9): 512M, 1G, 2G, 4G, 8G,
        /// 512MB, 4GB o bytes crudos. Limita SOLO lo que el runtime mantiene
        /// residente a propósito (KV, scratch, estados, overhead) — el modelo
        /// vive en mmap paginado por demanda y NO cuenta contra el presupuesto
        /// (mapped != resident). Sin la opción: sin límite (modo Fase 8).
        #[arg(long)]
        memory_budget: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Inspect { gguf, no_tensors } => {
            if let Err(e) = cmd_inspect(&gguf, no_tensors) {
                eprintln!("unltd inspect: {e}");
                std::process::exit(1);
            }
        }
        Cmd::MinForward {
            model,
            tokens,
            oracle_dir,
            legacy_prec,
        } => {
            if let Err(e) = cmd_min_forward(&model, &tokens, oracle_dir.as_deref(), legacy_prec) {
                eprintln!("unltd min-forward: {e}");
                std::process::exit(1);
            }
        }
        Cmd::ForwardOracle {
            model,
            out_dir,
            oracle_dir,
            debug_nodes,
        } => {
            if let Err(e) = cmd_forward_oracle(
                &model,
                &out_dir,
                oracle_dir.as_deref(),
                debug_nodes.as_deref(),
            ) {
                eprintln!("unltd forward-oracle: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Tokenize { model, text } => {
            if let Err(e) = cmd_tokenize(&model, &text) {
                eprintln!("unltd tokenize: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            memory_budget,
        } => {
            if let Err(e) = cmd_run(
                &model,
                &prompt,
                max_tokens,
                temperature,
                memory_budget.as_deref(),
            ) {
                eprintln!("unltd run: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn cmd_inspect(gguf: &Path, no_tensors: bool) -> Result<(), LoadError> {
    let r = GgufReader::open(gguf)?;

    println!("GGUF file : {}", gguf.display());
    println!("file size : {} ({} bytes)", human(r.file_size), r.file_size);
    println!("version   : {}", r.version);
    println!("n_kv      : {}", r.metadata.len());
    println!("n_tensors : {}", r.tensors().len());
    println!(
        "data start: {} ({} bytes, tensor data begins here)",
        human(r.data_start),
        r.data_start
    );

    println!();
    println!("metadata:");
    for (k, v) in &r.metadata {
        println!("  {k} = {}", v.describe());
    }

    if !no_tensors {
        println!();
        println!("tensors ({}):", r.tensors().len());
        println!(
            "  {:<55} {:>14} {:>9} {:>12} {:>14} {:>7}",
            "name", "dims", "type", "offset", "bytes", "%file"
        );
        for t in r.tensors() {
            let dims = t
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            // bytes EXACTOS: inspect existe para verificar contra el oráculo,
            // no para leer bonito (el resumen ya muestra los tamaños legibles).
            let bytes = t
                .nbytes
                .map(|b| b.to_string())
                .unwrap_or_else(|| "?".to_string());
            let pct = t
                .nbytes
                .map(|b| 100.0 * b as f64 / r.file_size as f64)
                .unwrap_or(0.0);
            println!(
                "  {:<55} {:>14} {:>9} {:>12} {:>14} {:>6.2}%",
                t.name, dims, t.dtype, t.offset, bytes, pct
            );
        }
    }

    println!();
    println!("summary:");
    let (total, n_unknown) = r.tensor_bytes_summary();
    println!(
        "  total tensor bytes: {} ({} bytes, {:.2}% of file)",
        human(total),
        total,
        100.0 * total as f64 / r.file_size as f64
    );
    if n_unknown > 0 {
        println!("  WARNING: {n_unknown} tensor(s) with unknown ggml type — la suma es un piso, no el total");
    }
    let mut sorted: Vec<_> = r.tensors().iter().collect();
    sorted.sort_by_key(|t| std::cmp::Reverse(t.nbytes.unwrap_or(0)));
    println!("  top tensors:");
    for t in sorted.iter().take(5) {
        let bytes = t.nbytes.map(human).unwrap_or_else(|| "?".to_string());
        println!("    {:<55} {:>9} {:>14}", t.name, t.dtype, bytes);
    }
    let unknown: Vec<_> = r.unknown_type_tensors().collect();
    if !unknown.is_empty() {
        println!("  unknown-type tensors (nbytes = None, nunca adivinado):");
        for t in unknown.iter().take(10) {
            println!("    {:<55} type id {}", t.name, t.ggml_type_id);
        }
    }
    match r.get_str("general.architecture") {
        Some(a) => println!("architecture hint: {a}"),
        None => println!("architecture hint: (ausente)"),
    }

    Ok(())
}

/// Tokeniza texto con el tokenizador del GGUF (Fase 7). Imprime Token IDs /
/// Pieces / texto decodificado. La puerta de la fase es que los IDs coincidan
/// con los del oráculo llama.cpp para el mismo texto.
fn cmd_tokenize(model: &Path, text: &str) -> Result<(), LoadError> {
    let reader = GgufReader::open(model)?;
    let tok = Gpt2Tokenizer::from_gguf(&reader)?;

    println!("modelo  : {}", model.display());
    println!("tokenizer: {} (pre: {})", tok.kind(), tok.pre());
    println!("vocab   : {}", tok.vocab_size());
    match (tok.bos(), tok.eos()) {
        (Some(b), Some(e)) => println!("bos/eos : {b} / {e}"),
        (None, Some(e)) => println!("bos/eos : (ninguno) / {e}"),
        (Some(b), None) => println!("bos/eos : {b} / (ninguno)"),
        (None, None) => println!("bos/eos : (ninguno) / (ninguno)"),
    }
    println!("text    : {text:?}");

    let mut ids = Vec::new();
    tok.encode(text, &mut ids).map_err(|e| match e {
        TokenError::PieceMissing(p) => LoadError::corrupt(format!(
            "encode: pieza {p:?} fuera del vocab (fallback por char incluido)"
        )),
        other => LoadError::corrupt(format!("encode: {other}")),
    })?;

    println!();
    println!("tokens ({}):", ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let piece = tok.piece(id).unwrap_or("<fuera de rango>");
        let mut dec = Vec::new();
        tok.decode(&[id], &mut dec)
            .map_err(|e| LoadError::corrupt(format!("decode: {e}")))?;
        println!("  [{i:>3}] id {id:<7} piece {piece:?} decoded {dec:?}");
    }
    let mut decoded = Vec::new();
    tok.decode(&ids, &mut decoded)
        .map_err(|e| LoadError::corrupt(format!("decode: {e}")))?;
    println!();
    println!("ids     : {ids:?}");
    println!("decoded : {:?}", String::from_utf8_lossy(&decoded));
    Ok(())
}

/// Tokens del prompt con el que se generó el oráculo de ornith
/// ("The capital of France is", raw, add_bos=false).
const ORACLE_PROMPT_TOKENS: [u32; 5] = [760, 6511, 314, 9338, 369];

/// Tokens generados por el oráculo para ese prompt (llama-server Prism,
/// greedy temp 0, raw -no-cnv, n_predict 20, -ngl 0). Fuente única de verdad:
/// `benchmarks/reference/ornith-greedy-tokens.txt` (stop_type: limit, sin EOS).
const ORACLE_GENERATED_TOKENS: [u32; 20] = [
    11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751,
    13, 198, 760,
];

/// Generación greedy multi-token (Fase 8): tokenize raw → prefill token a
/// token → loop decode (norm → logits → argmax → append). Sin KV cache
/// incremental adicional: la sesión YA es el KV cache incremental del motor
/// (~14 s/token marginales, medido en Fase 6).
///
/// Puerta: los 20 tokens greedy coinciden con el oráculo (tabla
/// `step | UNLTD | oracle | match`). Si divergen, se reporta el PRIMER token
/// distinto y se sigue generando para completar la tabla (política de la
/// directiva: aislar el primer mismatch, no detenerse a investigar acá).
fn cmd_run(
    model: &Path,
    prompt: &str,
    max_tokens: u32,
    temperature: f64,
    memory_budget: Option<&str>,
) -> Result<(), LoadError> {
    if temperature != 0.0 || temperature.is_nan() {
        return Err(LoadError::corrupt(format!(
            "temperatura {temperature}: solo 0 (greedy determinista) está implementado (Fase 8)"
        )));
    }

    let reader = GgufReader::open(model)?;
    let tok = Gpt2Tokenizer::from_gguf(&reader)?;
    let fwd = Qwen35Forward::open(MappedWeights::open(model)?)?;

    // Tokenizado RAW (sin BOS): el mismo modo que el oráculo (-no-cnv, sin
    // add_bos). ornith no declara bos_token_id, así que no hay política de BOS.
    let mut prompt_ids = Vec::new();
    tok.encode(prompt, &mut prompt_ids).map_err(|e| match e {
        TokenError::PieceMissing(p) => LoadError::corrupt(format!(
            "encode: pieza {p:?} fuera del vocab (fallback por char incluido)"
        )),
        other => LoadError::corrupt(format!("encode: {other}")),
    })?;
    if prompt_ids.is_empty() {
        return Err(LoadError::corrupt("prompt vacío"));
    }

    let ctx = prompt_ids.len() + max_tokens as usize;

    // ---- Fase 9: presupuesto de memoria controlada. El plan se suma ENTERO
    // ANTES de asignar nada (contrato: negarse con números, nunca OOM-kill).
    let mut kv_bytes = 0u64;
    for il in 0..fwd.cfg.n_layer {
        if fwd.cfg.is_full_attn(il) {
            kv_bytes += (ctx * fwd.cfg.n_head_kv * fwd.cfg.head_dim_v * 2 * 4) as u64;
        }
    }
    let (recurrent_bytes, scratch_bytes, runtime_bytes) = controlled_plan(&fwd, &tok, &reader, ctx);
    // El estado recurrente (conv ring + GDN) es memoria CONTROLADA residente
    // deliberadamente (la asigna `new_session`), igual que el tokenizer: se
    // contabiliza en runtime_bytes. mandatory = KV + scratch + runtime_total.
    let runtime_total = runtime_bytes.saturating_add(recurrent_bytes);
    let mandatory_bytes = kv_bytes
        .saturating_add(scratch_bytes)
        .saturating_add(runtime_total);
    let budget = match memory_budget {
        Some(s) => Some(
            parse_size(s).map_err(|e| LoadError::corrupt(format!("--memory-budget {s:?}: {e}")))?,
        ),
        None => None,
    };
    let acc = budget.map(|b| {
        let mut a = MemoryAccounting::new(b, mandatory_bytes);
        a.kv_cache_bytes = kv_bytes;
        a.scratch_bytes = scratch_bytes;
        a.runtime_bytes = runtime_total;
        // weight_buffer_bytes y weight_cache_bytes quedan en 0: los pesos son
        // VISTAS sobre el mmap (Fase 10), sin copias heap ni cache de dequant.
        a
    });
    if let Some(a) = &acc {
        if a.mandatory_bytes > a.configured_budget {
            println!("REFUSING TO RUN — el presupuesto no cubre el mínimo mandatorio:");
            println!(
                "  Memory budget    : {} ({:?} bytes)",
                human(a.configured_budget),
                a.configured_budget
            );
            println!(
                "  Required minimum : {} ({:?} bytes) = weight buffers + weight cache + KV + scratch + runtime (runtime incluye estado recurrente)",
                human(a.mandatory_bytes),
                a.mandatory_bytes
            );
            println!("  KV cache         : {}", human(a.kv_cache_bytes));
            println!("  Recurrent state  : {}", human(recurrent_bytes));
            println!("  Scratch (peak)   : {}", human(a.scratch_bytes));
            println!(
                "  Runtime overhead : {} (tokenizer + índice GGUF + estado recurrente)",
                human(a.runtime_bytes)
            );
            println!(
                "  Layer buffers    : {} (pesos = vistas sobre mmap, sin copias)",
                human(a.weight_buffer_bytes)
            );
            std::process::exit(2);
        }
    }

    println!("modelo  : {}", model.display());
    println!("arch    : qwen35 (Fases 8-10, greedy temperatura 0)");
    println!("prompt  : {prompt:?}");
    println!(
        "tokens  : {prompt_ids:?} ({}, raw sin BOS)",
        prompt_ids.len()
    );
    println!("eos     : {:?}", tok.eos());
    println!("max     : {max_tokens}");
    println!("ctx     : {ctx} (prompt + max_tokens, KV cache pre-dimensionado)");

    match &acc {
        Some(a) => print_memory_plan("plan", a, reader.file_size, recurrent_bytes),
        // Sin presupuesto: línea informativa previa (modo Fase 8, sin cambios).
        None => println!(
            "ram     : {} modelo (mmap, paginado) + {} KV cache + activaciones O(n_embd={})",
            human(reader.file_size),
            human(kv_bytes),
            fwd.cfg.n_embd
        ),
    }

    let mut session = fwd.new_session(ctx);

    let mut gen = GreedyLoop::new(
        |t, logits| {
            let emb = fwd.embed(&[t])?;
            let l_out = fwd.step(&mut session, &emb, None, None)?;
            let rn = fwd.output_norm(&l_out)?;
            *logits = fwd.output_logits(&rn)?;
            Ok(())
        },
        tok.eos(),
        max_tokens,
    );

    let t0 = std::time::Instant::now();
    gen.prefill(&prompt_ids)?;
    let t_prefill = t0.elapsed();
    println!(
        "prefill : {} tokens en {:.1}s ({:.2} s/token)",
        prompt_ids.len(),
        t_prefill.as_secs_f64(),
        t_prefill.as_secs_f64() / prompt_ids.len() as f64
    );

    println!();
    println!("step | UNLTD                            | oracle | match");
    let mut generated: Vec<u32> = Vec::new();
    let mut n_match = 0usize;
    let mut first_mismatch: Option<(usize, u32, u32)> = None;
    // Los logits del último token del prefill deciden el argmax del paso 0.
    let mut decision_logits: Vec<f32> = gen.logits().to_vec();
    let t1 = std::time::Instant::now();
    while let Some(t) = gen.next_token()? {
        let step = generated.len();
        generated.push(t);
        let oracle = ORACLE_GENERATED_TOKENS.get(step).copied();
        let piece = tok.piece(t).unwrap_or("<oob>");
        let ok = oracle == Some(t);
        if ok {
            n_match += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some((step, t, oracle.unwrap_or(u32::MAX)));
            // Los logits que DECIDIERON este argmax (snapshot antes del
            // forward del token): miden si el token del oráculo estaba cerca
            // (flip de argmax por divergencia numérica documentada) o lejos.
            let (ti, top) = top5(&decision_logits);
            println!("       top-5 decisión: [{ti}] {top:?}");
            if let Some(o) = oracle {
                match decision_logits
                    .iter()
                    .enumerate()
                    .find(|(i, _)| *i == o as usize)
                {
                    Some((_, &v)) => {
                        let gap = decision_logits[ti] - v;
                        println!("       oráculo {o} en logit {v:.6} (gap al argmax {gap:.6})");
                    }
                    None => println!("       oráculo {o} fuera del vocab"),
                }
            }
        }
        println!(
            "{:>4} | {:<6} {:<24.24} | {:>6} | {}",
            step,
            t,
            piece,
            oracle.map(|o| o.to_string()).unwrap_or_else(|| "-".into()),
            if ok { "MATCH" } else { "DIFF" }
        );
        // Snapshot para el próximo paso (los logits vigentes deciden el
        // argmax de la PRÓXIMA iteración).
        decision_logits = gen.logits().to_vec();
    }
    let t_gen = t1.elapsed();

    // top-5 del último token (informativo: la puerta es el argmax, los valores
    // difieren del oráculo dentro del contrato numérico de la Fase 6).
    let (top_idx, top) = top5(gen.logits());
    println!("top-5   : [{top_idx}] {top:?} (informativo, la puerta es el argmax)");

    let mut decoded = Vec::new();
    tok.decode(&generated, &mut decoded)
        .map_err(|e| LoadError::corrupt(format!("decode: {e}")))?;
    println!("decoded : {:?}", String::from_utf8_lossy(&decoded));
    println!();
    println!("matches : {n_match}/{max_tokens} contra el oráculo");
    let n_forward = if generated.is_empty() {
        0
    } else {
        generated.len() - 1
    };
    let s_per_tok = if n_forward > 0 {
        t_gen.as_secs_f64() / n_forward as f64
    } else {
        0.0
    };
    println!(
        "tiempos : prefill {:.1}s (TTFT ~= prefill, el token 1 sale del último logit del prefill); \
         decode {:.1}s para {} forwards ({:.2} s/token); total {:.1}s",
        t_prefill.as_secs_f64(),
        t_gen.as_secs_f64(),
        n_forward,
        s_per_tok,
        (t_prefill + t_gen).as_secs_f64(),
    );

    // ---- Fase 9: reporte final con cifras MEDIDAS del proceso (RSS e IO),
    // no del plan. El peak RSS puede superar el presupuesto sin violarlo:
    // incluye páginas del modelo cacheadas por el SO (demand paging, Fase 10).
    if let Some(a) = &acc {
        println!();
        print_memory_plan("final", a, reader.file_size, recurrent_bytes);
        let (peak_rss, rss_now, read_bytes) = measure_process(std::process::id());
        match (peak_rss, rss_now) {
            (Some(peak), Some(now)) => {
                println!(
                    "  peak RSS (medido)   : {} (PeakWorkingSet64; incluye páginas del modelo cacheadas por el SO)",
                    human(peak)
                );
                println!("  working set actual   : {}", human(now));
            }
            _ => println!(
                "  peak RSS (medido)   : Unavailable (PowerShell no disponible — no se inventan números)"
            ),
        }
        match read_bytes {
            Some(b) => println!(
                "  bytes read (proceso): {} (GetProcessIoCounters; incluye exe/DLLs, no solo el modelo)",
                human(b)
            ),
            None => println!(
                "  bytes read (proceso): Unavailable (GetProcessIoCounters no disponible)"
            ),
        }
    }

    match first_mismatch {
        None => {
            println!("\nRUN PASS: {n_match}/{max_tokens} tokens greedy == oráculo.");
            Ok(())
        }
        Some((step, mine, oracle)) => {
            eprintln!(
                "\nRUN INVALID — primer mismatch en el token generado {step}: \
                 UNLTD {mine} vs oráculo {oracle}"
            );
            std::process::exit(3);
        }
    }
}

/// Plan de la memoria controlada (Fase 9), excluido el KV (fórmula propia):
/// estado recurrente (conv ring + GDN) y pico estimado del scratch por paso.
/// `runtime_bytes` mide las estructuras reales (tokenizer + índice GGUF); el
/// llamador SUMA el estado recurrente a runtime (es memoria controlada
/// residente, asignada por `new_session`). Los pesos NO aparecen: son vistas
/// sobre el mmap, sin materialización en heap (Fase 10) —
/// `weight_buffer_bytes` y `weight_cache_bytes` quedan en 0.
fn controlled_plan(
    fwd: &Qwen35Forward,
    tok: &Gpt2Tokenizer,
    reader: &GgufReader,
    ctx: usize,
) -> (u64, u64, u64) {
    let c = &fwd.cfg;
    let n = c.n_embd;
    let n_ff = c.n_ff;
    let n_v = c.n_v_heads();
    let hd = c.head_dim_linear();
    let d_conv = c.d_conv();
    let mut n_recr = 0usize;
    for il in 0..c.n_layer {
        if !c.is_full_attn(il) {
            n_recr += 1;
        }
    }
    // conv ring (conv_kernel-1)×d_conv + estado GDN n_v×hd×hd, por capa
    // recurrente, en f32 (ver Qwen35Forward::new_session).
    let recurrent = n_recr as u64 * ((c.conv_kernel - 1) * d_conv + n_v * hd * hd) as u64 * 4;

    // Scratch: suma conservadora de los Vecs f32 transitorios de UN paso
    // (step + ambas capas + FFN, ver qwen35_forward.rs) + logits del bucle
    // greedy (loop + snapshot del CLI) + índice del sort top-5 (u64/vocab).
    // Es el pico de activaciones; no crece con el contexto salvo scores.
    let vocab = fwd.vocab();
    let elems = 18 * n as u64 // out/x/z/y/l_out/residual/attn_out/gdn_out/z_silu/q/gate/qfull/f/embed
        + 3 * n_ff as u64 // FFN gate/up/down
        + 8 * d_conv as u64 // qkv/conv_out/mix + kernel conv_kernel·d_conv
        + 5 * n_v as u64 // beta_raw/alpha/beta/beta_biased/sp
        + 2 * c.n_head_kv as u64 * c.head_dim_v as u64 // k + v
        + c.n_head as u64 * ctx as u64; // scores de attn
    let scratch = elems * 4 + 2 * vocab as u64 * 4 + vocab as u64 * 8;

    // Runtime: heap real del tokenizer + índice GGUF (strings de metadata y
    // tabla de tensores). Estimación conservadora; el RSS es el veredicto.
    let runtime = tok.heap_bytes().saturating_add(gguf_index_bytes(reader));

    (recurrent, scratch, runtime)
}

/// Estimación conservadora del heap del índice GGUF (metadata + tabla de
/// tensores). El reader parsea SOLO header+índice; los datos de los tensores
/// viven en el mmap y no se copian.
fn gguf_index_bytes(r: &GgufReader) -> u64 {
    let mut b = 0u64;
    for (k, v) in &r.metadata {
        b += (k.len() as u64).saturating_add(v.describe().len() as u64);
    }
    for t in r.tensors() {
        b += (t.name.len() as u64).saturating_add(96); // shape/offset/dtype ≈ 96 B
    }
    b
}

/// Reporte de memoria del contrato Fase 9: file size, mapped virtual, pesos
/// controlados (buffer + cache), KV, scratch, runtime, presupuesto y los tres
/// métodos del componente de contabilidad. `phase` = "plan" (antes de
/// asignar) o "final" (con las cifras medidas que el llamador agrega).
/// `recurrent_bytes` es solo display: ya está incluido en runtime_bytes.
fn print_memory_plan(phase: &str, a: &MemoryAccounting, file_size: u64, recurrent_bytes: u64) {
    println!("memoria ({phase}, Fase 9):");
    println!(
        "  file size            : {} ({} bytes)",
        human(file_size),
        file_size
    );
    println!(
        "  mapped (virtual)     : {} (mmap del archivo completo — NO cuenta contra el presupuesto; mapped != resident)",
        human(file_size)
    );
    println!(
        "  configured budget    : {} ({} bytes, SOLO memoria controlada)",
        human(a.configured_budget),
        a.configured_budget
    );
    println!("  controlled:");
    println!(
        "    weight_buffer_bytes: {} (pesos = vistas sobre mmap, sin copias heap)",
        human(a.weight_buffer_bytes)
    );
    println!(
        "    weight_cache_bytes : {} (sin cache de dequant)",
        human(a.weight_cache_bytes)
    );
    println!("    kv_cache_bytes     : {}", human(a.kv_cache_bytes));
    println!(
        "    scratch_bytes      : {} (pico estimado del paso: activaciones + logits)",
        human(a.scratch_bytes)
    );
    println!(
        "    runtime_bytes      : {} (tokenizer + índice GGUF + estado recurrente)",
        human(a.runtime_bytes)
    );
    println!(
        "    recurrent state    : {} (conv ring + GDN, incluido en runtime_bytes)",
        human(recurrent_bytes)
    );
    println!(
        "  used_controlled_bytes: {}",
        human(a.used_controlled_bytes())
    );
    println!("  available_budget     : {}", human(a.available_budget()));
    println!("  mandatory_bytes      : {}", human(a.mandatory_bytes));
    println!("  budget_respected     : {}", a.budget_respected());
}

/// RSS e IO del proceso propio vía PowerShell (API de Windows existente, sin
/// dependencias nuevas): PeakWorkingSet64/WorkingSet64 de Get-Process y
/// GetProcessIoCounters de kernel32 vía Add-Type. Best-effort: cualquier
/// fallo devuelve `None` y el reporte dice "Unavailable" (no se inventan
/// números). Devuelve (peak_rss, rss_actual, bytes_leídos).
fn measure_process(pid: u32) -> (Option<u64>, Option<u64>, Option<u64>) {
    let script = format!(
        "$p = Get-Process -Id {pid}; \
         Write-Output ('RSS ' + $p.PeakWorkingSet64 + ' ' + $p.WorkingSet64); \
         try {{ \
           Add-Type -TypeDefinition 'public static class UIo {{ [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)] public struct C {{ public ulong ro, wo, oo, rb, wb, ob; }} [System.Runtime.InteropServices.DllImport(\"kernel32.dll\")] public static extern bool GetProcessIoCounters(System.IntPtr h, out C c); }}'; \
           $c = New-Object UIo+C; \
           if ([UIo]::GetProcessIoCounters($p.Handle, [ref]$c)) {{ Write-Output ('IO ' + $c.rb + ' ' + $c.ro) }} else {{ Write-Output 'IO UNAVAILABLE' }} \
         }} catch {{ Write-Output ('IO UNAVAILABLE ' + $_.Exception.Message) }}"
    );
    let parse_pair = |out: &std::process::Output| -> (Option<u64>, Option<u64>, Option<u64>) {
        let mut peak = None;
        let mut rss = None;
        let mut read = None;
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let mut it = line.split_whitespace();
            match it.next() {
                Some("RSS") => {
                    peak = it.next().and_then(|v| v.parse().ok());
                    rss = it.next().and_then(|v| v.parse().ok());
                }
                Some("IO") => {
                    read = it.next().and_then(|v| v.parse().ok());
                }
                _ => {}
            }
        }
        (peak, rss, read)
    };
    match std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-Command", &script])
        .output()
    {
        Ok(out) if out.status.success() => parse_pair(&out),
        Ok(out) => {
            eprintln!(
                "memoria: PowerShell falló ({}) — RSS/IO Unavailable",
                String::from_utf8_lossy(&out.stderr).trim()
            );
            (None, None, None)
        }
        Err(e) => {
            eprintln!("memoria: PowerShell no disponible ({e}) — RSS/IO Unavailable");
            (None, None, None)
        }
    }
}

/// Tolerancia de attn_norm contra el oráculo: el motor reduce pairwise en f64
/// y el oráculo secuencial en f32 (l2_norm de ggml); con |x| ~ O(1) la
/// diferencia de orden + precisión cabe en 1e-5 (docs/AUDIT.md §3.3).
const NORM_TOL: f32 = 1e-5;

fn cmd_min_forward(
    model: &Path,
    tokens: &str,
    oracle_dir: Option<&Path>,
    legacy_prec: bool,
) -> Result<(), LoadError> {
    let ids: Vec<u32> = tokens
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<u32>()
                .map_err(|e| LoadError::corrupt(format!("token inválido '{s}': {e}")))
        })
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err(LoadError::corrupt("--tokens vacío".to_string()));
    }

    let mf = MinForward::open(MappedWeights::open(model)?)?;
    let n = mf.cfg.n_embd;
    println!("modelo     : {}", model.display());
    println!("arch       : qwen35 (Fase 5)");
    println!("n_embd     : {n}");
    println!("vocab      : {}", mf.vocab());
    println!("tokens     : {ids:?}");

    let emb = mf.embed(&ids)?;
    let norm = mf.attn_norm_rows(&emb)?;
    println!(
        "embed      : {} filas x {n} (max |x| = {:.6})",
        ids.len(),
        max_abs(&emb)
    );
    println!(
        "attn_norm  : {} filas x {n} (max |x| = {:.6})",
        ids.len(),
        max_abs(&norm)
    );

    if ids.as_slice() != ORACLE_PROMPT_TOKENS.as_slice() {
        eprintln!(
            "WARNING: los tokens no son el prompt del oráculo {ORACLE_PROMPT_TOKENS:?}; \
             la comparación fila a fila fallará en la puerta de embeddings."
        );
    }

    let Some(dir) = oracle_dir else {
        println!("\n(sin --oracle-dir: forward ejercitado, sin comparación)");
        return Ok(());
    };

    // Tolerancias por precisión del oráculo:
    // - %.9g (prec9): embeddings bit-exactos (0.0, mismo dequantize Q4_K) y
    //   attn_norm ≤ 1e-5 (pairwise-f64 vs secuencial-f32, documentado);
    // - %12.4f (legacy): el redondeo de impresión es el límite — 5e-5 por
    //   valor (FLT_DECIMAL_DIG-3) en embeddings, 1e-4 en attn_norm.
    let (emb_tol, norm_tol) = if legacy_prec {
        (5e-5f32, 1e-4f32)
    } else {
        (0.0f32, NORM_TOL)
    };

    // 1) embeddings. Con bins prec9 `0.0` ES la igualdad bit a bit, no una cota.
    let oracle_emb = read_f32_bin(&dir.join("ornith-model.input_embed.f32.bin"), emb.len())?;
    let (e_max, e_at) = max_abs_diff(&emb, &oracle_emb);
    println!("\npuerta embeddings (tolerancia {emb_tol:.1e}):");
    println!("  max |diff| = {e_max:.9} en [{e_at}]");
    if e_max > emb_tol {
        return fail(
            "embeddings",
            &format!("max |diff| = {e_max:.9} > {emb_tol}"),
        );
    }
    println!(
        "  PASS{}",
        if legacy_prec {
            " (límite de redondeo %12.4f)"
        } else {
            ": bit-idénticos al oráculo"
        }
    );

    // 2) attn_norm: ≤ 1e-5 con bins prec9, ≤ 1e-4 con legacy.
    let oracle_norm = read_f32_bin(&dir.join("ornith-attn_norm-0.f32.bin"), norm.len())?;
    let (n_max, n_at) = max_abs_diff(&norm, &oracle_norm);
    println!("puerta attn_norm (tolerancia {norm_tol:.1e}):");
    println!("  max |diff| = {n_max:.9} en [{n_at}]");
    if n_max > norm_tol {
        return fail(
            "attn_norm",
            &format!("max |diff| = {n_max:.9} > {norm_tol}"),
        );
    }
    println!("  PASS");

    // 3) cabeza de salida: argmax == oráculo. La magnitud del diff refleja la
    // cuantización Q8_K de activaciones del oráculo (documentada), no se gata.
    let oracle_rn = read_f32_bin(&dir.join("ornith-result_norm.f32.bin"), n)?;
    let logits = mf.output_logits(&oracle_rn)?;
    let oracle_out = read_f32_bin(&dir.join("ornith-result_output.f32.bin"), mf.vocab())?;
    let (mine_idx, mine_top) = top5(&logits);
    let (ora_idx, ora_top) = top5(&oracle_out);
    let (l_max, l_at) = max_abs_diff(&logits, &oracle_out);
    println!("puerta logits (argmax == oráculo):");
    println!("  max |diff| = {l_max:.9} en [{l_at}] (Q8_K de activaciones, informativo)");
    println!("  argmax     : mine {mine_idx}, oráculo {ora_idx}");
    println!("  top-5      : mine {mine_top:?}");
    println!("               ora  {ora_top:?}");
    if mine_idx != ora_idx {
        return fail("logits", &format!("argmax {mine_idx} != oráculo {ora_idx}"));
    }
    println!("  PASS");

    println!("\nMIN-FORWARD PASS: las tres puertas en verde.");
    Ok(())
}

/// Prefill completo Fase 6: los 5 tokens del oráculo por las 32 capas.
///
/// Escribe `engine-layers.f32.bin` (formato de compare-layers-oracle.py:
/// [u32 n_layer][u32 n_tokens][u32 n_embd] y por (capa, token) los tres
/// vectores l_out, attn_residual, linear_attn_out en f32 LE) y un summary de
/// cola (result_norm, argmax, top-5). Con `--oracle-dir` (bins prec9) gata:
/// result_norm ≤ 1e-4 y argmax == oráculo (11751).
fn cmd_forward_oracle(
    model: &Path,
    out_dir: &Path,
    oracle_dir: Option<&Path>,
    debug_nodes: Option<&Path>,
) -> Result<(), LoadError> {
    let fwd = Qwen35Forward::open(MappedWeights::open(model)?)?;
    let n = fwd.cfg.n_embd;
    let n_layer = fwd.cfg.n_layer;
    let n_tokens = ORACLE_PROMPT_TOKENS.len();

    println!("modelo : {}", model.display());
    println!("arch   : qwen35 (Fase 6, 32 capas)");
    println!("tokens : {ORACLE_PROMPT_TOKENS:?} (ctx de sesión = {n_tokens})");

    let mut session = fwd.new_session(n_tokens);
    let mut per_token: Vec<LayerDump> = Vec::with_capacity(n_tokens);
    let mut nodes_all: Vec<NodeCapture> = Vec::with_capacity(n_tokens);
    let mut final_l_out = Vec::new();
    let t0 = std::time::Instant::now();
    for (t, &tok) in ORACLE_PROMPT_TOKENS.iter().enumerate() {
        let emb = fwd.embed(&[tok])?;
        let mut dump = LayerDump {
            attn_residual: vec![0.0f32; n_layer * n],
            l_out: vec![0.0f32; n_layer * n],
            linear_attn_out: vec![0.0f32; n_layer * n],
        };
        let mut nodes = NodeCapture::default();
        let nodes_ref = debug_nodes.map(|_| &mut nodes);
        final_l_out = fwd.step(&mut session, &emb, Some(&mut dump), nodes_ref)?;
        println!(
            "  token {t} (id {tok}) listo en {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        per_token.push(dump);
        nodes_all.push(nodes);
    }

    std::fs::create_dir_all(out_dir)
        .map_err(|e| LoadError::corrupt(format!("crear {}: {e}", out_dir.display())))?;
    let bin_path = out_dir.join("engine-layers.f32.bin");
    write_layers_bin(&bin_path, &per_token, n_layer, n_tokens, n)?;
    println!(
        "capas    : {} ({} bytes)",
        bin_path.display(),
        bin_path.metadata().map(|m| m.len()).unwrap_or(0)
    );

    if let Some(dn) = debug_nodes {
        write_nodes_bin(dn, &nodes_all, n_tokens)?;
        println!(
            "nodos    : {} ({} bytes)",
            dn.display(),
            dn.metadata().map(|m| m.len()).unwrap_or(0)
        );
    }

    // Cola: result_norm + logits del último token.
    let rn = fwd.output_norm(&final_l_out)?;
    let logits = fwd.output_logits(&rn)?;
    let (mine_idx, mine_top) = top5(&logits);
    println!("cola     : result_norm max |x| = {:.6}", max_abs(&rn));
    println!("           argmax = {mine_idx}, top-5 = {mine_top:?}");
    if let Some(dir) = oracle_dir {
        let oracle_rn = read_f32_bin(&dir.join("ornith-result_norm.f32.bin"), n)?;
        let oracle_out = read_f32_bin(&dir.join("ornith-result_output.f32.bin"), fwd.vocab())?;
        let (ora_idx, ora_top) = top5(&oracle_out);
        let (rn_max, rn_at) = max_abs_diff(&rn, &oracle_rn);
        let (l_max, l_at) = max_abs_diff(&logits, &oracle_out);
        println!("puerta result_norm (tolerancia 1e-4):");
        println!("  max |diff| = {rn_max:.9} en [{rn_at}]");
        if rn_max > 1e-4 {
            return fail("result_norm", &format!("max |diff| = {rn_max:.9} > 1e-4"));
        }
        println!("  PASS");
        println!("puerta logits (argmax == oráculo):");
        println!("  max |diff| = {l_max:.9} en [{l_at}] (Q8_K del oráculo + diffs de capas, informativo)");
        println!("  argmax     : mine {mine_idx}, oráculo {ora_idx}");
        println!("  top-5      : mine {mine_top:?}");
        println!("               ora  {ora_top:?}");
        if mine_idx != ora_idx {
            return fail("logits", &format!("argmax {mine_idx} != oráculo {ora_idx}"));
        }
        println!("  PASS");
    }

    println!("\nFORWARD-ORACLE OK: ahora comparar capas con compare-layers-oracle.py.");
    Ok(())
}

/// Escribe el binario de capas: header (3×u32) + por (capa, token) los tres
/// vectores l_out / attn_residual / linear_attn_out en f32 LE, capa-major.
fn write_layers_bin(
    path: &Path,
    per_token: &[LayerDump],
    n_layer: usize,
    n_tokens: usize,
    n: usize,
) -> Result<(), LoadError> {
    let mut buf: Vec<u8> = Vec::with_capacity(12 + n_layer * n_tokens * 3 * n * 4);
    buf.extend_from_slice(&(n_layer as u32).to_le_bytes());
    buf.extend_from_slice(&(n_tokens as u32).to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for il in 0..n_layer {
        for t in 0..n_tokens {
            let base = il * n;
            let push = |buf: &mut Vec<u8>, v: &[f32]| {
                for x in v {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            };
            push(&mut buf, &per_token[t].l_out[base..base + n]);
            push(&mut buf, &per_token[t].attn_residual[base..base + n]);
            push(&mut buf, &per_token[t].linear_attn_out[base..base + n]);
        }
    }
    std::fs::write(path, buf)
        .map_err(|e| LoadError::corrupt(format!("escribir {}: {e}", path.display())))
}

/// Escribe los nodos capturados: [u32 n_tokens] y por token [u32 n_nodes],
/// por nodo [u32 name_len][name bytes][u32 ne0][f32×ne0] (LE). Los nodos son
/// los intermedios del camino recurrente capturados en `Qwen35Forward::step`.
fn write_nodes_bin(
    path: &Path,
    per_token: &[NodeCapture],
    n_tokens: usize,
) -> Result<(), LoadError> {
    assert_eq!(per_token.len(), n_tokens);
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(n_tokens as u32).to_le_bytes());
    for cap in per_token {
        let tok = &cap.per_token; // un Vec<(name, vals)> por token
                                  // un solo token por step: per_token siempre tiene 1 entrada
        let nodes = tok.first().map(|v| v.as_slice()).unwrap_or(&[]);
        buf.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        for (name, vals) in nodes {
            let name = name.as_bytes();
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name);
            buf.extend_from_slice(&(vals.len() as u32).to_le_bytes());
            for x in vals {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    std::fs::write(path, buf)
        .map_err(|e| LoadError::corrupt(format!("escribir {}: {e}", path.display())))
}

/// Fallo numérico de una puerta: exit 3 (distinto de errores de carga) +
/// `RUN INVALID`, el contrato de la CLI para corridas no fiables.
fn fail(gate: &str, detail: &str) -> Result<(), LoadError> {
    eprintln!("\nMIN-FORWARD FAIL ({gate}): {detail}");
    eprintln!("RUN INVALID — no continuar a Fase 6 con este estado.");
    std::process::exit(3);
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0f32;
    let mut at = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
            at = i;
        }
    }
    (m, at)
}

fn top5(v: &[f32]) -> (usize, Vec<(usize, f32)>) {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    (idx[0], idx.iter().take(5).map(|&i| (i, v[i])).collect())
}

fn read_f32_bin(path: &Path, expect: usize) -> Result<Vec<f32>, LoadError> {
    let bytes = std::fs::read(path)
        .map_err(|e| LoadError::corrupt(format!("oráculo {}: {e}", path.display())))?;
    if bytes.len() != expect * 4 {
        return Err(LoadError::corrupt(format!(
            "oráculo {}: {} bytes, se esperaban {}",
            path.display(),
            bytes.len(),
            expect * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
