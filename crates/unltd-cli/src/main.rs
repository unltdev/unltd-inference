//! CLI de unltd. Subcomandos por fase del ROADMAP:
//! - `inspect` (Fase 2): header, metadata y tabla de tensores de un GGUF;
//! - `min-forward` (Fase 5): embedding + attn_norm + cabeza de salida, validado
//!   contra los bins del oráculo llama-eval-callback;
//! - `tokenize` / `run` (Fase 6): stubs explícitos, no silenciosos.
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
use unltd_generation::{LayerDump, MinForward, NodeCapture, Qwen35Forward};
use unltd_model_loader::{GgufReader, MappedWeights};

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

    /// Tokeniza texto con el tokenizador del modelo (Fase 6).
    Tokenize {
        /// Modelo: archivo .gguf.
        model: PathBuf,

        #[arg(long)]
        text: String,
    },

    /// Genera texto con el motor (Fase 5).
    Run {
        /// Modelo: archivo .gguf.
        model: PathBuf,

        #[arg(long)]
        prompt: String,

        /// Tokens a generar.
        #[arg(long, default_value_t = 20)]
        max_tokens: u32,

        /// Temperatura (0 = greedy determinista, el default).
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,

        /// Presupuesto total del motor (GB). `auto` lo dimensiona de la RAM disponible.
        #[arg(long, default_value = "auto")]
        budget: String,
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
        Cmd::MinForward { model, tokens, oracle_dir, legacy_prec } => {
            if let Err(e) = cmd_min_forward(&model, &tokens, oracle_dir.as_deref(), legacy_prec) {
                eprintln!("unltd min-forward: {e}");
                std::process::exit(1);
            }
        }
        Cmd::ForwardOracle { model, out_dir, oracle_dir, debug_nodes } => {
            if let Err(e) =
                cmd_forward_oracle(&model, &out_dir, oracle_dir.as_deref(), debug_nodes.as_deref())
            {
                eprintln!("unltd forward-oracle: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Tokenize { .. } | Cmd::Run { .. } => {
            eprintln!(
                "unltd: este comando llega en la Fase 7 (ver docs/ROADMAP.md). \
                 Hoy existen `inspect` (Fase 2), `min-forward` (Fase 5) y \
                 `forward-oracle` (Fase 6)."
            );
            std::process::exit(2);
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

/// Tokens del prompt con el que se generó el oráculo de ornith
/// ("The capital of France is", raw, add_bos=false).
const ORACLE_PROMPT_TOKENS: [u32; 5] = [760, 6511, 314, 9338, 369];

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
    let (emb_tol, norm_tol) = if legacy_prec { (5e-5f32, 1e-4f32) } else { (0.0f32, NORM_TOL) };

    // 1) embeddings. Con bins prec9 `0.0` ES la igualdad bit a bit, no una cota.
    let oracle_emb = read_f32_bin(&dir.join("ornith-model.input_embed.f32.bin"), emb.len())?;
    let (e_max, e_at) = max_abs_diff(&emb, &oracle_emb);
    println!("\npuerta embeddings (tolerancia {emb_tol:.1e}):");
    println!("  max |diff| = {e_max:.9} en [{e_at}]");
    if e_max > emb_tol {
        return fail("embeddings", &format!("max |diff| = {e_max:.9} > {emb_tol}"));
    }
    println!("  PASS{}", if legacy_prec { " (límite de redondeo %12.4f)" } else { ": bit-idénticos al oráculo" });

    // 2) attn_norm: ≤ 1e-5 con bins prec9, ≤ 1e-4 con legacy.
    let oracle_norm = read_f32_bin(&dir.join("ornith-attn_norm-0.f32.bin"), norm.len())?;
    let (n_max, n_at) = max_abs_diff(&norm, &oracle_norm);
    println!("puerta attn_norm (tolerancia {norm_tol:.1e}):");
    println!("  max |diff| = {n_max:.9} en [{n_at}]");
    if n_max > norm_tol {
        return fail("attn_norm", &format!("max |diff| = {n_max:.9} > {norm_tol}"));
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
        println!("  token {t} (id {tok}) listo en {:.1}s", t0.elapsed().as_secs_f64());
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
