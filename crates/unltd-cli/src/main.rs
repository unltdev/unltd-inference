//! CLI de unltd. Subcomandos por fase del ROADMAP:
//! - `inspect` (Fase 2): header, metadata y tabla de tensores de un GGUF;
//! - `tokenize` / `run` (Fase 5): stubs explícitos, no silenciosos.
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
use unltd_model_loader::GgufReader;

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

    /// Tokeniza texto con el tokenizador del modelo (Fase 5).
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
        Cmd::Tokenize { .. } | Cmd::Run { .. } => {
            eprintln!(
                "unltd: este comando llega en la Fase 5 (ver docs/ROADMAP.md). \
                 Hoy solo existe `inspect` (Fase 2)."
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
