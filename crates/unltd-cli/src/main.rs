//! CLI de unltd. Esqueleto: las opciones reflejan el diseño de `docs/ARCHITECTURE.md`
//! y `docs/MEMORY-DESIGN.md`; el cuerpo llega en el Stage 0 del ROADMAP.
//!
//! Contratos de la CLI (heredados de k3_run.c, ver docs/AUDIT.md):
//! - el banner de memoria es un PLAN que se imprime ANTES de asignar; el PEAK RSS se
//!   imprime al final y es la cifra autoritativa;
//! - negarse con números ("necesita X, hay Y") antes de un OOM-kill;
//! - un experto caído = exit code distinto + "RUN INVALID";
//! - la salida de texto se imprime como bloque, no streameada (un multi-byte cortado
//!   no es UTF-8 válido).

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "unltd", version, about = "Disk-first CPU inference runtime")]
struct Cli {
    /// Modelo: archivo .gguf o directorio con safetensors.
    model: PathBuf,

    /// Presupuesto total del motor (GB). `auto` lo dimensiona de la RAM disponible.
    #[arg(long, default_value = "auto")]
    budget: String,

    /// Presupuesto del tronco (GB); por defecto lo decide el preset.
    #[arg(long)]
    trunk_gb: Option<f64>,

    /// Presupuesto de la cache de expertos (GB).
    #[arg(long)]
    cache_gb: Option<f64>,

    /// Prompt de texto (una de --prompt/--ids, nunca ambas).
    #[arg(long)]
    prompt: Option<String>,

    /// Prompt como ids (canal reproducible de los tests).
    #[arg(long)]
    ids: Option<String>,

    /// Tokens a generar.
    #[arg(long, default_value_t = 8)]
    gen: u32,

    /// Grabar el trace de accesos a expertos para replay offline (sim_cache).
    #[arg(long)]
    dump_cache_trace: Option<PathBuf>,
}

fn main() {
    let _cli = Cli::parse();
    eprintln!(
        "unltd: esqueleto del workspace. El motor arranca en el Stage 0.\n\
         Ver docs/ROADMAP.md para el plan y docs/ARCHITECTURE.md para el diseño."
    );
    std::process::exit(2);
}
