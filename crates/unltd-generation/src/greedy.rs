//! Bucle greedy determinista (Fase 8): temperatura 0, sin sampling.
//!
//! Contrato (heredado de la Fase 6 y de docs/AUDIT.md):
//! - **Determinismo**: el mismo prompt + motor + pesos producen la MISMA
//!   secuencia, corrida tras corrida. La estabilidad del argmax es la puerta
//!   aceptada contra el oráculo (la divergencia F32/AVX2 de valores internos
//!   está documentada en docs/PHASE-6-CHECKPOINT.md y NO se re-audita).
//! - **EOS detiene** el bucle (el EOS queda incluido en la salida, como en
//!   llama.cpp), `max_tokens` es el tope.
//! - El bucle no conoce el motor: recibe un `forward` que procesa UN token
//!   (embed + step + output_norm + output_logits) y deja los logits en `out`.
//!   El CLI lo alimenta con `Qwen35Forward`; los tests, con un modelo sintético.

use unltd_core::LoadError;

/// Índice del máximo de `v`; en caso de empate, el PRIMERO (determinista).
/// `v` no puede estar vacío (lo garantiza `GreedyLoop`, que exige prefill ≥ 1).
pub fn argmax(v: &[f32]) -> usize {
    assert!(!v.is_empty(), "argmax: logits vacíos");
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Estado de un bucle greedy: prefill token a token y generación
/// `next_token` por paso. Mantiene los logits del último token procesado
/// (útil para tablas y top-5 del CLI).
pub struct GreedyLoop<F> {
    forward: F,
    eos: Option<u32>,
    remaining: u32,
    logits: Vec<f32>,
    /// true si el bucle terminó por EOS (falso si terminó por max_tokens).
    pub stopped_by_eos: bool,
}

impl<F: FnMut(u32, &mut Vec<f32>) -> Result<(), LoadError>> GreedyLoop<F> {
    /// `forward` procesa un token y deja sus logits en el segundo argumento.
    pub fn new(forward: F, eos: Option<u32>, max_tokens: u32) -> Self {
        Self { forward, eos, remaining: max_tokens, logits: Vec::new(), stopped_by_eos: false }
    }

    /// Prefill: alimenta el prompt token a token (el KV cache incremental del
    /// motor acumula el estado). Los logits finales quedan en `self.logits`.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<(), LoadError> {
        if prompt.is_empty() {
            return Err(LoadError::corrupt(
                "greedy: prompt vacío — no hay logits para arrancar el decode",
            ));
        }
        for &t in prompt {
            (self.forward)(t, &mut self.logits)?;
        }
        Ok(())
    }

    /// Próximo token greedy, o `None` si el bucle terminó (EOS o max_tokens).
    pub fn next_token(&mut self) -> Result<Option<u32>, LoadError> {
        if self.stopped_by_eos || self.remaining == 0 {
            return Ok(None);
        }
        let next = argmax(&self.logits) as u32;
        self.remaining -= 1;
        if Some(next) == self.eos {
            self.stopped_by_eos = true;
            return Ok(Some(next));
        }
        // Sin token siguiente (límite alcanzado), no se re-envía el forward:
        // el motor no calcula logits que nadie va a leer.
        if self.remaining == 0 {
            return Ok(Some(next));
        }
        (self.forward)(next, &mut self.logits)?;
        Ok(Some(next))
    }

    /// Logits del último token procesado (prompt o generado).
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Modelo sintético: vocab = 5, el token siguiente es `(t + 1) % 5`.
    /// `seen` registra el orden exacto de tokens que recibe el forward
    /// (prefill + generación) — es el test del contrato de orden.
    fn next_mod5(seen: &mut Vec<u32>) -> impl FnMut(u32, &mut Vec<f32>) -> Result<(), LoadError> + '_ {
        move |t, out| {
            seen.push(t);
            let next = (t + 1) % 5;
            *out = vec![0.0f32; 5];
            out[next as usize] = 1.0;
            Ok(())
        }
    }

    #[test]
    fn argmax_first_max_on_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, 0.0, -0.5]), 1);
        assert_eq!(argmax(&[7.5]), 0);
    }

    /// Append + stop por EOS: el EOS queda incluido y detiene el bucle.
    #[test]
    fn stops_at_eos() {
        let mut seen = Vec::new();
        let out = {
            let mut g = GreedyLoop::new(next_mod5(&mut seen), Some(0), 10);
            g.prefill(&[1, 2]).expect("prefill");
            let mut out = Vec::new();
            while let Some(t) = g.next_token().expect("paso") {
                out.push(t);
            }
            assert!(g.stopped_by_eos);
            out
        };
        assert_eq!(out, vec![3, 4, 0]);
        // forward recibió: prompt [1, 2] y generados [3, 4] (0 = EOS no se re-envía).
        assert_eq!(seen, vec![1, 2, 3, 4]);
    }

    /// Sin EOS: el bucle para por max_tokens, con la secuencia completa.
    #[test]
    fn stops_at_max_tokens() {
        let mut seen = Vec::new();
        let out = {
            let mut g = GreedyLoop::new(next_mod5(&mut seen), None, 7);
            g.prefill(&[4]).expect("prefill");
            let mut out = Vec::new();
            while let Some(t) = g.next_token().expect("paso") {
                out.push(t);
            }
            assert!(!g.stopped_by_eos);
            out
        };
        assert_eq!(out, vec![0, 1, 2, 3, 4, 0, 1]);
        // El 7º generado NO se re-envía (límite alcanzado, no hay token siguiente):
        // 6 forwards de generación en total.
        assert_eq!(seen, vec![4, 0, 1, 2, 3, 4, 0]);
    }

    /// Determinismo: dos corridas con el mismo modelo producen lo mismo.
    #[test]
    fn deterministic_generation() {
        let run = || {
            let mut seen = Vec::new();
            let mut g = GreedyLoop::new(next_mod5(&mut seen), None, 12);
            g.prefill(&[2, 3]).expect("prefill");
            let mut out = Vec::new();
            while let Some(t) = g.next_token().expect("paso") {
                out.push(t);
            }
            out
        };
        assert_eq!(run(), run());
    }

    /// max_tokens = 0: no genera nada (y no toca el forward).
    #[test]
    fn zero_max_tokens() {
        let mut seen = Vec::new();
        let mut g = GreedyLoop::new(next_mod5(&mut seen), None, 0);
        g.prefill(&[1]).expect("prefill");
        assert_eq!(g.next_token().expect("paso"), None);
        assert!(!g.stopped_by_eos);
    }

    /// Prompt vacío: negativa, no un argmax sobre logits vacíos.
    #[test]
    fn refuses_empty_prompt() {
        let mut seen = Vec::new();
        let mut g = GreedyLoop::new(next_mod5(&mut seen), None, 5);
        assert!(g.prefill(&[]).is_err());
    }

    /// `logits()` expone los logits del último token (el contrato del CLI
    /// para el top-5): tras prefill [1, 2] el argmax debe ser 3.
    #[test]
    fn logits_visible_after_prefill() {
        let mut seen = Vec::new();
        let mut g = GreedyLoop::new(next_mod5(&mut seen), None, 1);
        g.prefill(&[1, 2]).expect("prefill");
        assert_eq!(argmax(g.logits()), 3);
    }
}
