//! The real decision engine: ONNX graph execution (`VonSession`) plus an
//! operation-for-operation port of the Python backend's decision math
//! (`option_marker_backend.py`): calibration-map temperatures, the zero-shot
//! noul debias dual pass, fp32 softmax and the API rounding envelope.

use std::path::Path;
use std::sync::Mutex;

use indexmap::IndexMap;
use tokenizers::Tokenizer;
use von_core::{Answer, EngineError, QuestionBackend, round_half_even};

use crate::session::VonSession;
use crate::temperature::{Calibration, NoulPrior};
use crate::{VON_SPECIAL, load_tokenizer};
use von_core::pack_sequence;

const NOUL_POS_DEFAULT: &str = "Yes, condition holds true.";
const NOUL_NEG_DEFAULT: &str = "No, condition is false.";

pub struct OrtEngine {
    session: VonSession,
    tokenizer: Tokenizer,
    calib: Calibration,
    device: crate::session::Device,
    /// Runtime override of the fitted zero-shot noul prior (the goldens'
    /// `prior_*`/`probe_only` rows inject the synthetic prior exactly the way
    /// `capture_golden.py` pokes `backend._noul_prior`).
    noul_prior_override: Mutex<Option<NoulPrior>>,
}

impl OrtEngine {
    pub fn from_artifacts(onnx: &Path, snapshot: &Path) -> Result<Self, EngineError> {
        Self::from_artifacts_with_device(onnx, snapshot, crate::session::Device::Cpu)
    }

    pub fn from_artifacts_with_device(
        onnx: &Path,
        snapshot: &Path,
        device: crate::session::Device,
    ) -> Result<Self, EngineError> {
        let session = VonSession::from_file_with_device(onnx, device)?;
        let tokenizer = load_tokenizer(snapshot)?;
        let calib = Calibration::load(snapshot)?;
        Ok(OrtEngine {
            session,
            tokenizer,
            calib,
            device,
            noul_prior_override: Mutex::new(None),
        })
    }

    pub fn device(&self) -> crate::session::Device {
        self.device
    }

    pub fn effective_noul_prior(&self) -> Option<NoulPrior> {
        self.noul_prior_override
            .lock()
            .expect("noul prior lock poisoned")
            .or(self.calib.noul_prior)
    }

    pub fn set_noul_prior(&self, prior: Option<NoulPrior>) {
        *self
            .noul_prior_override
            .lock()
            .expect("noul prior lock poisoned") = prior;
    }

    /// One forward pass over a packed sequence: encode, find `[MASK]`
    /// positions, run the graph. Returns one logit per option (f64 copies of
    /// the f32 graph outputs, like `Tensor.tolist()`).
    fn forward_logits(&self, packed_text: &str) -> Result<Vec<f64>, EngineError> {
        let enc = self
            .tokenizer
            .encode(packed_text, true)
            .map_err(|e| EngineError(format!("tokenization failed: {e}")))?;
        let mask_positions: Vec<usize> = enc
            .get_ids()
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == VON_SPECIAL.mask)
            .map(|(i, _)| i)
            .collect();
        self.session
            .forward_with_masks(enc.get_ids(), &mask_positions)
            .map(|f| f.logits)
    }

    /// Port of `_effective_temperature`: override -> scalar default -> the
    /// input-conditioned map, with the entropy feature computed in fp32
    /// (`torch.softmax(logits.float())`) and the `log_tokens` feature counted
    /// by the real tokenizer with `add_special_tokens=False`.
    fn effective_temperature(
        &self,
        logits: &[f64],
        state_text: &str,
        n_options: usize,
        override_: Option<f64>,
    ) -> f64 {
        if let Some(t) = override_ {
            return t;
        }
        let Some(map) = &self.calib.map else {
            return self.calib.default_temp;
        };

        let probs = softmax_f32(logits);
        let n = probs.len().max(1);
        let ent = if n > 1 {
            let mut sum = 0f32;
            for p in &probs {
                sum += p * p.max(1e-12).ln();
            }
            f64::from(-sum) / (n as f64).ln()
        } else {
            0.0
        };

        let tokens = self
            .tokenizer
            .encode(state_text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
            .max(1);
        let feats = [
            (map.bias, 1.0),
            (map.entropy, ent),
            (map.log_tokens, (tokens as f64).log10() / 4.0),
            (map.n_options, n_options as f64 / 8.0),
        ];
        let raw = feats
            .iter()
            .fold(0.0, |acc, (param, feat)| acc + param * feat);
        map.hi.min(map.lo.max(raw))
    }
}

/// fp32 softmax, matching `torch.softmax` on a float32 tensor: subtract the
/// max, exponentiate, normalize — then widen to f64 exactly like
/// `.cpu().tolist()`.
fn softmax_f32(logits: &[f64]) -> Vec<f32> {
    let xs: Vec<f32> = logits.iter().map(|l| *l as f32).collect();
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Temperature-scaled fp32 probabilities for the API fields.
fn scaled_probs(logits: &[f64], eff_temp: f64) -> Vec<f64> {
    let t = eff_temp.max(1e-4) as f32;
    let scaled: Vec<f64> = logits.iter().map(|l| f64::from(*l as f32 / t)).collect();
    softmax_f32(&scaled).into_iter().map(f64::from).collect()
}

/// `torch.argmax` contract: the FIRST maximal index (an explicit strict `>`
/// fold; Rust's `max_by` keeps the last max).
fn argmax_first(logits: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate().skip(1) {
        if *v > logits[best] {
            best = i;
        }
    }
    best
}

/// Top1 − top2 of the *unrounded* probabilities, clamped to [0, 1].
fn top_gap_confidence(probs: &[f64]) -> f64 {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = sorted.first().copied().unwrap_or(0.0);
    let second = sorted.get(1).copied().unwrap_or(0.0);
    round_half_even((top - second).clamp(0.0, 1.0), 3)
}

impl QuestionBackend for OrtEngine {
    fn eval_choice(&self, state_text: &str, q: &von_types::Choice) -> Result<Answer, EngineError> {
        let options: Vec<&String> = q.criteria.keys().collect();
        if options.is_empty() {
            return Ok(Answer::Choice {
                choice: String::new(),
                probabilities: IndexMap::new(),
                confidence: 0.0,
            });
        }

        // `desc.strip() if desc else opt.strip()`: a null or empty criteria
        // value falls back to the option name.
        let descriptions: Vec<String> = options
            .iter()
            .map(|opt| {
                q.criteria
                    .get(*opt)
                    .and_then(|d| d.as_deref())
                    .filter(|d| !d.is_empty())
                    .map(str::trim)
                    .map(str::to_string)
                    .unwrap_or_else(|| opt.trim().to_string())
            })
            .collect();
        let option_refs: Vec<&str> = descriptions.iter().map(String::as_str).collect();

        let packed = pack_sequence(state_text, &q.instructions, &option_refs);
        let logits = self.forward_logits(&packed)?;
        let eff_temp = self.effective_temperature(&logits, state_text, options.len(), None);
        let probs = scaled_probs(&logits, eff_temp);

        let best_idx = argmax_first(&logits);
        let mut probabilities = IndexMap::new();
        for (opt, p) in options.iter().zip(&probs) {
            probabilities.insert((*opt).clone(), round_half_even(*p, 4));
        }
        // A literal "[MASK]" in the packed text creates phantom option
        // positions; if the argmax lands on one, Python's `options[best_idx]`
        // raises IndexError and the server maps str(exc) to a 422 detail.
        // Reproduce that contract instead of panicking on the slice index.
        let choice = options
            .get(best_idx)
            .ok_or_else(|| EngineError("list index out of range".to_string()))?;
        Ok(Answer::Choice {
            choice: (*choice).clone(),
            probabilities,
            confidence: top_gap_confidence(&probs),
        })
    }

    fn eval_noul(&self, state_text: &str, q: &von_types::Noul) -> Result<Answer, EngineError> {
        let crit = q.criteria.as_ref();
        let pos_raw = crit.and_then(|m| m.get("true")).map(String::as_str);
        let neg_raw = crit.and_then(|m| m.get("false")).map(String::as_str);
        // Python truthiness: an empty string criterion counts as absent.
        let has_explicit =
            pos_raw.is_some_and(|s| !s.is_empty()) || neg_raw.is_some_and(|s| !s.is_empty());
        let pos_desc = match pos_raw {
            Some(s) if !s.is_empty() => s,
            _ => NOUL_POS_DEFAULT,
        };
        let neg_desc = match neg_raw {
            Some(s) if !s.is_empty() => s,
            _ => NOUL_NEG_DEFAULT,
        };
        let descriptions = [pos_desc, neg_desc];

        let packed = pack_sequence(state_text, &q.instructions, &descriptions);
        let mut logits = self.forward_logits(&packed)?;

        // Zero-shot debias: a second forward pass on an empty state cancels
        // the intrinsic positive-polarity prior. The backend keeps this
        // arithmetic entirely in fp32 tensor scalars (Python floats
        // weak-promote), so `a * bias + b` computes in f32 here too.
        if !has_explicit {
            let null_packed = pack_sequence("", &q.instructions, &descriptions);
            let null_logits = self.forward_logits(&null_packed)?;
            let bias = null_logits[0] as f32 - null_logits[1] as f32;
            let correction: f32 = match self.effective_noul_prior() {
                Some(prior) => prior.a as f32 * bias + prior.b as f32,
                None => 0.7f32 * bias,
            };
            logits[0] = f64::from(logits[0] as f32 - correction);
            // py: `logits = torch.stack([logits[0] - correction, logits[1]])`
            // rebuilds the tensor with exactly two entries, dropping phantom
            // "[MASK]" positions from the softmax denominator.
            logits.truncate(2);
        }

        let eff_temp = self.effective_temperature(&logits, state_text, 2, None);
        let probs = scaled_probs(&logits, eff_temp);
        let prob_true = round_half_even(probs[0].clamp(0.0, 1.0), 4);
        Ok(Answer::Noul { noul: prob_true })
    }

    fn eval_score(&self, state_text: &str, q: &von_types::Score) -> Result<Answer, EngineError> {
        if q.criteria.is_empty() {
            return Ok(Answer::Score {
                score: 0.0,
                confidence: 0.0,
                legend: IndexMap::new(),
                probabilities: IndexMap::new(),
            });
        }

        let mut legend = IndexMap::new();
        let mut descriptions = Vec::with_capacity(q.criteria.len());
        for (i, item) in q.criteria.iter().enumerate() {
            let desc = von_core::score_level_description(item)?;
            legend.insert(i.to_string(), desc.clone());
            descriptions.push(desc);
        }
        let option_refs: Vec<&str> = descriptions.iter().map(String::as_str).collect();

        let packed = pack_sequence(state_text, &q.instructions, &option_refs);
        let logits = self.forward_logits(&packed)?;
        let eff_temp = self.effective_temperature(&logits, state_text, descriptions.len(), None);
        let probs = scaled_probs(&logits, eff_temp);

        // `sum(i * p for i, p in enumerate(probs))` — f64 over the widened
        // probabilities.
        let weighted: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        let mut probabilities = IndexMap::new();
        for (i, p) in probs.iter().enumerate() {
            probabilities.insert(i.to_string(), round_half_even(*p, 4));
        }
        Ok(Answer::Score {
            score: round_half_even(weighted, 2),
            confidence: top_gap_confidence(&probs),
            legend,
            probabilities,
        })
    }
}
