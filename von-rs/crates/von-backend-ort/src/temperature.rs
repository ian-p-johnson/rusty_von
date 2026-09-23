//! Calibration state: scalar temperature, the input-conditioned calibration
//! map, and the fitted zero-shot noul prior — ported with the exact validation
//! predicates of `option_marker_backend.py` (§3.3): malformed values drop
//! wholesale to safe fallbacks, out-of-sane-range bounds warn at load.

use std::path::Path;

use serde_json::Value;
use von_core::EngineError;

pub const TEMP_SANITY_MAX: f64 = 50.0;

/// Python `%g`-style rendering for the load warning (integer values render
/// without a fractional part); stderr-only, not a wire artifact.
fn py_g(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

#[derive(Debug, Clone)]
pub struct Calibration {
    pub default_temp: f64,
    pub map: Option<CalibMap>,
    pub noul_prior: Option<NoulPrior>,
}

#[derive(Debug, Clone)]
pub struct CalibMap {
    pub bias: f64,
    pub entropy: f64,
    pub log_tokens: f64,
    pub n_options: f64,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NoulPrior {
    pub a: f64,
    pub b: f64,
}

/// Python `float(value)` over the JSON domain: numbers pass through, bools
/// coerce to 1.0/0.0, numeric strings parse (Python also accepts its own
/// spellings like `"1_0.5"`, which we deliberately do not — the calibration
/// file is machine-written), anything else fails coercion.
fn python_float(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Port of `_validate_calibration_map`: coerce every entry to f64; any
/// non-numeric entry drops the whole map; the map must carry at least one
/// feature coefficient; `lo`/`hi` default to 0.5/12.0; `lo > hi` drops it.
pub fn validate_calibration_map(raw: Option<&Value>) -> Option<CalibMap> {
    let raw = raw?;
    let obj = raw.as_object()?;
    let mut out = std::collections::HashMap::<String, f64>::new();
    for (k, v) in obj {
        out.insert(k.clone(), python_float(v)?);
    }
    if !["bias", "entropy", "log_tokens", "n_options"]
        .iter()
        .any(|k| out.contains_key(*k))
    {
        return None;
    }
    let lo = out.get("lo").copied().unwrap_or(0.5);
    let hi = out.get("hi").copied().unwrap_or(12.0);
    if lo > hi {
        return None;
    }
    if lo <= 0.0 || hi > TEMP_SANITY_MAX {
        eprintln!(
            "[von] warning: calibration map bounds [{}, {}] are outside the sane range (0, {}]; confidences may be distorted.",
            py_g(lo),
            py_g(hi),
            py_g(TEMP_SANITY_MAX)
        );
    }
    Some(CalibMap {
        bias: out.get("bias").copied().unwrap_or(0.0),
        entropy: out.get("entropy").copied().unwrap_or(0.0),
        log_tokens: out.get("log_tokens").copied().unwrap_or(0.0),
        n_options: out.get("n_options").copied().unwrap_or(0.0),
        lo,
        hi,
    })
}

/// Port of `_validate_noul_prior`: absent or malformed always falls back to
/// the hardcoded 0.7*prior branch.
pub fn validate_noul_prior(raw: Option<&Value>) -> Option<NoulPrior> {
    let raw = raw?;
    let obj = raw.as_object()?;
    let a = python_float(obj.get("a")?)?;
    let b = python_float(obj.get("b")?)?;
    Some(NoulPrior { a, b })
}

impl Calibration {
    /// Load `marker_calibration.json` with the backend's all-or-nothing
    /// fallback semantics: any read/parse failure leaves the defaults.
    pub fn load(snapshot: &Path) -> Result<Self, EngineError> {
        let path = snapshot.join("marker_calibration.json");
        let data: Value = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).map_err(|e| {
                EngineError(format!("marker_calibration.json parse failed: {e}"))
            })?,
            Err(e) => {
                return Err(EngineError(format!(
                    "marker_calibration.json missing ({}): {e}",
                    path.display()
                )))
            }
        };
        let default_temp = data
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        Ok(Calibration {
            default_temp,
            map: validate_calibration_map(data.get("calibration_map")),
            noul_prior: validate_noul_prior(data.get("noul_zero_shot_prior")),
        })
    }
}
