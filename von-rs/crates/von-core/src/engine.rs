//! Engine orchestrator: aliases, model stamping, the `Engine` trait and the
//! Stage 1 stub engine (structure-faithful, uniform-probability outputs).

use indexmap::IndexMap;
use serde_json::Value;
use von_types::{Answer, SystemOneResponse, parse_question};

use crate::fmt::format_state;
use crate::rounding::round_half_even;
use crate::usage::compute_usage;

pub const VON_VERSION: &str = "1.1";

pub const VON_CURRENT_ALIASES: [&str; 6] =
    ["von-1.1", "1.1", "von", "default", "latest", "von-latest"];

pub const VON_MODEL_ID: &str = "von-1.1.0";

pub fn is_supported_alias(name: &str) -> bool {
    let normalized = name.trim().to_lowercase();
    VON_CURRENT_ALIASES.contains(&normalized.as_str())
}

pub fn resolved_model_id() -> String {
    format!("von-{VON_VERSION}.0")
}

pub fn unknown_model_message(name: &str) -> String {
    let mut aliases: Vec<&str> = VON_CURRENT_ALIASES.to_vec();
    aliases.sort_unstable();
    format!(
        "Unknown model '{name}'. Von {VON_VERSION} is the only model; accepted aliases: {}.",
        aliases.join(", ")
    )
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct EngineError(pub String);

pub trait Engine {
    fn evaluate(
        &self,
        state: &Value,
        questions: &IndexMap<String, Value>,
        model: Option<&str>,
    ) -> Result<SystemOneResponse, EngineError>;
}

fn resolve_model(model: Option<&str>) -> String {
    match model {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => VON_MODEL_ID.to_string(),
    }
}

pub struct StubEngine;

impl Engine for StubEngine {
    fn evaluate(
        &self,
        state: &Value,
        questions: &IndexMap<String, Value>,
        model: Option<&str>,
    ) -> Result<SystemOneResponse, EngineError> {
        let state_str = format_state(state);
        let mut answers = IndexMap::new();
        let mut total_q_chars = 0usize;

        for (q_id, q_data) in questions {
            let mut warnings = Vec::new();
            let q = parse_question(q_data, &mut warnings).map_err(|e| EngineError(e.detail()))?;
            for w in warnings {
                eprintln!("[von] warning: {w}");
            }

            let answer = match &q {
                von_types::Question::Choice(qc) => evaluate_choice_stub(qc)?,
                von_types::Question::Noul(qn) => evaluate_noul_stub(qn),
                von_types::Question::Score(qs) => evaluate_score_stub(qs)?,
            };
            total_q_chars += q.instructions().chars().count();
            answers.insert(q_id.clone(), answer);
        }

        let usage = compute_usage(&state_str, total_q_chars, answers.len());
        Ok(SystemOneResponse {
            model: resolve_model(model),
            answers,
            usage,
        })
    }
}

fn uniform_probs(n: usize, ndigits: u32) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let p = 1.0 / n as f64;
    vec![round_half_even(p, ndigits); n]
}

fn top_gap_confidence(probs: &[f64]) -> f64 {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = sorted.first().copied().unwrap_or(0.0);
    let second = sorted.get(1).copied().unwrap_or(0.0);
    round_half_even((top - second).clamp(0.0, 1.0), 3)
}

fn evaluate_choice_stub(q: &von_types::Choice) -> Result<Answer, EngineError> {
    let options: Vec<&String> = q.criteria.keys().collect();
    if options.is_empty() {
        return Ok(Answer::Choice {
            choice: String::new(),
            probabilities: IndexMap::new(),
            confidence: 0.0,
        });
    }
    let probs = uniform_probs(options.len(), 4);
    let mut probabilities = IndexMap::new();
    for (opt, p) in options.iter().zip(&probs) {
        probabilities.insert((*opt).clone(), *p);
    }
    Ok(Answer::Choice {
        choice: options[0].clone(),
        probabilities,
        confidence: top_gap_confidence(&probs),
    })
}

fn evaluate_noul_stub(q: &von_types::Noul) -> Answer {
    let _ = q;
    Answer::Noul {
        noul: round_half_even(0.5f64.clamp(0.0, 1.0), 4),
    }
}

fn python_join_error_message(items: &[Value]) -> Option<String> {
    for (i, item) in items.iter().enumerate() {
        if !item.is_string() {
            let py_type = match item {
                Value::Null => "NoneType",
                Value::Bool(_) => "bool",
                Value::Number(n) if n.is_i64() || n.is_u64() => "int",
                Value::Number(_) => "float",
                Value::Array(_) => "list",
                Value::Object(_) => "dict",
                Value::String(_) => unreachable!(),
            };
            return Some(format!(
                "sequence item {i}: expected str instance, {py_type} found"
            ));
        }
    }
    None
}

fn score_level_description(item: &von_types::ScoreCriterion) -> Result<String, EngineError> {
    match item {
        von_types::ScoreCriterion::Text(text) => Ok(text.trim().to_string()),
        von_types::ScoreCriterion::Map(map) => {
            let what = match map.get("what") {
                Some(v) => von_types::py_str(v),
                None => String::new(),
            };
            let ex_str = match map.get("examples") {
                None => String::new(),
                Some(Value::Null) => String::new(),
                Some(Value::String(s)) if s.is_empty() => String::new(),
                Some(Value::String(s)) => {
                    let joined: Vec<String> = s.chars().map(|c| c.to_string()).collect();
                    format!(" Examples: {}", joined.join(", "))
                }
                Some(Value::Array(items)) => {
                    if items.is_empty() {
                        String::new()
                    } else if let Some(msg) = python_join_error_message(items) {
                        return Err(EngineError(msg));
                    } else {
                        let joined: Vec<String> = items
                            .iter()
                            .map(|v| v.as_str().unwrap_or_default().to_string())
                            .collect();
                        format!(" Examples: {}", joined.join(", "))
                    }
                }
                Some(other) => {
                    return Err(EngineError(format!(
                        "'{}' object is not iterable",
                        von_types::type_name(other)
                    )));
                }
            };
            Ok(format!("{what}{ex_str}").trim().to_string())
        }
    }
}

fn evaluate_score_stub(q: &von_types::Score) -> Result<Answer, EngineError> {
    if q.criteria.is_empty() {
        return Ok(Answer::Score {
            score: 0.0,
            confidence: 0.0,
            legend: IndexMap::new(),
            probabilities: IndexMap::new(),
        });
    }
    let mut legend = IndexMap::new();
    for (i, item) in q.criteria.iter().enumerate() {
        legend.insert(i.to_string(), score_level_description(item)?);
    }
    let probs = uniform_probs(q.criteria.len(), 4);
    let weighted: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
    let mut probabilities = IndexMap::new();
    for (i, p) in probs.iter().enumerate() {
        probabilities.insert(i.to_string(), *p);
    }
    Ok(Answer::Score {
        score: round_half_even(weighted, 2),
        confidence: top_gap_confidence(&probs),
        legend,
        probabilities,
    })
}

pub fn decide<E: Engine>(
    engine: &E,
    state: &Value,
    choices: &[String],
    instructions: &str,
    model: Option<&str>,
) -> Result<Answer, EngineError> {
    let mut seen = std::collections::HashSet::new();
    let has_dup = choices.iter().any(|c| !seen.insert(c));
    if has_dup {
        let listed = von_types::py_repr(&Value::Array(
            choices.iter().map(|c| Value::String(c.clone())).collect(),
        ));
        return Err(EngineError(format!(
            "Duplicate choices found in options list: {listed}"
        )));
    }
    let criteria: serde_json::Map<String, Value> =
        choices.iter().map(|c| (c.clone(), Value::Null)).collect();
    let question = serde_json::json!({
        "type": "choice",
        "instructions": instructions,
        "criteria": Value::Object(criteria),
    });
    let mut questions = IndexMap::new();
    questions.insert("decision".to_string(), question);
    let resp = engine.evaluate(state, &questions, model)?;
    resp.get("decision")
        .cloned()
        .ok_or_else(|| EngineError("missing decision answer".to_string()))
}

pub fn judge<E: Engine>(
    engine: &E,
    state: &Value,
    instructions: &str,
    criteria: Option<IndexMap<String, String>>,
    model: Option<&str>,
) -> Result<f64, EngineError> {
    let criteria_value = match criteria {
        Some(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect(),
        ),
        None => Value::Null,
    };
    let question = serde_json::json!({
        "type": "noul",
        "instructions": instructions,
        "criteria": criteria_value,
    });
    let mut questions = IndexMap::new();
    questions.insert("judgment".to_string(), question);
    let resp = engine.evaluate(state, &questions, model)?;
    match resp.get("judgment") {
        Some(Answer::Noul { noul }) => Ok(*noul),
        _ => Ok(0.0),
    }
}

pub fn rate<E: Engine>(
    engine: &E,
    state: &Value,
    criteria: Vec<Value>,
    instructions: &str,
    model: Option<&str>,
) -> Result<Answer, EngineError> {
    let question = serde_json::json!({
        "type": "score",
        "instructions": instructions,
        "criteria": Value::Array(criteria),
    });
    let mut questions = IndexMap::new();
    questions.insert("rating".to_string(), question);
    let resp = engine.evaluate(state, &questions, model)?;
    resp.get("rating")
        .cloned()
        .ok_or_else(|| EngineError("missing rating answer".to_string()))
}
