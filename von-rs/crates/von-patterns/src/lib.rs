//! High-level composable decision patterns, parity-ported from
//! `src/von/patterns.py` against the `Engine` trait.

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;
use von_core::{Engine, EngineError, round_half_even};
use von_types::{Answer, Choice, Question, SystemOneResponse};

#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct PatternError(pub String);

impl From<EngineError> for PatternError {
    fn from(e: EngineError) -> Self {
        PatternError(e.0)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceGateOutput {
    pub automatic: IndexMap<String, Answer>,
    pub escalate: IndexMap<String, Answer>,
    pub response: SystemOneResponse,
}

pub fn confidence_gate<E: Engine>(
    engine: &E,
    state: &Value,
    questions: &IndexMap<String, Value>,
    threshold: f64,
) -> Result<ConfidenceGateOutput, PatternError> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PatternError(format!(
            "threshold must be in [0.0, 1.0], got {threshold}"
        )));
    }
    let resp = engine.evaluate(state, questions, None)?;
    let mut automatic = IndexMap::new();
    let mut escalate = IndexMap::new();
    for (q_id, ans) in &resp.answers {
        let conf = ans.confidence();
        if conf >= threshold {
            automatic.insert(q_id.clone(), ans.clone());
        } else {
            escalate.insert(q_id.clone(), ans.clone());
        }
    }
    Ok(ConfidenceGateOutput {
        automatic,
        escalate,
        response: resp,
    })
}

pub type RouteHandlers<'a> = IndexMap<String, Box<dyn Fn(&Answer) -> Value + 'a>>;

pub fn route<E: Engine>(
    engine: &E,
    state: &Value,
    question: &Question,
    routes: &RouteHandlers<'_>,
    default: Option<&dyn Fn(&Answer) -> Value>,
    min_confidence: f64,
) -> Result<Value, PatternError> {
    let Choice {
        instructions,
        criteria,
    } = match question {
        Question::Choice(c) => c,
        _ => {
            return Err(PatternError(
                "question must be an instance of von.Choice".to_string(),
            ));
        }
    };
    let _ = criteria;
    let mut questions = IndexMap::new();
    questions.insert(
        "route_question".to_string(),
        serde_json::json!({
            "type": "choice",
            "instructions": instructions,
            "criteria": serde_json::Value::Object(
                criteria.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect(),
            ),
        }),
    );
    let resp = engine.evaluate(state, &questions, None)?;
    let Some(
        ans @ Answer::Choice {
            choice: chosen_id,
            confidence,
            ..
        },
    ) = resp.get("route_question")
    else {
        return Err(PatternError("missing route_question answer".to_string()));
    };
    let chosen_id = chosen_id.clone();
    let confidence = *confidence;
    let ans = ans.clone();

    let handler = routes.get(&chosen_id);
    match handler {
        Some(h) if confidence >= min_confidence => Ok(h(&ans)),
        _ => match default {
            Some(d) => Ok(d(&ans)),
            None => Ok(serde_json::to_value(&ans).expect("answer serialization cannot fail")),
        },
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BreakdownEntry {
    pub raw: Value,
    pub normalized: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompositeScoreOutput {
    pub score: f64,
    pub breakdown: IndexMap<String, BreakdownEntry>,
    pub response: SystemOneResponse,
}

pub fn composite_score<E: Engine>(
    engine: &E,
    state: &Value,
    questions: &IndexMap<String, Value>,
    weights: Option<&IndexMap<String, f64>>,
    normalize: bool,
) -> Result<CompositeScoreOutput, PatternError> {
    let resp = engine.evaluate(state, questions, None)?;
    let w_map = weights.cloned().unwrap_or_default();

    let mut total_weighted_sum = 0.0f64;
    let mut total_weights = 0.0f64;
    let mut breakdown = IndexMap::new();

    for (q_id, ans) in &resp.answers {
        let normalized_val: Option<f64> = match ans {
            Answer::Score { score, legend, .. } => {
                let max_level = std::cmp::max(1, legend.len() as i64 - 1) as f64;
                Some(score / max_level)
            }
            Answer::Noul { noul } => Some(*noul),
            Answer::Choice { .. } => None,
        };
        if let Some(nv) = normalized_val {
            let w = w_map.get(q_id).copied().unwrap_or(1.0);
            total_weighted_sum += nv * w;
            total_weights += w;
            let raw = match ans {
                Answer::Score { score, .. } => serde_json::json!(score),
                Answer::Noul { noul } => serde_json::json!(noul),
                Answer::Choice { .. } => serde_json::to_value(ans).unwrap(),
            };
            breakdown.insert(
                q_id.clone(),
                BreakdownEntry {
                    raw,
                    normalized: round_half_even(nv, 4),
                    weight: w,
                },
            );
        }
    }

    let final_score = if normalize && total_weights > 0.0 {
        total_weighted_sum / total_weights
    } else {
        total_weighted_sum
    };

    Ok(CompositeScoreOutput {
        score: round_half_even(final_score, 4),
        breakdown,
        response: resp,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct TwoStageOutput {
    pub category: String,
    pub category_confidence: f64,
    pub choice: String,
    pub choice_confidence: f64,
    pub combined_confidence: f64,
}

pub fn two_stage_choice<E: Engine>(
    engine: &E,
    state: &Value,
    taxonomy: &IndexMap<String, IndexMap<String, Option<String>>>,
    instructions_category: &str,
    instructions_option: &str,
) -> Result<TwoStageOutput, PatternError> {
    let cat_criteria: serde_json::Map<String, Value> = taxonomy
        .keys()
        .map(|cat| {
            (
                cat.clone(),
                serde_json::json!(format!("Category for {cat} operations and topics")),
            )
        })
        .collect();
    let mut questions = IndexMap::new();
    questions.insert(
        "category".to_string(),
        serde_json::json!({
            "type": "choice",
            "instructions": instructions_category,
            "criteria": Value::Object(cat_criteria),
        }),
    );
    let cat_resp = engine.evaluate(state, &questions, None)?;
    let Some(Answer::Choice {
        choice: top_cat,
        confidence: cat_conf,
        ..
    }) = cat_resp.get("category")
    else {
        return Err(PatternError("missing category answer".to_string()));
    };
    let top_cat = top_cat.clone();
    let cat_conf = *cat_conf;

    let sub_criteria: serde_json::Map<String, Value> = taxonomy
        .get(&top_cat)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, serde_json::json!(v)))
        .collect();
    let opt_instructions = instructions_option.replace("{category}", &top_cat);
    let mut questions = IndexMap::new();
    questions.insert(
        "option".to_string(),
        serde_json::json!({
            "type": "choice",
            "instructions": opt_instructions,
            "criteria": Value::Object(sub_criteria),
        }),
    );
    let opt_resp = engine.evaluate(state, &questions, None)?;
    let Some(Answer::Choice {
        choice: chosen_opt,
        confidence: opt_conf,
        ..
    }) = opt_resp.get("option")
    else {
        return Err(PatternError("missing option answer".to_string()));
    };
    let chosen_opt = chosen_opt.clone();
    let opt_conf = *opt_conf;

    Ok(TwoStageOutput {
        category: top_cat,
        category_confidence: cat_conf,
        choice: chosen_opt,
        choice_confidence: opt_conf,
        combined_confidence: round_half_even(cat_conf * opt_conf, 4),
    })
}
