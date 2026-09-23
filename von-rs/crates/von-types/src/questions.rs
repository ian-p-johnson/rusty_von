//! Question schemas mirroring `von/types.py`, including the legacy
//! `pos_criteria`/`neg_criteria` fold-with-warning validator.

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::pyerror::{PydanticError, ValidationEntry};

pub const LEGACY_NOUL_CRITERIA: [(&str, &str); 2] =
    [("pos_criteria", "true"), ("neg_criteria", "false")];

pub const LEGACY_DEPRECATION_WARNING: &str =
    "Noul pos_criteria/neg_criteria are deprecated; use criteria={'true': ..., 'false': ...}.";

#[derive(Debug, Clone, PartialEq)]
pub struct Noul {
    pub instructions: String,
    pub criteria: Option<IndexMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub instructions: String,
    pub criteria: IndexMap<String, Option<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScoreCriterion {
    Text(String),
    Map(Map<String, Value>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub instructions: String,
    pub criteria: Vec<ScoreCriterion>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    Noul(Noul),
    Choice(Choice),
    Score(Score),
}

impl Question {
    pub fn instructions(&self) -> &str {
        match self {
            Question::Noul(q) => &q.instructions,
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
        }
    }
}

#[derive(Debug, Clone)]
pub enum QuestionError {
    Value(String),
    Validation(PydanticError),
}

impl QuestionError {
    pub fn detail(&self) -> String {
        match self {
            QuestionError::Value(msg) => msg.clone(),
            QuestionError::Validation(err) => err.render(),
        }
    }
}

fn json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn get_str_field(
    raw: &Map<String, Value>,
    field: &str,
    errors: &mut Vec<ValidationEntry>,
    input_repr: &Value,
) -> Option<String> {
    match raw.get(field) {
        None => {
            errors.push(ValidationEntry::missing(
                vec![field.to_string()],
                input_repr.clone(),
            ));
            None
        }
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            errors.push(ValidationEntry::string_type(
                vec![field.to_string()],
                other.clone(),
            ));
            None
        }
    }
}

fn fold_legacy_criteria(
    raw: &mut Map<String, Value>,
    warnings: &mut Vec<String>,
) -> Result<(), String> {
    if !raw
        .keys()
        .any(|k| LEGACY_NOUL_CRITERIA.iter().any(|(l, _)| l == k))
    {
        return Ok(());
    }
    let mut criteria: Map<String, Value> = match raw.get("criteria") {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::Null) | None => Map::new(),
        Some(_) => {
            for (legacy, key) in LEGACY_NOUL_CRITERIA {
                if raw.contains_key(legacy) {
                    raw.shift_remove(legacy);
                    let _ = key;
                }
            }
            return Ok(());
        }
    };
    for (legacy, key) in LEGACY_NOUL_CRITERIA {
        if !raw.contains_key(legacy) {
            continue;
        }
        let value = raw.shift_remove(legacy).unwrap();
        if criteria.contains_key(key) {
            return Err(format!(
                "Noul got both '{legacy}' and criteria['{key}']; use criteria only."
            ));
        }
        if json_truthy(&value) {
            criteria.insert(key.to_string(), value);
        }
    }
    warnings.push(LEGACY_DEPRECATION_WARNING.to_string());
    raw.insert(
        "criteria".to_string(),
        if criteria.is_empty() {
            Value::Null
        } else {
            Value::Object(criteria)
        },
    );
    Ok(())
}

pub fn parse_question(raw: &Value, warnings: &mut Vec<String>) -> Result<Question, QuestionError> {
    let Some(map) = raw.as_object() else {
        return Err(QuestionError::Value(format!(
            "'{}' object has no attribute 'get'",
            crate::pyval::type_name(raw)
        )));
    };
    let q_type = match map.get("type") {
        None | Some(Value::Null) => "choice".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::pyval::py_str(other),
    };
    match q_type.as_str() {
        "noul" => parse_noul(map, warnings)
            .map_err(QuestionError::Validation)
            .map(Question::Noul),
        "choice" => parse_choice(map)
            .map_err(QuestionError::Validation)
            .map(Question::Choice),
        "score" => parse_score(map)
            .map_err(QuestionError::Validation)
            .map(Question::Score),
        _ => Err(QuestionError::Value(format!(
            "Unknown question type '{q_type}'"
        ))),
    }
}

pub fn parse_noul(
    raw: &Map<String, Value>,
    warnings: &mut Vec<String>,
) -> Result<Noul, PydanticError> {
    let original = Value::Object(raw.clone());
    let mut work = raw.clone();
    if let Err(msg) = fold_legacy_criteria(&mut work, warnings) {
        return Err(PydanticError {
            model: "Noul",
            entries: vec![ValidationEntry::value_error(msg, original)],
        });
    }
    let mut errors = Vec::new();
    let instructions = get_str_field(&work, "instructions", &mut errors, &original);
    let mut criteria = None;
    match work.get("criteria") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            let mut crit = IndexMap::new();
            for (k, v) in map {
                match v {
                    Value::String(s) => {
                        crit.insert(k.clone(), s.clone());
                    }
                    other => {
                        errors.push(ValidationEntry::string_type(
                            vec!["criteria".to_string(), k.clone()],
                            other.clone(),
                        ));
                    }
                }
            }
            criteria = Some(crit);
        }
        Some(other) => {
            errors.push(ValidationEntry::dict_type(
                vec!["criteria".to_string()],
                other.clone(),
            ));
        }
    }
    if !errors.is_empty() {
        return Err(PydanticError {
            model: "Noul",
            entries: errors,
        });
    }
    Ok(Noul {
        instructions: instructions.unwrap_or_default(),
        criteria,
    })
}

pub fn parse_choice(raw: &Map<String, Value>) -> Result<Choice, PydanticError> {
    let original = Value::Object(raw.clone());
    let mut errors = Vec::new();
    let instructions = get_str_field(raw, "instructions", &mut errors, &original);
    let mut criteria = IndexMap::new();
    match raw.get("criteria") {
        None => errors.push(ValidationEntry::missing(
            vec!["criteria".to_string()],
            original,
        )),
        Some(Value::Object(map)) => {
            for (k, v) in map {
                match v {
                    Value::String(s) => {
                        criteria.insert(k.clone(), Some(s.clone()));
                    }
                    Value::Null => {
                        criteria.insert(k.clone(), None);
                    }
                    other => {
                        errors.push(ValidationEntry::string_type(
                            vec!["criteria".to_string(), k.clone()],
                            other.clone(),
                        ));
                    }
                }
            }
        }
        Some(other) => {
            errors.push(ValidationEntry::dict_type(
                vec!["criteria".to_string()],
                other.clone(),
            ));
        }
    }
    if !errors.is_empty() {
        return Err(PydanticError {
            model: "Choice",
            entries: errors,
        });
    }
    Ok(Choice {
        instructions: instructions.unwrap_or_default(),
        criteria,
    })
}

pub fn parse_score(raw: &Map<String, Value>) -> Result<Score, PydanticError> {
    let original = Value::Object(raw.clone());
    let mut errors = Vec::new();
    let instructions = get_str_field(raw, "instructions", &mut errors, &original);
    let mut criteria = Vec::new();
    match raw.get("criteria") {
        None => errors.push(ValidationEntry::missing(
            vec!["criteria".to_string()],
            original,
        )),
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(s) => criteria.push(ScoreCriterion::Text(s.clone())),
                    Value::Object(map) => criteria.push(ScoreCriterion::Map(map.clone())),
                    other => {
                        errors.push(ValidationEntry::string_type(
                            vec!["criteria".to_string(), i.to_string(), "str".to_string()],
                            other.clone(),
                        ));
                        errors.push(ValidationEntry::dict_type(
                            vec![
                                "criteria".to_string(),
                                i.to_string(),
                                "dict[str,any]".to_string(),
                            ],
                            other.clone(),
                        ));
                    }
                }
            }
        }
        Some(other) => {
            errors.push(ValidationEntry::list_type(
                vec!["criteria".to_string()],
                other.clone(),
            ));
        }
    }
    if !errors.is_empty() {
        return Err(PydanticError {
            model: "Score",
            entries: errors,
        });
    }
    Ok(Score {
        instructions: instructions.unwrap_or_default(),
        criteria,
    })
}
