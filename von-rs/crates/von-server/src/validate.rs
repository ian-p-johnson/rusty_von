//! FastAPI/Starlette-compatible request validation for `/v1/systemone`.

use serde_json::{Value, json};

use crate::pyjson_scan::{JsonScanError, scan};

pub struct RequestModel {
    pub model: String,
    pub state: Value,
    pub questions: indexmap::IndexMap<String, Value>,
}

fn entry(kind: &str, loc: Value, msg: &str, input: Value) -> Value {
    json!({"type": kind, "loc": loc, "msg": msg, "input": input})
}

pub fn json_invalid_body(err: &JsonScanError) -> String {
    json!({
        "detail": [{
            "type": "json_invalid",
            "loc": ["body", err.pos],
            "msg": "JSON decode error",
            "input": {},
            "ctx": {"error": err.msg}
        }]
    })
    .to_string()
}

fn error_body(entries: Vec<Value>) -> String {
    json!({"detail": entries}).to_string()
}

pub enum ParsedRequest {
    Ok(RequestModel),
    Err(String),
}

pub fn validate_body(raw: &str) -> ParsedRequest {
    if raw.trim().is_empty() {
        return ParsedRequest::Err(error_body(vec![entry(
            "missing",
            json!(["body"]),
            "Field required",
            Value::Null,
        )]));
    }
    let body = match scan(raw) {
        Ok(v) => v,
        Err(e) => return ParsedRequest::Err(json_invalid_body(&e)),
    };
    if !body.is_object() {
        return ParsedRequest::Err(error_body(vec![entry(
            "model_attributes_type",
            json!(["body"]),
            "Input should be a valid dictionary or object to extract fields from",
            body,
        )]));
    }
    let map = body.as_object().unwrap();
    let mut entries: Vec<Value> = Vec::new();

    let model = match map.get("model") {
        None => "von-latest".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            entries.push(entry(
                "string_type",
                json!(["body", "model"]),
                "Input should be a valid string",
                other.clone(),
            ));
            String::new()
        }
    };

    let state = match map.get("state") {
        Some(v) => v.clone(),
        None => {
            entries.push(entry(
                "missing",
                json!(["body", "state"]),
                "Field required",
                body.clone(),
            ));
            Value::Null
        }
    };

    let mut questions = indexmap::IndexMap::new();
    match map.get("questions") {
        Some(Value::Object(qmap)) => {
            for (q_id, q_val) in qmap {
                if !q_val.is_object() {
                    entries.push(entry(
                        "dict_type",
                        json!(["body", "questions", q_id]),
                        "Input should be a valid dictionary",
                        q_val.clone(),
                    ));
                    continue;
                }
                questions.insert(q_id.clone(), q_val.clone());
            }
        }
        Some(Value::Null) => {
            entries.push(entry(
                "dict_type",
                json!(["body", "questions"]),
                "Input should be a valid dictionary",
                Value::Null,
            ));
        }
        Some(other) => {
            entries.push(entry(
                "dict_type",
                json!(["body", "questions"]),
                "Input should be a valid dictionary",
                other.clone(),
            ));
        }
        None => {
            entries.push(entry(
                "missing",
                json!(["body", "questions"]),
                "Field required",
                body.clone(),
            ));
        }
    }

    if !entries.is_empty() {
        return ParsedRequest::Err(error_body(entries));
    }
    ParsedRequest::Ok(RequestModel {
        model,
        state,
        questions,
    })
}
