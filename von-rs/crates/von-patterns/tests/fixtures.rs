use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;
use von_core::{Engine, EngineError, StubEngine};
use von_patterns::{RouteHandlers, composite_score, confidence_gate, route, two_stage_choice};
use von_types::{SystemOneResponse, parse_question};

fn load() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/patterns.json");
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn case_output(name: &str) -> Value {
    for c in load()["cases"].as_array().unwrap() {
        if c["name"].as_str() == Some(name) {
            return c["output"].clone();
        }
    }
    panic!("pattern case {name} missing")
}

#[derive(Clone)]
struct FakeEngine {
    default: Value,
}

impl Engine for FakeEngine {
    fn evaluate(
        &self,
        _state: &Value,
        _questions: &IndexMap<String, Value>,
        _model: Option<&str>,
    ) -> Result<SystemOneResponse, EngineError> {
        Ok(serde_json::from_value(self.default.clone()).unwrap())
    }
}

fn gate_engine() -> FakeEngine {
    FakeEngine {
        default: serde_json::json!({
            "model": "von-1.1.0",
            "answers": {
                "q_confident": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 0.95},
                "q_unsure": {"type": "score", "score": 2.0, "confidence": 0.6, "legend": {"0": "l", "1": "m", "2": "h"}, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
                "q_noul": {"type": "noul", "noul": 0.4}
            },
            "usage": {"input_tokens": 20, "output_tokens": 3}
        }),
    }
}

fn composite_engine() -> FakeEngine {
    FakeEngine {
        default: serde_json::json!({
            "model": "von-1.1.0",
            "answers": {
                "rating": {"type": "score", "score": 3.2, "confidence": 0.9, "legend": {"0": "a", "1": "b", "2": "c", "3": "d"}, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}},
                "judgment": {"type": "noul", "noul": 0.55},
                "decision": {"type": "choice", "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0}
            },
            "usage": {"input_tokens": 30, "output_tokens": 3}
        }),
    }
}

fn route_engine() -> FakeEngine {
    FakeEngine {
        default: serde_json::json!({
            "model": "von-1.1.0",
            "answers": {
                "route_question": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.8}
            },
            "usage": {"input_tokens": 10, "output_tokens": 1}
        }),
    }
}

fn two_stage_engine() -> impl Engine {
    #[derive(Clone)]
    struct TwoStage;
    impl Engine for TwoStage {
        fn evaluate(
            &self,
            _state: &Value,
            questions: &IndexMap<String, Value>,
            _model: Option<&str>,
        ) -> Result<SystemOneResponse, EngineError> {
            if questions.contains_key("category") {
                Ok(serde_json::from_value(serde_json::json!({
                    "model": "von-1.1.0",
                    "answers": {"category": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.9}},
                    "usage": {"input_tokens": 10, "output_tokens": 1}
                }))
                .unwrap())
            } else {
                Ok(serde_json::from_value(serde_json::json!({
                    "model": "von-1.1.0",
                    "answers": {"option": {"type": "choice", "choice": "invoice", "probabilities": {"invoice": 0.8, "refund": 0.2}, "confidence": 0.6}},
                    "usage": {"input_tokens": 11, "output_tokens": 1}
                }))
                .unwrap())
            }
        }
    }
    TwoStage
}

fn to_value<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap()
}

#[test]
fn confidence_gate_matches_python() {
    let engine = gate_engine();
    let out = confidence_gate(&engine, &Value::Null, &IndexMap::new(), 0.8).unwrap();
    assert_eq!(to_value(&out), case_output("confidence_gate"));
}

#[test]
fn confidence_gate_threshold_validation() {
    let engine = gate_engine();
    let err = confidence_gate(&engine, &Value::Null, &IndexMap::new(), 1.5).unwrap_err();
    assert_eq!(err.0, "threshold must be in [0.0, 1.0], got 1.5");
}

#[test]
fn route_cases_match_python() {
    let engine = route_engine();
    let question = parse_question(
        &serde_json::json!({"type": "choice", "instructions": "Which?", "criteria": {"billing": null, "tech": null}}),
        &mut Vec::new(),
    )
    .unwrap();

    let mut handlers: RouteHandlers = IndexMap::new();
    handlers.insert(
        "billing".to_string(),
        Box::new(|ans: &von_types::Answer| {
            let von_types::Answer::Choice { choice, .. } = ans else {
                unreachable!()
            };
            serde_json::json!(format!("handled:{choice}"))
        }),
    );
    let out = route(&engine, &Value::Null, &question, &handlers, None, 0.0).unwrap();
    assert_eq!(out, case_output("route_hit"));

    let no_handlers: RouteHandlers = IndexMap::new();
    let out = route(&engine, &Value::Null, &question, &no_handlers, None, 0.0).unwrap();
    assert_eq!(out, case_output("route_no_handler_returns_answer"));

    let default = |_ans: &von_types::Answer| serde_json::json!("defaulted");
    let out = route(
        &engine,
        &Value::Null,
        &question,
        &no_handlers,
        Some(&default),
        0.9,
    )
    .unwrap();
    assert_eq!(out, case_output("route_default_low_conf"));
}

#[test]
fn route_rejects_non_choice() {
    let engine = route_engine();
    let question = parse_question(
        &serde_json::json!({"type": "noul", "instructions": "i"}),
        &mut Vec::new(),
    )
    .unwrap();
    let no_handlers: RouteHandlers = IndexMap::new();
    let err = route(&engine, &Value::Null, &question, &no_handlers, None, 0.0).unwrap_err();
    assert_eq!(err.0, "question must be an instance of von.Choice");
}

#[test]
fn composite_score_matches_python() {
    let engine = composite_engine();
    let mut weights = IndexMap::new();
    weights.insert("rating".to_string(), 2.0);
    let out = composite_score(
        &engine,
        &Value::Null,
        &IndexMap::new(),
        Some(&weights),
        true,
    )
    .unwrap();
    assert_eq!(to_value(&out), case_output("composite_score"));
}

#[test]
fn two_stage_choice_matches_python() {
    let engine = two_stage_engine();
    let mut taxonomy: IndexMap<String, IndexMap<String, Option<String>>> = IndexMap::new();
    taxonomy.insert(
        "billing".to_string(),
        IndexMap::from([("invoice".to_string(), None), ("refund".to_string(), None)]),
    );
    taxonomy.insert(
        "tech".to_string(),
        IndexMap::from([("bug".to_string(), None)]),
    );
    let out = two_stage_choice(
        &engine,
        &Value::Null,
        &taxonomy,
        "Which broad category best matches the state?",
        "Which specific sub-option applies within {category}?",
    )
    .unwrap();
    assert_eq!(to_value(&out), case_output("two_stage_choice"));
}

#[test]
fn stub_engine_feeds_patterns() {
    let engine = StubEngine;
    let questions: IndexMap<String, Value> = IndexMap::from([(
        "check".to_string(),
        serde_json::json!({"type": "noul", "instructions": "Is it up?"}),
    )]);
    let out = confidence_gate(&engine, &Value::String("s".into()), &questions, 0.8).unwrap();
    assert!(out.automatic.contains_key("check"));
    assert!(out.escalate.is_empty());
    assert_eq!(
        out.response.get("check"),
        Some(&von_types::Answer::Noul { noul: 0.5 })
    );
}
