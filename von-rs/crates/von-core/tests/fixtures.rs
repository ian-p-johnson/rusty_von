use indexmap::IndexMap;
use serde_json::Value;
use von_core::{Engine, StubEngine};
use von_types::Answer;

fn load(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn state_formatting_matches_python() {
    let doc = load("state_formatting.json");
    for case in doc["cases"].as_array().unwrap() {
        let input = &case["input"];
        let formatted = von_core::format_state(input);
        assert_eq!(
            formatted,
            case["formatted"].as_str().unwrap(),
            "format_state({input})"
        );
        let py_str = von_core::py_str(input);
        assert_eq!(py_str, case["str"].as_str().unwrap(), "py_str({input})");
    }
}

#[test]
fn pack_sequence_matches_python() {
    let doc = load("pack_sequence.json");
    for case in doc["cases"].as_array().unwrap() {
        let state = case["state"].as_str().unwrap();
        let question = case["question"].as_str().unwrap();
        let options: Vec<&str> = case["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let packed = von_core::pack_sequence(state, question, &options);
        assert_eq!(packed, case["packed"].as_str().unwrap(), "case {case:?}");
    }
}

#[test]
fn rounding_matches_python() {
    let doc = load("rounding.json");
    for case in doc["cases"].as_array().unwrap() {
        let value = case["value"].as_f64().unwrap();
        let nd = case["ndigits"].as_u64().unwrap() as u32;
        let got = von_core::round_half_even(value, nd);
        let expected: f64 = case["expected"].as_str().unwrap().parse().unwrap();
        assert_eq!(got.to_bits(), expected.to_bits(), "round({value}, {nd})");
    }
}

#[test]
fn usage_accounting_matches_python() {
    let doc = load("usage.json");
    for case in doc["cases"].as_array().unwrap() {
        let state = case["state"].as_str().unwrap();
        let qchars = case["q_chars"].as_u64().unwrap() as usize;
        let n = case["n_answers"].as_u64().unwrap() as usize;
        let usage = von_core::compute_usage(state, qchars, n);
        assert_eq!(
            usage.input_tokens,
            case["usage"]["input_tokens"].as_u64().unwrap(),
            "state={state:?}"
        );
        assert_eq!(
            usage.output_tokens,
            case["usage"]["output_tokens"].as_u64().unwrap()
        );
    }
}

#[test]
fn json_writer_matches_python_json_dumps() {
    let doc = load("json_dumps.json");
    for case in doc["cases"].as_array().unwrap() {
        let value = &case["value"];
        assert_eq!(
            von_core::to_python_json(value),
            case["compact"].as_str().unwrap(),
            "compact {value}"
        );
        assert_eq!(
            von_core::to_python_json_indent(value, 2),
            case["indent2"].as_str().unwrap(),
            "indent2 {value}"
        );
    }
}

fn api_error_of(id: &str) -> String {
    let doc = load("api_errors.json");
    for c in doc["cases"].as_array().unwrap() {
        if c["id"].as_str() == Some(id) {
            return c["message"].as_str().unwrap().to_string();
        }
    }
    panic!("case {id} missing");
}

fn questions_map(raw: Value) -> IndexMap<String, Value> {
    let mut m = IndexMap::new();
    if let Value::Object(map) = raw {
        for (k, v) in map {
            m.insert(k, v);
        }
    }
    m
}

#[test]
fn stub_engine_error_contracts_match_python() {
    let engine = StubEngine;

    let err = engine
        .evaluate(
            &Value::Null,
            &questions_map(serde_json::json!({"q": {"type": "bogus", "instructions": "?"}})),
            None,
        )
        .unwrap_err();
    assert_eq!(err.0, "Unknown question type 'bogus'");

    let err = engine
        .evaluate(
            &Value::Null,
            &questions_map(serde_json::json!({"q": {"type": 5}})),
            None,
        )
        .unwrap_err();
    assert_eq!(err.0, "Unknown question type '5'");

    let err = engine
        .evaluate(
            &Value::Null,
            &questions_map(serde_json::json!({"q": {"type": "score", "instructions": "i", "criteria": [{"what": "w", "examples": [1, "two"]}]}})),
            None,
        )
        .unwrap_err();
    assert_eq!(err.0, api_error_of("score_join_int"));

    let err = engine
        .evaluate(
            &Value::Null,
            &questions_map(serde_json::json!({"q": {"type": "score", "instructions": "i", "criteria": [{"what": "w", "examples": ["one", null]}]}})),
            None,
        )
        .unwrap_err();
    assert_eq!(err.0, api_error_of("score_join_null"));

    let err = engine
        .evaluate(
            &Value::Null,
            &questions_map(serde_json::json!({"q": {"type": "score", "instructions": "i", "criteria": [{"what": "w", "examples": ["one", true]}]}})),
            None,
        )
        .unwrap_err();
    assert_eq!(err.0, api_error_of("score_join_bool"));

    let choices = vec!["yes".to_string(), "yes".to_string()];
    let err = von_core::decide(&engine, &Value::Null, &choices, "i", None).unwrap_err();
    assert_eq!(err.0, api_error_of("duplicate_choices"));
}

#[test]
fn engine_constants_match_python() {
    assert_eq!(
        api_error_of("unknown_model"),
        von_core::unknown_model_message("option-marker")
    );
    assert_eq!(von_core::VON_MODEL_ID, "von-1.1.0");
    assert_eq!(von_core::resolved_model_id(), "von-1.1.0");
    for alias in ["von-1.1", "1.1", "von", "default", "latest", "von-latest"] {
        assert!(von_core::is_supported_alias(alias));
        assert!(von_core::is_supported_alias(&alias.to_uppercase()));
    }
    assert!(!von_core::is_supported_alias("option-marker"));
}

#[test]
fn stub_engine_envelope_shape_and_stub_values() {
    let engine = StubEngine;
    let questions = questions_map(serde_json::json!({
        "pick": {"type": "choice", "instructions": "Which?", "criteria": {"a": "Alpha", "b": "Beta", "c": null}},
        "judge": {"type": "noul", "instructions": "Is it?"},
        "rate": {"type": "score", "instructions": "Rate:", "criteria": ["low", "mid", "high", "top"]},
        "none": {"type": "choice", "instructions": "Empty", "criteria": {}}
    }));
    let resp = engine
        .evaluate(&Value::String("s".into()), &questions, Some("von-1.1.0"))
        .unwrap();
    assert_eq!(resp.model, "von-1.1.0");
    let keys: Vec<&String> = resp.answers.keys().collect();
    assert_eq!(keys, ["pick", "judge", "rate", "none"]);
    assert_eq!(resp.usage.input_tokens, 6);
    assert_eq!(resp.usage.output_tokens, 4);

    match resp.get("pick").unwrap() {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            assert_eq!(choice, "a");
            let vals: Vec<f64> = probabilities.values().copied().collect();
            assert_eq!(vals, [0.3333, 0.3333, 0.3333]);
            assert_eq!(*confidence, 0.0);
        }
        _ => panic!(),
    }
    match resp.get("judge").unwrap() {
        Answer::Noul { noul } => assert_eq!(*noul, 0.5),
        _ => panic!(),
    }
    match resp.get("rate").unwrap() {
        Answer::Score {
            score,
            legend,
            probabilities,
            confidence,
        } => {
            assert_eq!(*score, 1.5);
            assert_eq!(*confidence, 0.0);
            let legend_vals: Vec<&String> = legend.values().collect();
            assert_eq!(
                legend_vals,
                [
                    &"low".to_string(),
                    &"mid".to_string(),
                    &"high".to_string(),
                    &"top".to_string()
                ]
            );
            assert_eq!(probabilities.len(), 4);
        }
        _ => panic!(),
    }
    match resp.get("none").unwrap() {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            assert_eq!(choice, "");
            assert!(probabilities.is_empty());
            assert_eq!(*confidence, 0.0);
        }
        _ => panic!(),
    }
}

#[test]
fn stub_engine_validation_errors_are_pydantic_shaped() {
    let engine = StubEngine;
    let err = engine
        .evaluate(
            &Value::Null,
            &questions_map(serde_json::json!({"q": {"type": "noul"}})),
            None,
        )
        .unwrap_err();
    assert!(
        err.0
            .starts_with("1 validation error for Noul\ninstructions\n  Field required")
    );
}
