use indexmap::IndexMap;
use serde_json::Value;
use von_types::{
    SystemOneResponse, parse_choice, parse_noul, parse_question, parse_score, py_repr,
};

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load(name: &str) -> Value {
    let path = fixtures_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn envelope_rendering_matches_fastapi_bytes() {
    let doc = load("envelopes.json");
    let cases = doc["cases"].as_array().unwrap();
    for case in cases {
        let input = &case["input"];
        let resp: SystemOneResponse = serde_json::from_value(input.clone()).unwrap();
        let rendered = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            rendered,
            case["rendered"].as_str().unwrap(),
            "case {}",
            case["name"]
        );
    }
}

#[test]
fn roundtrip_deserialize_preserves_answer_order() {
    let doc = load("envelopes.json");
    let cases = doc["cases"].as_array().unwrap();
    for case in cases {
        let input = &case["input"];
        let resp: SystemOneResponse = serde_json::from_value(input.clone()).unwrap();
        let keys: Vec<&String> = resp.answers.keys().collect();
        let expected: Vec<&String> = input["answers"].as_object().unwrap().keys().collect();
        assert_eq!(keys, expected);
        let re: SystemOneResponse =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();
        assert_eq!(re, resp);
    }
}

fn detail_of(cases: &Value, id: &str) -> String {
    for c in cases.as_array().unwrap() {
        if c["id"].as_str() == Some(id) {
            return c["message"].as_str().unwrap().to_string();
        }
    }
    panic!("case {id} not found");
}

fn model_detail(model: &str, raw: &Value) -> (String, Vec<String>) {
    let mut warnings = Vec::new();
    let result = match model {
        "Noul" => parse_noul(raw.as_object().unwrap(), &mut warnings).map(|_| ()),
        "Choice" => parse_choice(raw.as_object().unwrap()).map(|_| ()),
        "Score" => parse_score(raw.as_object().unwrap()).map(|_| ()),
        _ => panic!(),
    };
    match result {
        Ok(()) => (String::new(), warnings),
        Err(e) => (e.render(), warnings),
    }
}

#[test]
fn pydantic_validation_messages_match_python() {
    let cases = load("api_errors.json")["cases"].clone();

    let (detail, _) = model_detail("Noul", &serde_json::json!({"instructions": 5}));
    assert_eq!(detail, detail_of(&cases, "noul_instructions_int"));

    let (detail, _) = model_detail(
        "Choice",
        &serde_json::json!({"instructions": "?", "criteria": 5}),
    );
    assert_eq!(detail, detail_of(&cases, "choice_criteria_int"));

    let (detail, _) = model_detail(
        "Choice",
        &serde_json::json!({"instructions": "?", "criteria": null}),
    );
    assert_eq!(detail, detail_of(&cases, "choice_criteria_null"));

    let (detail, _) = model_detail(
        "Choice",
        &serde_json::json!({"instructions": "?", "criteria": "not a dict"}),
    );
    assert_eq!(detail, detail_of(&cases, "choice_criteria_wrong_type"));

    let (detail, _) = model_detail(
        "Choice",
        &serde_json::json!({"instructions": "?", "criteria": {"a": 5}}),
    );
    assert_eq!(detail, detail_of(&cases, "choice_criteria_int_value"));

    let (detail, _) = model_detail(
        "Score",
        &serde_json::json!({"instructions": "?", "criteria": {"a": 1}}),
    );
    assert_eq!(detail, detail_of(&cases, "score_criteria_dict"));

    let (detail, _) = model_detail(
        "Score",
        &serde_json::json!({"instructions": "?", "criteria": ["a", 7]}),
    );
    assert_eq!(detail, detail_of(&cases, "score_criteria_int_item"));

    let (detail, _) = model_detail(
        "Noul",
        &serde_json::json!({"criteria": {"true": "t", "false": 3}}),
    );
    assert_eq!(detail, detail_of(&cases, "noul_criteria_int_value"));
}

#[test]
fn legacy_criteria_fold_matches_python() {
    let cases = load("api_errors.json")["cases"].clone();

    let mut warnings = Vec::new();
    let q = parse_question(
        &serde_json::json!({"type": "noul", "instructions": "Is it down?", "pos_criteria": "Also down"}),
        &mut warnings,
    )
    .unwrap();
    assert_eq!(
        warnings,
        vec![von_types::LEGACY_DEPRECATION_WARNING.to_string()]
    );
    let expected: Value =
        serde_json::from_str(&detail_of(&cases, "noul_legacy_fold_result")).unwrap();
    match q {
        von_types::Question::Noul(n) => {
            let actual = serde_json::json!({
                "type": "noul",
                "instructions": n.instructions,
                "criteria": n.criteria,
            });
            assert_eq!(actual, expected);
        }
        _ => panic!(),
    }

    let mut warnings = Vec::new();
    let q = parse_question(
        &serde_json::json!({
            "type": "noul", "instructions": "Is it down?",
            "pos_criteria": "Down", "neg_criteria": "Up"
        }),
        &mut warnings,
    )
    .unwrap();
    assert_eq!(warnings.len(), 1);
    match q {
        von_types::Question::Noul(n) => assert_eq!(
            n.criteria.unwrap(),
            IndexMap::from([
                ("true".to_string(), "Down".to_string()),
                ("false".to_string(), "Up".to_string())
            ])
        ),
        _ => panic!(),
    }

    let mut warnings = Vec::new();
    let q = parse_question(
        &serde_json::json!({"type": "noul", "instructions": "Is it down?", "pos_criteria": ""}),
        &mut warnings,
    )
    .unwrap();
    assert_eq!(warnings.len(), 1);
    match q {
        von_types::Question::Noul(n) => assert_eq!(n.criteria, None),
        _ => panic!(),
    }

    let (detail, _) = model_detail(
        "Noul",
        &serde_json::json!({
            "instructions": "Is it down?",
            "criteria": {"true": "Down"}, "pos_criteria": "Also down", "neg_criteria": "Up"
        }),
    );
    assert_eq!(detail, detail_of(&cases, "noul_legacy_conflict"));

    let (detail, _) = model_detail(
        "Noul",
        &serde_json::json!({
            "instructions": "Is it down?",
            "criteria": {"false": "F"}, "neg_criteria": "Up"
        }),
    );
    assert_eq!(detail, detail_of(&cases, "noul_legacy_conflict_false"));
}

#[test]
fn unknown_question_type_is_a_value_error() {
    let cases = load("api_errors.json")["cases"].clone();
    let mut warnings = Vec::new();
    let err = parse_question(
        &serde_json::json!({"type": "bogus", "instructions": "?"}),
        &mut warnings,
    )
    .unwrap_err();
    assert_eq!(err.detail(), detail_of(&cases, "unknown_qtype"));

    let mut warnings = Vec::new();
    let err = parse_question(&serde_json::json!({"type": 5}), &mut warnings).unwrap_err();
    assert_eq!(err.detail(), "Unknown question type '5'");
}

#[test]
fn truncate_repr_cuts_middle_over_51_chars() {
    let long: String = "x".repeat(60);
    let t = von_types::truncate_repr(&format!("{{{long}}}"));
    assert_eq!(t.chars().count(), 52);
    assert!(t.contains("..."));
    let short = py_repr(&serde_json::json!({"type": "noul"}));
    assert_eq!(von_types::truncate_repr(&short), short);
}
