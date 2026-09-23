use indexmap::IndexMap;
use von_presets::{email_preset, moderation_preset, security_preset, triage_preset};

fn load() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/presets.json");
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn case_by_name(name: &str) -> serde_json::Value {
    for c in load()["cases"].as_array().unwrap() {
        if c["name"].as_str() == Some(name) {
            return c.clone();
        }
    }
    panic!("preset case {name} missing")
}

fn assert_preset(name: &str, preset: IndexMap<String, serde_json::Value>) {
    let case = case_by_name(name);
    let rendered =
        serde_json::to_string(&serde_json::Value::Object(preset.into_iter().collect())).unwrap();
    assert_eq!(
        rendered,
        case["rendered"].as_str().unwrap(),
        "preset {name}"
    );
}

#[test]
fn presets_match_python_byte_for_byte() {
    assert_preset("triage", triage_preset());
    assert_preset("email_default", email_preset(None));
    let mut custom = IndexMap::new();
    custom.insert(
        "custom".to_string(),
        "Custom category description".to_string(),
    );
    assert_preset("email_custom", email_preset(Some(custom)));
    assert_preset("moderation", moderation_preset());
    assert_preset("security", security_preset());
}
