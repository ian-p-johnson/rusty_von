//! Python-`str()`-compatible state rendering and `_format_state`.

use serde_json::Value;
pub use von_types::{py_repr, py_str};

pub fn format_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{k}: {}", py_str(v)))
                .collect();
            parts.join("\n")
        }
        other => py_str(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrors_python_semantics() {
        assert_eq!(format_state(&Value::String("abc".into())), "abc");
        assert_eq!(
            format_state(&serde_json::json!({"a": 1, "b": [true, null]})),
            "a: 1\nb: [True, None]"
        );
        assert_eq!(format_state(&serde_json::json!(null)), "None");
    }
}
