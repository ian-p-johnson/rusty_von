//! `json.dumps`-compatible JSON rendering (ensure_ascii=True defaults).

use serde_json::Value;

pub fn to_python_json(v: &Value) -> String {
    write_value(v, 0, None)
}

pub fn to_python_json_indent(v: &Value, indent: usize) -> String {
    write_value(v, 0, Some(indent))
}

fn write_value(v: &Value, depth: usize, indent: Option<usize>) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                von_types::repr_f64(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => escape_string(s),
        Value::Array(items) => {
            if items.is_empty() {
                return "[]".to_string();
            }
            let inner: Vec<String> = items
                .iter()
                .map(|item| write_value(item, depth + 1, indent))
                .collect();
            match indent {
                Some(w) => {
                    let pad = " ".repeat(w * (depth + 1));
                    let close = " ".repeat(w * depth);
                    format!("[\n{pad}{}\n{close}]", inner.join(&format!(",\n{pad}")))
                }
                None => format!("[{}]", inner.join(", ")),
            }
        }
        Value::Object(map) => {
            if map.is_empty() {
                return "{}".to_string();
            }
            let inner: Vec<String> = map
                .iter()
                .map(|(k, val)| {
                    format!(
                        "{}: {}",
                        escape_string(k),
                        write_value(val, depth + 1, indent)
                    )
                })
                .collect();
            match indent {
                Some(w) => {
                    let pad = " ".repeat(w * (depth + 1));
                    let close = " ".repeat(w * depth);
                    format!("{{\n{pad}{}\n{close}}}", inner.join(&format!(",\n{pad}")))
                }
                None => format!("{{{}}}", inner.join(", ")),
            }
        }
    }
}

pub fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x7f => out.push(c),
            c => {
                let cp = c as u32;
                if cp <= 0xffff {
                    out.push_str(&format!("\\u{:04x}", cp));
                } else {
                    let v = cp - 0x10000;
                    let hi = 0xd800 + (v >> 10);
                    let lo = 0xdc00 + (v & 0x3ff);
                    out.push_str(&format!("\\u{:04x}\\u{:04x}", hi, lo));
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_like_json_dumps() {
        assert_eq!(escape_string("café"), "\"caf\\u00e9\"");
        assert_eq!(escape_string("😀"), "\"\\ud83d\\ude00\"");
        assert_eq!(escape_string("a\u{1}b"), "\"a\\u0001b\"");
        assert_eq!(escape_string("\u{7f}"), "\"\\u007f\"");
    }

    #[test]
    fn compact_and_indented_shapes() {
        let v = serde_json::json!({"a": 1, "b": [1, 2]});
        assert_eq!(to_python_json(&v), r#"{"a": 1, "b": [1, 2]}"#);
        assert_eq!(
            to_python_json_indent(&v, 2),
            "{\n  \"a\": 1,\n  \"b\": [\n    1,\n    2\n  ]\n}"
        );
        assert_eq!(to_python_json_indent(&serde_json::json!({}), 2), "{}");
        assert_eq!(to_python_json_indent(&serde_json::json!([]), 2), "[]");
        assert_eq!(to_python_json(&serde_json::json!(1e16)), "1e+16");
    }
}
