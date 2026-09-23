//! Pydantic-ValidationError-compatible rendering for wire-pinned 422 bodies.

use serde_json::Value;

use crate::pyval::{py_repr, truncate_repr};

#[derive(Debug, Clone)]
pub struct ValidationEntry {
    pub loc: Vec<String>,
    pub kind: &'static str,
    pub msg: String,
    pub input: Value,
}

impl ValidationEntry {
    pub fn missing(loc: Vec<String>, input: Value) -> Self {
        Self {
            loc,
            kind: "missing",
            msg: "Field required".to_string(),
            input,
        }
    }

    pub fn string_type(loc: Vec<String>, input: Value) -> Self {
        Self {
            loc,
            kind: "string_type",
            msg: "Input should be a valid string".to_string(),
            input,
        }
    }

    pub fn dict_type(loc: Vec<String>, input: Value) -> Self {
        Self {
            loc,
            kind: "dict_type",
            msg: "Input should be a valid dictionary".to_string(),
            input,
        }
    }

    pub fn list_type(loc: Vec<String>, input: Value) -> Self {
        Self {
            loc,
            kind: "list_type",
            msg: "Input should be a valid list".to_string(),
            input,
        }
    }

    pub fn value_error(msg: String, input: Value) -> Self {
        Self {
            loc: Vec::new(),
            kind: "value_error",
            msg,
            input,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PydanticError {
    pub model: &'static str,
    pub entries: Vec<ValidationEntry>,
}

impl PydanticError {
    pub fn render(&self) -> String {
        let n = self.entries.len();
        let mut out = format!(
            "{n} validation error{} for {}",
            if n == 1 { "" } else { "s" },
            self.model
        );
        for e in &self.entries {
            out.push('\n');
            if e.kind == "value_error" {
                out.push_str(&format!(
                    "  Value error, {} [type={}, input_value={}, input_type={}]",
                    e.msg,
                    e.kind,
                    truncate_repr(&py_repr(&e.input)),
                    crate::pyval::type_name(&e.input)
                ));
            } else {
                out.push_str(&format!(
                    "{}\n  {} [type={}, input_value={}, input_type={}]",
                    e.loc.join("."),
                    e.msg,
                    e.kind,
                    truncate_repr(&py_repr(&e.input)),
                    crate::pyval::type_name(&e.input)
                ));
            }
            out.push_str(&format!(
                "\n    For further information visit https://errors.pydantic.dev/2.13/v/{}",
                e.kind
            ));
        }
        out
    }
}
