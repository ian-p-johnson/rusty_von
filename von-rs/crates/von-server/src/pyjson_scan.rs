//! A JSON scanner that reproduces Python `json.loads` error messages and
//! character offsets, so `json_invalid` 422 bodies are byte-identical.

pub struct JsonScanError {
    pub msg: String,
    pub pos: usize,
}

pub fn scan(text: &str) -> Result<serde_json::Value, JsonScanError> {
    let chars: Vec<char> = text.chars().collect();
    let mut p = Parser { chars, pos: 0 };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.pos < p.chars.len() {
        return Err(p.err("Extra data"));
    }
    Ok(value)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn err(&self, msg: &str) -> JsonScanError {
        JsonScanError {
            msg: msg.to_string(),
            pos: self.pos,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(
            self.peek(),
            Some(' ') | Some('\t') | Some('\n') | Some('\r')
        ) {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Result<serde_json::Value, JsonScanError> {
        match self.peek() {
            None => Err(self.err("Expecting value")),
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => self.parse_string().map(serde_json::Value::String),
            Some('t') => self.parse_literal("true", serde_json::Value::Bool(true)),
            Some('f') => self.parse_literal("false", serde_json::Value::Bool(false)),
            Some('n') => self.parse_literal("null", serde_json::Value::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(_) => Err(self.err("Expecting value")),
        }
    }

    fn parse_literal(
        &mut self,
        lit: &str,
        value: serde_json::Value,
    ) -> Result<serde_json::Value, JsonScanError> {
        for expected in lit.chars() {
            match self.peek() {
                Some(c) if c == expected => self.pos += 1,
                _ => return Err(self.err("Expecting value")),
            }
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<serde_json::Value, JsonScanError> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            text.parse::<f64>()
                .map(|f| serde_json::json!(f))
                .map_err(|_| JsonScanError {
                    msg: "Expecting value".to_string(),
                    pos: start,
                })
        } else {
            text.parse::<i64>()
                .map(|i| serde_json::json!(i))
                .map_err(|_| JsonScanError {
                    msg: "Expecting value".to_string(),
                    pos: start,
                })
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonScanError> {
        let start = self.pos;
        self.pos += 1;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => {
                    self.pos = start;
                    return Err(self.err("Unterminated string starting at"));
                }
                Some('"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some('\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some('"') => out.push('"'),
                        Some('\\') => out.push('\\'),
                        Some('/') => out.push('/'),
                        Some('b') => out.push('\u{8}'),
                        Some('f') => out.push('\u{c}'),
                        Some('n') => out.push('\n'),
                        Some('r') => out.push('\r'),
                        Some('t') => out.push('\t'),
                        Some('u') => {
                            self.pos += 1;
                            let hex: String = self.chars
                                [self.pos..(self.pos + 4).min(self.chars.len())]
                                .iter()
                                .collect();
                            if hex.chars().count() != 4 {
                                self.pos = start;
                                return Err(self.err("Unterminated string starting at"));
                            }
                            let cp = u32::from_str_radix(&hex, 16).unwrap_or(0xfffd);
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            self.pos += 3;
                        }
                        _ => {
                            self.pos = start;
                            return Err(self.err("Unterminated string starting at"));
                        }
                    }
                    self.pos += 1;
                }
                Some(c) => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<serde_json::Value, JsonScanError> {
        let mut map = serde_json::Map::new();
        self.pos += 1;
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(serde_json::Value::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') {
                if self.peek().is_none() {
                    return Err(self.err("Expecting value"));
                }
                return Err(self.err("Expecting property name enclosed in double quotes"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(':') {
                if self.peek().is_none() {
                    return Err(self.err("Expecting value"));
                }
                return Err(self.err("Expecting ':' delimiter"));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some('}') => {
                    self.pos += 1;
                    return Ok(serde_json::Value::Object(map));
                }
                None => return Err(self.err("Expecting ',' delimiter")),
                Some(_) => return Err(self.err("Expecting ',' delimiter")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<serde_json::Value, JsonScanError> {
        let mut items = Vec::new();
        self.pos += 1;
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(serde_json::Value::Array(items));
        }
        loop {
            self.skip_ws();
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some(']') => {
                    self.pos += 1;
                    return Ok(serde_json::Value::Array(items));
                }
                _ => return Err(self.err("Expecting ',' delimiter")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_error_positions() {
        let err = scan(r#"{"model": "von-latest", "state": "#).unwrap_err();
        assert_eq!(err.msg, "Expecting value");
        assert_eq!(err.pos, 33);

        let err = scan(r#"{"model": , }"#).unwrap_err();
        assert_eq!(err.msg, "Expecting value");
        assert_eq!(err.pos, 10);

        let err = scan(r#"{"state": "s", "questions": {}} extra"#).unwrap_err();
        assert_eq!(err.msg, "Extra data");
        assert_eq!(err.pos, 32);

        assert!(scan(r#"{"a": 1}"#).is_ok());
    }
}
