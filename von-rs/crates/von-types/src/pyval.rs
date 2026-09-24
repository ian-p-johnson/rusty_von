//! Python-compatible value rendering for JSON values.
//!
//! Every wire-visible string in von flows through Python's `str()`/`repr()`
//! semantics, so the Rust side must render JSON values the way CPython does.

use serde_json::Value;

pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                repr_f64(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, val)| format!("{}: {}", repr_str(k), py_repr(val)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

pub fn repr_str(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let (quote, escape_quote) = if has_single && !has_double {
        ('"', '"')
    } else {
        ('\'', '\'')
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == escape_quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

pub fn repr_f64(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let neg = x < 0.0;
    let abs = x.abs();
    let sci = format!("{:e}", abs);
    let (mantissa, exp_str) = sci.split_once('e').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if !(-4..16).contains(&exp) {
        let mut mant = String::new();
        mant.push_str(&digits[..1]);
        if digits.len() > 1 {
            mant.push('.');
            mant.push_str(&digits[1..]);
        }
        let (sign, mag) = if exp < 0 { ('-', -exp) } else { ('+', exp) };
        out.push_str(&format!("{mant}e{sign}{mag:02}"));
    } else if exp >= digits.len() as i32 - 1 {
        out.push_str(digits);
        out.push_str(&"0".repeat(exp as usize - (digits.len() as i32 - 1) as usize));
        out.push_str(".0");
    } else if exp >= 0 {
        let point = (exp + 1) as usize;
        out.push_str(&digits[..point]);
        out.push('.');
        out.push_str(&digits[point..]);
    } else {
        out.push_str("0.");
        out.push_str(&"0".repeat((-exp - 1) as usize));
        out.push_str(digits);
    }
    out
}

/// pydantic-core `write_truncated_to_limited_bytes` (tools.rs): the repr is
/// truncated by BYTE length, not chars — max 50 bytes, head from byte 0 to
/// `floor_char_boundary(25)`, tail from `ceil_char_boundary(len - 24)` to the
/// end, joined with "...".
pub fn truncate_repr(s: &str) -> String {
    const MAX_LEN: usize = 50;
    if s.len() <= MAX_LEN {
        return s.to_string();
    }
    let mid_point = MAX_LEN.div_ceil(2); // 25
    format!(
        "{}...{}",
        &s[..floor_char_boundary(s, mid_point)],
        &s[ceil_char_boundary(s, s.len() - (mid_point - 1))..]
    )
}

fn is_utf8_char_boundary(b: u8) -> bool {
    // Bit magic equivalent to: b < 128 || b >= 192
    (b as i8) >= -0x40
}

fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let lower_bound = index.saturating_sub(3);
    let new_index = s.as_bytes()[lower_bound..=index]
        .iter()
        .rposition(|b| is_utf8_char_boundary(*b));
    lower_bound + new_index.unwrap()
}

fn ceil_char_boundary(s: &str, index: usize) -> usize {
    let upper_bound = Ord::min(index + 4, s.len());
    s.as_bytes()[index..upper_bound]
        .iter()
        .position(|b| is_utf8_char_boundary(*b))
        .map_or(upper_bound, |pos| pos + index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        assert_eq!(repr_f64(1.5), "1.5");
        assert_eq!(repr_f64(3.0), "3.0");
        assert_eq!(repr_f64(-0.0), "-0.0");
        assert_eq!(repr_f64(0.1), "0.1");
        assert_eq!(repr_f64(0.30000000000000004), "0.30000000000000004");
        assert_eq!(repr_f64(1e15), "1000000000000000.0");
        assert_eq!(repr_f64(1e16), "1e+16");
        assert_eq!(repr_f64(9999999999999998.0), "9999999999999998.0");
        assert_eq!(repr_f64(1e-4), "0.0001");
        assert_eq!(repr_f64(1e-5), "1e-05");
        assert_eq!(repr_f64(1.2345678901234567e-7), "1.2345678901234566e-07");
        assert_eq!(repr_f64(2.675), "2.675");
        assert_eq!(repr_f64(1e300), "1e+300");
        assert_eq!(repr_f64(5e-324), "5e-324");
    }

    #[test]
    fn repr_switches_quotes() {
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("say \"hi\""), "'say \"hi\"'");
        assert_eq!(repr_str("both ' and \""), "'both \\' and \"'");
    }
}
