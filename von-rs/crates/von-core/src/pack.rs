//! Option-marker sequence packing (pure string logic; mask/sep pinned to the
//! real tokenizer via fixtures).

pub const MASK_TOKEN: &str = "[MASK]";
pub const SEP_TOKEN: &str = "[SEP]";

pub fn pack_sequence(state: &str, question: &str, options: &[&str]) -> String {
    pack_sequence_with(state, question, options, MASK_TOKEN, SEP_TOKEN)
}

pub fn pack_sequence_with(
    state: &str,
    question: &str,
    options: &[&str],
    mask: &str,
    sep: &str,
) -> String {
    let prefix = if !question.is_empty() {
        format!("{question} {state}").trim().to_string()
    } else {
        state.trim().to_string()
    };
    let opts_packed = options
        .iter()
        .map(|opt| format!("{mask} {}", opt.trim()))
        .collect::<Vec<_>>()
        .join(" ");
    format!("{prefix} {sep} {opts_packed}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_pack() {
        assert_eq!(
            pack_sequence("state text", "question?", &["a", "b"]),
            "question? state text [SEP] [MASK] a [MASK] b"
        );
        assert_eq!(pack_sequence("s", "", &["a"]), "s [SEP] [MASK] a");
        assert_eq!(pack_sequence("", "q", &[]), "q [SEP] ");
        assert_eq!(pack_sequence("  x  ", "", &["  y  "]), "x [SEP] [MASK] y");
    }
}
