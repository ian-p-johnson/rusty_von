//! Special-token IDs, pinned from the Python `AutoTokenizer` (fixture
//! `tokens.json`), never discovered by alias probing.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialIds {
    pub mask: u32,
    pub sep: u32,
    pub cls: u32,
    pub pad: u32,
}

/// The pinned IDs for `wfzyx/von` (ModernBERT-large vocab, 50280 regular
/// tokens + specials). These must match `tokens.json > tokenizer`.
pub const VON_SPECIAL: SpecialIds = SpecialIds {
    mask: 50284,
    sep: 50282,
    cls: 50281,
    pad: 50283,
};
