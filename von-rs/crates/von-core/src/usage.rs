//! Token usage accounting (`input_tokens = max(1, len//4)` floors).

use von_types::Usage;

pub fn compute_usage(state_str: &str, total_q_chars: usize, n_answers: usize) -> Usage {
    let state_tokens = std::cmp::max(1, state_str.chars().count() / 4);
    let q_tokens = std::cmp::max(1, total_q_chars / 4);
    Usage {
        input_tokens: (state_tokens + q_tokens) as u64,
        output_tokens: n_answers as u64,
    }
}
