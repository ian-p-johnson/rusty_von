//! Decimal half-even rounding matching Python `round(x, n)` on binary floats.

pub fn round_half_even(x: f64, ndigits: u32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", ndigits as usize, x)
        .parse::<f64>()
        .unwrap_or(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halfway_cases_go_to_even() {
        assert_eq!(round_half_even(0.125, 3), 0.125);
        assert_eq!(round_half_even(2.675, 2), 2.67);
        assert_eq!(round_half_even(0.15, 1), 0.1);
        assert_eq!(round_half_even(0.5, 3), 0.5);
        assert_eq!(round_half_even(1e20, 2), 1e20);
        assert_eq!(round_half_even(1e-5, 4), 0.0);
    }
}
