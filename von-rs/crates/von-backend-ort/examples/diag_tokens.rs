//! Diagnostic: compare Rust tokenization against fixtures/tokens.json and
//! print the first divergence context. Run:
//!   cargo run -p von-backend-ort --example diag_tokens

use std::path::Path;

use von_backend_ort::{load_tokenizer, snapshot_dir};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tokens.json");
    let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root).unwrap()).unwrap();
    let snapshot = snapshot_dir().unwrap();
    let tok = load_tokenizer(&snapshot).unwrap();

    for enc in fx["encodings"].as_array().unwrap() {
        let text = enc["text"].as_str().unwrap();
        let want: Vec<u32> = enc["input_ids"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got_ids = tok.encode(text, true).unwrap().get_ids().to_vec();
        if got_ids == want {
            println!("{}: OK ({} tokens)", enc["id"].as_str().unwrap(), want.len());
            continue;
        }
        println!(
            "{}: DIVERGE rust={} python={}",
            enc["id"].as_str().unwrap(),
            got_ids.len(),
            want.len()
        );
        let n = got_ids.len().min(want.len());
        let mut i = 0;
        while i < n && got_ids[i] == want[i] {
            i += 1;
        }
        let lo = i.saturating_sub(8);
        let hi = (i + 8).min(n.max(got_ids.len().min(want.len())));
        println!("first divergence at token {i}");
        println!("  rust  [{lo}..{hi}]: {:?}", &got_ids[lo..hi.min(got_ids.len())]);
        println!("  python[{lo}..{hi}]: {:?}", &want[lo..hi.min(want.len())]);
        println!("  rust  text: {:?}", tok.decode(&got_ids[lo..hi.min(got_ids.len())], false));
        println!("  python text: {:?}", tok.decode(&want[lo..hi.min(want.len())], false));
    }
}
