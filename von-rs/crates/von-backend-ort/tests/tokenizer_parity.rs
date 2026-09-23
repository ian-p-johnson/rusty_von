//! M1 gate: the Rust `tokenizers` crate must reproduce the Python tokenizer's
//! token IDs exactly for every packed text in the golden corpus, and the
//! `add_special_tokens=False` state counts used by the calibration feature.
//! Fixture: `von-rs/fixtures/tokens.json` (from `benchmarks/dump_tokens.py`).

use std::path::{Path, PathBuf};

use von_backend_ort::{VON_SPECIAL, load_tokenizer, snapshot_dir};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .canonicalize()
        .expect("fixtures dir exists")
}

#[test]
fn special_ids_match_fixture() {
    let fx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures_dir().join("tokens.json")).unwrap(),
    )
    .unwrap();
    let meta = &fx["tokenizer"];
    assert_eq!(VON_SPECIAL.mask, meta["mask_token_id"].as_u64().unwrap() as u32);
    assert_eq!(VON_SPECIAL.sep, meta["sep_token_id"].as_u64().unwrap() as u32);
    assert_eq!(VON_SPECIAL.cls, meta["cls_token_id"].as_u64().unwrap() as u32);
    assert_eq!(VON_SPECIAL.pad, meta["pad_token_id"].as_u64().unwrap() as u32);

    // Pin the tokenizer.json bytes against the manifest-captured hash so a
    // silent cache refresh cannot invalidate the corpus.
    let snapshot = snapshot_dir().unwrap();
    let bytes = std::fs::read(snapshot.join("tokenizer.json")).unwrap();
    use sha2::Digest;
    let got = sha2::Sha256::digest(&bytes);
    let want = meta["tokenizer_json_sha256"].as_str().unwrap();
    assert_eq!(format!("{got:x}"), want, "tokenizer.json drifted from the pinned revision");
}

#[test]
fn token_ids_match_python_for_every_golden_encoding() {
    let fx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures_dir().join("tokens.json")).unwrap(),
    )
    .unwrap();

    let snapshot = snapshot_dir().unwrap();
    let tok = load_tokenizer(&snapshot).unwrap();

    let mut checked = 0usize;
    for enc in fx["encodings"].as_array().unwrap() {
        let text = enc["text"].as_str().unwrap();
        let want_ids: Vec<u32> = enc["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text, true).unwrap();
        assert_eq!(
            got.get_ids(),
            want_ids.as_slice(),
            "token ID divergence for {}",
            enc["id"].as_str().unwrap()
        );
        // Mask positions must line up too (the head gathers at these).
        let want_masks: Vec<usize> = enc["mask_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let got_masks: Vec<usize> = got
            .get_ids()
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == VON_SPECIAL.mask)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(got_masks, want_masks, "mask positions diverge for {}", enc["id"].as_str().unwrap());
        checked += 1;
    }
    assert_eq!(checked, 52, "expected the full encoding corpus");

    for sc in fx["state_token_counts"].as_array().unwrap() {
        let state = sc["state"].as_str().unwrap();
        let want = sc["count"].as_u64().unwrap() as usize;
        let got = tok.encode(state, false).unwrap().get_ids().len();
        assert_eq!(
            got, want,
            "add_special_tokens=False count diverges for {}",
            sc["id"].as_str().unwrap()
        );
    }
    assert_eq!(fx["state_token_counts"].as_array().unwrap().len(), 45);
}
