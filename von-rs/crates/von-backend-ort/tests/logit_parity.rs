//! M2 gate: the exported ONNX graph, run through `ort` on CPU fp32, must
//! reproduce the oracle's pre-softmax logits (goldens/logits.jsonl) within
//! the plan's logit gate: max abs delta <= 1e-4. This is the same L-infinity
//! gate laya executed for the same architecture family (the plan's rel clause
//! at 1e-5 is unreachable in the ORT-vs-libtorch 1e-5..1e-4 band; the binding
//! downstream gates are argmax agreement and exact-after-rounding
//! probabilities).

use von_backend_ort::{VonSession, load_tokenizer, snapshot_dir};

fn artifacts_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts")
        .canonicalize()
        .expect("artifacts dir exists; run von-rs/scripts/export_onnx.py first")
}

const LOGIT_GATE: f64 = 1e-4;

#[test]
fn onnx_logits_match_python_oracle() {
    let logits_corpus: Vec<serde_json::Value> = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../goldens/logits.jsonl")
            .canonicalize()
            .unwrap(),
    )
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str(l).unwrap())
    .collect();

    let session = VonSession::from_file(&artifacts_dir().join("von-option-marker.onnx")).unwrap();
    let tok = load_tokenizer(&snapshot_dir().unwrap()).unwrap();

    let mut worst_abs = 0.0f64;
    let mut worst_id = String::new();
    let mut rows_checked = 0usize;

    for row in &logits_corpus {
        let id = row["id"].as_str().unwrap();
        // Choice/score probes record "logits"; noul probes record
        // "raw_logits". Zero-shot noul additionally carries the second
        // (null-state) pass; both graph executions must match.
        let fields: [(&str, &str); 2] =
            [("packed_text", "raw"), ("null_packed_text", "null_logits")];
        for (text_key, logits_kind) in fields {
            let Some(text) = row.get(text_key).and_then(|v| v.as_str()) else {
                continue;
            };
            let want: Vec<f64> = match logits_kind {
                "raw" => row
                    .get("logits")
                    .or_else(|| row.get("raw_logits"))
                    .expect("logits array present"),
                _ => row.get("null_logits").expect("null_logits array present"),
            }
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
            let enc = tok.encode(text, true).unwrap();
            let mask_positions: Vec<usize> = enc
                .get_ids()
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == von_backend_ort::VON_SPECIAL.mask)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                mask_positions.len(),
                want.len(),
                "mask count != option count for {id}/{text_key}"
            );
            let got = session
                .forward_with_masks(enc.get_ids(), &mask_positions)
                .unwrap()
                .logits;
            assert_eq!(
                got.len(),
                want.len(),
                "logit count mismatch for {id}/{text_key}"
            );
            for (g, w) in got.iter().zip(&want) {
                let d = (g - w).abs();
                if d > worst_abs {
                    worst_abs = d;
                    worst_id = format!("{id}/{text_key}");
                }
                assert!(
                    d <= LOGIT_GATE,
                    "logit gate failed for {id}/{text_key}: got {g}, want {w}, delta {d}"
                );
            }
        }
        rows_checked += 1;
    }
    assert_eq!(
        rows_checked, 45,
        "expected every golden probe row to be replayed"
    );
    println!("max |Δ logit| = {worst_abs:.3e} ({worst_id})");
}
