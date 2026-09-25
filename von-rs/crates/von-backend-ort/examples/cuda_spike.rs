//! Minimal CUDA spike: load the GPU dylib, build a CUDA session over the von
//! graph, run one tiny forward. Bisection tool for the Stage 4 CUDA hang.
//!
//! Usage: cargo run -p von-backend-ort --example cuda_spike -- <onnx path>
//! Env: ORT_DYLIB_PATH (or the vendored ort-gpu tree), VON_SPIKE_OPTIONS=1
//! applies the session options the real engine uses (opt level etc.).

use std::path::PathBuf;

use ort::session::Session;

fn find_gpu_dylib() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(p));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut base = manifest.clone();
    for _ in 0..3 {
        let candidate = base.join("third_party/ort-gpu");
        if candidate.is_dir() {
            let mut stack = vec![candidate];
            let mut libs = Vec::new();
            while let Some(dir) = stack.pop() {
                let Ok(entries) = std::fs::read_dir(&dir) else {
                    continue;
                };
                for entry in entries.filter_map(|e| e.ok()) {
                    let p = entry.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("libonnxruntime.so"))
                        .unwrap_or(false)
                    {
                        libs.push(p);
                    }
                }
            }
            libs.sort();
            return libs.pop();
        }
        base = base.parent()?.to_path_buf();
    }
    None
}

fn main() {
    let onnx = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/von-option-marker.onnx".to_string());
    let dylib = find_gpu_dylib().expect("no GPU dylib");
    eprintln!("[spike] dylib: {}", dylib.display());

    ort::init_from(&dylib).expect("init").commit();
    eprintln!("[spike] runtime loaded");

    use ort::session::builder::GraphOptimizationLevel;
    let ep = ort::ep::CUDA::default().with_device_id(0).build();
    let mut builder = Session::builder()
        .expect("builder")
        .with_execution_providers([ep])
        .expect("ep");
    if std::env::var("VON_SPIKE_OPTIONS").ok().as_deref() == Some("1") {
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("opt");
    }
    let mut session = builder.commit_from_file(&onnx).expect("commit");
    eprintln!("[spike] session committed");

    let ids: Vec<i64> = (0..32)
        .map(|i| if i == 0 { 50284 } else { 100 + i as i64 })
        .collect();
    let ones = vec![1i64; 32];
    let pos: Vec<i64> = vec![1, 2];

    let a = ort::value::Tensor::from_array((vec![1, 32], ids)).unwrap();
    let b = ort::value::Tensor::from_array((vec![1, 32], ones)).unwrap();
    let c = ort::value::Tensor::from_array((vec![1, 2], pos)).unwrap();
    eprintln!("[spike] tensors ready");

    let out = session
        .run(ort::inputs!["input_ids" => a, "attention_mask" => b, "mask_pos" => c])
        .expect("run");
    eprintln!("[spike] run complete");

    let (_shape, data) = out["logits"].try_extract_tensor::<f32>().unwrap();
    let heads: Vec<f32> = data.iter().copied().take(4).collect();
    eprintln!("[spike] logits[0..4]: {:?}", heads);
}
