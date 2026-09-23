//! The ONNX Runtime session over the exported fused Option-Marker graph
//! (`von-rs/scripts/export_onnx.py`). The graph *is* the Python graph, so the
//! encoder/scorer math cannot drift; this module only feeds it the same
//! tensors the backend does: `input_ids`, an all-ones `attention_mask` (the
//! backend never pads a single packed sequence), and `[MASK]` positions.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

use ort::session::{Session, builder::GraphOptimizationLevel};
use von_core::EngineError;

/// ONNX Runtime library loading. With the `load-dynamic` feature the runtime
/// must come from a shared library; we deliberately load the **same ORT build
/// the Python oracle's onnxruntime wheel ships**, because kernel-level
/// differences between ORT builds are a real numerics axis (the pyke
/// -downloaded 1.2x build drifts ~3e-4 from the torch oracle on this graph;
/// the 1.30 wheel dylib stays under ~5e-5). Resolution order:
///
/// 1. `$ORT_DYLIB_PATH` (explicit; Stage 4 points this at the
///    onnxruntime-gpu wheel for the Blackwell sm_120 fix),
/// 2. the oracle venv's CPU wheel dylib found relative to the repo root,
/// 3. ort's own default search names.
static RUNTIME_INIT: Once = Once::new();
static RUNTIME_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

fn find_venv_dylib() -> Option<PathBuf> {
    let capi = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.venv/lib")
        .read_dir()
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("site-packages/onnxruntime/capi"))
        .find(|p| p.is_dir())?;
    let mut libs: Vec<PathBuf> = std::fs::read_dir(&capi)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(p.file_name().and_then(|n| n.to_str()), Some(n) if n.starts_with("libonnxruntime.so"))
        })
        .collect();
    libs.sort();
    libs.pop()
}

fn ensure_runtime() -> Result<(), EngineError> {
    RUNTIME_INIT.call_once(|| {
        let result = (|| -> Result<(), String> {
            let path = match std::env::var("ORT_DYLIB_PATH") {
                Ok(p) if !p.is_empty() => PathBuf::from(p),
                _ => find_venv_dylib().ok_or_else(|| {
                    "no ONNX Runtime dylib found: set ORT_DYLIB_PATH (e.g. the \
                     onnxruntime wheel's libonnxruntime.so) or install the wheel \
                     into .venv"
                        .to_string()
                })?,
            };
            ort::init_from(&path)
                .map_err(|e| format!("failed to load ORT dylib {}: {e}", path.display()))?
                .commit();
            Ok(())
        })();
        let _ = RUNTIME_RESULT.set(result);
    });
    match RUNTIME_RESULT.get().expect("runtime init ran") {
        Ok(()) => Ok(()),
        Err(msg) => Err(EngineError(msg.clone())),
    }
}

pub struct VonSession {
    session: Mutex<Session>,
}

/// One forward pass over a packed sequence: `(logits, mask_positions)` ->
/// raw f32 logits, one per option, widened to f64 at the boundary exactly
/// like `Tensor.tolist()`.
pub struct Forward {
    pub logits: Vec<f64>,
}

impl VonSession {
    pub fn from_file(path: &Path) -> Result<Self, EngineError> {
        ensure_runtime()?;
        let session = Session::builder()
            .map_err(|e| EngineError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EngineError(e.to_string()))?
            .commit_from_file(path)
            .map_err(ort_err)?;
        Ok(VonSession {
            session: Mutex::new(session),
        })
    }

    /// The runtime entry point: one packed sequence plus its `[MASK]`
    /// positions; returns one logit per option in position order.
    pub fn forward_with_masks(
        &self,
        input_ids: &[u32],
        mask_positions: &[usize],
    ) -> Result<Forward, EngineError> {
        let n = input_ids.len() as i64;
        let k = mask_positions.len() as i64;
        let ids: Vec<i64> = input_ids.iter().map(|t| i64::from(*t)).collect();
        let ones = vec![1i64; input_ids.len()];
        let pos: Vec<i64> = mask_positions.iter().map(|p| *p as i64).collect();

        let input_ids = ort::value::Tensor::from_array((vec![1, n], ids)).map_err(ort_err)?;
        let attention_mask = ort::value::Tensor::from_array((vec![1, n], ones)).map_err(ort_err)?;
        let mask_pos = ort::value::Tensor::from_array((vec![1, k], pos)).map_err(ort_err)?;

        let mut session = self.session.lock().expect("session lock poisoned");
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "mask_pos" => mask_pos,
            ])
            .map_err(ort_err)?;

        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let dims = shape.to_vec();
        debug_assert_eq!(dims.first().copied(), Some(1), "batch is always 1");
        let logits = data.iter().map(|f| f64::from(*f)).collect();
        Ok(Forward { logits })
    }
}

fn ort_err(e: ort::Error) -> EngineError {
    EngineError(e.to_string())
}
