//! ONNX Runtime backend for the von Option-Marker engine (**Stage 2, Route A**):
//! runs the exported fused encoder+scorer graph (`von-rs/scripts/export_onnx.py`)
//! through `ort` and reproduces the Python backend's decision math
//! (`src/von/backends/option_marker_backend.py`) operation-for-operation.

pub mod engine;
pub mod session;
pub mod special;
pub mod temperature;

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;
use von_core::EngineError;

pub use engine::OrtEngine;
pub use session::VonSession;
pub use special::{SpecialIds, VON_SPECIAL};
pub use temperature::{Calibration, NoulPrior};

/// Locate the pinned HF snapshot directory (config/tokenizer/calibration files).
///
/// Resolution order: `$VON_SNAPSHOT_DIR`, else `$VON_HF_REVISION` resolved
/// inside the standard HF cache (the same pin the Python oracle honors — one
/// env var pins both sides of the differential harness), else the
/// lexicographically newest snapshot under the cache for repo `wfzyx/von`
/// (the cache is content-addressed and shared with the Python oracle by
/// design — PORTING_RUST.md §7.1).
pub fn snapshot_dir() -> Result<PathBuf, EngineError> {
    if let Ok(dir) = std::env::var("VON_SNAPSHOT_DIR") {
        let p = PathBuf::from(dir);
        return if p.is_dir() {
            Ok(p)
        } else {
            Err(EngineError(format!(
                "VON_SNAPSHOT_DIR is not a directory: {}",
                p.display()
            )))
        };
    }
    let home = std::env::var("HOME")
        .map_err(|_| EngineError("cannot resolve home directory".to_string()))?;
    let snapshots = Path::new(&home).join(".cache/huggingface/hub/models--wfzyx--von/snapshots");
    if let Ok(rev) = std::env::var("VON_HF_REVISION")
        && !rev.is_empty()
    {
        let p = snapshots.join(&rev);
        return if p.is_dir() {
            Ok(p)
        } else {
            Err(EngineError(format!(
                "VON_HF_REVISION {rev} has no snapshot under {}",
                snapshots.display()
            )))
        };
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .map_err(|e| EngineError(format!("HF cache missing ({}): {e}", snapshots.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries
        .pop()
        .ok_or_else(|| EngineError(format!("no snapshots under {}", snapshots.display())))
}

/// Load the fast tokenizer (the same Rust `tokenizers` implementation the
/// Python backend calls into) from a snapshot directory.
///
/// The shipped `tokenizer.json` stores a 512-token truncation and a
/// BatchLongest padding config. The Python backend calls the bare tokenizer
/// (`tok(text, return_tensors="pt")`), which applies neither — transformers
/// only enables truncation/padding when explicitly asked — and the full
/// 8192-context sequence is the entire point of the long-state parity case.
/// Mirror that call behavior here.
pub fn load_tokenizer(snapshot: &Path) -> Result<Tokenizer, EngineError> {
    let mut tok = Tokenizer::from_file(snapshot.join("tokenizer.json"))
        .map_err(|e| EngineError(format!("tokenizer load failed: {e}")))?;
    let _ = tok.with_truncation(None);
    tok.with_padding(None);
    Ok(tok)
}
