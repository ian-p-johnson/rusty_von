//! The ONNX Runtime session over the exported fused Option-Marker graph
//! (`von-rs/scripts/export_onnx.py`). The graph *is* the Python graph, so the
//! encoder/scorer math cannot drift; this module only feeds it the same
//! tensors the backend does: `input_ids`, an all-ones `attention_mask` (the
//! backend never pads a single packed sequence), and `[MASK]` positions.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, Once, OnceLock};

use ort::session::{Session, builder::GraphOptimizationLevel};
use von_core::EngineError;

/// Compute device for the session, mirroring the Python runtime's
/// `device.py` semantics at the granularity the ORT runtime supports:
/// CUDA/ROCm both land on the CUDA execution provider (device.py maps
/// rocm/hip -> cuda the same way); MPS/DirectML have no wired provider and
/// are rejected at the CLI layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda { device_id: i32 },
}

/// ONNX Runtime library loading. With the `load-dynamic` feature the runtime
/// must come from a shared library; we deliberately load the **same ORT build
/// the Python oracle's onnxruntime wheel ships** for CPU work, because
/// kernel-level differences between ORT builds are a real numerics axis (the
/// pyke-downloaded 1.2x build drifts ~3e-4 from the torch oracle on this
/// graph; the 1.30 wheel dylib stays under ~5e-5). Resolution order:
///
/// 1. `$ORT_DYLIB_PATH` (explicit),
/// 2. for [`Device::Cuda`]: the **onnxruntime-gpu** wheel's dylib — the CPU
///    wheel's build has no CUDA EP, and ort-sys's downloaded CUDA kernels
///    lack sm_120/Blackwell SASS entirely (Stage 4; see `fetch_ort_gpu.sh`),
/// 3. for [`Device::Cpu`]: the oracle venv's CPU wheel dylib found relative
///    to the repo root,
/// 4. ort's own default search names.
///
/// The runtime is process-global and fixed on first use: one device per
/// process, exactly like the Python server (`VON_DEVICE` is read once at
/// startup there).
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
    find_dylib_in(&capi)
}

/// Locate the onnxruntime-gpu wheel's dylib: `$VON_ORT_GPU_DYLIB`, else the
/// vendored `third_party/ort-gpu` tree (see `scripts/fetch_ort_gpu.sh`),
/// searched from the manifest dir upward like the artifact candidates.
fn find_gpu_dylib() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("VON_ORT_GPU_DYLIB")
        && !p.is_empty()
    {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "VON_ORT_GPU_DYLIB is not a file: {}",
            path.display()
        ));
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut base = manifest;
    for _ in 0..3 {
        let candidate = base.join("third_party/ort-gpu");
        if candidate.is_dir()
            && let Some(dylib) = find_dylib_in_tree(&candidate)
        {
            return Ok(dylib);
        }
        base = base.parent().unwrap_or(base);
    }
    Err(
        "no onnxruntime-gpu dylib found for the CUDA execution provider. Fetch it once with \
         `bash von-rs/scripts/fetch_ort_gpu.sh` (vendored under von-rs/third_party/ort-gpu, \
         gitignored) or point VON_ORT_GPU_DYLIB at the wheel's libonnxruntime.so"
            .to_string(),
    )
}

fn find_dylib_in(dir: &Path) -> Option<PathBuf> {
    let mut libs: Vec<PathBuf> = std::fs::read_dir(dir)
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

/// Wheel layout search: the vendored tree nests the dylibs under
/// `onnxruntime/capi/`; walk a bounded depth below `root`.
fn find_dylib_in_tree(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut libs: Vec<PathBuf> = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().and_then(|n| n.to_str()) != Some("__pycache__") {
                    stack.push(path);
                }
            } else if matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some(n) if n.starts_with("libonnxruntime.so")
            ) {
                libs.push(path);
            }
        }
    }
    libs.sort();
    libs.pop()
}

fn ensure_runtime(device: Device) -> Result<(), EngineError> {
    RUNTIME_INIT.call_once(|| {
        let result = init_runtime(device);
        let _ = RUNTIME_RESULT.set(result);
    });
    match RUNTIME_RESULT.get().expect("runtime init ran") {
        Ok(()) => Ok(()),
        Err(msg) => Err(EngineError(msg.clone())),
    }
}

fn init_runtime(device: Device) -> Result<(), String> {
    let (path, _capi): (PathBuf, Option<PathBuf>) = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) if !p.is_empty() => {
            let path = PathBuf::from(&p);
            let capi = path.parent().map(|d| d.to_path_buf());
            (path, capi)
        }
        _ => match device {
            Device::Cuda { .. } => {
                let path = find_gpu_dylib()?;
                let capi = path.parent().map(|d| d.to_path_buf());
                (path, capi)
            }
            Device::Cpu => {
                let path = find_venv_dylib().ok_or_else(|| {
                    "no ONNX Runtime dylib found: set ORT_DYLIB_PATH (e.g. the \
                     onnxruntime wheel's libonnxruntime.so) or install the wheel \
                     into .venv"
                        .to_string()
                })?;
                let capi = path.parent().map(|d| d.to_path_buf());
                (path, capi)
            }
        },
    };
    // NOTE: no dlopen preloading of the provider libraries here — loading
    // `libonnxruntime_providers_cuda.so` before the main ORT library segfaults
    // in its ELF initializers. ORT resolves the providers relative to its own
    // directory once the main dylib is loaded; if a deployment layout defeats
    // that, set LD_LIBRARY_PATH to the wheel's capi directory (the laya
    // recipe) — that is an environment concern, not a code path.
    ort::init_from(&path)
        .map_err(|e| format!("failed to load ORT dylib {}: {e}", path.display()))?
        .commit();
    Ok(())
}

pub struct VonSession {
    /// Session pool. `run(&mut self)` in ort requires exclusive access, and
    /// this port keeps that conservatively safe: size 1 (the default) is the
    /// exact Stage 2/3 sequential semantics whose parity is proven; larger
    /// pools (VON_SESSION_POOL) allow concurrent forwards, gated by the same
    /// parity instruments (each ORT run is per-request stateless).
    pool: Mutex<VecDeque<Session>>,
    pool_cv: Condvar,
}

/// One forward pass over a packed sequence: `(logits, mask_positions)` ->
/// raw f32 logits, one per option, widened to f64 at the boundary exactly
/// like `Tensor.tolist()`.
pub struct Forward {
    pub logits: Vec<f64>,
}

impl VonSession {
    pub fn from_file(path: &Path) -> Result<Self, EngineError> {
        Self::from_file_with_device(path, Device::Cpu)
    }

    pub fn from_file_with_device(path: &Path, device: Device) -> Result<Self, EngineError> {
        let pool_size = pool_size();
        ensure_runtime(device)?;
        let mut sessions = VecDeque::with_capacity(pool_size);
        for _ in 0..pool_size {
            sessions.push_back(build_session(path, device)?);
        }
        Ok(VonSession {
            pool: Mutex::new(sessions),
            pool_cv: Condvar::new(),
        })
    }

    /// The runtime entry point: one packed sequence plus its `[MASK]`
    /// positions; returns one logit per option in position order.
    ///
    /// Checks a session out of the pool, runs the graph, checks it back in.
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
        let trace = std::env::var("VON_TRACE").ok().as_deref() == Some("1");
        if trace {
            eprintln!("[trace] tensors built; checking out session");
        }

        let mut guard = self.pool.lock().expect("session pool poisoned");
        while guard.is_empty() {
            guard = self
                .pool_cv
                .wait(guard)
                .expect("session pool poisoned (wait)");
        }
        let mut session = guard.pop_front().expect("checked non-empty");
        drop(guard); // release the pool lock for the duration of the run
        if trace {
            eprintln!("[trace] session checked out; running");
        }

        let run = session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "mask_pos" => mask_pos,
        ]);
        if trace {
            eprintln!("[trace] run returned");
        }
        // SessionOutputs borrows the session, so extract before check-in;
        // the helper confines the borrow so `session` can move back (the
        // borrow must not extend past `extract`).
        fn extract(
            run: Result<ort::session::SessionOutputs<'_>, ort::Error>,
        ) -> Result<Vec<f64>, EngineError> {
            let trace = std::env::var("VON_TRACE").ok().as_deref() == Some("1");
            let outputs = run.map_err(ort_err)?;
            if trace {
                eprintln!("[trace] outputs mapped");
            }
            let (shape, data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            if trace {
                eprintln!("[trace] tensor extracted");
            }
            debug_assert_eq!(shape.first().copied(), Some(1), "batch is always 1");
            let out = data.iter().map(|f| f64::from(*f)).collect();
            if trace {
                eprintln!("[trace] extracted to host vec");
            }
            Ok(out)
        }
        let extracted = extract(run);
        if trace {
            eprintln!("[trace] extract done; checking in");
        }

        self.pool
            .lock()
            .expect("session pool poisoned")
            .push_back(session);
        self.pool_cv.notify_one();

        Ok(Forward { logits: extracted? })
    }
}

/// Pool size: `VON_SESSION_POOL` (default 1 = sequential, the parity-proven
/// default; 1 is also clamped for values < 1).
fn pool_size() -> usize {
    std::env::var("VON_SESSION_POOL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| n.max(1))
        .unwrap_or(1)
}

fn build_session(path: &Path, device: Device) -> Result<Session, EngineError> {
    let mut builder = Session::builder()
        .map_err(|e| EngineError(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| EngineError(e.to_string()))?;
    // Debug/tuning knobs (VON_ORT_* env): the CUDA EP + thread-pool defaults
    // deadlocked on first run (Stage 4 investigation); these allow bisecting
    // thread-pool configurations without a rebuild.
    if let Ok(n) = std::env::var("VON_ORT_INTRA_THREADS")
        && let Ok(n) = n.parse::<i32>() {
            builder = builder
                .with_config_entry("session.intra_op.thread_num", n.to_string())
                .map_err(|e| EngineError(e.to_string()))?;
        }
    if std::env::var("VON_ORT_NO_SPIN").ok().as_deref() == Some("1") {
        builder = builder
            .with_intra_op_spinning(false)
            .map_err(|e| EngineError(e.to_string()))?;
    }
    if let Device::Cuda { device_id } = device {
        #[cfg(feature = "cuda")]
        {
            // Strict device semantics, mirroring device.py (which raises when
            // torch.cuda is unavailable): check the driver for a usable GPU
            // up front. The EP registration itself hard-errors when the
            // provider library or CUDA runtime cannot be loaded. CPU fallback
            // stays ENABLED because ORT deliberately places shape/striding
            // ops on the CPU EP even in a healthy CUDA session.
            // NOTE: cuInit here (driver probe) deadlocked the first CUDA run
            // — an early cuInit from a bare dlopen'ed libcuda conflicts with
            // ORT's own driver init. GPU presence is verified differently:
            // see gpu_present().
            if !gpu_present() {
                return Err(EngineError(
                    "no NVIDIA GPU visible (checked /proc/driver/nvidia and nvidia-smi)"
                        .to_string(),
                ));
            }
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(device_id)
                    // Grow the arena by exactly what is requested instead of
                    // doubling: long sequences need multi-GB attention
                    // buffers and the next-power-of-two strategy wastes VRAM
                    // a 12GB laptop GPU does not have.
                    .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                    .build()])
                .map_err(|e| EngineError(e.to_string()))?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device_id;
            return Err(EngineError(
                "CUDA requested but this build lacks the `cuda` feature".to_string(),
            ));
        }
    }
    builder.commit_from_file(path).map_err(ort_err)
}

/// Driver-level GPU presence check WITHOUT touching libcuda (cuInit before
/// ORT's own driver init deadlocks the first CUDA run — found the hard way).
/// /proc/driver/nvidia lists one directory per GPU on any loaded-driver box;
/// nvidia-smi is the fallback.
fn gpu_present() -> bool {
    if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in entries.filter_map(|e| e.ok()) {
            if entry.path().is_dir() {
                return true;
            }
        }
    }
    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false)
}

fn ort_err(e: ort::Error) -> EngineError {
    EngineError(e.to_string())
}
