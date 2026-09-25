use clap::{Parser, Subcommand};
use std::path::PathBuf;

use von_core::{Engine, StubEngine, to_python_json_indent};

#[derive(Parser)]
#[command(
    name = "von",
    version = "1.0.0",
    about = "Von - Open Source System One Decision Model"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

fn alias_choices_vec() -> Vec<&'static str> {
    let mut aliases: Vec<&'static str> = von_core::VON_CURRENT_ALIASES.to_vec();
    aliases.sort();
    aliases
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Von System One HTTP server.
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "von-1.1", value_parser = clap::builder::PossibleValuesParser::new(alias_choices_vec()))]
        model: String,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = false)]
        reload: bool,
    },
    /// Classify input text among discrete choices.
    Decide {
        text: String,
        #[arg(short = 'c', long, required = true)]
        choices: String,
        #[arg(
            short = 'i',
            long,
            default_value = "Which option best describes the input?"
        )]
        instructions: String,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long)]
        base_url: Option<String>,
    },
    /// Evaluate a yes/no judgment (Noul) and return the probability.
    Judge {
        text: String,
        #[arg(short = 'i', long, required = true)]
        instructions: String,
        #[arg(long, default_value = "")]
        pos: String,
        #[arg(long, default_value = "")]
        neg: String,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long)]
        base_url: Option<String>,
    },
    /// Rate text on an ordered multi-level scale (Score).
    Rate {
        text: String,
        #[arg(short = 'l', long, required = true)]
        levels: String,
        #[arg(
            short = 'i',
            long,
            default_value = "Rate where the state falls on this scale:"
        )]
        instructions: String,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long)]
        base_url: Option<String>,
    },
    /// Evaluate a JSON request file containing state and questions.
    Eval {
        request_file: PathBuf,
        #[arg(long)]
        base_url: Option<String>,
    },
}

/// Stage 4 device surface: `auto` resolves to CUDA when the onnxruntime-gpu
/// dylib is available (falling back to CPU with a note, like `device.py`'s
/// auto path), `cuda|rocm|hip` all execute on the CUDA execution provider
/// (device.py maps rocm/hip the same way), and `mps|dml|directml` stay
/// rejected — those EPs are platform-specific (macOS/Windows) and not wired.
fn resolve_device(device: &str) -> Result<von_backend_ort::Device, String> {
    let lowered = device.to_ascii_lowercase();
    match lowered.as_str() {
        "" | "auto" => {
            if find_gpu_dylib_available() {
                Ok(von_backend_ort::Device::Cuda { device_id: 0 })
            } else {
                Ok(von_backend_ort::Device::Cpu)
            }
        }
        "cpu" => Ok(von_backend_ort::Device::Cpu),
        "cuda" | "rocm" | "hip" => Ok(von_backend_ort::Device::Cuda { device_id: 0 }),
        "mps" => Err(
            "Error: device 'mps' is not available in the Rust runtime (no MPS execution provider; supported: auto, cpu, cuda, rocm, hip)."
                .to_string(),
        ),
        "dml" | "directml" => Err(
            "Error: device 'dml' is not available in the Rust runtime (DirectML is Windows-only; supported: auto, cpu, cuda, rocm, hip)."
                .to_string(),
        ),
        other => Err(format!(
            "Error: device '{other}' is not a valid device (expected one of: auto, cpu, cuda, rocm, hip, mps, dml, directml)."
        )),
    }
}

fn find_gpu_dylib_available() -> bool {
    // Cheap existence probe without initializing the runtime: mirror the
    // engine's resolution order (env var, then the vendored ort-gpu tree).
    if let Ok(p) = std::env::var("VON_ORT_GPU_DYLIB")
        && !p.is_empty()
    {
        return std::path::Path::new(&p).is_file();
    }
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut base = manifest;
    for _ in 0..3 {
        if base.join("third_party/ort-gpu").is_dir() {
            return true;
        }
        base = base.parent().unwrap_or(base);
    }
    false
}

fn resolve_base_url(explicit: Option<&str>) -> Option<String> {
    if let Some(url) = explicit {
        return Some(url.trim_end_matches('/').to_string());
    }
    std::env::var("VON_BASE_URL")
        .ok()
        .map(|u| u.trim_end_matches('/').to_string())
}

fn exit_err(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn remote_system_one(base_url: &str, payload: &serde_json::Value) -> serde_json::Value {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|e| exit_err(&format!("Error: HTTP client construction failed: {e}")));
    let mut req = client
        .post(format!("{base_url}/v1/systemone"))
        .header("Content-Type", "application/json")
        .json(payload);
    if let Ok(key) = std::env::var("VON_API_KEY") {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    let resp = req
        .send()
        .unwrap_or_else(|e| exit_err(&format!("Error: request failed: {e}")));
    let status = resp.status();
    let text = resp
        .text()
        .unwrap_or_else(|e| exit_err(&format!("Error: reading response failed: {e}")));
    if !status.is_success() {
        exit_err(&format!("Error: server returned {status}: {text}"));
    }
    serde_json::from_str(&text)
        .unwrap_or_else(|e| exit_err(&format!("Error: invalid JSON response: {e}")))
}

fn run_system_one(
    base_url: Option<String>,
    state: serde_json::Value,
    questions: indexmap::IndexMap<String, serde_json::Value>,
    model: &str,
    device: von_backend_ort::Device,
) -> serde_json::Value {
    let payload = serde_json::json!({
        "model": model,
        "state": state,
        "questions": serde_json::Value::Object(questions.into_iter().collect()),
    });
    if let Some(url) = base_url {
        return remote_system_one(&url, &payload);
    }
    let engine = build_engine(device);
    let questions_map: indexmap::IndexMap<String, serde_json::Value> = payload["questions"]
        .as_object()
        .expect("questions object")
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let resp = engine
        .evaluate(&state, &questions_map, Some(model))
        .unwrap_or_else(|e| exit_err(&format!("Error: {e}")));
    serde_json::to_value(&resp).expect("response serialization cannot fail")
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve {
            host,
            port,
            model,
            device,
            reload,
        } => {
            if reload {
                exit_err("Error: --reload is not supported by the Rust server.");
            }
            let resolved = resolve_device(&device).unwrap_or_else(|e| exit_err(&e));
            println!(
                "Starting Von Decision Server [{} on {}] on http://{}:{}",
                model,
                match resolved {
                    von_backend_ort::Device::Cuda { .. } => "CUDA",
                    von_backend_ort::Device::Cpu => "CPU",
                },
                host,
                port
            );
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            runtime.block_on(async move {
                let router = von_server::build_engine_router(build_engine(resolved));
                let listener = tokio::net::TcpListener::bind((host.as_str(), port))
                    .await
                    .unwrap_or_else(|e| {
                        exit_err(&format!("Error: cannot bind {host}:{port}: {e}"))
                    });
                von_server::serve_router(listener, router)
                    .await
                    .expect("server error");
            });
        }
        Commands::Decide {
            text,
            choices,
            instructions,
            device,
            base_url,
        } => {
            let resolved = resolve_device(&device).unwrap_or_else(|e| exit_err(&e));
            let opts: Vec<String> = choices
                .split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect();
            if opts.is_empty() {
                exit_err("Error: At least one choice must be provided.");
            }
            let mut seen = std::collections::HashSet::new();
            if opts.iter().any(|c| !seen.insert(c)) {
                let listed = von_core::py_repr(&serde_json::Value::Array(
                    opts.iter()
                        .map(|c| serde_json::Value::String(c.clone()))
                        .collect(),
                ));
                exit_err(&format!(
                    "Error: Duplicate choices found in options list: {listed}"
                ));
            }
            let criteria: indexmap::IndexMap<String, serde_json::Value> = opts
                .iter()
                .map(|c| (c.clone(), serde_json::Value::Null))
                .collect();
            let mut questions = indexmap::IndexMap::new();
            questions.insert(
                "decision".to_string(),
                serde_json::json!({"type": "choice", "instructions": instructions, "criteria": criteria}),
            );
            let answer = run_system_one(
                resolve_base_url(base_url.as_deref()),
                serde_json::json!(text),
                questions,
                "von-latest",
                resolved,
            );
            let out = serde_json::json!({
                "choice": answer["answers"]["decision"]["choice"],
                "confidence": answer["answers"]["decision"]["confidence"],
                "probabilities": answer["answers"]["decision"]["probabilities"],
            });
            println!("{}", to_python_json_indent(&out, 2));
            if std::env::var("VON_TRACE").ok().as_deref() == Some("1") {
                use std::io::Write;
                eprintln!("[trace] decide printed; flushing stdout");
                std::io::stdout().flush().ok();
                eprintln!("[trace] stdout flushed; exiting");
            }
        }
        Commands::Judge {
            text,
            instructions,
            pos,
            neg,
            device,
            base_url,
        } => {
            let resolved = resolve_device(&device).unwrap_or_else(|e| exit_err(&e));
            let mut criteria = indexmap::IndexMap::new();
            if !pos.is_empty() {
                criteria.insert("true".to_string(), serde_json::json!(pos));
            }
            if !neg.is_empty() {
                criteria.insert("false".to_string(), serde_json::json!(neg));
            }
            let mut questions = indexmap::IndexMap::new();
            questions.insert(
                "judgment".to_string(),
                serde_json::json!({
                    "type": "noul",
                    "instructions": instructions.clone(),
                    "criteria": if criteria.is_empty() { serde_json::Value::Null } else { serde_json::Value::Object(criteria.into_iter().collect()) },
                }),
            );
            let answer = run_system_one(
                resolve_base_url(base_url.as_deref()),
                serde_json::json!(text),
                questions,
                "von-latest",
                resolved,
            );
            let out = serde_json::json!({
                "type": "noul",
                "instructions": instructions,
                "noul": answer["answers"]["judgment"]["noul"],
            });
            println!("{}", to_python_json_indent(&out, 2));
        }
        Commands::Rate {
            text,
            levels,
            instructions,
            device,
            base_url,
        } => {
            let resolved = resolve_device(&device).unwrap_or_else(|e| exit_err(&e));
            let lvl_list: Vec<serde_json::Value> = levels
                .split(',')
                .map(|l| serde_json::json!(l.trim()))
                .filter(|l| l.as_str().map(|s| !s.is_empty()).unwrap_or(false))
                .collect();
            if lvl_list.len() < 2 {
                exit_err("Error: At least two levels must be provided.");
            }
            let mut questions = indexmap::IndexMap::new();
            questions.insert(
                "rating".to_string(),
                serde_json::json!({"type": "score", "instructions": instructions, "criteria": lvl_list}),
            );
            let answer = run_system_one(
                resolve_base_url(base_url.as_deref()),
                serde_json::json!(text),
                questions,
                "von-latest",
                resolved,
            );
            let decision = &answer["answers"]["rating"];
            let out = serde_json::json!({
                "type": "score",
                "score": decision["score"],
                "confidence": decision["confidence"],
                "legend": decision["legend"],
                "probabilities": decision["probabilities"],
            });
            println!("{}", to_python_json_indent(&out, 2));
        }
        Commands::Eval {
            request_file,
            base_url,
        } => {
            let data: serde_json::Value = match std::fs::read_to_string(&request_file) {
                Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                    exit_err(&format!(
                        "Error: {} is not valid JSON: {e}",
                        request_file.display()
                    ))
                }),
                Err(e) => exit_err(&format!(
                    "Error: cannot read {}: {e}",
                    request_file.display()
                )),
            };
            let state = data.get("state").cloned();
            let questions = data.get("questions").cloned();
            let (state, questions) = match (state, questions) {
                (Some(s), Some(q)) => (s, q),
                _ => exit_err("Error: JSON must contain 'state' and 'questions' fields."),
            };
            let model = data
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("von-latest")
                .to_string();
            let q_map: indexmap::IndexMap<String, serde_json::Value> = questions
                .as_object()
                .expect("questions object")
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let resp = run_system_one(
                resolve_base_url(base_url.as_deref()),
                state,
                q_map,
                &model,
                resolve_device("auto").unwrap_or(von_backend_ort::Device::Cpu),
            );
            println!("{}", to_python_json_indent(&resp, 2));
        }
    }
}

/// Engine resolution for CLI commands and `serve`: the real ONNX backend
/// when the exported artifact is present (`VON_ONNX` or the repo-relative
/// default), else the Stage 1 stub so the CLI stays usable without weights.
/// A Cuda request is strict: a failed load exits — never silently serve stub
/// numbers from the wrong device.
fn build_engine(device: von_backend_ort::Device) -> std::sync::Arc<dyn Engine> {
    let candidates: Vec<PathBuf> = match std::env::var("VON_ONNX") {
        Ok(p) if !p.is_empty() => vec![PathBuf::from(p)],
        _ => vec![
            PathBuf::from("von-rs/artifacts/von-option-marker.onnx"),
            PathBuf::from("artifacts/von-option-marker.onnx"),
        ],
    };
    for path in candidates {
        if path.is_file() {
            match von_backend_ort::OrtEngine::from_artifacts_with_device(
                &path,
                &von_backend_ort::snapshot_dir().expect("HF snapshot resolution"),
                device,
            ) {
                Ok(engine) => {
                    eprintln!(
                        "[von] engine: ONNX backend ({})",
                        path.canonicalize()
                            .unwrap_or_else(|_| path.clone())
                            .display()
                    );
                    return std::sync::Arc::new(engine);
                }
                Err(e) => {
                    if matches!(device, von_backend_ort::Device::Cuda { .. }) {
                        exit_err(&format!("Error: CUDA engine load failed: {e}"));
                    }
                    eprintln!(
                        "[von] warning: ONNX backend failed to load ({e}); using stub engine"
                    );
                    return std::sync::Arc::new(StubEngine);
                }
            }
        }
    }
    if matches!(device, von_backend_ort::Device::Cuda { .. }) {
        exit_err("Error: CUDA requested but no ONNX artifact was found");
    }
    eprintln!("[von] warning: no ONNX artifact found (set VON_ONNX); using stub engine");
    std::sync::Arc::new(StubEngine)
}
