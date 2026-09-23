//! Pure-logic port of the von runtime: state rendering, packing, rounding,
//! usage accounting, engine orchestration and the Stage 1 stub engine.

pub mod engine;
pub mod fmt;
pub mod pack;
pub mod pyjson;
pub mod rounding;
pub mod usage;

pub use engine::{
    Engine, EngineError, QuestionBackend, StubEngine, VON_CURRENT_ALIASES, VON_MODEL_ID,
    VON_VERSION, decide, evaluate_with, is_supported_alias, judge, python_join_error_message, rate,
    resolved_model_id, score_level_description, unknown_model_message,
};
pub use fmt::format_state;
pub use pack::pack_sequence;
pub use pyjson::{to_python_json, to_python_json_indent};
pub use rounding::round_half_even;
pub use usage::compute_usage;
pub use von_types::{Answer, Question, SystemOneResponse, Usage, py_repr, py_str};
