//! Answer, usage and response-envelope schemas (field order is wire-pinned).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

pub type NoulAnswer = Answer;
pub type ChoiceAnswer = Answer;
pub type ScoreAnswer = Answer;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Answer {
    #[serde(rename = "noul")]
    Noul { noul: f64 },
    #[serde(rename = "choice")]
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    #[serde(rename = "score")]
    Score {
        score: f64,
        confidence: f64,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f64>,
    },
}

impl Answer {
    pub fn confidence(&self) -> f64 {
        match self {
            Answer::Noul { .. } => 1.0,
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => *confidence,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("answer serialization cannot fail")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

impl SystemOneResponse {
    pub fn get(&self, q_id: &str) -> Option<&Answer> {
        self.answers.get(q_id)
    }
}
