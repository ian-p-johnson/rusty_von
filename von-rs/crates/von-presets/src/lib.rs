//! Static preset question definitions, parity-ported from `src/von/presets.py`.

use indexmap::IndexMap;
use serde_json::{Value, json};

pub type Preset = IndexMap<String, Value>;

fn obj(pairs: &[(&str, Value)]) -> Value {
    let mut map = serde_json::Map::new();
    for (k, v) in pairs {
        map.insert((*k).to_string(), v.clone());
    }
    Value::Object(map)
}

fn noul(instructions: &str, criteria: &[(&str, &str)]) -> Value {
    obj(&[
        ("type", json!("noul")),
        ("instructions", json!(instructions)),
        (
            "criteria",
            if criteria.is_empty() {
                Value::Null
            } else {
                Value::Object(
                    criteria
                        .iter()
                        .map(|(k, v)| ((*k).to_string(), json!(v)))
                        .collect(),
                )
            },
        ),
    ])
}

fn choice(instructions: &str, criteria: &[(&str, &str)]) -> Value {
    obj(&[
        ("type", json!("choice")),
        ("instructions", json!(instructions)),
        (
            "criteria",
            Value::Object(
                criteria
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), json!(v)))
                    .collect(),
            ),
        ),
    ])
}

fn score(instructions: &str, criteria: &[&str]) -> Value {
    obj(&[
        ("type", json!("score")),
        ("instructions", json!(instructions)),
        (
            "criteria",
            Value::Array(criteria.iter().map(|c| json!(c)).collect()),
        ),
    ])
}

pub fn triage_preset() -> Preset {
    let mut m = IndexMap::new();
    m.insert(
        "intent".to_string(),
        choice(
            "What is the primary customer intent in the message?",
            &[
                (
                    "refund",
                    "Requesting money back, refund, or duplicate billing reversal",
                ),
                (
                    "technical_help",
                    "Reporting a bug, API error, 500 downtime, or integration issue",
                ),
                (
                    "billing_question",
                    "Questions about invoices, subscription plans, or payment methods",
                ),
                (
                    "cancellation",
                    "Requesting account closure, cancellation, or downgrading",
                ),
                (
                    "general_info",
                    "Inquiring about documentation, pricing tiers, or how-to guidance",
                ),
            ],
        ),
    );
    m.insert(
        "is_urgent".to_string(),
        noul(
            "Does the customer communicate extreme urgency, critical outage, or impending deadline?",
            &[
                ("true", "Urgent, production down, emergency, immediate attention needed"),
                ("false", "Routine question, low priority, general feedback"),
            ],
        ),
    );
    m.insert(
        "frustration".to_string(),
        score(
            "Rate the customer frustration level.",
            &[
                "Calm and polite",
                "Slightly concerned or asking for status",
                "Visibly frustrated or annoyed",
                "Extremely angry, threatening legal action or cancellation",
            ],
        ),
    );
    m.insert(
        "churn_risk".to_string(),
        noul(
            "Does the message indicate high risk of the customer leaving or churning?",
            &[
                (
                    "true",
                    "Threatening to switch to competitors, cancel contract, or stop using product",
                ),
                (
                    "false",
                    "Committed user asking for help, no mention of leaving",
                ),
            ],
        ),
    );
    m
}

pub fn email_preset(custom_categories: Option<IndexMap<String, String>>) -> Preset {
    let categories: Vec<(String, String)> = match custom_categories {
        Some(map) => map.into_iter().collect(),
        None => vec![
            (
                "billing".to_string(),
                "Invoices, payments, credit cards, pricing questions".to_string(),
            ),
            (
                "engineering".to_string(),
                "Bug reports, API failures, stack traces, system outages".to_string(),
            ),
            (
                "sales".to_string(),
                "Enterprise demos, contract inquiries, volume discounts".to_string(),
            ),
            (
                "security".to_string(),
                "Phishing reports, suspicious access, vulnerability disclosures".to_string(),
            ),
            (
                "general".to_string(),
                "General questions or uncategorized inquiries".to_string(),
            ),
        ],
    };
    let mut m = IndexMap::new();
    m.insert(
        "destination".to_string(),
        choice(
            "Which internal team should handle this email?",
            &categories
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>(),
        ),
    );
    m.insert(
        "is_spam_or_phishing".to_string(),
        noul(
            "Is this email an unsolicited sales pitch, scam, or phishing attempt?",
            &[
                (
                    "true",
                    "Spam, promotional blast, credential harvesting, phishing",
                ),
                ("false", "Legitimate user or customer inquiry"),
            ],
        ),
    );
    m.insert(
        "priority".to_string(),
        score(
            "What priority level should be assigned to this email?",
            &[
                "Low: Newsletter, informational, no action required",
                "Medium: Standard inquiry with 24-48hr SLA",
                "High: Blocking issue affecting paying customer",
                "Critical: Security breach, legal threat, or severe production impact",
            ],
        ),
    );
    m
}

pub fn moderation_preset() -> Preset {
    let mut m = IndexMap::new();
    m.insert(
        "policy_violation".to_string(),
        choice(
            "Does this content violate acceptable use policies?",
            &[
                (
                    "clean",
                    "Content is safe, constructive, and follows community guidelines",
                ),
                (
                    "harassment",
                    "Direct personal attacks, bullying, threats, or hate speech",
                ),
                (
                    "spam",
                    "Repetitive links, commercial spam, crypto scams, or bot text",
                ),
                (
                    "sensitive",
                    "Explicit adult content, graphic violence, or illegal goods",
                ),
            ],
        ),
    );
    m.insert(
        "should_block".to_string(),
        noul(
            "Should this content be immediately blocked from publication?",
            &[
                ("true", "Clear violation requiring immediate rejection"),
                (
                    "false",
                    "Safe or borderline content that can be published or reviewed",
                ),
            ],
        ),
    );
    m.insert(
        "severity".to_string(),
        score(
            "Rate the severity of the content risk.",
            &[
                "Safe: Compliant content",
                "Low: Minor profanity or mild uncivil behavior",
                "Medium: Aggressive tone, self-promotion, or borderline spam",
                "High: Severe violation, harassment, or malicious payload",
            ],
        ),
    );
    m
}

pub fn security_preset() -> Preset {
    let mut m = IndexMap::new();
    m.insert(
        "event_type".to_string(),
        choice(
            "Classify the observed security or authentication anomaly.",
            &[
                (
                    "benign",
                    "Expected user activity, legitimate IP change, or normal login",
                ),
                (
                    "credential_stuffing",
                    "Rapid succession of failed logins across multiple accounts",
                ),
                (
                    "brute_force",
                    "Repeated failed attempts targeting a single high-value account",
                ),
                (
                    "privilege_escalation",
                    "Attempting unauthorized administrative or sudo operations",
                ),
                (
                    "data_exfiltration",
                    "Abnormal volume of export requests or bulk database downloads",
                ),
            ],
        ),
    );
    m.insert(
        "is_threat".to_string(),
        noul(
            "Does this state represent an active, confirmed malicious security threat?",
            &[
                (
                    "true",
                    "Active cyber attack, intrusion, or unauthorized compromise",
                ),
                (
                    "false",
                    "Normal operational glitch, user error, or benign variance",
                ),
            ],
        ),
    );
    m.insert(
        "severity".to_string(),
        score(
            "Rate the incident severity.",
            &[
                "Informational: Logged for audit trail, no action",
                "Warning: Suspicious variance, rate-limit triggered",
                "Elevated: Incident responder paged for triage",
                "Critical: Active breach, immediate token revocation and IP ban",
            ],
        ),
    );
    m
}
