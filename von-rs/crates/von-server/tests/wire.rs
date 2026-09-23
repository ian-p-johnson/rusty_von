#![allow(clippy::await_holding_lock)]

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;
use von_server::build_router;

static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load_wire() -> Vec<Value> {
    let path = fixtures_dir().join("wire_errors.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

async fn send(
    method: &str,
    path: &str,
    body: Option<&str>,
    auth: Option<&str>,
    extra_headers: &[(String, String)],
) -> (StatusCode, bytes::Bytes, Vec<(String, String)>) {
    let router = build_router();
    let builder = Request::builder()
        .method(Method::from_bytes(method.as_bytes()).unwrap())
        .uri(path);
    let mut req = match body {
        Some(b) => builder
            .header("content-type", "application/json")
            .body(Body::from(b.to_string()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    if let Some(a) = auth {
        req.headers_mut()
            .insert("authorization", a.parse().unwrap());
    }
    for (k, v) in extra_headers {
        let hv: axum::http::HeaderValue = v.parse().unwrap();
        req.headers_mut().insert(
            axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
            hv,
        );
    }
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    let headers: Vec<(String, String)> = resp
        .headers()
        .iter()
        .map(|(k, v)| {
            (
                k.as_str().to_string(),
                v.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, body, headers)
}

#[tokio::test]
async fn wire_gate_replays_all_captured_cases() {
    let _guard = guard();
    let cases = load_wire();
    assert!(cases.len() >= 30);
    for case in cases {
        let id = case["id"].as_str().unwrap().to_string();
        if id.starts_with("auth_") {
            unsafe { std::env::set_var("VON_API_KEY", "sekrit") };
        } else {
            unsafe { std::env::remove_var("VON_API_KEY") };
        }
        let method = case["method"].as_str().unwrap().to_string();
        let path = case["path"].as_str().unwrap().to_string();
        let body = case["body"].as_str().map(|s| s.to_string());
        let auth = case["auth"].as_str().map(|s| s.to_string());
        let expected_status = case["status"].as_u64().unwrap() as u16;
        let expected_body = case["response"].as_str().unwrap();
        let expected_headers = case["response_headers"].as_object().unwrap();

        let extra: Vec<(String, String)> = case["request_headers"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter(|(k, _)| k.as_str() != "content-type" && k.as_str() != "authorization")
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let _ = &extra;
        let (status, got_body, headers) =
            send(&method, &path, body.as_deref(), auth.as_deref(), &extra).await;

        assert_eq!(status.as_u16(), expected_status, "status for {id}");
        let got = String::from_utf8_lossy(&got_body);
        assert_eq!(got, expected_body, "body for {id}");

        for (k, v) in expected_headers {
            let expected_val = v.as_str().unwrap();
            let actual = headers
                .iter()
                .find(|(hk, _)| hk.eq_ignore_ascii_case(k))
                .map(|(_, hv)| hv.as_str());
            assert_eq!(actual, Some(expected_val), "header {k} for {id}");
        }
    }
}

#[tokio::test]
async fn stub_engine_positive_path_envelope() {
    let _guard = guard();
    let (status, body, _) = send(
        "POST",
        "/v1/systemone",
        Some(r#"{"model": "von-latest", "state": "Customer requests refund", "questions": {"decision": {"type": "choice", "instructions": "Which one?", "criteria": {"only": "The single possible answer"}}}}"#),
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8_lossy(&body).to_string();
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["model"], "von-1.1.0");
    assert_eq!(v["answers"]["decision"]["type"], "choice");
    assert_eq!(v["answers"]["decision"]["choice"], "only");
    assert_eq!(v["answers"]["decision"]["probabilities"]["only"], 1.0);
    assert_eq!(v["answers"]["decision"]["confidence"], 1.0);
    assert_eq!(v["usage"]["input_tokens"], 8);
    assert_eq!(v["usage"]["output_tokens"], 1);
    let expected_order = ["model", "answers", "usage"];
    let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    assert_eq!(keys, expected_order);
    let answer_keys: Vec<&str> = v["answers"]["decision"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    assert_eq!(
        answer_keys,
        ["type", "choice", "probabilities", "confidence"]
    );
    assert_eq!(
        text,
        r#"{"model":"von-1.1.0","answers":{"decision":{"type":"choice","choice":"only","probabilities":{"only":1.0},"confidence":1.0}},"usage":{"input_tokens":8,"output_tokens":1}}"#
    );
}

#[tokio::test]
async fn stub_engine_fanout_usage_matches_formula() {
    let _guard = guard();
    let (status, body, _) = send(
        "POST",
        "/v1/systemone",
        Some(r#"{"model": "von", "state": {"env": "prod", "retries": 3.5}, "questions": {"a": {"type": "noul", "instructions": "Is it up?"}, "b": {"type": "score", "instructions": "Rate:", "criteria": ["low", "high"]}}}"#),
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).unwrap();
    let answers = v["answers"].as_object().unwrap();
    let keys: Vec<&String> = answers.keys().collect();
    assert_eq!(keys, ["a", "b"]);
    assert_eq!(v["model"], "von-1.1.0");
    assert_eq!(v["usage"]["output_tokens"], 2);
    assert_eq!(v["usage"]["input_tokens"], 8);
}

#[tokio::test]
async fn engine_level_422s_match_golden_messages() {
    let _guard = guard();
    let (status, body, _) = send(
        "POST",
        "/v1/systemone",
        Some(r#"{"state": "s", "questions": {"q": {"type": "bogus", "instructions": "?"}}}"#),
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        String::from_utf8_lossy(&body),
        r#"{"detail":"Unknown question type 'bogus'"}"#
    );

    let (status, body, _) = send(
        "POST",
        "/v1/systemone",
        Some(r#"{"state": "s", "questions": {"q": {"type": "score", "instructions": "i", "criteria": [{"what": "w", "examples": [1, "two"]}]}}}"#),
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        String::from_utf8_lossy(&body),
        r#"{"detail":"sequence item 0: expected str instance, int found"}"#
    );
}

#[tokio::test]
async fn auth_gate_matches_python_when_key_set() {
    let _guard = guard();
    unsafe { std::env::set_var("VON_API_KEY", "sekrit") };
    let cases: Vec<(String, Option<String>, u16, &str)> = vec![
        (
            "auth_no_header".into(),
            None,
            401,
            r#"{"detail":"Missing or invalid Bearer token"}"#,
        ),
        (
            "auth_bad_token".into(),
            Some("Bearer wrong".into()),
            401,
            r#"{"detail":"Unauthorized: invalid API key"}"#,
        ),
        (
            "auth_not_bearer".into(),
            Some("Basic sekrit".into()),
            401,
            r#"{"detail":"Missing or invalid Bearer token"}"#,
        ),
        (
            "auth_empty_token".into(),
            Some("Bearer ".into()),
            401,
            r#"{"detail":"Unauthorized: invalid API key"}"#,
        ),
    ];
    for (id, auth, status, expected) in cases {
        let (got_status, body, _) = send(
            "POST",
            "/v1/systemone",
            Some(r#"{"state": "s", "questions": {}}"#),
            auth.as_deref(),
            &[],
        )
        .await;
        assert_eq!(got_status.as_u16(), status, "{id}");
        assert_eq!(String::from_utf8_lossy(&body), expected, "{id}");
    }
    unsafe { std::env::remove_var("VON_API_KEY") };

    let (status, _, _) = send(
        "POST",
        "/v1/systemone",
        Some(r#"{"state": "s", "questions": {}}"#),
        Some("Bearer sekrit"),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn no_auth_required_without_env_key() {
    let _guard = guard();
    unsafe { std::env::remove_var("VON_API_KEY") };
    let (status, _, _) = send(
        "POST",
        "/v1/systemone",
        Some(r#"{"state": "s", "questions": {}}"#),
        None,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn cors_preflight_matches_starlette() {
    let _guard = guard();
    let router = build_router();
    let req = Request::builder()
        .method(Method::OPTIONS)
        .uri("/v1/systemone")
        .header("origin", "http://example.com")
        .header("access-control-request-method", "POST")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()["access-control-allow-origin"], "*");
    assert_eq!(
        resp.headers()["access-control-allow-methods"],
        "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"
    );
    assert_eq!(resp.headers()["access-control-max-age"], "600");
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(String::from_utf8_lossy(&body), "OK");
}

#[tokio::test]
async fn golden_200_requests_produce_correct_status_and_shape() {
    let _guard = guard();
    let req_path = fixtures_dir()
        .parent()
        .unwrap()
        .join("../goldens/requests.jsonl");
    let resp_path = fixtures_dir()
        .parent()
        .unwrap()
        .join("../goldens/responses.jsonl");
    let text = std::fs::read_to_string(&req_path).unwrap();
    let resp_text = std::fs::read_to_string(&resp_path).unwrap();
    let mut golden_status = std::collections::HashMap::new();
    for line in resp_text.lines().filter(|l| !l.trim().is_empty()) {
        let r: Value = serde_json::from_str(line).unwrap();
        golden_status.insert(
            r["id"].as_str().unwrap().to_string(),
            r["status"].as_u64().unwrap() as u16,
        );
    }
    let mut checked = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let req: Value = serde_json::from_str(line).unwrap();
        let id = req["id"].as_str().unwrap().to_string();
        if req["method"].as_str() != Some("POST") {
            continue;
        }
        let body_str = req["body"].as_str().unwrap();
        let body: Value = match serde_json::from_str(body_str) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if body.get("state").is_none() || body.get("questions").is_none() {
            continue;
        }
        if id.starts_with("auth_") {
            continue;
        }
        let (status, resp_body, _) = send("POST", "/v1/systemone", Some(body_str), None, &[]).await;
        let expected_status = golden_status
            .get(id.as_str())
            .copied()
            .unwrap_or_else(|| panic!("{id} missing from goldens/responses.jsonl"));
        assert_eq!(status.as_u16(), expected_status, "{id}");
        if expected_status != 200 {
            continue;
        }
        let v: Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(v["model"], "von-1.1.0", "{id}");
        assert!(v["usage"]["input_tokens"].as_u64().unwrap() >= 1, "{id}");
        assert_eq!(
            v["usage"]["output_tokens"],
            body["questions"].as_object().unwrap().len() as u64,
            "{id}"
        );
        checked += 1;
    }
    assert!(checked >= 40, "only checked {checked} golden requests");
}
