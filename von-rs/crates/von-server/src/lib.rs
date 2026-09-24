//! FastAPI-compatible von decision server (Stage 2: real engine via
//! `build_engine_router`, stub retained for wire-gate tests).

pub mod pyjson_scan;
pub mod validate;

use std::env;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::Response;
use axum::routing::MethodRouter;
use von_core::{Engine, StubEngine};

#[derive(Clone, Debug)]
pub struct CorsConfig {
    wildcard: bool,
    origins: Vec<String>,
}

impl CorsConfig {
    pub fn from_env() -> Self {
        let raw = env::var("VON_CORS_ORIGINS").unwrap_or_else(|_| "*".to_string());
        let origins: Vec<String> = raw
            .split(',')
            .map(|o| o.trim().to_string())
            .filter(|o| !o.is_empty())
            .collect();
        let wildcard = origins == ["*"];
        CorsConfig { wildcard, origins }
    }
}

/// Stub-engine router (Stage 1 wire-gate fixture; not for deployment).
pub fn build_router() -> Router {
    build_engine_router(Arc::new(StubEngine))
}

/// Router backed by any engine — the Stage 2 wiring point for the ONNX
/// backend.
pub fn build_engine_router(engine: Arc<dyn Engine>) -> Router {
    let router = Router::new()
        .route("/", health_route())
        .route("/health", health_route())
        .route("/v1/models", models_route())
        .route("/v1/systemone", system_one_route())
        .fallback(not_found)
        .with_state(engine);
    router.layer(axum::middleware::from_fn(cors_middleware))
}

fn detail_response(status: StatusCode, body: String) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

fn method_not_allowed(allow: &str, method: &Method) -> Response {
    let builder = Response::builder()
        .status(StatusCode::METHOD_NOT_ALLOWED)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ALLOW, allow);
    if *method == Method::HEAD {
        return builder.body(Body::empty()).unwrap();
    }
    builder
        .body(Body::from(r#"{"detail":"Method Not Allowed"}"#))
        .unwrap()
}

async fn not_found() -> Response {
    detail_response(
        StatusCode::NOT_FOUND,
        r#"{"detail":"Not Found"}"#.to_string(),
    )
}

fn health_body() -> String {
    r#"{"status":"ok","service":"von-decision-server","version":"1.1.0","engine":"von-1.1","homage":"John von Neumann & Ludwig von Mises"}"#
        .to_string()
}

async fn health_handler(method: axum::http::Method) -> Response {
    if method == Method::HEAD {
        return method_not_allowed("GET", &method);
    }
    detail_response(StatusCode::OK, health_body())
}

fn health_route<S>() -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    MethodRouter::new()
        .get(health_handler)
        .fallback(|method: axum::http::Method| async move { method_not_allowed("GET", &method) })
}

async fn models_handler(method: axum::http::Method) -> Response {
    if method == Method::HEAD {
        return method_not_allowed("GET", &method);
    }
    let body = serde_json::json!({
        "models": [
            {"name": "von-latest", "description": "Current Von System One decision model", "release_date": "2026-09-21"},
            {"name": "von-1.1.0", "description": "Von 1.1 stable release", "release_date": "2026-09-21"},
            {"name": "jev-latest", "description": "TypeSafe Jev compatibility alias", "release_date": "2026-09-21"}
        ],
        "object": "list",
        "data": [
            {"id": "von-latest", "object": "model", "owned_by": "von"},
            {"id": "von-1.1.0", "object": "model", "owned_by": "von"},
            {"id": "jev-latest", "object": "model", "owned_by": "von"}
        ]
    });
    detail_response(StatusCode::OK, body.to_string())
}

fn models_route<S>() -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    MethodRouter::new()
        .get(models_handler)
        .fallback(|method: axum::http::Method| async move { method_not_allowed("GET", &method) })
}

fn check_auth(headers: &HeaderMap) -> Option<Response> {
    let expected_key = env::var("VON_API_KEY").ok()?;
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let Some(auth) = auth else {
        return Some(detail_response(
            StatusCode::UNAUTHORIZED,
            r#"{"detail":"Missing or invalid Bearer token"}"#.to_string(),
        ));
    };
    if !auth.starts_with("Bearer ") {
        return Some(detail_response(
            StatusCode::UNAUTHORIZED,
            r#"{"detail":"Missing or invalid Bearer token"}"#.to_string(),
        ));
    }
    let token = auth
        .split_once("Bearer ")
        .map(|(_, rest)| rest)
        .unwrap_or("")
        .trim();
    if token != expected_key {
        return Some(detail_response(
            StatusCode::UNAUTHORIZED,
            r#"{"detail":"Unauthorized: invalid API key"}"#.to_string(),
        ));
    }
    None
}

async fn system_one_handler(State(engine): State<Arc<dyn Engine>>, request: Request) -> Response {
    // Ordering mirrors FastAPI: the request body is parsed and validated by
    // the framework before the endpoint function (where the auth check lives
    // in von's Python server) ever runs, so a malformed/invalid body yields
    // 422 even when the request would also fail auth. Auth second, then eval.
    let (parts, body) = request.into_parts();
    let body = match axum::body::to_bytes(body, 64 * 1024 * 1024).await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => {
            return detail_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validate::json_invalid_body(&pyjson_scan::JsonScanError {
                    msg: "Expecting value".to_string(),
                    pos: 0,
                }),
            );
        }
    };
    // FastAPI checks a missing body before any content-type dispatch: an
    // empty body is "Field required" regardless of headers.
    if body.is_empty() {
        return detail_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            validate::missing_body(),
        );
    }
    if !validate::content_type_is_json(
        parts.headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()),
    ) {
        // Non-JSON content types bypass request.json() entirely: the raw
        // body reaches pydantic as bytes and fails the model check.
        return detail_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            validate::model_attributes_type_body(&body),
        );
    }
    let parsed = validate::validate_body(&body);
    match parsed {
        validate::ParsedRequest::Err(detail) => {
            detail_response(StatusCode::UNPROCESSABLE_ENTITY, detail)
        }
        validate::ParsedRequest::Ok(req) => {
            if let Some(resp) = check_auth(&parts.headers) {
                return resp;
            }
            match engine.evaluate(
                &req.state,
                &req.questions,
                Some(von_core::resolved_model_id().as_str()),
            ) {
                Ok(resp) => detail_response(StatusCode::OK, serde_json::to_string(&resp).unwrap()),
                Err(e) => detail_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    serde_json::to_string(&serde_json::json!({"detail": e.0})).unwrap(),
                ),
            }
        }
    }
}

fn system_one_route() -> MethodRouter<Arc<dyn Engine>> {
    MethodRouter::new()
        .post(system_one_handler)
        .fallback(|method: axum::http::Method| async move { method_not_allowed("POST", &method) })
}

async fn cors_middleware(req: Request, next: axum::middleware::Next) -> Response {
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let is_preflight = req.method() == Method::OPTIONS
        && origin.is_some()
        && req.headers().contains_key("access-control-request-method");

    if is_preflight {
        return preflight_response(&origin);
    }

    let mut response = next.run(req).await;
    if let Some(origin) = origin {
        append_cors_headers(&mut response, &origin);
    }
    response
}

fn preflight_response(origin: &Option<String>) -> Response {
    let cors = CorsConfig::from_env();
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .header(
            "access-control-allow-methods",
            "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
        )
        .header("access-control-max-age", "600");
    if cors.wildcard {
        builder = builder.header("access-control-allow-origin", "*");
    } else if let Some(origin) = origin.as_ref().filter(|o| cors.origins.contains(o)) {
        builder = builder
            .header("access-control-allow-origin", origin)
            .header("access-control-allow-credentials", "true")
            .header(header::VARY, "Origin");
    }
    builder.body(Body::from("OK")).unwrap()
}

fn append_cors_headers(response: &mut Response, origin: &str) {
    let cors = CorsConfig::from_env();
    let headers = response.headers_mut();
    if cors.wildcard {
        headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
    } else if cors.origins.iter().any(|o| o == origin) {
        if let Ok(v) = HeaderValue::from_str(origin) {
            headers.insert("access-control-allow-origin", v);
        }
        headers.insert(
            "access-control-allow-credentials",
            HeaderValue::from_static("true"),
        );
        headers.insert(header::VARY, HeaderValue::from_static("Origin"));
    }
}

pub async fn serve_router(
    listener: tokio::net::TcpListener,
    router: Router,
) -> std::io::Result<()> {
    axum::serve(listener, router).await
}
