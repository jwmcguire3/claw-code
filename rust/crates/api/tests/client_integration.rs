use std::collections::HashMap;
use std::sync::Arc;

use api::{AnthropicClient, InputMessage, MessageRequest, OutputContentBlock, StreamEvent};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[tokio::test]
async fn send_message_posts_json_and_parses_response() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let body = concat!(
        "{",
        "\"id\":\"msg_test\",",
        "\"type\":\"message\",",
        "\"role\":\"assistant\",",
        "\"content\":[{\"type\":\"text\",\"text\":\"Hello from Claude\"}],",
        "\"model\":\"claude-3-7-sonnet-latest\",",
        "\"stop_reason\":\"end_turn\",",
        "\"stop_sequence\":null,",
        "\"usage\":{\"input_tokens\":12,\"output_tokens\":4}",
        "}"
    );
    let server = spawn_server(state.clone(), http_response("application/json", body)).await;

    let client = AnthropicClient::new("test-key")
        .with_auth_token(Some("proxy-token".to_string()))
        .with_base_url(server.base_url());
    let response = client
        .send_message(&sample_request(false))
        .await
        .expect("request should succeed");

    assert_eq!(response.id, "msg_test");
    assert_eq!(
        response.content,
        vec![OutputContentBlock::Text {
            text: "Hello from Claude".to_string(),
        }]
    );

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/messages");
    assert_eq!(
        request.headers.get("x-api-key").map(String::as_str),
        Some("test-key")
    );
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer proxy-token")
    );
    assert_eq!(
        request.headers.get("anthropic-version").map(String::as_str),
        Some("2023-06-01")
    );
    let body: serde_json::Value =
        serde_json::from_str(&request.body).expect("request body should be json");
    assert_eq!(
        body.get("model").and_then(serde_json::Value::as_str),
        Some("claude-3-7-sonnet-latest")
    );
    assert!(
        body.get("stream").is_none(),
        "non-stream request should omit stream=false"
    );
}

#[tokio::test]
async fn stream_message_parses_sse_events() {
    let state = Arc::new(Mutex::new(Vec::<CapturedRequest>::new()));
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-7-sonnet-latest\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
        "data: [DONE]\n\n"
    );
    let server = spawn_server(state.clone(), http_response("text/event-stream", sse)).await;

    let client = AnthropicClient::new("test-key")
        .with_auth_token(Some("proxy-token".to_string()))
        .with_base_url(server.base_url());
    let mut stream = client
        .stream_message(&sample_request(false))
        .await
        .expect("stream should start");

    let mut events = Vec::new();
    while let Some(event) = stream
        .next_event()
        .await
        .expect("stream event should parse")
    {
        events.push(event);
    }

    assert_eq!(events.len(), 5);
    assert!(matches!(events[0], StreamEvent::MessageStart(_)));
    assert!(matches!(events[1], StreamEvent::ContentBlockStart(_)));
    assert!(matches!(events[2], StreamEvent::ContentBlockDelta(_)));
    assert!(matches!(events[3], StreamEvent::ContentBlockStop(_)));
    assert!(matches!(events[4], StreamEvent::MessageStop(_)));

    let captured = state.lock().await;
    let request = captured.first().expect("server should capture request");
    assert!(request.body.contains("\"stream\":true"));
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and network access"]
async fn live_stream_smoke_test() {
    let client = AnthropicClient::from_env().expect("ANTHROPIC_API_KEY must be set");
    let mut stream = client
        .stream_message(&MessageRequest {
            model: std::env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-3-7-sonnet-latest".to_string()),
            max_tokens: 32,
            messages: vec![InputMessage::user_text(
                "Reply with exactly: hello from rust",
            )],
            system: None,
            stream: false,
        })
        .await
        .expect("live stream should start");

    let mut saw_start = false;
    let mut saw_follow_up = false;
    let mut event_kinds = Vec::new();
    while let Some(event) = stream
        .next_event()
        .await
        .expect("live stream should yield events")
    {
        match event {
            StreamEvent::MessageStart(_) => {
                saw_start = true;
                event_kinds.push("message_start");
            }
            StreamEvent::ContentBlockStart(_) => {
                saw_follow_up = true;
                event_kinds.push("content_block_start");
            }
            StreamEvent::ContentBlockDelta(_) => {
                saw_follow_up = true;
                event_kinds.push("content_block_delta");
            }
            StreamEvent::ContentBlockStop(_) => {
                saw_follow_up = true;
                event_kinds.push("content_block_stop");
            }
            StreamEvent::MessageStop(_) => {
                saw_follow_up = true;
                event_kinds.push("message_stop");
            }
        }
    }

    assert!(
        saw_start,
        "expected a message_start event; got {event_kinds:?}"
    );
    assert!(
        saw_follow_up,
        "expected at least one follow-up stream event; got {event_kinds:?}"
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

struct TestServer {
    base_url: String,
    join_handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    fn base_url(&self) -> String {
        self.base_url.clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.join_handle.abort();
    }
}

async fn spawn_server(state: Arc<Mutex<Vec<CapturedRequest>>>, response: String) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener should bind");
    let address = listener
        .local_addr()
        .expect("listener should have local addr");
    let join_handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("server should accept");
        let mut buffer = Vec::new();
        let mut header_end = None;

        loop {
            let mut chunk = [0_u8; 1024];
            let read = socket
                .read(&mut chunk)
                .await
                .expect("request read should succeed");
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(position) = find_header_end(&buffer) {
                header_end = Some(position);
                break;
            }
        }

        let header_end = header_end.expect("request should include headers");
        let (header_bytes, remaining) = buffer.split_at(header_end);
        let header_text = String::from_utf8(header_bytes.to_vec()).expect("headers should be utf8");
        let mut lines = header_text.split("\r\n");
        let request_line = lines.next().expect("request line should exist");
        let mut parts = request_line.split_whitespace();
        let method = parts.next().expect("method should exist").to_string();
        let path = parts.next().expect("path should exist").to_string();
        let mut headers = HashMap::new();
        let mut content_length = 0_usize;
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let (name, value) = line.split_once(':').expect("header should have colon");
            let value = value.trim().to_string();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().expect("content length should parse");
            }
            headers.insert(name.to_ascii_lowercase(), value);
        }

        let mut body = remaining[4..].to_vec();
        while body.len() < content_length {
            let mut chunk = vec![0_u8; content_length - body.len()];
            let read = socket
                .read(&mut chunk)
                .await
                .expect("body read should succeed");
            if read == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..read]);
        }

        state.lock().await.push(CapturedRequest {
            method,
            path,
            headers,
            body: String::from_utf8(body).expect("body should be utf8"),
        });

        socket
            .write_all(response.as_bytes())
            .await
            .expect("response write should succeed");
    });

    TestServer {
        base_url: format!("http://{address}"),
        join_handle,
    }
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn http_response(content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn sample_request(stream: bool) -> MessageRequest {
    MessageRequest {
        model: "claude-3-7-sonnet-latest".to_string(),
        max_tokens: 64,
        messages: vec![InputMessage::user_text("Say hello")],
        system: None,
        stream,
    }
}
