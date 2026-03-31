use crate::error::ApiError;
use crate::sse::SseParser;
use crate::types::{MessageRequest, MessageResponse, StreamEvent};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    auth_token: Option<String>,
    base_url: String,
}

impl AnthropicClient {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            auth_token: None,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        Ok(Self::new(read_api_key(|key| std::env::var(key))?)
            .with_auth_token(std::env::var("ANTHROPIC_AUTH_TOKEN").ok())
            .with_base_url(
                std::env::var("ANTHROPIC_BASE_URL")
                    .ok()
                    .or_else(|| std::env::var("CLAUDE_CODE_API_BASE_URL").ok())
                    .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            ))
    }

    #[must_use]
    pub fn with_auth_token(mut self, auth_token: Option<String>) -> Self {
        self.auth_token = auth_token.filter(|token| !token.is_empty());
        self
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let request = MessageRequest {
            stream: false,
            ..request.clone()
        };
        let response = self.send_raw_request(&request).await?;
        let response = expect_success(response).await?;
        response
            .json::<MessageResponse>()
            .await
            .map_err(ApiError::from)
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        let response = self
            .send_raw_request(&request.clone().with_streaming())
            .await?;
        let response = expect_success(response).await?;
        Ok(MessageStream {
            response,
            parser: SseParser::new(),
            pending: std::collections::VecDeque::new(),
            done: false,
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut request_builder = self
            .http
            .post(format!(
                "{}/v1/messages",
                self.base_url.trim_end_matches('/')
            ))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json");

        if let Some(auth_token) = &self.auth_token {
            request_builder = request_builder.bearer_auth(auth_token);
        }

        request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)
    }
}

fn read_api_key(
    getter: impl FnOnce(&str) -> Result<String, std::env::VarError>,
) -> Result<String, ApiError> {
    match getter("ANTHROPIC_API_KEY") {
        Ok(api_key) if api_key.is_empty() => Err(ApiError::MissingApiKey),
        Ok(api_key) => Ok(api_key),
        Err(std::env::VarError::NotPresent) => Err(ApiError::MissingApiKey),
        Err(error) => Err(ApiError::from(error)),
    }
}

#[derive(Debug)]
pub struct MessageStream {
    response: reqwest::Response,
    parser: SseParser,
    pending: std::collections::VecDeque<StreamEvent>,
    done: bool,
}

impl MessageStream {
    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            if self.done {
                let remaining = self.parser.finish()?;
                self.pending.extend(remaining);
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }

            match self.response.chunk().await? {
                Some(chunk) => {
                    self.pending.extend(self.parser.push(&chunk)?);
                }
                None => {
                    self.done = true;
                }
            }
        }
    }
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_else(|_| String::new());
    Err(ApiError::UnexpectedStatus { status, body })
}

#[cfg(test)]
mod tests {
    use std::env::VarError;

    use crate::types::MessageRequest;

    #[test]
    fn read_api_key_requires_presence() {
        let error = super::read_api_key(|_| Err(VarError::NotPresent))
            .expect_err("missing key should error");
        assert!(matches!(error, crate::error::ApiError::MissingApiKey));
    }

    #[test]
    fn read_api_key_requires_non_empty_value() {
        let error = super::read_api_key(|_| Ok(String::new())).expect_err("empty key should error");
        assert!(matches!(error, crate::error::ApiError::MissingApiKey));
    }

    #[test]
    fn with_auth_token_drops_empty_values() {
        let client = super::AnthropicClient::new("test-key").with_auth_token(Some(String::new()));
        assert!(client.auth_token.is_none());
    }

    #[test]
    fn message_request_stream_helper_sets_stream_true() {
        let request = MessageRequest {
            model: "claude-3-7-sonnet-latest".to_string(),
            max_tokens: 64,
            messages: vec![],
            system: None,
            stream: false,
        };

        assert!(request.with_streaming().stream);
    }
}
