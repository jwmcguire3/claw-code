use std::env::VarError;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum ApiError {
    MissingApiKey,
    InvalidApiKeyEnv(VarError),
    Http(reqwest::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    UnexpectedStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    InvalidSseFrame(&'static str),
}

impl Display for ApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingApiKey => {
                write!(
                    f,
                    "ANTHROPIC_API_KEY is not set; export it before calling the Anthropic API"
                )
            }
            Self::InvalidApiKeyEnv(error) => {
                write!(f, "failed to read ANTHROPIC_API_KEY: {error}")
            }
            Self::Http(error) => write!(f, "http error: {error}"),
            Self::Io(error) => write!(f, "io error: {error}"),
            Self::Json(error) => write!(f, "json error: {error}"),
            Self::UnexpectedStatus { status, body } => {
                write!(f, "anthropic api returned {status}: {body}")
            }
            Self::InvalidSseFrame(message) => write!(f, "invalid sse frame: {message}"),
        }
    }
}

impl std::error::Error for ApiError {}

impl From<reqwest::Error> for ApiError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<std::io::Error> for ApiError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<VarError> for ApiError {
    fn from(value: VarError) -> Self {
        Self::InvalidApiKeyEnv(value)
    }
}
